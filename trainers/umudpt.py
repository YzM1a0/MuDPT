import torch
import sys

sys.path.append("..")
import clip
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from yacs.config import CfgNode
from torch.cuda.amp import GradScaler, autocast
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip.simple_tokenizer import SimpleTokenizer as Tokenizer

_tokenizer = Tokenizer()


def load_clip_to_cpu(cfg: CfgNode = None):
    if not cfg.MODEL.BACKBONE.PATH:
        backbone_name = cfg.MODEL.BACKBONE.NAME
        url = clip._MODELS[backbone_name]
        print(f"Download backbone: {backbone_name} from {url}")
        model_path = clip._download(url)
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = clip.build_model(state_dict or model.state_dict(), cfg=cfg)
    else:
        model_path = cfg.MODEL.BACKBONE.PATH
        print(f"Loading CLIP backbone: {cfg.MODEL.BACKBONE.NAME} from {model_path}")
        model, preprocess = clip.load(model_path, device="cpu", cfg=cfg)

    return model


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LightTransformer(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class UMuDPTPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)  # CLASS NAMES
        n_ctx = cfg.TRAINER.UMUDPT.N_CTX  # CONTEXT LENGTH
        ctx_init = cfg.TRAINER.UMUDPT.CTX_INIT  # INITIALIZE CONTEXT
        dtype = clip_model.dtype  # torch.float32
        ctx_dim = clip_model.ln_final.weight.shape[0]  # context dimension: 512
        clip_imsize = clip_model.visual.input_resolution  # clip image size: 224
        cfg_imsize = cfg.INPUT.SIZE[0]  # config image size: 224
        visual_ctx_dim = clip_model.visual.positional_embedding.shape[1]

        assert cfg.TRAINER.UMUDPT.DEEP_PROMPT_DEPTH > 0, "PROMPT_DEPTH should be > 0"
        self.deep_prompts_depth = cfg.TRAINER.UMUDPT.DEEP_PROMPT_DEPTH

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = " ".join(ctx_init.split()[:n_ctx])
        else:
            # random initialization
            print(f"Initializing A Generic Context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(['X'] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Depth of deep prompt: {self.deep_prompts_depth}")

        # initialize deep prompts (n_ctx, nctx_dim)
        self.deep_prompts = nn.Parameter(torch.empty(self.deep_prompts_depth - 1, n_ctx, ctx_dim))
        nn.init.normal_(self.deep_prompts, std=0.02)

        # light transformer for t2v prompts
        self.ln_pre = LayerNorm(ctx_dim)
        self.self_attn = LightTransformer(d_model=ctx_dim, n_head=ctx_dim // 64)
        self.ln_post = LayerNorm(ctx_dim)
        self.visual_proj = nn.Linear(in_features=ctx_dim, out_features=visual_ctx_dim)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts], dim=0)  # (n_cls, 77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # (n_cls, 77, ctx_dim)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS . EOS ~

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.dtype = dtype
        self.ctx_dim = ctx_dim
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, ctx_dim)
                ctx,  # (n_cls, n_ctx, ctx_dim)
                suffix,  # (n_cls, *, ctx_dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx  # context parameters
        if ctx.dim() == 2:  # generic context (n_ctx, n_dim)
            ctx = ctx.unsqueeze(0).expand(self.n_cls, self.n_ctx, self.ctx_dim)

        prefix = self.token_prefix  # SOS
        suffix = self.token_suffix  # CLS . EOS ~
        ctx = self.construct_prompts(ctx, prefix, suffix)

        visual_prompts = torch.cat([self.ctx.unsqueeze(0), self.deep_prompts], dim=0)
        visual_prompts = self.ln_pre(visual_prompts)
        visual_prompts = visual_prompts.permute(1, 0, 2)
        visual_prompts = self.self_attn(visual_prompts)
        visual_prompts = visual_prompts.permute(1, 0, 2)
        visual_prompts = self.ln_post(visual_prompts)
        visual_prompts = self.visual_proj(visual_prompts)

        return ctx, self.deep_prompts, visual_prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, deep_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        combined = [x, deep_prompts, 0]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.umudpt_prompt_learner = UMuDPTPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.umudpt_prompt_learner.tokenized_prompts  # (n_cls, 77)
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts  # (n_cls, 77)
        prompts, text_deep_prompts, visual_deep_prompts = self.umudpt_prompt_learner()

        image_features = self.image_encoder(image.type(self.dtype), visual_deep_prompts[:1, :, :], visual_deep_prompts[1:, :, :])
        text_features = self.text_encoder(prompts, tokenized_prompts, text_deep_prompts)  # (n_cls, 1024)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class UMuDPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.UMUDPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.UMUDPT.PREC in ["fp32", "amp"]:
            clip_model.float()

        print(f"Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_optimize = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_optimize not in name:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("UnifiedMultimodalDeepPromptTuning", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.UMUDPT.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus = {device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.UMUDPT.PREC

        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            # "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "umudpt_prompt_learner.token_prefix" in state_dict:
                del state_dict["umudpt_prompt_learner.token_prefix"]

            if "umudpt_prompt_learner.token_suffix" in state_dict:
                del state_dict["umudpt_prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
