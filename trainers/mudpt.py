import torch
import sys
sys.path.append("..")
import copy
import clip
import os.path as osp
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from collections import OrderedDict
from torch.cuda.amp import GradScaler, autocast
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip.simple_tokenizer import SimpleTokenizer as Tokenizer

_tokenizer = Tokenizer()


def load_clip_to_cpu(cfg):
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


class MuDPTPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)  # CLASS NAMES
        n_ctx = cfg.TRAINER.MUDPT.N_CTX  # CONTEXT LENGTH
        ctx_init = cfg.TRAINER.MUDPT.CTX_INIT  # INITIALIZE CONTEXT
        dtype = clip_model.dtype  # torch.float32
        ctx_dim = clip_model.ln_final.weight.shape[0]  # context dimension: 512
        clip_imsize = clip_model.visual.input_resolution  # clip image size: 224
        cfg_imsize = cfg.INPUT.SIZE[0]  # config image size: 224

        assert cfg.TRAINER.MUDPT.DEEP_PROMPT_DEPTH > 0, "PROMPT_DEPTH should be > 0"
        self.deep_prompts_depth = cfg.TRAINER.MUDPT.DEEP_PROMPT_DEPTH

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

        self.embed_projection = nn.Linear(in_features=ctx_dim, out_features=clip_model.visual.positional_embedding.shape[1])
        self.deep_prompts = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_dim)) for _ in range(self.deep_prompts_depth - 1)])
        for parameter in self.deep_prompts:
            nn.init.normal_(parameter, std=0.02)

        layer = nn.Linear(ctx_dim, clip_model.visual.proj.shape[0])
        self.deep_projections = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.deep_prompts_depth - 1)])

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts], dim=0)  # (n_cls, 77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # (n_cls, 77, ctx_dim)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS . EOS ~

        self.n_cls = n_cls
        self.n_ctx = n_ctx
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
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform prompts to 768 for the visual side
        visual_prompts = []
        for index, layer in enumerate(self.deep_projections):
            visual_prompts.append(layer(self.deep_prompts[index]))

        t2v_shared_ctx = self.embed_projection(self.ctx)

        return prompts, t2v_shared_ctx, self.deep_prompts, visual_prompts


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
        self.mudpt_prompt_learner = MuDPTPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.mudpt_prompt_learner.tokenized_prompts  # (n_cls, 77)
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.deep_prompts_depth = self.mudpt_prompt_learner.deep_prompts_depth

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts  # (n_cls, 77)
        prompts, shared_ctx, text_deep_prompts, t2v_visual_prompts = self.mudpt_prompt_learner()

        image_features, text_prompts = self.image_encoder(image.type(self.dtype), t2v_visual_prompts)
        for index in range(self.deep_prompts_depth - 1):
            text_prompts[index] = text_deep_prompts[index] + text_prompts[index]

        text_features = self.text_encoder(prompts, tokenized_prompts, text_prompts)  # (n_cls, 1024)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class MuDPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MUDPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MUDPT.PREC in ["fp32", "amp"]:
            clip_model.float()

        print(f"Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_optimize = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_optimize not in name:
                if "visual_ctx" in name:
                    param.requires_grad_(True)
                else:
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
        self.register_model("MultimodalDeepPromptTuning", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.MUDPT.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus = {device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.MUDPT.PREC

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
            "acc": compute_accuracy(output, label)[0].item(),
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
            if "mudpt_prompt_learner.token_prefix" in state_dict:
                del state_dict["mudpt_prompt_learner.token_prefix"]

            if "mudpt_prompt_learner.token_suffix" in state_dict:
                del state_dict["mudpt_prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)