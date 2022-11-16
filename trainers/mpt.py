import sys
import torch
sys.path.append("..")

import clip
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from clip import clip
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

        model = clip.build_model(state_dict or model.state_dict())
    else:
        model_path = cfg.MODEL.BACKBONE.PATH
        print(f"Loading CLIP backbone: {cfg.MODEL.BACKBONE.NAME} from {model_path}")
        model, preprocess = clip.load(model_path, device="cpu", cfg=cfg)

    return model


class TextPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        text_n_ctx = cfg.TRAINER.MPT.DEEP_TEXT_N_CTX
        ctx_init = cfg.TRAINER.MPT.TEXT_CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = text_n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(text_n_ctx, ctx_dim, dtype=dtype)  # (16, 512)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * text_n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) of deep visual prompt: {cfg.TRAINER.MPT.DEEP_VISUAL_N_CTX}")
        print(f"Number of context words (tokens) of deep text prompt: {cfg.TRAINER.MPT.DEEP_TEXT_N_CTX}")
        print(f"Number of depth of deep visual prompt: {cfg.TRAINER.MPT.VISUAL_PROMPT_DEPTH}")
        print(f"Number of depth of deep text prompt: {cfg.TRAINER.MPT.TEXT_PROMPT_DEPTH}")

        self.visual_ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + text_n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = text_n_ctx
        self.tokenized_prompted = tokenized_prompts

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
        ctx = self.visual_ctx  # (16, 512)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # (37, 16, 512)

        prefix = self.token_prefix  # (37, 1, 512)
        suffix = self.token_suffix  # (37, 60, 512)
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding  # (77, 512)
        self.ln_final = clip_model.ln_final  # LayerNorm((512,))
        self.text_projection = clip_model.text_projection  # (512, 1024)
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)  # (37, 77, 512) + (77, 512)
        x = x.permute(1, 0, 2)  # NLD -> LND (37, 77, 512) -> (77, 37, 512)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (77, 37, 512) -> (37, 77, 512)
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        t = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]  # (37, 512)
        x = t @ self.text_projection

        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.text_prompt_learner = TextPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.text_prompt_learner.tokenized_prompted
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts  # (n_cls, 77)
        prompts = self.text_prompt_learner()  # (n_cls, 77, 512)

        image_features = self.image_encoder(image.type(self.dtype))
        text_features = self.text_encoder(prompts, tokenized_prompts)  # (n_cls, 1024)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class MPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MPT.PREC in ["fp32", "amp"]:
            clip_model.float()

        print(f"Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_optimize = "visual_ctx"
        for name, param in self.model.named_parameters():
            if name_to_optimize not in name:
                if "ctx" in name:
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
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MPT.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus = {device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.MPT.PREC

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
            if "text_prompt_learner.token_prefix" in state_dict:
                del state_dict["text_prompt_learner.token_prefix"]

            if "text_prompt_learner.token_suffix" in state_dict:
                del state_dict["text_prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)