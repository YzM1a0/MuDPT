import sys

from collections import OrderedDict
from typing import Tuple, Union
from yacs.config import CfgNode
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Optional, List
from trainers.uumudpt import LightTransformer

sys.path.append("..")


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
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


class ResidualAttentionBlock_VPT(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, nth_layer: int = 0, is_text_layer: bool = False, need_prompt: bool = False, cfg: CfgNode = None):
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

        self.is_text_layer = is_text_layer
        if nth_layer > 0:
            self.need_prompt = need_prompt
            if self.need_prompt:
                if self.is_text_layer:
                    self.text_n_ctx = eval(f"cfg.TRAINER.{cfg.TRAINER.NAME}.DEEP_TEXT_N_CTX")
                    ctx_vectors = torch.empty(self.text_n_ctx, d_model)
                else:
                    self.img_n_ctx = eval(f"cfg.TRAINER.{cfg.TRAINER.NAME}.DEEP_VISUAL_N_CTX")
                    ctx_vectors = torch.empty(self.img_n_ctx, d_model)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.visual_ctx = nn.Parameter(ctx_vectors)

        else:
            self.need_prompt = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        if self.need_prompt:
            if self.is_text_layer:
                if self.text_n_ctx > 0:
                    prefix = x[:1, :, :]
                    suffix = x[1 + self.text_n_ctx:, :, :]
                    textual_context = self.visual_ctx.to(x.dtype) + torch.zeros(x.shape[1], self.visual_ctx.shape[0], self.visual_ctx.shape[1], dtype=x.dtype, device=x.device)
                    textual_context = textual_context.permute(1, 0, 2)
                    x = torch.cat([prefix, textual_context, suffix], dim=0)
            else:
                prefix = x[0: x.shape[0] - self.img_n_ctx, :, :]
                visual_ctx = self.visual_ctx.to(x.dtype) + torch.zeros(x.shape[1], self.visual_ctx.shape[0], self.visual_ctx.shape[1], dtype=x.dtype, device=x.device)
                visual_ctx = visual_ctx.permute(1, 0, 2)
                x = torch.cat([prefix, visual_ctx], dim=0)
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_MuDPT(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, nth_layer: int = 0, is_text_layer: bool = False, cfg: CfgNode = None):
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

        self.is_text_layer = is_text_layer
        self.prompt_nctx = eval(f"cfg.TRAINER.{str.upper(cfg.TRAINER.NAME)}.N_CTX")
        self.is_first_layer = True if nth_layer == 0 else False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs: Optional[List]):
        x = inputs[0]  # text: (77, 100, 512)  visual: (201, 16, 768)
        prompts = inputs[1]
        nth_layer = inputs[2]
        if not self.is_first_layer:
            if prompts.shape[0] > 0:
                if self.is_text_layer:
                    if nth_layer < prompts.shape[0]:
                        prefix = x[:1, :, :]  # text: (1, 100, 512)
                        suffix = x[1 + self.prompt_nctx:, :, :]
                        text_ctx = prompts[nth_layer: nth_layer + 1, :, :]
                        text_ctx = text_ctx.expand(x.shape[1], -1, -1).to(x.dtype).to(x.device)
                        text_ctx = text_ctx.permute(1, 0, 2)
                        x = torch.cat([prefix, text_ctx, suffix], dim=0)
                        nth_layer += 1
                else:
                    if nth_layer < prompts.shape[0]:
                        prefix = x[: x.shape[0] - self.prompt_nctx, :, :]
                        visual_ctx = prompts[nth_layer: nth_layer + 1, :, :]
                        visual_ctx = visual_ctx.expand(x.shape[1], -1, -1).to(x.dtype).to(x.device)
                        visual_ctx = visual_ctx.permute(1, 0, 2)
                        x = torch.cat([prefix, visual_ctx], dim=0)
                        nth_layer += 1

        x = x + self.attention(self.ln_1(x))  # text: (77, 100, 512)
        x = x + self.mlp(self.ln_2(x))
        return [x, prompts, nth_layer]


class ResidualAttentionBlock_UMuDPT(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, nth_layer: int = 0, is_text_layer: bool = False, cfg: CfgNode = None):
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

        self.is_text_layer = is_text_layer
        self.prompt_nctx = eval(f"cfg.TRAINER.{str.upper(cfg.TRAINER.NAME)}.N_CTX")
        self.is_first_layer = True if nth_layer == 0 else False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs: Optional[List]):
        x = inputs[0]  # text: (77, 100, 512)  visual: (201, 16, 768)
        prompts = inputs[1]
        nth_layer = inputs[2]
        if not self.is_first_layer:
            if prompts.shape[0] > 0:
                if self.is_text_layer:
                    if nth_layer < prompts.shape[0]:
                        prefix = x[:1, :, :]  # text: (1, 100, 512)
                        suffix = x[1 + self.prompt_nctx:, :, :]
                        text_ctx = prompts[nth_layer: nth_layer + 1, :, :]
                        text_ctx = text_ctx.expand(x.shape[1], -1, -1).to(x.dtype).to(x.device)
                        text_ctx = text_ctx.permute(1, 0, 2)
                        x = torch.cat([prefix, text_ctx, suffix], dim=0)
                        nth_layer += 1
                else:
                    if nth_layer < prompts.shape[0]:
                        prefix = x[: x.shape[0] - self.prompt_nctx, :, :]
                        visual_ctx = prompts[nth_layer: nth_layer + 1, :, :]
                        visual_ctx = visual_ctx.expand(x.shape[1], -1, -1).to(x.dtype).to(x.device)
                        visual_ctx = visual_ctx.permute(1, 0, 2)
                        x = torch.cat([prefix, visual_ctx], dim=0)
                        nth_layer += 1

        x = x + self.attention(self.ln_1(x))  # text: (77, 100, 512)
        x = x + self.mlp(self.ln_2(x))
        return [x, prompts, nth_layer]


class ResidualAttentionBlock_UUMuDPT(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, nth_layer: int = 0, is_text_layer: bool = False, cfg: CfgNode = None):
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

        self.is_text_layer = is_text_layer
        self.prompt_nctx = eval(f"cfg.TRAINER.{str.upper(cfg.TRAINER.NAME)}.N_CTX")
        self.is_first_layer = True if nth_layer == 0 else False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs: Optional[List]):
        x = inputs[0]  # text: (77, 100, 512)  visual: (201, 16, 768)
        prompts = inputs[1]
        nth_layer = inputs[2]
        if not self.is_first_layer:
            if prompts.shape[0] > 0:
                if self.is_text_layer:
                    if nth_layer < prompts.shape[0]:
                        prefix = x[:1, :, :]  # text: (1, 100, 512)
                        suffix = x[1 + self.prompt_nctx:, :, :]
                        text_ctx = prompts[nth_layer: nth_layer + 1, :, :]
                        text_ctx = text_ctx.expand(x.shape[1], -1, -1).to(x.dtype).to(x.device)
                        text_ctx = text_ctx.permute(1, 0, 2)
                        x = torch.cat([prefix, text_ctx, suffix], dim=0)
                        nth_layer += 1
                else:
                    if nth_layer < prompts.shape[0]:
                        prefix = x[: x.shape[0] - self.prompt_nctx, :, :]
                        visual_ctx = prompts[nth_layer: nth_layer + 1, :, :]
                        visual_ctx = visual_ctx.expand(x.shape[1], -1, -1).to(x.dtype).to(x.device)
                        visual_ctx = visual_ctx.permute(1, 0, 2)
                        x = torch.cat([prefix, visual_ctx], dim=0)
                        nth_layer += 1

        x = x + self.attention(self.ln_1(x))  # text: (77, 100, 512)
        x = x + self.mlp(self.ln_2(x))
        return [x, prompts, nth_layer]


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompt_depth: int = 0, is_text_layer: bool = False, cfg: CfgNode = None):
        super().__init__()
        self.width = width
        self.layers = layers

        if cfg is not None:
            trainer = cfg.TRAINER.NAME
            if trainer in ["VPT", "MPT"]:
                self.resblocks = nn.Sequential(*[
                    ResidualAttentionBlock_VPT(width, heads, attn_mask, i, is_text_layer=is_text_layer, need_prompt=True, cfg=cfg) if i < prompt_depth else
                    ResidualAttentionBlock_VPT(width, heads, attn_mask, i, is_text_layer=is_text_layer, need_prompt=False, cfg=cfg) for i in range(layers)
                ])

            elif trainer == "MuDPT":
                self.resblocks = nn.Sequential(*[
                    ResidualAttentionBlock_MuDPT(width, heads, attn_mask, i, is_text_layer, cfg=cfg) for i in range(layers)
                ])

            elif trainer == "UMuDPT":
                self.resblocks = nn.Sequential(*[
                    ResidualAttentionBlock_UMuDPT(width, heads, attn_mask, i, is_text_layer, cfg=cfg) for i in range(layers)
                ])

            elif trainer == "UUMuDPT":
                self.resblocks = nn.Sequential(*[
                    ResidualAttentionBlock_UUMuDPT(width, heads, attn_mask, i, is_text_layer, cfg=cfg) for i in range(layers)
                ])

            else:
                raise NotImplementedError(f"{cfg.TRAIER.NAME} is not implemented")

        else:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, cfg: CfgNode = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # vit_b16: 768
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))  # vit_b16: (197, 768)

        self.img_prompt_depth = 0
        self.img_prompt = False
        if cfg is not None:
            trainer = cfg.TRAINER.NAME
            img_prompt_depth = eval(f"cfg.TRAINER.{trainer}.VISUAL_PROMPT_DEPTH")
            if 0 < img_prompt_depth <= 12:
                self.img_prompt = True
                self.img_prompt_depth = img_prompt_depth
                img_n_ctx = eval(f"cfg.TRAINER.{trainer}.DEEP_VISUAL_N_CTX")
                ctx_vectors = torch.empty(img_n_ctx, width)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.visual_ctx = nn.Parameter(ctx_vectors)

        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, prompt_depth=self.img_prompt_depth, cfg=cfg)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        # add image prompts
        if self.img_prompt:
            visual_ctx = self.visual_ctx.to(x.dtype) + torch.zeros(x.shape[0], self.visual_ctx.shape[0], self.visual_ctx.shape[1], dtype=x.dtype, device=x.device)
            x = torch.cat([x, visual_ctx], dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionTransformer_MuDPT(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, cfg: CfgNode = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # vit_b16: 768
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))  # vit_b16: (197, 768)

        self.deep_prompts_depth = eval(f"cfg.TRAINER.{str.upper(cfg.TRAINER.NAME)}.DEEP_PROMPT_DEPTH")
        n_ctx = eval(f"cfg.TRAINER.{str.upper(cfg.TRAINER.NAME)}.N_CTX")
        ctx_vectors = torch.empty(n_ctx, width)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.visual_ctx = nn.Parameter(ctx_vectors)

        self.visual_ctx_deep_prompts = nn.Parameter(torch.empty(self.deep_prompts_depth - 1, n_ctx, width))
        nn.init.normal_(self.visual_ctx_deep_prompts, std=0.02)

        self.visual_ctx_deep_projections = nn.Linear(in_features=width, out_features=output_dim)

        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, cfg=cfg)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_prompt, t2v_visual_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # add image prompts
        visual_prompt = self.visual_ctx.unsqueeze(0) + shared_prompt
        visual_prompt = visual_prompt.expand(x.shape[0], -1, -1).to(x.dtype).to(x.device)
        x = torch.cat([x, visual_prompt], dim=1)  # (16, 205, 768)
        visual_deep_prompts = t2v_visual_prompts + self.visual_ctx_deep_prompts

        text_prompts = self.visual_ctx_deep_projections(self.visual_ctx_deep_prompts)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        output = self.transformer([x, visual_deep_prompts, 0])
        x = output[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, text_prompts


class VisionTransformer_MaPLe(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, cfg: CfgNode = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # vit_b16: 768
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))  # vit_b16: (197, 768)

        self.img_prompt = True
        self.deep_prompts_depth = eval(f"cfg.TRAINER.{str.upper(cfg.TRAINER.NAME)}.DEEP_PROMPT_DEPTH")
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, cfg=cfg)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # add image prompts
        if self.img_prompt:
            shared_ctx = shared_ctx.to(x.dtype) + torch.zeros(x.shape[0], shared_ctx.shape[0], shared_ctx.shape[1], dtype=x.dtype, device=x.device)
            x = torch.cat([x, shared_ctx], dim=1)  # (16, 205, 768)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        output = self.transformer([x, compound_deeper_prompts, 0])
        x = output[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionTransformer_UMuDPT(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, cfg: CfgNode = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # vit_b16: 768
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))  # vit_b16: (197, 768)

        self.deep_prompts_depth = eval(f"cfg.TRAINER.{str.upper(cfg.TRAINER.NAME)}.DEEP_PROMPT_DEPTH")

        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, cfg=cfg)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # add image prompts
        shared_ctx = shared_ctx.expand(x.shape[0], -1, -1).to(x.dtype).to(x.device)
        x = torch.cat([x, shared_ctx], dim=1)  # (16, 205, 768)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        output = self.transformer([x, deeper_prompts, 0])
        x = output[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionTransformer_UUMuDPT(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, cfg: CfgNode = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # vit_b16: 768
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))  # vit_b16: (197, 768)

        self.deep_prompts_depth = eval(f"cfg.TRAINER.{str.upper(cfg.TRAINER.NAME)}.DEEP_PROMPT_DEPTH")
        n_ctx = eval(f"cfg.TRAINER.{str.upper(cfg.TRAINER.NAME)}.N_CTX")
        ctx_vectors = torch.empty(n_ctx, width)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.visual_ctx = nn.Parameter(ctx_vectors)

        self.visual_ctx_deep_prompts = nn.Parameter(torch.empty(self.deep_prompts_depth - 1, n_ctx, width))
        nn.init.normal_(self.visual_ctx_deep_prompts, std=0.02)

        self.visual_ctx_ln_intra_pre = LayerNorm(width)
        self.visual_ctx_self_attn = LightTransformer(d_model=width, n_head=width // 64)
        self.visual_ctx_ln_intra_post = LayerNorm(width)
        self.visual_ctx_text_proj = nn.Linear(in_features=width, out_features=output_dim)

        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, cfg=cfg)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # add image prompts
        visual_ctx = self.visual_ctx.unsqueeze(0)
        shared_ctx = shared_ctx + visual_ctx
        shared_ctx = shared_ctx.expand(x.shape[0], -1, -1).to(x.dtype).to(x.device)
        # shared_ctx = shared_ctx.to(x.dtype) + torch.zeros(x.shape[0], shared_ctx.shape[1], shared_ctx.shape[2], dtype=x.dtype, device=x.device)
        x = torch.cat([x, shared_ctx], dim=1)  # (16, 205, 768)
        visual_deeper_prompts = deeper_prompts + self.visual_ctx_deep_prompts

        # visual prompts to textual prompts
        textual_prompts = self.visual_ctx_deep_prompts
        textual_prompts = self.visual_ctx_ln_intra_pre(textual_prompts)
        textual_prompts = textual_prompts.permute(1, 0, 2)
        textual_prompts = self.visual_ctx_self_attn(textual_prompts)
        textual_prompts = textual_prompts.permute(1, 0, 2)
        textual_prompts = self.visual_ctx_ln_intra_post(textual_prompts)
        textual_prompts = self.visual_ctx_text_proj(textual_prompts)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        output = self.transformer([x, visual_deeper_prompts, 0])
        x = output[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, textual_prompts


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 cfg: CfgNode = None):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            if cfg is not None:
                if cfg.TRAINER.NAME == "MuDPT":
                    self.visual = VisionTransformer_MuDPT(
                        input_resolution=image_resolution,
                        patch_size=vision_patch_size,
                        width=vision_width,
                        layers=vision_layers,
                        heads=vision_heads,
                        output_dim=embed_dim,
                        cfg=cfg,
                    )

                elif cfg.TRAINER.NAME == "MaPLe":
                    self.visual = VisionTransformer_MaPLe(
                        input_resolution=image_resolution,
                        patch_size=vision_patch_size,
                        width=vision_width,
                        layers=vision_layers,
                        heads=vision_heads,
                        output_dim=embed_dim,
                        cfg=cfg,
                    )

                elif cfg.TRAINER.NAME == "UMuDPT":
                    self.visual = VisionTransformer_UMuDPT(
                        input_resolution=image_resolution,
                        patch_size=vision_patch_size,
                        width=vision_width,
                        layers=vision_layers,
                        heads=vision_heads,
                        output_dim=embed_dim,
                        cfg=cfg,
                    )

                elif cfg.TRAINER.NAME == "UUMuDPT":
                    self.visual = VisionTransformer_UUMuDPT(
                        input_resolution=image_resolution,
                        patch_size=vision_patch_size,
                        width=vision_width,
                        layers=vision_layers,
                        heads=vision_heads,
                        output_dim=embed_dim,
                        cfg=cfg,
                    )

                else:
                    self.visual = VisionTransformer(
                        input_resolution=image_resolution,
                        patch_size=vision_patch_size,
                        width=vision_width,
                        layers=vision_layers,
                        heads=vision_heads,
                        output_dim=embed_dim,
                        cfg=cfg,
                    )
            else:
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    cfg=cfg,
                )

        if cfg is not None:
            trainer = cfg.TRAINER.NAME
            if trainer in ["VPT", "MPT"]:
                text_prompt_depth = eval(f"cfg.TRAINER.{trainer}.TEXT_PROMPT_DEPTH")
            else:
                text_prompt_depth = 0
        else:
            text_prompt_depth = 0

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            is_text_layer=True,
            prompt_depth=text_prompt_depth,
            cfg=cfg,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, cfg: CfgNode = None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]  # vit_b16: 768
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])  # vit_b16: 12
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]  # vit_b16: 16
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)  # vit_b16: 14
        image_resolution = vision_patch_size * grid_size  # vit_b16: 224
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]  # vit_b16: 512
    context_length = state_dict["positional_embedding"].shape[0]  # vit_b16: 77
    vocab_size = state_dict["token_embedding.weight"].shape[0]  # vit_b16: 49408
    transformer_width = state_dict["ln_final.weight"].shape[0]  # vit_b16: 512
    transformer_heads = transformer_width // 64  # vit_b16: 8
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))  # vit_b16: 12

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        cfg,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)

    miss_keys, _ = model.load_state_dict(state_dict, strict=False)
    print(f'Weights not found: {miss_keys}')
    return model.eval()
