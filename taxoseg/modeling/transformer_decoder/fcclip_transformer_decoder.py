"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py
"""

import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import numpy as np
import torch
import torch.utils.checkpoint as cp
from torch import nn, Tensor
from torch.nn import functional as F
from timm.models.layers import DropPath

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry


from .position_encoding import PositionEmbeddingSine


TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskFormer.
"""


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


def get_classification_logits(x, text_classifier, logit_scale, num_templates=None):
    # x in shape of [B, *, C]
    # text_classifier in shape of [num_classes, C]
    # logit_scale is a learnable scalar https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L201
    # return: [B, *, num_classes]
    x = F.normalize(x, dim=-1)
    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    pred_logits = logit_scale * x @ text_classifier.T # B, *, N + 1
    # max ensembel as in OpenSeg/ODISE
    final_pred_logits = []
    cur_idx = 0
    for num_t in num_templates:
        final_pred_logits.append(pred_logits[:, :, cur_idx: cur_idx + num_t].max(-1).values)
        cur_idx += num_t
    final_pred_logits.append(pred_logits[:, :, -1]) # the last classifier is for void
    final_pred_logits = torch.stack(final_pred_logits, dim=-1)
    return final_pred_logits

# Ref: https://github.com/NVlabs/ODISE/blob/e97b06c424c575fec9fc5368dd4b3e050d91abc4/odise/modeling/meta_arch/odise.py#L923
class MaskPooling(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        x,
        mask,
        *,
        detach: bool = True,
        binarize: bool = True,
        use_sigmoid: bool = False,
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        """
        Args:
            x: [B, C, H, W]
            mask: [B, Q, H, W]
        """
        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if detach:
            with torch.no_grad():
                mask = mask.detach()
                if use_sigmoid:
                    mask = mask.sigmoid()
                if binarize:
                    mask = (mask > 0).to(mask.dtype)
                if normalize:
                    denorm = mask.sum(dim=(-1, -2), keepdim=True) + eps
                    mask = mask / denorm
        else:
            if use_sigmoid:
                mask = mask.sigmoid()
            if binarize:
                mask = (mask > 0).to(mask.dtype)
            if normalize:
                denorm = mask.sum(dim=(-1, -2), keepdim=True) + eps
                mask = mask / denorm

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask,
        )
        return mask_pooled_x

class MaskAdapterHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mask_in_chans: int,
        num_channels: int,
        num_output_maps: int,
        condition_dim: int = 0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_output_maps = num_output_maps

        self.fuse = nn.Conv2d(in_channels, num_channels, 1)
        self.cnext1 = ConvNextBlock(num_channels)
        self.cnext2 = ConvNextBlock(num_channels)
        self.cnext3 = ConvNextBlock(num_channels)
        self.norm = nn.LayerNorm(num_channels)
        self.final = nn.Conv2d(num_channels, num_output_maps, 1)

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, in_channels, kernel_size=1),
        )
        self.condition_proj = None
        self.condition_scale = None
        if condition_dim > 0:
            self.condition_proj = nn.Linear(condition_dim, in_channels)
            # Start from zero to preserve previous behavior before CAP learns useful conditions.
            self.condition_scale = nn.Parameter(torch.zeros(1))

    def forward(self, mask_features, masks, condition=None):
        """
        Args:
            mask_features: [B, C, H, W]
            masks: [B, Q, H, W]
            condition: [B, D]
        """
        b, q, h, w = masks.shape
        feat = mask_features.unsqueeze(1).repeat(1, q, 1, 1, 1).reshape(b * q, -1, h, w)
        masks = masks.reshape(b * q, 1, h, w)
        masks = F.interpolate(masks.float(), size=(h * 4, w * 4), mode='bilinear', align_corners=False)
        masks = self.mask_downscaling(masks)
        outputs = feat + masks
        if condition is not None and self.condition_proj is not None:
            cond = self.condition_proj(condition).to(outputs.dtype)
            cond = cond.unsqueeze(1).expand(-1, q, -1).reshape(b * q, -1, 1, 1)
            outputs = outputs + self.condition_scale.to(outputs.dtype) * cond

        def _inner_forward(x):
            x = self.fuse(x)
            x = self.cnext1(x)
            x = self.cnext2(x)
            x = self.cnext3(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.final(x)
            return x

        if self.use_checkpoint and self.training:
            try:
                # torch>=2.0 supports use_reentrant kwarg; older versions do not.
                outputs = cp.checkpoint(_inner_forward, outputs, use_reentrant=False)
            except (TypeError, ValueError):
                outputs = cp.checkpoint(_inner_forward, outputs)
        else:
            outputs = _inner_forward(outputs)

        return outputs.reshape(b, q * self.num_output_maps, h, w)

class ConvNextBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()
        x = shortcut + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        clip_embedding_dim: int,
        mask_adapter_on: bool,
        mask_adapter_mask_in_chans: int,
        mask_adapter_num_channels: int,
        mask_adapter_num_output_maps: int,
        mask_adapter_mask_threshold: float,
        mask_adapter_use_checkpoint: bool,
        cap_on: bool,
        cap_cond_source: str,
        cap_clip_weight: float,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        # if self.mask_classification:
        #     self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # class embeddings
        self.mask_pooling = MaskPooling()
        self.mask_adapter_on = mask_adapter_on
        self.cap_on = bool(cap_on and mask_adapter_on)
        self.cap_cond_source = str(cap_cond_source).lower()
        self.cap_clip_weight = float(cap_clip_weight)
        if self.mask_adapter_on:
            self.mask_adapter = MaskAdapterHead(
                in_channels=mask_dim,
                mask_in_chans=mask_adapter_mask_in_chans,
                num_channels=mask_adapter_num_channels,
                num_output_maps=mask_adapter_num_output_maps,
                condition_dim=(mask_dim if self.cap_on else 0),
                use_checkpoint=mask_adapter_use_checkpoint,
            )
        else:
            self.mask_adapter = None
        self.mask_adapter_num_output_maps = mask_adapter_num_output_maps
        self.mask_adapter_mask_threshold = mask_adapter_mask_threshold
        self._mask_pooling_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim))
        self.class_embed = MLP(hidden_dim, hidden_dim, clip_embedding_dim, 3)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if self.cap_on:
            self.cap_mask_proj = nn.Linear(mask_dim, mask_dim)
            self.cap_clip_proj = nn.Linear(mask_dim, mask_dim)
            self.cap_cond_norm = nn.LayerNorm(mask_dim)
            self.cap_strong_hidden_to_cond = (
                nn.Linear(hidden_dim, mask_dim)
                if self.cap_cond_source == "strong_h"
                else None
            )
            self.cap_strong_logits_to_cond = (
                nn.Linear(mask_dim, mask_dim)
                if self.cap_cond_source == "strong_z"
                else None
            )
            self.cap_strong_fuse = (
                nn.Linear(mask_dim * 2, mask_dim)
                if self.cap_cond_source in ("strong_h", "strong_z")
                else None
            )
        else:
            self.cap_mask_proj = None
            self.cap_clip_proj = None
            self.cap_cond_norm = None
            self.cap_strong_hidden_to_cond = None
            self.cap_strong_logits_to_cond = None
            self.cap_strong_fuse = None

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["clip_embedding_dim"] = cfg.MODEL.FC_CLIP.EMBED_DIM
        ret["mask_adapter_on"] = cfg.MODEL.FC_CLIP.MASK_ADAPTER_ON
        ret["mask_adapter_mask_in_chans"] = cfg.MODEL.FC_CLIP.MASK_ADAPTER_MASK_IN_CHANNELS
        ret["mask_adapter_num_channels"] = cfg.MODEL.FC_CLIP.MASK_ADAPTER_NUM_CHANNELS
        ret["mask_adapter_num_output_maps"] = cfg.MODEL.FC_CLIP.MASK_ADAPTER_NUM_OUTPUT_MAPS
        ret["mask_adapter_mask_threshold"] = cfg.MODEL.FC_CLIP.MASK_ADAPTER_MASK_THRESHOLD
        ret["mask_adapter_use_checkpoint"] = cfg.MODEL.FC_CLIP.MASK_ADAPTER_USE_CHECKPOINT
        ret["cap_on"] = cfg.MODEL.FC_CLIP.CAP_ON
        ret["cap_cond_source"] = cfg.MODEL.FC_CLIP.CAP_COND_SOURCE
        ret["cap_clip_weight"] = cfg.MODEL.FC_CLIP.CAP_CLIP_WEIGHT
        return ret

    def forward(
        self,
        x,
        mask_features,
        mask=None,
        text_classifier=None,
        num_templates=None,
        text_classifier_long=None,
        num_templates_long=None,
        text_classifier_crop=None,
        num_templates_crop=None,
        text_classifier_disease=None,
        num_templates_disease=None,
        global_crop_on: bool = False,
        clip_vis_dense=None,
    ):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_class_long = []
        predictions_mask = []
        predictions_class_crop = []
        predictions_class_disease = []
        predictions_mask_adapter = []
        predictions_mask_adapter_weights = []

        global_crop_logits = None
        if global_crop_on and text_classifier_crop is not None:
            global_crop_logits = self._forward_global_crop(
                mask_features, text_classifier_crop, num_templates_crop
            )
        cap_condition = None
        if self.cap_on and self.mask_adapter_on:
            cap_condition = self._build_cap_condition(
                mask_features=mask_features,
                clip_vis_dense=clip_vis_dense,
                text_classifier_crop=text_classifier_crop,
                num_templates_crop=num_templates_crop,
            )

        # prediction heads on learnable query features
        (
            outputs_class,
            outputs_class_long,
            outputs_mask,
            attn_mask,
            outputs_class_crop,
            outputs_class_disease,
            outputs_mask_adapter,
            outputs_mask_adapter_weights,
        ) = self.forward_prediction_heads(
            output,
            mask_features,
            attn_mask_target_size=size_list[0],
            text_classifier=text_classifier,
            num_templates=num_templates,
            text_classifier_long=text_classifier_long,
            num_templates_long=num_templates_long,
            text_classifier_crop=text_classifier_crop,
            num_templates_crop=num_templates_crop,
            text_classifier_disease=text_classifier_disease,
            num_templates_disease=num_templates_disease,
            cap_condition=cap_condition,
            use_adapter=False,
        )
        predictions_class.append(outputs_class)
        if outputs_class_long is not None:
            predictions_class_long.append(outputs_class_long)
        predictions_mask.append(outputs_mask)
        if outputs_mask_adapter is not None:
            predictions_mask_adapter.append(outputs_mask_adapter)
        if outputs_mask_adapter_weights is not None:
            predictions_mask_adapter_weights.append(outputs_mask_adapter_weights)
        if outputs_class_crop is not None:
            predictions_class_crop.append(outputs_class_crop)
        if outputs_class_disease is not None:
            predictions_class_disease.append(outputs_class_disease)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            (
                outputs_class,
                outputs_class_long,
                outputs_mask,
                attn_mask,
                outputs_class_crop,
                outputs_class_disease,
                outputs_mask_adapter,
                outputs_mask_adapter_weights,
            ) = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                text_classifier=text_classifier,
                num_templates=num_templates,
                text_classifier_long=text_classifier_long,
                num_templates_long=num_templates_long,
                text_classifier_crop=text_classifier_crop,
                num_templates_crop=num_templates_crop,
                text_classifier_disease=text_classifier_disease,
                num_templates_disease=num_templates_disease,
                cap_condition=cap_condition,
                use_adapter=(i == self.num_layers - 1),
            )
            predictions_class.append(outputs_class)
            if outputs_class_long is not None:
                predictions_class_long.append(outputs_class_long)
            predictions_mask.append(outputs_mask)
            if outputs_mask_adapter is not None:
                predictions_mask_adapter.append(outputs_mask_adapter)
            if outputs_mask_adapter_weights is not None:
                predictions_mask_adapter_weights.append(outputs_mask_adapter_weights)
            if outputs_class_crop is not None:
                predictions_class_crop.append(outputs_class_crop)
            if outputs_class_disease is not None:
                predictions_class_disease.append(outputs_class_disease)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        if predictions_class_long:
            out["pred_logits_long"] = predictions_class_long[-1]
            out["aux_outputs_long"] = self._set_aux_class_loss(predictions_class_long)
        if predictions_class_crop:
            out["pred_logits_crop"] = predictions_class_crop[-1]
        if predictions_class_disease:
            out["pred_logits_disease"] = predictions_class_disease[-1]
        if global_crop_logits is not None:
            out["global_crop_logits"] = global_crop_logits
        if predictions_mask_adapter:
            out["pred_masks_adapter"] = predictions_mask_adapter[-1]
        if predictions_mask_adapter_weights:
            out["pred_masks_adapter_weights"] = predictions_mask_adapter_weights[-1]
        return out

    def _extract_global_crop_hidden(self, mask_features):
        global_feat = F.adaptive_avg_pool2d(mask_features, 1)
        global_feat = global_feat.flatten(2).transpose(1, 2)  # B, 1, C
        global_feat = self._mask_pooling_proj(global_feat)
        return global_feat.squeeze(1)

    def _forward_global_crop(self, mask_features, text_classifier_crop, num_templates_crop):
        global_feat = self._extract_global_crop_hidden(mask_features).unsqueeze(1)
        global_embed = self.class_embed(global_feat)
        global_logits = get_classification_logits(
            global_embed, text_classifier_crop, self.logit_scale, num_templates_crop
        )
        return global_logits.squeeze(1)

    def _build_cap_condition(
        self,
        mask_features,
        clip_vis_dense=None,
        text_classifier_crop=None,
        num_templates_crop=None,
    ):
        if not self.cap_on:
            return None

        g_mask = F.adaptive_avg_pool2d(mask_features, 1).flatten(1)
        g_mask = self.cap_mask_proj(g_mask)
        g_clip_in = g_mask
        alpha = 0.0
        if clip_vis_dense is not None:
            g_clip_in = F.adaptive_avg_pool2d(clip_vis_dense, 1).flatten(1)
            alpha = float(np.clip(self.cap_clip_weight, 0.0, 1.0))
        g_clip_in = F.adaptive_avg_pool1d(
            g_clip_in.unsqueeze(1), self.cap_mask_proj.in_features
        ).squeeze(1)
        g_clip = self.cap_clip_proj(g_clip_in)
        g = alpha * g_clip + (1.0 - alpha) * g_mask
        g = self.cap_cond_norm(g)

        if self.cap_cond_source == "weak_fuse":
            return g

        crop_hidden = self._extract_global_crop_hidden(mask_features)
        if self.cap_cond_source == "strong_h" and self.cap_strong_hidden_to_cond is not None:
            g_crop = self.cap_strong_hidden_to_cond(crop_hidden)
        elif self.cap_cond_source == "strong_z" and self.cap_strong_logits_to_cond is not None:
            if text_classifier_crop is not None and num_templates_crop is not None:
                crop_embed = self.class_embed(crop_hidden.unsqueeze(1))
                crop_logits = get_classification_logits(
                    crop_embed, text_classifier_crop, self.logit_scale, num_templates_crop
                ).squeeze(1)
            else:
                crop_logits = crop_hidden
            crop_logits = F.adaptive_avg_pool1d(
                crop_logits.unsqueeze(1), self.cap_mask_proj.in_features
            ).squeeze(1)
            g_crop = self.cap_strong_logits_to_cond(crop_logits)
        else:
            logging.getLogger(__name__).warning(
                "Unknown CAP condition source '%s', fallback to weak_fuse.",
                self.cap_cond_source,
            )
            return g

        if self.cap_strong_fuse is None:
            return g
        fused = self.cap_strong_fuse(torch.cat([g, g_crop], dim=-1))
        return self.cap_cond_norm(fused)

    def forward_prediction_heads(
        self,
        output,
        mask_features,
        attn_mask_target_size,
        text_classifier,
        num_templates,
        text_classifier_long=None,
        num_templates_long=None,
        text_classifier_crop=None,
        num_templates_crop=None,
        text_classifier_disease=None,
        num_templates_disease=None,
        cap_condition=None,
        use_adapter: bool = True,
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # segmentation head
        outputs_mask_adapter = None
        outputs_mask_adapter_weights = None
        if self.mask_adapter_on and use_adapter:
            mask_in = (outputs_mask.sigmoid() > self.mask_adapter_mask_threshold).to(outputs_mask.dtype)
            outputs_maps = self.mask_adapter(mask_features, mask_in, condition=cap_condition)
            b, n, h, w = outputs_mask.shape
            num_maps = self.mask_adapter_num_output_maps
            outputs_mask_adapter = outputs_maps.view(b, n, num_maps, h, w).mean(dim=2)

            maps_for_pooling = F.interpolate(
                outputs_maps, size=mask_features.shape[-2:], mode="bilinear", align_corners=False
            )
            b, c = mask_features.shape[:2]
            total_maps = maps_for_pooling.size(1)
            num_instances = total_maps // num_maps
            maps_for_pooling = F.softmax(
                F.logsigmoid(maps_for_pooling).view(b, total_maps, -1), dim=-1
            )
            outputs_mask_adapter_weights = maps_for_pooling.view(
                b, num_instances, num_maps, -1
            ).mean(dim=2).view(b, num_instances, h, w)
            pooled = torch.bmm(
                maps_for_pooling, mask_features.view(b, c, -1).permute(0, 2, 1)
            )
            maskpool_embeddings = pooled.view(
                b, num_instances, num_maps, c
            ).mean(dim=2).contiguous()
        else:
            maskpool_embeddings = self.mask_pooling(x=mask_features, mask=outputs_mask) # [B, Q, C]
        maskpool_embeddings = self._mask_pooling_proj(maskpool_embeddings)
        class_embed = self.class_embed(maskpool_embeddings + decoder_output)
        outputs_class = get_classification_logits(class_embed, text_classifier, self.logit_scale, num_templates)
        outputs_class_long = None
        if text_classifier_long is not None:
            outputs_class_long = get_classification_logits(
                class_embed, text_classifier_long, self.logit_scale, num_templates_long
            )
        outputs_class_crop = None
        outputs_class_disease = None
        if text_classifier_crop is not None:
            outputs_class_crop = get_classification_logits(
                class_embed, text_classifier_crop, self.logit_scale, num_templates_crop
            )
        if text_classifier_disease is not None:
            outputs_class_disease = get_classification_logits(
                class_embed, text_classifier_disease, self.logit_scale, num_templates_disease
            )

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return (
            outputs_class,
            outputs_class_long,
            outputs_mask,
            attn_mask,
            outputs_class_crop,
            outputs_class_disease,
            outputs_mask_adapter,
            outputs_mask_adapter_weights,
        )

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

    @torch.jit.unused
    def _set_aux_class_loss(self, outputs_class):
        return [{"pred_logits": a} for a in outputs_class[:-1]]
