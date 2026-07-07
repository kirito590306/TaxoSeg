# -*- coding: utf-8 -*-
"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/config.py
"""
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


def add_fcclip_config(cfg):
    # TaxoSeg model config
    cfg.MODEL.FC_CLIP = CN()

    # Base model settings
    cfg.MODEL.FC_CLIP.CLIP_MODEL_NAME = "convnext_large_d_320"
    cfg.MODEL.FC_CLIP.CLIP_PRETRAINED_WEIGHTS = "laion2b_s29b_b131k_ft_soup"
    cfg.MODEL.FC_CLIP.EMBED_DIM = 768  # CLIP text/vision embedding dimension
    cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA = 0.4  # seen-class out-vocabulary ensemble weight
    cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA = 0.8  # unseen-class out-vocabulary ensemble weight
    cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK = False  # enable ensemble only when the mask has valid pixels

    # Innovation 1: Structured A2F-HKD
    cfg.MODEL.FC_CLIP.HIER_CROP_WEIGHT = 1.0  # crop-level classification loss weight
    cfg.MODEL.FC_CLIP.HIER_DISEASE_WEIGHT = 1.0  # disease-level classification loss weight
    cfg.MODEL.FC_CLIP.A2F_HKD_ON = True  # training-time Aux-to-Fine hierarchical distillation
    cfg.MODEL.FC_CLIP.A2F_HKD_WEIGHT = 2.0  # Aux-to-Fine distillation loss weight
    cfg.MODEL.FC_CLIP.A2F_HKD_WARMUP_ITERS = 0  # Aux-to-Fine distillation warmup iterations
    cfg.MODEL.FC_CLIP.A2F_HKD_ALPHA = 2.0  # crop teacher fusion strength
    cfg.MODEL.FC_CLIP.A2F_HKD_BETA = 1.0  # disease teacher fusion strength
    cfg.MODEL.FC_CLIP.A2F_HKD_TEMP = 0.8  # distillation temperature
    cfg.MODEL.FC_CLIP.A2F_HKD_FUSION = "poe"  # teacher fusion type: poe / sum
    cfg.MODEL.FC_CLIP.A2F_HKD_EPS = 1e-6  # numerical stability term
    cfg.MODEL.FC_CLIP.A2F_HKD_CROP_CONF_TAU = 0.0  # crop teacher confidence threshold
    cfg.MODEL.FC_CLIP.A2F_HKD_DISEASE_CONF_TAU = 0.0  # disease teacher confidence threshold
    cfg.MODEL.FC_CLIP.A2F_HKD_FINE_MARGIN_TAU = 0.0  # fine top1-top2 margin gate threshold

    # Innovation 2: CAP (includes global crop head + conditioned adapter)
    cfg.MODEL.FC_CLIP.GLOBAL_CROP_WEIGHT = 0.1  # global crop branch loss weight
    cfg.MODEL.FC_CLIP.GLOBAL_CROP_WARMUP_ITERS = 10000  # global crop branch warmup iterations
    cfg.MODEL.FC_CLIP.GLOBAL_CROP_WARMUP_OFF_ITERS = 5000  # global crop branch warmup-off iterations
    cfg.MODEL.FC_CLIP.MASK_ADAPTER_ON = True  # CAP implementation path: conditioned adapter
    cfg.MODEL.FC_CLIP.MASK_ADAPTER_MASK_IN_CHANNELS = 16  # adapter mask downscaling base channels
    cfg.MODEL.FC_CLIP.MASK_ADAPTER_NUM_CHANNELS = 256  # adapter ConvNeXt backbone channels
    cfg.MODEL.FC_CLIP.MASK_ADAPTER_NUM_OUTPUT_MAPS = 16  # generated weight maps per instance
    cfg.MODEL.FC_CLIP.MASK_ADAPTER_MASK_THRESHOLD = 0.5  # adapter input mask threshold
    cfg.MODEL.FC_CLIP.MASK_ADAPTER_USE_CHECKPOINT = True  # use checkpointing in adapter to save memory
    cfg.MODEL.FC_CLIP.MASK_ADAPTER_REG_WEIGHT = 0.05  # adapter weight-map regularization loss weight
    cfg.MODEL.FC_CLIP.MASK_ADAPTER_REG_WARMUP_ITERS = 3000  # adapter regularization warmup iterations
    cfg.MODEL.FC_CLIP.MASK_ADAPTER_REG_DECAY_ITERS = 20000  # adapter regularization decay iterations
    cfg.MODEL.FC_CLIP.CAP_ON = True  # CAP master switch, decoupled from global_crop_on
    cfg.MODEL.FC_CLIP.CAP_COND_SOURCE = "strong_z"  # weak_fuse / strong_h / strong_z
    cfg.MODEL.FC_CLIP.CAP_CLIP_WEIGHT = 0.8  # F_clip fusion ratio in weak_fuse mode

    # Innovation 3: Dual-branch Long Prompt
    cfg.MODEL.FC_CLIP.FINE_PROMPT_FUSION_ON = True  # fine short/long prompt fusion switch
    cfg.MODEL.FC_CLIP.FINE_SHORT_WEIGHT = 1.0  # fine short branch logits fusion weight
    cfg.MODEL.FC_CLIP.FINE_LONG_WEIGHT = 1.0  # fine long branch logits fusion weight
    cfg.MODEL.FC_CLIP.FINE_LONG_WARMUP_ITERS = 0  # fine long branch weight warmup iterations
