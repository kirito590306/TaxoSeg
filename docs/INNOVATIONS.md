# TaxoSeg Method Notes

TaxoSeg uses the dataset taxonomy file to improve wheat disease semantic segmentation. The default WheatField-Seg configuration enables the components below.

## 1. Structured A2F-HKD

Configuration switches:

```yaml
HIER_CROP_WEIGHT: 1.0
HIER_DISEASE_WEIGHT: 1.0
A2F_HKD_ON: True
A2F_HKD_WEIGHT: 2.0
A2F_HKD_FUSION: "poe"
```

These options enable crop/disease auxiliary hierarchy supervision and Aux-to-Fine hierarchical distillation.

## 2. Crop-Aware Prediction

Configuration switches:

```yaml
GLOBAL_CROP_WEIGHT: 0.1
MASK_ADAPTER_ON: True
CAP_ON: True
CAP_COND_SOURCE: "strong_z"
CAP_CLIP_WEIGHT: 0.8
```

These options enable the global crop branch and the conditioned mask adapter.

## 3. Dual-Branch Fine Prompt Fusion

Configuration switches:

```yaml
FINE_PROMPT_FUSION_ON: True
FINE_SHORT_WEIGHT: 1.0
FINE_LONG_WEIGHT: 1.0
```

The long fine-grained prompts are built from the dataset taxonomy file and fused with the short prompt branch.
