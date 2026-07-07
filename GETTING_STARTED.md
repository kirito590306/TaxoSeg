## Getting Started

See `README.md` or `README_zh-CN.md` for complete installation, dependency, and dataset instructions.

### Prepare Data

```bash
export TAXOSEG_DATASETS=/path/to/datasets
```

Expected dataset path:

```text
$TAXOSEG_DATASETS/WheatField-Seg/
```

The repository does not include dataset files or pretrained weights.

### Train

```bash
python train_net.py \
  --config-file configs/wheatfield/taxoseg_convnext_large_wheatfield_semseg.yaml \
  --num-gpus 1
```

### Evaluate

```bash
python train_net.py \
  --config-file configs/wheatfield/taxoseg_convnext_large_wheatfield_semseg.yaml \
  --eval-only \
  MODEL.WEIGHTS /path/to/checkpoint.pth
```
