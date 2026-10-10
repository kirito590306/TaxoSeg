<h1 align="center">TaxoSeg</h1>

<p align="center">
  Taxonomy-aware crop disease semantic segmentation via structured semantic prototypes
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8-blue" alt="Python 3.8">
  <img src="https://img.shields.io/badge/PyTorch-1.10.1-ee4c2c" alt="PyTorch 1.10.1">
  <img src="https://img.shields.io/badge/Detectron2-0.6-3b7ddd" alt="Detectron2 0.6">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License"></a>
</p>

## :seedling: Abstract

TaxoSeg is a Detectron2-based research codebase for taxonomy-aware crop disease semantic segmentation. It addresses field-scene disease segmentation where each fine-grained lesion category is defined by both crop identity and disease type. Built on a CLIP-based mask classification framework, TaxoSeg uses a fixed crop-disease semantic structure to connect fine-level labels with crop-side and disease-side factors during training, prediction, and diagnostic evaluation.

<p align="center">
  <img src="docs/assets/taxoseg_model_overview.png" width="94%" alt="TaxoSeg model overview">
</p>

## :sparkles: Main Features

- **Structured crop-disease prediction**: links each fine-grained lesion label to crop and disease factors for taxonomy-aware training and diagnosis-oriented evaluation.
- **A2F-HKD**: projects auxiliary crop and disease predictions into the fine-label space and uses their compatibility as a detached soft target for fine-grained classification.
- **CCMA**: injects crop-aware visual context into query-region classification to reduce ambiguity among visually similar disease symptoms.
- **Dual-branch text prototypes**: combine canonical class-name prompts with taxonomy-derived descriptor prompts for stronger CLIP semantic anchoring.
- **Open WheatField-Seg dataset**: provides an open wheat-field disease segmentation dataset for external evaluation.

## :computer: Deployment Preview

We are also developing a deployment prototype for practical crop disease analysis. The current interface supports TaxoSeg-based lesion segmentation and recognition, together with VLM-assisted disease identification and interactive agronomic analysis. This part is ongoing work for a follow-up study that combines TaxoSeg with disease-oriented vision-language models.

<p align="center">
  <img src="docs/assets/taxoseg_deployment_preview.png" width="94%" alt="TaxoSeg deployment preview">
</p>

## :wrench: Installation

Create a Python 3.8 environment:

```bash
conda create -n taxoseg python=3.8 -y
conda activate taxoseg
export PYTHONNOUSERSITE=1
```

```bash
python -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install detectron2 \
  -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```

Install the remaining dependencies and compile MSDeformAttn:

```bash
python -m pip install -r requirements.txt

cd taxoseg/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
```

## :rocket: Training and Evaluation

We will release the download link after the paper is formally published. Download `WheatField-Seg` separately and set the dataset root.

```bash
export TAXOSEG_DATASETS=/path/to/datasets
```

Train on one GPU:

```bash
python train_net.py \
  --config-file configs/wheatfield/taxoseg_convnext_large_wheatfield_semseg.yaml \
  --num-gpus 1
```

Train on multiple GPUs:

```bash
python train_net.py \
  --config-file configs/wheatfield/taxoseg_convnext_large_wheatfield_semseg.yaml \
  --num-gpus 8
```

Evaluate on the default validation split:

```bash
python train_net.py \
  --config-file configs/wheatfield/taxoseg_convnext_large_wheatfield_semseg.yaml \
  --eval-only \
  MODEL.WEIGHTS /path/to/checkpoint.pth
```

## :memo: License and Acknowledgements

TaxoSeg is released under the [Apache License 2.0](LICENSE). Portions of the implementation are adapted from Apache-2.0-compatible semantic segmentation and object detection projects, with upstream copyright notices retained in source-file headers where applicable.

See [NOTICE](NOTICE) for attribution details.
