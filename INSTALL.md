## Installation

For the full project overview, see `README.md` or `README_zh-CN.md`.

### Tested Environment

```text
Python:      3.8.13
CUDA:        11.1
PyTorch:     1.10.1+cu111
torchvision: 0.11.2+cu111
Detectron2:  0.6+cu111
```

### Setup

```bash
conda create -n taxoseg python=3.8 -y
conda activate taxoseg
export PYTHONNOUSERSITE=1
```

PyTorch, torchvision, and Detectron2 are installed separately because their wheels must match the CUDA and PyTorch versions.

```bash
python -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 \
  -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
python -m pip install -r requirements.txt
```

The tested Detectron2 build reports version `0.6+cu111`.

### Compile MSDeformAttn

```bash
cd taxoseg/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
```
