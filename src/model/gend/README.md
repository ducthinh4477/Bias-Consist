---
license: mit
library_name: transformers
tags:
- pytorch
- image-classification
---

# Deepfake Detection that Generalizes Across Benchmarks (WACV 2026)


[![arXiv Badge](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=FFF)](https://arxiv.org/abs/2508.06248)
[![GitHub Badge](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=fff)](https://github.com/yermandy/GenD)

This is the GenD (CLIP) model from Tab. 2 in the paper.

## Set up environment

``` bash
conda create --name GenD python=3.12 uv
conda activate GenD
uv pip install -r requirements.txt
```

## Usage

``` python
import requests
import torch
from PIL import Image

from src.hf.modeling_gend import GenD

model = GenD.from_pretrained("yermandy/GenD_CLIP_L_14")

urls = [
    "https://github.com/yermandy/deepfake-detection/blob/main/datasets/FF/DF/000_003/000.png?raw=true",
    "https://github.com/yermandy/deepfake-detection/blob/main/datasets/FF/real/000/000.png?raw=true",
]
images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
tensors = torch.stack([model.feature_extractor.preprocess(img) for img in images])
logits = model(tensors)
probs = logits.softmax(dim=-1)

print(probs)
```
