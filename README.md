# Frequency and Edge-Guided Segment Anything Model for Remote Sensing Image Semantic Segmentation, IEEE TGRS 2026
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

## 📋 Abstract

**FE-SAM (Frequency and Edge-guided SAM)** is a scalable and efficient framework designed for remote sensing image semantic segmentation (RSISS). While the Segment Anything Model (SAM) offers strong segmentation performance and generalization capabilities, existing SAM-based approaches face two critical limitations: (1) *insufficient adaptation of SAM's features to the diverse characteristics of land cover types*, and (2) *semantic ambiguity at object boundaries*. To address these challenges, FE-SAM introduces two key innovations: a **Frequency-Modulated Adapter (FMA)** that adaptively decomposes and modulates frequency-domain features to enhance informative high- and low-frequency components corresponding to different land cover types, and an **Edge Guided Refiner (EGRefiner)** that integrates multi-scale edge-enhanced information to improve fine-grained boundary delineation. Extensive experiments on three benchmark datasets demonstrate that FE-SAM outperforms state-of-the-art methods.

## 🏗️ Architecture Overview

<div align="center">
  <img src="https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-08-16_063700_466.jpg" alt="FE-SAM Architecture" width="800"/>
</div>


### Key Components:

- **Frequency-Modulated Adapter (FMA)**: Adaptively decomposes and modulates frequency-domain features to enhance informative components for different land cover types
- **Edge Guided Refiner (EGRefiner)**: Integrates multi-scale edge-enhanced information for fine-grained boundary delineation

## 🚀 Key Features

- 🎯 **Superior Performance**: State-of-the-art accuracy on remote sensing benchmarks
- ⚡ **Parameter Efficient**: Fine-tune only ~7.52% of model parameters with adapters
- 🔊 **Frequency-Domain Enhancement**: Adaptive modulation of frequency components for land cover diversity
- 📐 **Edge-Guided Refinement**: Multi-scale edge enhancement for precise boundary delineation
- 🌍 **Multi-dataset Validation**: Comprehensive evaluation on ISPRS Vaihingen and ISPRS Potsdam and LoveDA datasets
- 📊 **Scalable Framework**: Efficient and scalable design for practical deployment

## 📦 Installation

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/oucai/FE-SAM.git
cd FE-SAM

# Create conda environment
conda create -n fe-sam python=3.8
conda activate fe-sam

# Install PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Requirements

```text
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.3.0
scikit-learn>=1.0.0
tqdm>=4.62.0
tensorboard>=2.7.0
```

## 📊 Datasets

### ISPRS Vaihingen Dataset

```bash
# Download from ISPRS official website
wget https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/
# Extract to ./datasets/vaihingen/
```

### ISPRS Potsdam Dataset

```bash
# Download from ISPRS official website  
wget https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/
# Extract to ./datasets/potsdam/
```

### LoveDA Dataset

```bash
# Download from OpenEarthMap
wget https://zenodo.org/record/5706578
# Extract to ./datasets/loveda/
```

### Expected Dataset Structure

```
FE-SAM/
├── datasets/
│   ├── vaihingen/
│   │   ├── images/
│   │   ├── labels/
│   │   └── splits/
│   ├── potsdam/
│   │   ├── images/
│   │   ├── labels/
│   │   └── splits/
│   └── loveda/
│       ├── Train/
│       ├── Val/
│       └── Test/
```

## 🏋️ Pre-trained Models

Download the SAM backbone weights:

```bash
# SAM ViT-B backbone
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P sam_pretrain/

# SAM ViT-H backbone  
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P sam_pretrain/
```

## 🔧 Training

### Single GPU Training

```bash
# Vaihingen dataset
python adapter_vaihingen_train.py 

# Potsdam dataset  
python adapter_potsdam_train.py 

# LoveDA dataset
python adapter_loveda_train.py 
```

### Training Configuration

Modify the configuration files in `configs/` to adjust:

- Model architecture parameters
- Training hyperparameters  
- Data augmentation settings
- Loss function weights

## 🧪 Evaluation

### Quick Inference

```bash
# Vaihingen evaluation
python adapter_vaihingen_test.py \
    -c configs/adapter_sam_vh.py \
    -m /path/to/weights/fe_sam_vh.pth \
    -o /path/to/output \
    -t 'd4' \
    --rgb

# Potsdam evaluation
python adapter_potsdam_test.py \
    -c configs/adapter_sam_pd.py \
    -m /path/to/weights/fe_sam_pd.pth \
    -o /path/to/output \
    -t 'd4' \
    --rgb

# LoveDA evaluation  
python adapter_loveda_test.py \
    -c configs/adapter_sam_loveda.py \
    -m /path/to/weights/fe_sam_loveda.pth \
    -o /path/to/output \
    -t 'd4' \
    --rgb
```

### Evaluation Parameters

- `-c, --config`: Configuration file path
- `-m, --model`: Pre-trained model weights path
- `-o, --output`: Output directory for predictions
- `-t, --tta`: Test-time augmentation strategy ('d4', 'd8', or None)
- `--rgb`: Use RGB channels only (exclude IR/NIR)
- `--batch-size`: Inference batch size (default: 1)
- `--save-logits`: Save raw logits for ensemble methods

---

<div align="center">
  <sub>Built with ❤️ for the Remote Sensing Community</sub>
</div>
