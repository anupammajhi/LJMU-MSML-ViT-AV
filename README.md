<hr>
<hr>

# MEMORY EFFICIENT VISION TRANSFORMERS FOR REAL TIME OBJECT DETECTION IN AUTONOMOUS DRIVING

<hr>

# Introduction


## Video Presentation

[View on Youtube](https://www.youtube.com/watch?v=oe3Lj2SnpA4) 


## Dataset

Can be downloaded from the source: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

OR

Run the following script
```bash
python src/download_dataset.py
```
<hr>

# Installation Instructions

### Note:

For GPU support, install PyTorch and related libraries with the correct CUDA version.

Visit https://pytorch.org/get-started/locally/ and select your CUDA version.
Example for CUDA 12.2:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## Steps
1. Install PyTorch, torchvision, and torchaudio with CUDA support (see above).

2. Then install the rest:
```bash
    pip install -r requirements.txt
```
<hr>

# Code Structure

## 1. Download Dataset

```
cd src
python download_dataset.py
```

## 2. Training

```
cd src
python train.py
```

### Arguments (Optional)
  <ins>epochs</ins> (int): default=50
  - Number of epochs

  <ins>batch</ins> (int): default=25
  - Number of batches. 15 is a good number for small systems, 50 for big

  <ins>lrv (float): default=2e-4
  - Learning Rate

  <ins>model</ins> (str): default='vit'
  - Model to train. 
  - Choices=['vit', 'mobilevit', 'efficientvit', 'hybridvit']

  <ins>prune</ins> (bool): default='False'
  - Pruning for optimization.

  <ins>qat</ins> (bool): default='False'
  - Quantization for optimization

  <ins>distill</ins> (bool): default='False'
  - Knowledge distillation