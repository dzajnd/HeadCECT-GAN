# HeadCECT-GAN: Head Contrast-Enhanced CT Generation

This project implements **HeadCECT-GAN**, a specialized GAN model for generating **Contrast-Enhanced CT (CECT)** images from non-contrast head CT scans. 

It is built upon the architecture of **Reg-GAN** (Breaking the Dilemma of Medical Image-to-image Translation), incorporating registration networks to handle misalignment between non-contrast and contrast phases.

## Model Structure
The model consists of:
- **Generator**: A U-Net like architecture with a shared encoder and two decoders (one for the image, one for the deformation field).
- **Discriminator**: A patch-based discriminator.
- **Registration**: An optional registration module to align the generated CECT with the ground truth during training.

See [Model/HeadCECT_GAN.py](Model/HeadCECT_GAN.py) for details.

## Environment Requirements
- Python >= 3.6
- PyTorch >= 1.9.0
- visdom
- numpy
- scikit-image
- PyYAML
- OpenCV (cv2)
- Pillow (PIL)

## Usage

### 1. Data Preparation
Organize your dataset as follows:
- train/A/ : Non-contrast CT images (Source)
- train/B/ : Contrast-enhanced CT images (Target)
- val/A/   : Validation source images
- val/B/   : Validation target images

The default data loader expects .npy files normalized to [-1, 1] or standard image formats.

### 2. Configuration
Modify Yaml/HeadCECT_GAN.yaml to set your parameters:
- **dataroot**: Path to your training data.
- **val_dataroot**: Path to your validation data.
- **bidirect**: True for bidirectional cycle consistency (A->B and B->A).
- **regist**: True to enable the registration network (recommended for unaligned data).
- **noise_level**: Data augmentation noise level.

### 3. Visualization
Start the Visdom server to monitor training:
`ash
python -m visdom.server -p 6019
`
(Ensure the port matches the one in Yaml/HeadCECT_GAN.yaml)

### 4. Training
Run the training script with the configuration file:
`ash
python train.py --config Yaml/HeadCECT_GAN.yaml
`

### 5. Testing / Inference
To generate images using a trained model:
`ash
python test.py --config Yaml/HeadCECT_GAN.yaml
`
Output images will be saved to the directory specified in save_root in the yaml file.

## Acknowledgements
This code is based on the implementation of **Reg-GAN**.
If you find the base method useful, please cite:
`ibtex
@inproceedings{
kong2021breaking,
title={Breaking the Dilemma of Medical Image-to-image Translation},
author={Lingke Kong and Chenyu Lian and Detian Huang and ZhenJiang Li and Yanle Hu and Qichao Zhou},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=C0GmZH2RnVR}
}
`
"@
Set-Content -Path "D:\Research\AIGC-Med\HeadCECT-GAN\README.md" -Value @"
# HeadCECT-GAN

This repository contains the implementation of **HeadCECT-GAN**, a model designed to generate **Contrast-Enhanced CT (CECT)** images from non-contrast head CT scans.

## 1. Environment Setup

Ensure you have a Python environment (recommend 3.6+) with the following dependencies:
- PyTorch (>= 1.9.0)
- Visdom (visualization)
- Numpy, Scikit-image, PIL, OpenCV
- PyYAML

## 2. Data Preparation

Organize your dataset in the following structure:
- \	rain/A/\: Source images (Non-contrast CT)
- \	rain/B/\: Target images (Contrast-Enhanced CT)
- \al/A/\: Validation source images (Non-contrast CT)
- \al/B/\: Validation target images (Contrast-Enhanced CT)

The model expects input files (e.g., \.npy\) to be normalized to the range \[-1, 1]\.

## 3. Configuration

Modify Yaml/HeadCECT_GAN.yaml to set your training parameters:
- **dataroot**: Path to your dataset folder.
- **regist**: Set to \True\ to enable the registration network (recommended if A and B are not perfectly aligned).
- **bidirect**: Set to \True\ for bidirectional learning (A->B and B->A cycle consistency).
- **visdom port**: Default is 6019.

## 4. Training

1. **Start Visdom Server**:
   Open a terminal and run:
   \\\ash
   python -m visdom.server -p 6019
   \\\

2. **Run Training**:
   Open another terminal and run:
   \\\ash
   python train.py --config Yaml/HeadCECT_GAN.yaml
   \\\

## 5. Testing / Inference

To generate images using the trained model:
\\\ash
python test.py --config Yaml/HeadCECT_GAN.yaml
\\\
The results will be saved in the directory specified by \save_root\ in the config file.
