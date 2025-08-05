# StrokeAI: Accurate Classification and Segmentation of Acute Ischemic Stroke Lesions on Non-Contrast CT

This repository contains all code, models, and data processing scripts for the project **"Accurate Classification and Segmentation of Acute Ischemic Stroke Lesions on Non-Contrast CT with Deep Learning Methods."**

The goal of this project is to automate detection and localization of **Acute Ischemic Stroke (AIS)** lesions using deep learning models applied to **non-contrast CT (NCCT)** scans. It includes both **classification** (stroke vs. no stroke) and **segmentation** (localizing ischemic lesions).

## 🚀 Features

- **Classification**: ResNet50 and Vision Transformers (ViT) models trained to classify NCCT scans.
- **Segmentation**: U-Net variants (AUIS, classic Attention U-Net, UNet3D, ResUNet3D, SwinUNETR, nnUNet) for precise lesion localization.
- **Evaluation Metrics**: Dice score, Center-of-Mass (COM) distance, Bounding Box IoU.
- **Preprocessing pipeline**: Includes CT-to-MRI mapping, random cropping around lesions, and data normalization.
- **Multi-GPU support**: Training scripts designed for DistributedDataParallel (PyTorch) and multi-GPU training (TensorFlow).
