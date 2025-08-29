# Denoising Diffusion Probabilistic Models for FDF generation

This project implements Denoising Diffusion Probabilistic Models (DDPM) and their conditional variants, 
including classifier-free sampling and regressor-guided sampling, for studying the generizability of them on low and high Ka data.

## Project Structure

```
Denoising_Diffusion/
├── common/
│   ├── __init__.py
│   ├── schedules.py             # Beta schedule in DDPM
│   ├── preprocessing.py         # Log/BCT transforms
│   ├── data_utils.py            # Low/high Ka data loaders
│   ├── sampling.py              # Batch sampler
│   ├── plotting.py              # Comparison scatter and heatmaps
│   └── training_utils.py        # Training loop utilities
├── models/
│   ├── __init__.py
│   ├── DDPM_unet.py             # U-Net without condition
│   ├── cDDPM_unet.py            # U-Net with condition
│   └── Simple_regressor.py      # Regressor for guidance
├── samplers/
│   ├── __init__.py
│   ├── ddpm.py                  # DDPM + DDIM sampler
│   ├── ddpm_regressor.py        # DDIM with regressor guidance
│   └── cddpm.py                 # cDDIM sampler
├── training/
│   ├── train_ddpm.py            # Unconditional training script
│   ├── train_cddpm.py           # Conditional training script
│   └── train_regressor.py       # Regressor training script
├── inference/
│   ├── infer_ddpm.py            # Inference with/without regressor guidance
│   └── infer_cddpm.py           # Inference with cDDIM
├── Data/                        # Input data (*.npy)
├── Weight/                      # Model weights (*.pth)
└── requirements.txt             # Dependencies
```

## Installation

Clone this repository and install the dependencies:
```bash
git clone <your-repo-url>.git
cd DDPM
pip install -r requirements.txt
```

## Usage

### Training
- Unconditional DDPM:
```bash
python training/train_ddpm.py
```
- Conditional DDPM:
```bash
python training/train_cddpm.py
```
- Train regressor:
```bash
python training/train_regressor.py
```

### Inference
- Unconditional (with or without regressor guidance):
```bash
python inference/infer_ddpm.py
```
- Conditional:
```bash
python inference/infer_cddpm.py
```

## Reference

This implementation is mainly based on the following paper(s):
- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
- Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, ICML 2021.
- Nichol & Dhariwal, *Diffusion Models Beat GANs on Image Synthesis*, NeurIPS 2021.
- Ho et al., *Classifier-free Diffusion Guidance*, NeurIPS 2021.

Please cite these works if you use this code in your research.
