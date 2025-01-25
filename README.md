Denoising Diffusion Probabilistic Models Implementation in Pytorch
========

This repository implements [DDPM](https://arxiv.org/abs/2006.11239) with training and sampling methods of DDPM and unet architecture mimicking the stable diffusion unet used in diffusers library from huggingface.

# Quickstart
* ```python -m tools.train_ddpm``` for training ddpm
* ```python -m tools.sample_ddpm --config config/config.yaml --run_num 1``` for generating images

## Configuration
* ```config/config.yaml``` - Allows you to play with different components of ddpm  


