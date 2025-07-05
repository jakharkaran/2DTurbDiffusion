Denoising Diffusion Probabilistic Models (DDPM)
========

This repository implements Unconditional and Conditional [DDPM](https://arxiv.org/abs/2006.11239) with training and sampling methods of DDPM and unet architecture mimicking the stable diffusion unet used in diffusers library from huggingface.

# Quickstart - Single Node; Multi GPU
* ```torchrun --standalone --nproc_per_node=gpu tools/train_ddpm.py --config config/config.yaml``` for training ddpm
* ```torchrun --standalone --nproc_per_node=gpu tools/sample_ddpm.py --config config/config.yaml --run_num 1``` for sampling images
* ```python eval/analyze_model.py``` for evaluation of samples 

## Configuration
* ```config/config.yaml``` - ddpm training and sampling
* ```eval/config/onfig.yaml``` - evaluation of samples

## Install Relevant Packages
* ```pip3 install torch torchvision torchaudio scipy numpy matplotlib einops tqdm pyyaml```

## Install py2d
```
git clone https://github.com/envfluids/py2d.git
pip install -e ./
```
