#!/bin/bash

## Job Name
#SBATCH --job-name=myjob

## Resources
#SBATCH -A atm170004p
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=1
#SBATCH -t 48:00:00

## Export all environment variables to the job
#SBATCH --export=all

## Source bashrc and activate conda environment
# source /jet/home/jakhar/.bashrc
source /jet/home/jakhar/miniconda3/bin/activate /ocean/projects/atm170004p/jakhar/.envs/diffusion
module load cuda

## Run the python script
python -m tools.sample_ddpm --config config/config_bridges.yaml --run_num "$@"