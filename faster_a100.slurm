#!/bin/bash

#SBATCH --partition=gpu         # Queue (partition) name
#SBATCH -c 1                    # Total # of nodes
#SBATCH --ntasks=1              # Number of MPI ranks per node (one rank per GPU)
#SBATCH --mem=32G 
#SBATCH --gres=gpu:a100:1       # Number of GPUs to use
#SBATCH --time=48:00:00         # Total run time limit (hh:mm:ss)

## Export all environment variables to the job
#SBATCH --export=all

# Manage processing environment, load compilers, and applications

# set -x

ml CUDA/12.4.1
source /scratch/user/u.kj97065/miniconda3/bin/activate /scratch/user/u.kj97065/.conda/envs/diffusion

# Launch GPU code
python -m tools.sample_ddpm --config config/config_faster_a100.yaml --run_num "$@"