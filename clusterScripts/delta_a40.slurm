#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --account=bdiu-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --partition=gpuA40x4        # gpuA40x4 gpuA100x4
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # could be 1 for py-torch
#SBATCH --cpus-per-task=1           # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest          # select a cpu close to gpu on pci bus topology
#SBATCH -t 24:00:00
#SBATCH --export=all


module load cuda/12.4.0
source /work/nvme/bdiu/jakharkaran/miniconda3/bin/activate /work/nvme/bdiu/jakharkaran/miniconda3/envs/diffusion

# Launch GPU code
python -m tools.sample_ddpm --config config/config_delta.yaml --run_num "$@"

# cd ..
# Launch GPU code
# python -m tools.train_ddpm --config config/UDM_s1_Oa_lrC1e4_wd1e6_bilinear.yaml
# python -m tools.sample_ddpm --config config/UDM_s1_Oa_lrC1e4_wd1e6_bilinear.yaml

# python -m tools.train_ddpm --config "$@"
# python -m tools.sample_ddpm --config "$@"