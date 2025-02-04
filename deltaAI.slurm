#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --account=bdiu-dtai-gh

#SBATCH --partition=ghx4
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --mem-bind=verbose,local
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --export=all

module load cuda/12.6.1
source /projects/bdiu/jakharkaran/miniconda3/bin/activate /projects/bdiu/jakharkaran/miniconda3/envs/diffusion

# Launch GPU code
python -m tools.sample_ddpm --config config/config_deltaAI.yaml --run_num "$@"