import torch
import argparse
import yaml
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from torch.amp import autocast

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import barrier
import torch.distributed as dist

# Add the project root directory to sys.path for proper imports
# This ensures imports work regardless of whether script is run with python -m or torchrun
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from scheduler.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
from dataset.dataloader import CustomMatDataset
from eval.analysis.plots import save_image
from tools.logging_utils import log_print
from tools.distributed_data_parallel_utils import ddp_setup, ddp_cleanup

if torch.cuda.is_available():
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:  # Only rank 0 prints
        print("Number of GPUs available:", torch.cuda.device_count())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ID = int(os.environ["LOCAL_RANK"])
if device_ID == 0:  # Only rank 0 prints
    print("Device:", device, "  |  Device ID:", device_ID)

# Get distributed info early
if torch.distributed.is_initialized():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
else:
    world_size = 1
    rank = 0

def sample_turb(model, scheduler, train_config, sample_config, model_config, diffusion_config, dataset_config, logging_config, run_num):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """

    log_to_screen = logging_config['log_to_screen']
    diagnostic_logs = logging_config['diagnostic_logs']

    # -------- main-process seeding -------------------------------------------
    GLOBAL_ = sample_config['global_seed']
    if GLOBAL_ is not None:
        log_print(f"Setting global seed to {GLOBAL_}", log_to_screen=log_to_screen)
        random.seed(GLOBAL_)
        np.random.seed(GLOBAL_)
        torch.manual_seed(GLOBAL_)

    if 'mnist' in dataset_config['data_dir'].lower():
        # mnist dataset is not normalized
        pass
    elif dataset_config['normalize']:
        mean_std_data = np.load(dataset_config['data_dir'] + 'mean_std.npz')
        mean = np.asarray([mean_std_data['U_mean'], mean_std_data['V_mean']])
        std = np.asarray([mean_std_data['U_std'], mean_std_data['V_std']])

        mean_tensor = torch.tensor(mean.reshape(mean.shape[0], 1, 1)).to(device_ID)
        std_tensor = torch.tensor(std.reshape(std.shape[0], 1, 1)).to(device_ID)

    # Precompute timesteps once
    timesteps = [torch.tensor(i, device=device_ID).unsqueeze(0) for i in range(diffusion_config['num_timesteps'])]

    # Create directory for saving generated data if required
    if sample_config['save_data']:
        os.makedirs(os.path.join(train_config['save_dir'], 'data', run_num), exist_ok=True)

        # List all .npy files in the directory
        npy_files = [f for f in os.listdir(os.path.join(train_config['save_dir'], 'data', run_num)) if f.endswith('.npy')]

        if npy_files:
            # Extract numeric file numbers from filenames
            file_numbers = [int(os.path.splitext(f)[0]) for f in npy_files]
            largest_file_number = max(file_numbers)
            log_print(f"Data exists in the directory. Largest file number: {largest_file_number}", log_to_screen=log_to_screen)

        else:
            largest_file_number = -1  # No files found
            log_print("No .npy files found in the directory.", log_to_screen=log_to_screen)    

    # Check if all batches already exist
    if largest_file_number >= (sample_config['num_sample_batch']-1):
        log_print(f"All {sample_config['num_sample_batch']} batches already exist. Exiting.", log_to_screen=True)
        sys.exit(0)

    if diffusion_config['conditional']:
     # ----- NEW: seed with real frame t0 (selected by sample_file_start_idx) -----

        if npy_files:
            # If files exist, load the second-last one (avoiding errorsif last saved file is corrupted)
            log_print(f"Loading last saved batch: {largest_file_number}", log_to_screen=log_to_screen)
            largest_file_number -= 1

            idx_arr = [largest_file_number - step for step in range(0, dataset_config['num_prev_conditioning_steps'])]
            data_tensor_list = [np.load(os.path.join(train_config['save_dir'], 'data', run_num, f'{idx}.npy')) for idx in idx_arr] 
            # Each list in data_tensor_list has shape (B, C, H, W)

            # Concatenate each list along the channel dimension
            batch_cond = torch.from_numpy(np.concatenate(data_tensor_list, axis=1)) # [B, (T-1)*C, H, W]
            # print('idx_arr: ', idx_arr)
            # print('data_tensor shape: ', batch_cond.shape)
            # sys.exit()

        else:
            # Initiate dataloader
            seed_dataset = CustomMatDataset(dataset_config, train_config, sample_config, logging_config, training=False, conditional=True)
            t0_tensor = seed_dataset[0] # [T, C, H, W]

            T, C, H, W = t0_tensor.shape
            batch_cond = t0_tensor[1:, ...].reshape(1, (T-1)*C, H, W) # [T-1, C, H, W] -> [1, (T-1)*C, H, W]

            log_print(f'batch_cond shape: {batch_cond.shape}', log_to_screen=diagnostic_logs)

        batch_cond = batch_cond.float().to(device_ID)
    else:
        batch_cond = None


    # Loop over rounds of generation (each GPU generates one file per round)
    current_file_number = largest_file_number
    target_files = sample_config['num_sample_batch']

    while current_file_number < target_files:
        # Each GPU gets a unique file number in this round
        file_number = current_file_number + 1 + rank
        
        # Skip if this rank's file number exceeds the target
        if file_number >= target_files:
            break

        xt = torch.randn((sample_config['sample_batch_size'],
                        model_config['pred_channels'],
                        model_config['im_size'],
                        model_config['im_size'])).float().to(device_ID)
        
        if sample_config['sampler'] == 'ddpm':
            for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):

                t_tensor = timesteps[i].to(device_ID)  # Get the current timestep tensor

                with autocast('cuda'):
                    # Get prediction of noise
                    noise_pred = model(xt, t_tensor, cond=batch_cond)
                    
                    # Use scheduler to get x0 and xt-1
                    xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, t_tensor.squeeze(0))

        elif sample_config['sampler'] in ['dpm-solver', 'dpm-solver++']:

            # Continuous-time noise schedule built from *existing* betas
            ns = NoiseScheduleVP('discrete', 
                                  alphas_cumprod=scheduler.alpha_cum_prod)  # :contentReference[oaicite:0]{index=0}
 
            # Wrap the UNet so that it accepts *continuous* t ∈ (0, 1]
            model_kwargs = dict(cond=batch_cond)
            model_fn = model_wrapper(model,
                                      noise_schedule=ns,
                                      model_type='noise' if train_config['loss']=='noise' else 'x_start',
                                      model_kwargs  = model_kwargs,
                                      guidance_type = sample_config['dpm_guidance'],)              
             
            dpm_solver = DPM_Solver(model_fn, ns, algorithm_type="dpmsolver++")

            xt = dpm_solver.sample(xt,
                                    steps      = sample_config['dpm_steps'],
                                    t_start    = 1.0,                  # corresponds to DDPM t = num_timesteps-1
                                    t_end      = 1.0 / ns.total_N,     # ≈1e-3
                                    order      = sample_config['dpm_order'],
                                    method     = sample_config['dpm_method'],
                                    skip_type  = sample_config['dpm_skip'],)

                
        if sample_config['save_image'] or file_number < 5:
            diffusion_timestep = 0
            save_image(xt, diffusion_timestep, train_config, sample_config, dataset_config, run_num, file_number)

        if diffusion_config['conditional']:
            # Shift batch_cond to remove the last time step and add the new one at the beginning
            # batch_cond: shape [1, (T-1)*C, H, W]
            # Remove last C channels, prepend new C channels from xt
            _, C, H, W = xt.shape
            batch_cond = torch.cat(
                (xt.detach(), batch_cond[:, :-C, :, :].detach()),
                dim=1
            )

            log_print(f'xt shape: {xt.shape}', log_to_screen=diagnostic_logs)
            log_print(f'batch_cond shape: {batch_cond.shape}', log_to_screen=diagnostic_logs)

            xt_final = xt[:, :model_config['pred_channels'], :, :].detach()
        else:
            xt_final = xt.detach()

        # Synchronize all processes after sampling
        if torch.distributed.is_initialized():
            barrier(device_ids=[device_ID])

        if sample_config['save_data']:
            
            if 'mnist' in dataset_config['data_dir'].lower():
                pass
            elif dataset_config['normalize']:
                xt_final.mul_(std_tensor).add_(mean_tensor)

                xt_cpu = xt_final.cpu()


            # Fast parallel saving: each GPU saves its own file with sequential numbering
            np.save(os.path.join(train_config['save_dir'], 'data', run_num, f'{file_number}.npy'), xt_cpu.numpy())
            log_print(f"Rank {rank}: Saved file {file_number} ({xt_cpu.shape[0]} samples)", log_to_screen=log_to_screen)

        # Barrier to ensure all ranks complete this round before moving to next
        if torch.distributed.is_initialized():
            barrier(device_ids=[device_ID])
            
        # Update for next round
        current_file_number += world_size

    # Final synchronization to ensure all ranks complete before finishing
    if torch.distributed.is_initialized():
        barrier(device_ids=[device_ID])
        log_print(f"All ranks completed. Generated files up to {current_file_number + world_size - 1}", log_to_screen=log_to_screen)


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            log_to_screen(f"Error in configuration file: {exc}", Force=True)
            return
    
    # Extract logging config to control verbosity
    dataset_config = config['dataset_params']
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    sample_config = config['sample_params']
    logging_config = config['logging_params']
    run_num = args.run_num
    
    log_to_screen = logging_config['log_to_screen']
    diagnostic_logs = logging_config['diagnostic_logs']

    log_print('Configuration loaded successfully', log_to_screen=log_to_screen)
    log_print(f'Full config: {config}', log_to_screen=diagnostic_logs)
    ########################

    # Unconditional modeling has no conditioning channels
    if not diffusion_config['conditional']:
        model_config['cond_channels'] = 0
    
    # Create model and load checkpoint
    model = Unet(model_config)
    model = model.to(device_ID)

    # Wrap with DistributedDataParallel for multi-GPU training
    model = DDP(model, device_ids=[device_ID], output_device=device_ID)

    # Create output directories & task name
    if train_config['model_collapse']:
        task_name = train_config['task_name'] + '_' + train_config['model_collapse_type'] + '_' + str(train_config['model_collapse_gen'])
    else:
        task_name = train_config['task_name']
    
    # Make results path relative to project root
    train_config['save_dir'] = os.path.join(project_root, 'results', task_name)

    print(os.path.join(train_config['save_dir'], train_config['best_ckpt_name']))

    # Load weights/checkpoint if found
    if os.path.exists(os.path.join(train_config['save_dir'], train_config['best_ckpt_name'])):
        log_print(f'Loading trained model from {os.path.join(train_config["save_dir"], train_config["best_ckpt_name"])}', log_to_screen=log_to_screen)
        checkpoint = torch.load(os.path.join(train_config['save_dir'], train_config['best_ckpt_name']), weights_only=False)

        # Load model
        model.load_state_dict(checkpoint['model_state'])
    else:
        log_print(f"Checkpoint {train_config['best_ckpt_name']} not found in {train_config['save_dir']}. Exiting.", log_to_screen=True)
        sys.exit(0)

    model.eval()
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample_turb(model, scheduler, train_config, sample_config, model_config, diffusion_config, dataset_config, logging_config, run_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for DDPM image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/config.yaml', type=str,
                        help='Path to the configuration file')
    parser.add_argument('--run_num', dest='run_num', 
                        default='1', type=str,
                        help='Run number for the experiment')

    args = parser.parse_args()
    
    # Make config path relative to project root if it's a relative path
    if not os.path.isabs(args.config_path):
        args.config_path = os.path.join(project_root, args.config_path)

    ddp_setup() # Initialize distributed sampling environment
    try:
        # Call function with the parsed arguments
        infer(args)
    finally:
        ddp_cleanup()  # Clean up distributed sampling environment