import torch
import yaml
import argparse
import os, sys
import shutil
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import random
from pathlib import Path

from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
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

# Importing custom modules
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from dataset.dataloader import CustomMatDataset
from dataset.mnist_dataset import MnistDataset
from tools.util import SpectralDifferentiator, grad_norm, grad_max, generate_grid
from tools.logging_utils import log_print
from tools.distributed_data_parallel_utils import ddp_setup, ddp_cleanup

# Packages to outline architecture of the model
# from torchviz import make_dot
# from torchinfo import summary

ddp_setup() # Initialize distributed training environment

# Get distributed info early
if torch.distributed.is_initialized():
    world_size = dist.get_world_size()
    device_ID = dist.get_rank() # rank
else:
    world_size = 1
    device_ID = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device:", device, "  |  World size (# GPUs):", world_size, "  |  Device ID (Rank):", device_ID)

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            log_print(f"Error in configuration file: {exc}", log_to_screen=True)
            sys.exit(1)
    
    ########################
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    logging_config = config['logging_params']
    sample_config = config['sample_params']


    # Extract logging config to control verbosity
    log_to_screen = logging_config['log_to_screen']
    diagnostic_logs = logging_config['diagnostic_logs']
    
    log_print('Configuration loaded successfully', log_to_screen=log_to_screen)
    log_print(f'Full config: {config}', log_to_screen=diagnostic_logs)

    GLOBAL_SEED = train_config['global_seed']
    # -------- main-process seeding -------------------------------------------
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    # Create the noise scheduler
    diffusion_noise_scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Set the seed for reproducibility in the dataloader
    def worker_init_fn(worker_id):
        # Each worker gets a *unique* but *deterministic* stream
        worker_seed = GLOBAL_SEED + worker_id
        np.random.seed(worker_seed)          # NumPy ops in transforms
        random.seed(worker_seed)             # Python’s `random` calls
        torch.manual_seed(worker_seed)       # Torch ops inside the Dataset
    dl_gen = torch.Generator().manual_seed(GLOBAL_SEED)

    # Initiate dataloader

    # Validate effective batch size is evenly divisible by world size
    if train_config['effective_batch_size'] % world_size != 0:
        log_print(f"Warning: Effective batch size {train_config['effective_batch_size']} is not evenly divisible by world size {world_size}. "
                 f"This may result in data loss. Consider adjusting effective_batch_size to be a multiple of {world_size}.", 
                 log_to_screen=True)
    train_config['batch_size'] = train_config['effective_batch_size'] // world_size  # Effective batch size divided by number of GPUs

    if 'mnist' in dataset_config['data_dir'].lower():
        # MNIST dataset for testing
        log_print('** Using MNIST dataset for testing **', log_to_screen=log_to_screen)
        mnist = MnistDataset('train', im_path=dataset_config['data_dir'])
        turb_dataloader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=False, num_workers=4, 
                                     sampler=DistributedSampler(mnist, shuffle=True))
    else:
        # Turbulence dataset
        dataset = CustomMatDataset(dataset_config, train_config, sample_config, logging_config, conditional=diffusion_config['conditional'])
        turb_dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, \
                                    generator=dl_gen, worker_init_fn=worker_init_fn, sampler=DistributedSampler(dataset, shuffle=True)) # No shuffle for DistributedSampler
        
    # Ensuring 0 condtional channels if not conditional
    if not diffusion_config['conditional']:
        model_config['cond_channels'] = 0 

    # Instantiate the model
    model = Unet(model_config)
    model = model.to(device_ID) # Move model to device first, then wrap with DDP
    
    # Wrap with DistributedDataParallel for multi-GPU training
    model = DDP(model, device_ids=[device_ID], output_device=device_ID)
    model.train()
    
    # Create output directories &   ### Saving config file with the model weights
    if train_config['model_collapse']:
        save_dir = os.path.join(project_root, 'results', train_config['task_name'] + '_' + train_config['model_collapse_type'] + '_' + str(train_config['model_collapse_gen']))
    else:
        save_dir = os.path.join(project_root, 'results', train_config['task_name'])

    # 2D Turbulence System Parameter
    nx, ny = int(model_config['im_size']/dataset_config['downsample_factor']), int(model_config['im_size']/dataset_config['downsample_factor'])
    Lx, Ly = 2 * torch.pi, 2 * torch.pi
    # Kx, Ky, _, _, invKsq = initialize_wavenumbers_rfft2(nx, ny, Lx, Ly, INDEXING='ij')

    # --- Instantiate the Differentiator ---
    diff = SpectralDifferentiator(nx=nx, ny=ny, Lx=Lx, Ly=Ly, device=device_ID)

    if train_config['divergence_loss']:
        # Load the mean and std for de-normalization
        mean_std_data = np.load(dataset_config['data_dir'] + 'mean_std.npz')
        mean = np.asarray([mean_std_data['U_mean'], mean_std_data['V_mean']])
        std = np.asarray([mean_std_data['U_std'], mean_std_data['V_std']])

        mean_tensor = torch.tensor(mean.reshape(1, mean.shape[0], 1, 1)).to(device_ID)
        std_tensor = torch.tensor(std.reshape(1, std.shape[0], 1, 1)).to(device_ID)

        if train_config['divergence_loss_type'] == 'denoise_sample':
            # Precompute timesteps for denoising
            timesteps = [torch.tensor(i, device=device_ID).unsqueeze(0) for i in range(diffusion_config['num_timesteps'])]

    log_print(f'Saving config to: {save_dir}', log_to_screen=log_to_screen)
    if not os.path.exists(save_dir) and device_ID == 0:
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy(args.config_path, save_dir)

    # Specify training parameters
    num_epochs = train_config['num_epochs']

    if train_config['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=float(train_config['lr']), weight_decay=float(train_config['weight_decay']))
    elif train_config['optimizer'] == 'adamw':
        optimizer = AdamW(model.parameters(), lr=float(train_config['lr']), weight_decay=float(train_config['weight_decay']), fused=True)

    # Set learning rate scheduluer
    if train_config["scheduler"] == 'ReduceLROnPlateau':
        LRscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, cooldown=10, mode='min', min_lr=float(train_config['lr_min']))
    elif train_config["scheduler"] == 'CosineAnnealingLR':
        LRscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_config["num_epochs"]), eta_min=float(train_config['lr_min']))
    else:
       LRscheduler = None

    # Warm up epochs if using
    if train_config['warmup']:
        warmuplr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=train_config['warmup_start_factor'],
                                                              total_iters=train_config['warmup_total_iters'])

    # Set up logging with wandb - only on main process
    if logging_config['log_to_wandb'] and device_ID == 0:

        # Set up wandb
        wandb.login()
        wandb_id_path = Path(save_dir) / "wandb_id.txt"

        if wandb_id_path.exists():               # Resume run
            run_id  = wandb_id_path.read_text().strip()
            resume  = "allow"                    # "must" works too
        else:                                    # fresh run
            run_id  = wandb.util.generate_id()   # or uuid.uuid4().hex
            wandb_id_path.write_text(run_id)
            resume  = None                       # start fresh

        wandb.init(project=logging_config['wandb_project'], group=logging_config['wandb_group'],
                   name=logging_config['wandb_name'], config=config, id=run_id, resume=resume)

        # Watch model gradients with wandb
        wandb.watch(model, log="all", log_freq=logging_config['wandb_table_logging_interval'],)

    # Load checkpoint if found
    if os.path.exists(os.path.join(save_dir, train_config['ckpt_name'])):
        log_print('Loading existing checkpoint', log_to_screen=log_to_screen)
        checkpoint = torch.load(os.path.join(save_dir, train_config['ckpt_name']), weights_only=False)
        log_print(f'Checkpoint keys: {list(checkpoint.keys())}', log_to_screen=diagnostic_logs)
        
        # Load model and optimizer state
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Restore training state
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        iteration = checkpoint['iteration']
        
        log_print(f'Checkpoint loaded: epoch {start_epoch}, best_loss {best_loss:.2e}, iteration {iteration}', log_to_screen=True)
    else:
        start_epoch = 0
        best_loss = 1e6
        iteration = 0

    criterion = torch.nn.MSELoss()

    noise_cond = None
    # Run training
    for epoch_idx in range(start_epoch, num_epochs):
        epoch = epoch_idx + 1
        losses = []
        
        # Lists to store per-iteration metrics for batch logging to wandb
        iteration_metrics = []
        
        # Set epoch for DistributedSampler to ensure proper data shuffling across epochs
        if isinstance(turb_dataloader.sampler, DistributedSampler):
            log_print(f'Setting epoch for DistributedSampler: {epoch_idx}', log_to_screen=diagnostic_logs)
            turb_dataloader.sampler.set_epoch(epoch_idx)

        for batch_data in tqdm(turb_dataloader):

            iteration += 1

            optimizer.zero_grad()

            if diffusion_config['conditional']:
                B, T, C, H, W = batch_data.shape # [B, T, C, H, W]; T: t, t-1, t-2, ...; C: U, V

                # Split batch_data along T dimension
                batch_im = batch_data[:, 0, ...]      # [B, C, H, W] (first T)
                # [B, T-1, C, H, W] (remaining T), Stack T-1 and C into channel dimension -> [B, (T-1)*C, H, W]
                batch_cond = batch_data[:, 1:, ...].reshape(B, (T-1) * C, H, W)   
                batch_cond = batch_cond.float().to(device_ID)
            else:
                batch_cond = None

                if 'mnist' in dataset_config['data_dir'].lower():
                    # For MNIST dataset, batch_data is [B, C, H, W], add T dimension
                    batch_im = batch_data
                else:
                    # For unconditional model, batch_data is [B, 1, C, H, W], remove T dimension
                    batch_im = batch_data[:, 0, :, :, :] # [B, C, H, W]

            if train_config['divergence_loss'] and train_config['divergence_loss_type'] == 'denoise_sample' and train_config['loss'] == 'noise':
                # Calculate split sizes for the two branches
                batch_size = batch_im.shape[0]
                div_ratio = train_config['denoise_sample_batch_ratio']
                bs_div = int(batch_size * div_ratio)

                if bs_div == 0:
                    log_print('Warning: No data for divergence loss batch. Setting to 1.', log_to_screen=True)
                    bs_div=1

                bs_mse = batch_size - bs_div
                log_print(f'Batch split - MSE: {bs_mse}, Divergence: {bs_div}', log_to_screen=log_to_screen)

                im = batch_im[:bs_mse].float().to(device_ID)
                batch_div = batch_im[bs_mse:].float().to(device_ID)
            else:
                im = batch_im.float().to(device_ID)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device_ID)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device_ID)
            noisy_im = diffusion_noise_scheduler.add_noise(im, noise, t)

            model_out = model(noisy_im, t, cond=batch_cond)

            log_print(f'Model out shape:  {model_out.shape}', log_to_screen=diagnostic_logs)

            # make_dot(model_out, params=dict(model.named_parameters())).render("model_architecture", format="png")
            # summary(model, input_size=(10, 10))


            if train_config['loss'] == 'noise':
                # Noise is predicted by the model
                if diffusion_config['conditional']:

                    # noise_cond, noise_im  = torch.split(
                    #     noise, 
                    #     [model_config['im_channels'] * dataset_config['num_prev_conditioning_steps'],model_config['im_channels']], 
                    #     dim=1
                    # )
                    # print([model_config['pred_channels'] * dataset_config['num_prev_conditioning_steps'],model_config['pred_channels']])
                    # noise_cond, noise_im  = torch.split(
                    #     noise, 
                    #     [model_config['pred_channels'] * dataset_config['num_prev_conditioning_steps'],model_config['pred_channels']], 
                    #     dim=1
                    # )

                    # print('Noise shape: ', noise.shape, noise_cond.shape, noise_im.shape)
                    # sys.exit()
                    # noise_pred, noise_cond = model_out.split(2, dim=1)
                    # noise_cond = noise_cond.detach() 
                    # Exclude conditional channels from loss
                    # loss_mse = criterion(noise_pred, noise[:, :model_config['im_channels']//2,:,:])
                    # loss_mse = criterion(model_out, noise[:, :model_config['im_channels']//2,:,:])
                    loss_mse = criterion(model_out, noise)

                else:
                    loss_mse = criterion(model_out, noise) 

            elif train_config['loss'] == 'sample':
                # x0 (denoised sample) is predicted by the diffusion model
                if diffusion_config['conditional']:
                    # Exlude conditional channels from loss
                    loss_mse = criterion(model_out[:, :model_config['im_channels']//2,:,:], im[:, :model_config['im_channels']//2,:,:])
                else:
                    loss_mse = criterion(model_out, im) 

            ###### Divergence Loss
            if train_config['divergence_loss']:
                # model.eval()

                # Method # 1: If loss is in noise: Calculate x_0 directly from predicted_noise in a single step
                if train_config['divergence_loss_type'] == 'direct_sample' and train_config['loss'] == 'noise':
                    x0_pred = diffusion_noise_scheduler.sample_x0_from_noise(noisy_im, model_out, t) # Get x0 from xt and noise_pred

                # Method # 2: If loss is in noise: Calculate x_0 directly via denoising 
                if train_config['divergence_loss_type'] == 'denoise_sample' and train_config['loss'] == 'noise':

                    # Sample timestep
                    timesteps_div = [torch.tensor(i, device=device_ID).unsqueeze(0) for i in range(train_config['denoise_sample_timestep'])]
                    # Sample random noise
                    noise_div = torch.randn_like(batch_div).to(device_ID)

                    # Add noise to images according to timestep
                    t_div_noise = torch.full((batch_div.shape[0],), train_config['denoise_sample_timestep']-1, dtype=torch.int).to(device_ID)
                    noisy_im_div = diffusion_noise_scheduler.add_noise(batch_div, noise_div, t_div_noise)

                    for i in tqdm(reversed(range( train_config['denoise_sample_timestep']))):
                            
                        # Get prediction of noise
                        t_tensor = timesteps_div[i]
                        noise_pred_div = model(noisy_im_div, t_tensor)
                        
                        # Use scheduler to get x0 and xt-1
                        noisy_im_div, _ = diffusion_noise_scheduler.sample_prev_timestep_from_noise(noisy_im_div, noise_pred_div, t_tensor.squeeze(0))

                    x0_pred = noisy_im_div
                    # x0_pred = xt.detach().clone()

                    # del xt, noise_pred
                    # torch.cuda.empty_cache()

                    # Revierse diffusion process with model_out
                # Method # 2: If loss is in noise: Calculate x_0 directly via denoising 
                # if train_config['divergence_loss_type'] == 'denoise_sample' and train_config['loss'] == 'noise':
                #     with torch.inference_mode():

                #         # Compute Divergence
                #         xt = torch.randn((sample_config['batch_size'],
                #         model_config['im_channels'],
                #         model_config['im_size'],
                #         model_config['im_size'])).to(device_ID)

                #         for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
                                
                #             # with autocast('cuda'):
                #             # Get prediction of noise
                #             t_tensor = timesteps[i]
                #             noise_pred = model(xt, t_tensor)
                            
                #             # Use diffusion_noise_scheduler to get x0 and xt-1
                #             xt, _ = diffusion_noise_scheduler.sample_prev_timestep_from_noise(xt, noise_pred, t_tensor.squeeze(0))

                #         x0_pred = xt.detach().clone()

                #     del xt, noise_pred
                #     torch.cuda.empty_cache()

                # Method # 3: If loss is in sample
                if train_config['divergence_loss_type'] == 'direct_sample' and train_config['loss'] == 'sample':
                    x0_pred = model_out

                # De-normalize the data
                if dataset_config['normalize']:
                    x0_pred = x0_pred.mul(std_tensor).add(mean_tensor)
                    log_print(f'x0_pred shape: {x0_pred.shape}', log_to_screen=diagnostic_logs)
                    div = diff.divergence(x0_pred, spectral=False)

                loss_div = train_config['divergence_loss_weight'] * torch.mean(torch.abs(div))
                loss = loss_mse + loss_div


                # model.train()


            else:
                loss = loss_mse
                loss_div = 0.0  # Set to 0 when divergence loss is not used

            losses.append(loss.item())
            loss.backward()

            if train_config['clip_grad_norm'] is not None:
                # ——— gradient clipping & norm ———
                #  this returns the total norm before clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                    max_norm=float(train_config['clip_grad_norm']),
                )

            # Calculate maximum gradient value across all parameters
            batch_grad_norm = grad_norm(model)
            batch_grad_max = grad_max(model)
            
            # Store metrics for this iteration
            if logging_config['log_to_wandb'] and device_ID == 0:
                iter_metrics = {
                    "iteration": iteration,
                    "batch_loss": loss.item(),
                    "batch_loss_mse": loss_mse.item(),
                    "batch_grad_norm": batch_grad_norm,
                    "batch_grad_max": batch_grad_max,
                }
                
                # Add divergence loss if it's being used
                if train_config['divergence_loss']:
                    iter_metrics["batch_loss_div"] = loss_div.item() if hasattr(loss_div, 'item') else loss_div
                
                iteration_metrics.append(iter_metrics)

            optimizer.step()

        mean_epoch_loss = np.mean(losses)

        mean_loss_tensor = torch.tensor(mean_epoch_loss, device=device_ID)
        dist.all_reduce(mean_loss_tensor, op=dist.ReduceOp.AVG)
        global_mean_epoch_loss = mean_loss_tensor.item()

        # Adjust lr rate schedule if using
        if train_config["warmup"] and (epoch) < train_config["warmup_total_iters"]:
            warmuplr.step()
        else:
            if train_config["scheduler"] == 'ReduceLROnPlateau':
                LRscheduler.step(mean_epoch_loss)
            elif train_config["scheduler"] == 'CosineAnnealingLR':
                LRscheduler.step()

        # Synchronize all processes before saving checkpoints
        if torch.distributed.is_initialized():
            barrier(device_ids=[device_ID])

        # Save the model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_loss': min(best_loss, global_mean_epoch_loss),  # Use the actual best loss
            'iteration': iteration
        }

        if best_loss > global_mean_epoch_loss:
            best_loss = global_mean_epoch_loss
            if device_ID == 0:
                torch.save(checkpoint, os.path.join(save_dir, train_config['best_ckpt_name']))
            
        # Show epoch progress 
        log_print('Finished epoch:{} | Loss (mean/best) : {:.2e}/{:.2e}'.format(
            epoch, global_mean_epoch_loss, best_loss), log_to_screen=True)
        # Save the model
        if device_ID == 0:
            # Save the model
            torch.save(checkpoint, os.path.join(save_dir,
                                                    train_config['ckpt_name']))
        
        if logging_config['log_to_wandb'] and device_ID == 0:
            # First, log all the per-iteration metrics
            for i, metrics in enumerate(iteration_metrics):
                wandb.log(metrics, step=metrics['iteration'])
    
            # Then log the epoch summary
            epoch_summary = {
                "epoch": epoch,
                "epoch_loss": mean_epoch_loss,
                "epoch_loss_best": best_loss,
                "lr": optimizer.param_groups[0]['lr'],
            }
            
            # Log epoch summary with a special step that won't conflict
            wandb.log(epoch_summary, step=iteration)
    
    # Show training completion 
    log_print('Training Completed ...', log_to_screen=True)

    if logging_config['log_to_wandb'] and device_ID == 0:
        wandb.finish()
    

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser = argparse.ArgumentParser(description='Arguments for DDPM image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/config.yaml', type=str,
                        help='Path to the configuration file')
    args = parser.parse_args()
    
    # Make config path relative to project root if it's a relative path
    if not os.path.isabs(args.config_path):
        args.config_path = os.path.join(project_root, args.config_path)
    
    try:
        train(args)
    finally:
        ddp_cleanup()  # Clean up distributed training environment
