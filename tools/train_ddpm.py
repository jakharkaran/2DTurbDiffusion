import torch
import yaml
import argparse
import os, sys
import shutil
import numpy as np
from tqdm import tqdm
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast
import wandb
import matplotlib.pyplot as plt
import random

from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from dataset.dataloader import CustomMatDataset
from tools.util import SpectralDifferentiator, grad_norm, grad_max

# Packages to outline architecture of the model
# from torchviz import make_dot
# from torchinfo import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if torch.cuda.is_available():
    print("Number of GPUs available:", torch.cuda.device_count())

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Error in configuration file:", exc)
            sys.exit(1)
    
    print(f'Configuration loaded: {config}')


    ########################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    logging_config = config['logging_params']
    test_config = config['test_params']

    GLOBAL_SEED = train_config['global_seed']
    # -------- main-process seeding -------------------------------------------
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    if logging_config['log_to_wandb']:
        # Set up wandb
        wandb.login()
        wandb.init(project=logging_config['wandb_project'], group=logging_config['wandb_group'],
                   name=logging_config['wandb_name'], config=config)


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
    dataset = CustomMatDataset(dataset_config, train_config, test_config, conditional=diffusion_config['conditional'])
    turb_dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, \
                                 generator=dl_gen, worker_init_fn=worker_init_fn)
        
    # Instantiate the model
    model = Unet(model_config)
    
    # If multiple GPUs are available, wrap with DataParallel
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPUs available: {num_gpus}")
        if num_gpus > 1:
            print("Using DataParallel to run on multiple GPUs.")
            model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    model.train()

    # Watch model gradients with wandb
    if logging_config['log_to_wandb']:
        wandb.watch(model, log="all", log_freq=logging_config['wandb_table_logging_interval'],)
    
    # Create output directories &   ### Saving config file with the model weights
    if train_config['model_collapse']:
        save_dir =  os.path.join('results', train_config['task_name'] + '_' + train_config['model_collapse_type'] + '_' + str(train_config['model_collapse_gen']))
    else:
        save_dir = os.path.join('results',train_config['task_name'])

    # 2D Turbulence System Parameter
    nx, ny = int(model_config['im_size']/dataset_config['downsample_factor']), int(model_config['im_size']/dataset_config['downsample_factor'])
    Lx, Ly = 2 * torch.pi, 2 * torch.pi
    # Kx, Ky, _, _, invKsq = initialize_wavenumbers_rfft2(nx, ny, Lx, Ly, INDEXING='ij')

    # --- Instantiate the Differentiator ---
    diff = SpectralDifferentiator(nx=nx, ny=ny, Lx=Lx, Ly=Ly, device=device)

    if train_config['divergence_loss']:
        # Load the mean and std for de-normalization
        mean_std_data = np.load(dataset_config['data_dir'] + 'mean_std.npz')
        mean = np.asarray([mean_std_data['U_mean'], mean_std_data['V_mean']])
        std = np.asarray([mean_std_data['U_std'], mean_std_data['V_std']])

        mean_tensor = torch.tensor(mean.reshape(1, mean.shape[0], 1, 1)).to(device)
        std_tensor = torch.tensor(std.reshape(1, std.shape[0], 1, 1)).to(device)

        if train_config['divergence_loss_type'] == 'denoise_sample':
            # Precompute timesteps for denoising
            timesteps = [torch.tensor(i, device=device).unsqueeze(0) for i in range(diffusion_config['num_timesteps'])]

    print('*** Saving weights: ', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy(args.config_path, save_dir)
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(save_dir,train_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(save_dir,
                                                      train_config['ckpt_name']), map_location=device))

    # Specify training parameters
    num_epochs = train_config['num_epochs']

    if train_config['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=float(train_config['lr']), weight_decay=float(train_config['weight_decay']))
    elif train_config['optimizer'] == 'adamw':
        optimizer = AdamW(model.parameters(), lr=float(train_config['lr']), weight_decay=float(train_config['weight_decay']), fused=True)

    # Set learning rate scheduluer
    if train_config["scheduler"] == 'ReduceLROnPlateau':
        LRscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, cooldown=10, mode='min')
    elif train_config["scheduler"] == 'CosineAnnealingLR':
        LRscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_config["num_epochs"]), eta_min=float(train_config['lr_min']))
    else:
       LRscheduler = None

    # Warm up epochs if using
    if train_config['warmup']:
        warmuplr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=train_config['warmup_start_factor'],
                                                              total_iters=train_config['warmup_total_iters'])

    criterion = torch.nn.MSELoss()


    iteration = 0
    noise_cond = None
    # Run training
    best_loss = 1e6 # initialize
    for epoch_idx in range(num_epochs):
        losses = []

        for batch_data in tqdm(turb_dataloader):

            iteration += 1

            optimizer.zero_grad()

            if diffusion_config['conditional']:
                B, T, C, H, W = batch_data.shape # [B, T, C, H, W]; T: t, t-1, t-2, ...; C: U, V

                # Split batch_data along T dimension
                batch_im = batch_data[:, 0, ...]      # [B, C, H, W] (first T)
                # [B, T-1, C, H, W] (remaining T), Stack T-1 and C into channel dimension -> [B, (T-1)*C, H, W]
                batch_cond = batch_data[:, 1:, ...].reshape(B, (T-1) * C, H, W)      
            else:
                batch_cond = None
                # For unconditional model, batch_data is [B, 1, C, H, W], remove T dimension
                batch_im = batch_data[:, 0, :, :, :] # [B, C, H, W]

            if train_config['divergence_loss'] and train_config['divergence_loss_type'] == 'denoise_sample' and train_config['loss'] == 'noise':
                # Calculate split sizes for the two branches
                batch_size = batch_im.shape[0]
                div_ratio = train_config['denoise_sample_batch_ratio']
                bs_div = int(batch_size * div_ratio)

                if bs_div == 0:
                    print('Warning: No data for divergence loss batch. Setting to 1.')
                    bs_div=1

                bs_mse = batch_size - bs_div
                print('bs_mese:', bs_mse, 'bs_div:', bs_div)

                im = batch_im[:bs_mse].float().to(device)
                batch_div = batch_im[bs_mse:].float().to(device)
            else:
                im = batch_im.float().to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)


            # Add noise to images according to timestep
            # Have dataloader output x_init
            # model(noisy_im, t) -> model(noisy_im, t, x_init)
            # if diffusion_config['conditional']:
            #     noisy_im = scheduler.add_noise_partial(im, noise, t, n_cond=model_config['im_channels']//2)

            # else:
            noisy_im = diffusion_noise_scheduler.add_noise(im, noise, t)

            if diffusion_config['conditional']:
                # Concatenate the conditioning data with the noisy image
                batch_cond = batch_cond.float().to(device)
                model_in = torch.cat((noisy_im, batch_cond), dim=1)
            else:
                model_in = noisy_im

            model_out = model(model_in, t)

            # print('Model out: ', noise.shape)
            # sys.exit()

            # make_dot(model_out, params=dict(model.named_parameters())).render("model_architecture", format="png")
            # summary(model, input_size=(10, 10))

            # sys.exit()

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
                    timesteps_div = [torch.tensor(i, device=device).unsqueeze(0) for i in range(train_config['denoise_sample_timestep'])]
                    # Sample random noise
                    noise_div = torch.randn_like(batch_div).to(device)

                    # Add noise to images according to timestep
                    t_div_noise = torch.full((batch_div.shape[0],), train_config['denoise_sample_timestep']-1, dtype=torch.int).to(device)
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
                #         xt = torch.randn((test_config['batch_size'],
                #         model_config['im_channels'],
                #         model_config['im_size'],
                #         model_config['im_size'])).to(device)

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
                    print(x0_pred.shape)
                    div = diff.divergence(x0_pred, spectral=False)

                loss_div = train_config['divergence_loss_weight'] * torch.mean(torch.abs(div))
                loss = loss_mse + loss_div

                if logging_config['log_to_wandb']:
                    log_data["loss_div"] = loss_div # loggin for wandb

                # model.train()


            else:
                loss = loss_mse

            losses.append(loss.item())
            loss.backward()
            # # Trying with gradient clipping
            # ——— gradient clipping & norm ———
            # this returns the total norm before clipping:
            # grad_norm = torch.nn.utils.clip_grad_norm_(
            #     model.parameters(),
            #     max_norm=1.0,
            # )

            # Calculate maximum gradient value across all parameters
            batch_grad_norm = grad_norm(model)
            batch_grad_max = grad_max(model)


            optimizer.step()


            if logging_config['log_to_wandb']:
                batch_log = {
                    "iteration": iteration,
                    # "batch_loss_mse":  loss_mse.item(),
                    "batch_loss":      loss.item(),
                    "batch_grad_norm": batch_grad_norm,
                    "batch_grad_max":  batch_grad_max,
                }
                wandb.log(batch_log, step=iteration)

        mean_epoch_loss = np.mean(losses)

        # Adjust lr rate schedule if using
        if train_config["warmup"] and (epoch_idx + 1) < train_config["warmup_total_iters"]:
            warmuplr.step()
        else:
            if train_config["scheduler"] == 'ReduceLROnPlateau':
                LRscheduler.step(mean_epoch_loss)
            elif train_config["scheduler"] == 'CosineAnnealingLR':
                LRscheduler.step()
                if (epoch_idx + 1) >= train_config['num_epochs']:
                    print("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    break


        if best_loss > mean_epoch_loss:
            best_loss = mean_epoch_loss
            torch.save(model.state_dict(), os.path.join(save_dir,
                                                        train_config['best_ckpt_name']))
            
        print('Finished epoch:{} | Loss (mean/current/best) : {:.2e}/{:.2e}/{:.2e}'.format(
            epoch_idx + 1,
            mean_epoch_loss, loss, best_loss
        ))
        # Save the model
        torch.save(model.state_dict(), os.path.join(save_dir,
                                                    train_config['ckpt_name']))
        
        if logging_config['log_to_wandb']:
            epoch_log = {
                "epoch":     epoch_idx + 1,
                "epoch_loss": mean_epoch_loss,
                "epoch_loss_best": best_loss,
                "lr":        optimizer.param_groups[0]['lr'],
            }
            # here we use the epoch number as the step
            wandb.log(epoch_log, step=iteration)
    
    print('Training Completed ...')

    if logging_config['log_to_wandb']:
        wandb.finish()
    

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser = argparse.ArgumentParser(description='Arguments for DDPM image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/config.yaml', type=str,
                        help='Path to the configuration file')
    args = parser.parse_args()
    train(args)
