import torch
import yaml
import argparse
import os, sys
import shutil
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.amp import autocast
import wandb
import random

from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from dataset.dataloader import CustomMatDataset
from tools.util import SpectralDifferentiator

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
    
    print('Configuration loaded: {config}')


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
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
            
    def worker_init_fn(worker_id):
        # Each worker gets a *unique* but *deterministic* stream
        worker_seed = GLOBAL_SEED + worker_id
        np.random.seed(worker_seed)          # NumPy ops in transforms
        random.seed(worker_seed)             # Python’s `random` calls
        torch.manual_seed(worker_seed)       # Torch ops inside the Dataset
    dl_gen = torch.Generator().manual_seed(GLOBAL_SEED)

    # Initiate dataloader
    dataset = CustomMatDataset(dataset_config, train_config, test_config)
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
        wandb.watch(model)
    
    # Create output directories &   ### Saving config file with the model weights
    if train_config['model_collapse']:
        save_dir =  os.path.join('results', train_config['task_name'] + '_' + train_config['model_collapse_type'] + '_' + str(train_config['model_collapse_gen']))
    else:
        save_dir = os.path.join('results',train_config['task_name'])

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
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    if train_config['divergence_loss_type'] == 'denoise_sample':
        # Precompute timesteps for denoising
        timesteps = [torch.tensor(i, device=device).unsqueeze(0) for i in range(diffusion_config['num_timesteps'])]

    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 2D Trubulence System Parameter
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

    iteration = 0
    # Run training
    best_loss = 1e6 # initialize
    for epoch_idx in range(num_epochs):
        losses = []
        # for im in tqdm(mnist_loader):
        for im in tqdm(turb_dataloader):

            iteration += 1

            optimizer.zero_grad()
            im = im.float().to(device)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            model_out = model(noisy_im, t)

            # make_dot(model_out, params=dict(model.named_parameters())).render("model_architecture", format="png")
            # summary(model, input_size=(10, 10))

            # sys.exit()

            if train_config['loss'] == 'noise':
                loss_mse = criterion(model_out, noise) # Noise is predicted by the model
            elif train_config['loss'] == 'sample':
                loss_mse = criterion(model_out, im) # x0 (denoised sample) is predicted by the diffusion model

            if logging_config['log_to_wandb']:
                log_data = {
                    "epoch": epoch_idx + 1,
                    "loss_mse": loss_mse,
                    "iteration": iteration,
                }

            ###### Divergence Loss
            if train_config['divergence_loss']:
                model.eval()

                # Method # 1: If loss is in noise: Calculate x_0 directly from predicted_noise in a single step
                if train_config['divergence_loss_type'] == 'direct_sample' and train_config['loss'] == 'noise':
                    x0_pred = scheduler.sample_x0_from_noise(noisy_im, model_out, t) # Get x0 from xt and noise_pred

                # Method # 2: If loss is in noise: Calculate x_0 directly via denoising 
                if train_config['divergence_loss_type'] == 'denoise_sample' and train_config['loss'] == 'noise':
                    with torch.inference_mode():

                        # Compute Divergence
                        xt = torch.randn((test_config['batch_size'],
                        model_config['im_channels'],
                        model_config['im_size'],
                        model_config['im_size'])).to(device)

                        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
                                
                            # with autocast('cuda'):
                            # Get prediction of noise
                            t_tensor = timesteps[i]
                            noise_pred = model(xt, t_tensor)
                            
                            # Use scheduler to get x0 and xt-1
                            xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, t_tensor.squeeze(0))

                        x0_pred = xt.detach().clone()

                    del xt, noise_pred
                    torch.cuda.empty_cache()

                # Method # 3: If loss is in sample
                if train_config['divergence_loss_type'] == 'direct_sample' and train_config['loss'] == 'sample':
                    x0_pred = model_out

                # De-normalize the data
                if dataset_config['normalize']:
                    x0_pred = x0_pred.mul(std_tensor).add(mean_tensor)
                    div = diff.divergence(x0_pred, spectral=False)

                loss_div = train_config['divergence_loss_weight'] * torch.mean(torch.abs(div))
                loss = loss_mse + loss_div

                if logging_config['log_to_wandb']:
                    log_data["loss_div"] = loss_div # loggin for wandb

                model.train()


            else:
                loss = loss_mse

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if logging_config['log_to_wandb']:
                log_data["loss"] = loss
                log_data["best_loss"] = best_loss
                wandb.log(log_data) # Logging all data to wandb

        if best_loss > loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(save_dir,
                                                        train_config['best_ckpt_name']))
            
        print('Finished epoch:{} | Loss (mean/current/best) : {:.4f}/{:.4f}/{:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses), loss, best_loss
        ))
        # Save the model
        torch.save(model.state_dict(), os.path.join(save_dir,
                                                    train_config['ckpt_name']))
    
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
