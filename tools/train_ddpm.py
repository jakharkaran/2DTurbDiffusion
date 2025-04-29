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
import matplotlib.pyplot as plt

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

    if logging_config['log_to_wandb']:
        # Set up wandb
        wandb.login()
        wandb.init(project=logging_config['wandb_project'], group=logging_config['wandb_group'],
                   name=logging_config['wandb_name'], config=config)


    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
        
    # Initiate dataloader
    dataset = CustomMatDataset(dataset_config, train_config, test_config, conditional=diffusion_config['conditional'])
    turb_dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        
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
    noise_cond = None
    # Run training
    best_loss = 1e6 # initialize
    for epoch_idx in range(num_epochs):
        losses = []
        # for im in tqdm(mnist_loader):
        for batch_im in tqdm(turb_dataloader):

            iteration += 1

            optimizer.zero_grad()

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

            # fig, axes = plt.subplots(2,3, figsize=(15, 15))
            # Ut = im[0, 0, :, :].cpu().detach().numpy()
            # Vt = im[0, 1, :, :].cpu().detach().numpy()
            # Ut0 = im[0, 2, :, :].cpu().detach().numpy()
            # Vt0 = im[0, 3, :, :].cpu().detach().numpy()

            # vmaxU = np.max(np.abs(Ut))
            # vmaxV = np.max(np.abs(Vt))

            # axes[0,0].pcolorfast(Ut0, cmap='bwr', vmin=-vmaxU, vmax=vmaxU)
            # axes[0,1].pcolorfast(Ut, cmap='bwr', vmin=-vmaxU, vmax=vmaxU)
            # axes[0,2].pcolorfast(Ut-Ut0, cmap='bwr', vmin=-vmaxU, vmax=vmaxU)
            # axes[1,0].pcolorfast(Vt0, cmap='bwr', vmin=-vmaxV, vmax=vmaxV)
            # axes[1,1].pcolorfast(Vt, cmap='bwr', vmin=-vmaxV, vmax=vmaxV)
            # axes[1,2].pcolorfast(Vt-Vt0, cmap='bwr', vmin=-vmaxV, vmax=vmaxV)
            # for ax in axes.flatten():
            #     ax.set_aspect('equal')
            #     ax.axis('off')
            # plt.tight_layout()
            # fig.savefig(os.path.join(save_dir, 'a.jpg'), format='jpg', bbox_inches='tight', pad_inches=0)
            # sys.exit()


            # Add noise to images according to timestep
            # Have dataloader output x_init
            # model(noisy_im, t) -> model(noisy_im, t, x_init)
            if diffusion_config['conditional']:
                noisy_im = scheduler.add_noise_partial(im, noise, t, n_cond=model_config['im_channels']//2)
            else:
                noisy_im = scheduler.add_noise(im, noise, t)

            model_out = model(noisy_im, t)

            # make_dot(model_out, params=dict(model.named_parameters())).render("model_architecture", format="png")
            # summary(model, input_size=(10, 10))

            # sys.exit()

            if train_config['loss'] == 'noise':
                # Noise is predicted by the model
                if diffusion_config['conditional']:

                    noise_pred, noise_cond = model_out.split(2, dim=1)
                    noise_cond = noise_cond.detach() 
                    # Exclude conditional channels from loss
                    loss_mse = criterion(noise_pred, noise[:, :model_config['im_channels']//2,:,:])
                else:
                    loss_mse = criterion(model_out, noise) 

            elif train_config['loss'] == 'sample':
                # x0 (denoised sample) is predicted by the diffusion model
                if diffusion_config['conditional']:
                    # Exlude conditional channels from loss
                    loss_mse = criterion(model_out[:, :model_config['im_channels']//2,:,:], im[:, :model_config['im_channels']//2,:,:])
                else:
                    loss_mse = criterion(model_out, im) 

            if logging_config['log_to_wandb']:
                log_data = {
                    "epoch": epoch_idx + 1,
                    "loss_mse": loss_mse,
                    "iteration": iteration,
                    "noise_cond": torch.mean(torch.abs(noise_cond)),
                }

            ###### Divergence Loss
            if train_config['divergence_loss']:
                # model.eval()

                # Method # 1: If loss is in noise: Calculate x_0 directly from predicted_noise in a single step
                if train_config['divergence_loss_type'] == 'direct_sample' and train_config['loss'] == 'noise':
                    x0_pred = scheduler.sample_x0_from_noise(noisy_im, model_out, t) # Get x0 from xt and noise_pred

                # Method # 2: If loss is in noise: Calculate x_0 directly via denoising 
                if train_config['divergence_loss_type'] == 'denoise_sample' and train_config['loss'] == 'noise':

                    # Sample timestep
                    timesteps_div = [torch.tensor(i, device=device).unsqueeze(0) for i in range(train_config['denoise_sample_timestep'])]
                    # Sample random noise
                    noise_div = torch.randn_like(batch_div).to(device)

                    # Add noise to images according to timestep
                    t_div_noise = torch.full((batch_div.shape[0],), train_config['denoise_sample_timestep']-1, dtype=torch.int).to(device)
                    noisy_im_div = scheduler.add_noise(batch_div, noise_div, t_div_noise)

                    for i in tqdm(reversed(range( train_config['denoise_sample_timestep']))):
                            
                        # Get prediction of noise
                        t_tensor = timesteps_div[i]
                        noise_pred_div = model(noisy_im_div, t_tensor)
                        
                        # Use scheduler to get x0 and xt-1
                        noisy_im_div, _ = scheduler.sample_prev_timestep_from_noise(noisy_im_div, noise_pred_div, t_tensor.squeeze(0))

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
                            
                #             # Use scheduler to get x0 and xt-1
                #             xt, _ = scheduler.sample_prev_timestep_from_noise(xt, noise_pred, t_tensor.squeeze(0))

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
            optimizer.step()

            if logging_config['log_to_wandb']:
                log_data["loss"] = loss
                log_data["best_loss"] = best_loss
                wandb.log(log_data) # Logging all data to wandb

        if best_loss > loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(save_dir,
                                                        train_config['best_ckpt_name']))
            
        print('Finished epoch:{} | Loss (mean/current/best) : {:.2e}/{:.2e}/{:.2e}'.format(
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
