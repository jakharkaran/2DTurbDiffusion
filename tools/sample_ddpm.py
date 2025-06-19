import torch
import torchvision
import argparse
import yaml
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.amp import autocast
from tqdm import tqdm
import random

from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from dataset.dataloader import CustomMatDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if torch.cuda.is_available():
    print("Number of GPUs available:", torch.cuda.device_count())

def sample_turb(model, scheduler, train_config, test_config, model_config, diffusion_config, dataset_config, run_num):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """

    GLOBAL_SEED = train_config['global_seed']
    # -------- main-process seeding -------------------------------------------
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    if 'mnist' in dataset_config['data_dir'].lower():
        # mnist dataset is not normalized
        pass
    elif dataset_config['normalize']:
        mean_std_data = np.load(dataset_config['data_dir'] + 'mean_std.npz')
        mean = np.asarray([mean_std_data['U_mean'], mean_std_data['V_mean']])
        std = np.asarray([mean_std_data['U_std'], mean_std_data['V_std']])

        mean_tensor = torch.tensor(mean.reshape(mean.shape[0], 1, 1)).to(device)
        std_tensor = torch.tensor(std.reshape(std.shape[0], 1, 1)).to(device)

    # Precompute timesteps once
    timesteps = [torch.tensor(i, device=device).unsqueeze(0) for i in range(diffusion_config['num_timesteps'])]

    # Create directory for saving generated data if required
    if test_config['save_data']:
        os.makedirs(os.path.join(train_config['save_dir'], 'data', run_num), exist_ok=True)

        # List all .npy files in the directory
        npy_files = [f for f in os.listdir(os.path.join(train_config['save_dir'], 'data', run_num)) if f.endswith('.npy')]

        if npy_files:
            # Extract numeric file numbers from filenames
            file_numbers = [int(os.path.splitext(f)[0]) for f in npy_files]
            largest_file_number = max(file_numbers)
            print(f"Data exists in the directory. Largest file number: {largest_file_number}")

        else:
            largest_file_number = -1  # No files found
            print("No .npy files found in the directory.")    

    # Check if all batches already exist
    if largest_file_number >= (test_config['num_test_batch']-1):
        print(f"All {test_config['num_test_batch']} batches already exist. Exiting.")
        sys.exit(0)

    if diffusion_config['conditional']:
     # ----- NEW: seed with real frame t0 (selected by test_file_start_idx) -----

        if npy_files:
            # If files exist, load the second-last one (avoiding errorsif last saved file is corrupted)
            print(f"Loading last saved batch: {largest_file_number}")
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
            seed_dataset = CustomMatDataset(dataset_config, train_config, test_config, training=False, conditional=True)
            t0_tensor = seed_dataset[0] # [T, C, H, W]

            T, C, H, W = t0_tensor.shape
            batch_cond = t0_tensor[1:, ...].reshape(1, (T-1)*C, H, W) # [T-1, C, H, W] -> [1, (T-1)*C, H, W]

    else:
        batch_cond = None
    
    if diffusion_config['conditional']:
        print('batch_cond shape: ', batch_cond.shape)
        batch_cond = batch_cond.float().to(device)


    # Loop over the number of batches    
    for batch_count in range(largest_file_number+1,test_config['num_test_batch']):

        xt = torch.randn((test_config['batch_size'],
                        model_config['pred_channels'],
                        model_config['im_size'],
                        model_config['im_size'])).float().to(device)
        
        
        # if diffusion_config['conditional']:
        #     # model_in = torch.cat((xt, batch_cond), dim=1) # [B, C, H, W]
        #     # if batch_count == 0:
        #     #     xt_prev = t0_tensor[:model_config['im_channels']-2, :, :].to(device)
        #     # else:
        #     #     xt_prev = xt_final

        #     print('model_in shape: ', model_in.shape)

        # Create directories and figure objects for saving images if needed
        if test_config['save_image'] or batch_count < 5:

            os.makedirs(os.path.join(train_config['save_dir'], 'samples'), exist_ok=True)
            os.makedirs(os.path.join(train_config['save_dir'], 'samples', run_num), exist_ok=True)

            nrows = int(np.floor(np.sqrt(test_config['batch_size'])))

            figU, axesU = plt.subplots(nrows=nrows, ncols=nrows, figsize=(15, 15))

            if not 'mnist' in dataset_config['data_dir'].lower():
                figV, axesV = plt.subplots(nrows=nrows, ncols=nrows, figsize=(15, 15))

        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):

            t_tensor = timesteps[i]
            # print('********** ', t_tensor, t_tensor.squeeze(0))


            if diffusion_config['conditional']:
                model_in = torch.cat((xt, batch_cond), dim=1) # [B, C, H, W]
                
                # noise = torch.randn_like(xt_prev.unsqueeze(0)).to(device)
                # # print(xt_prev.unsqueeze(0).shape, noise.shape)
                # xt_prev_noisy = scheduler.add_noise(xt_prev.unsqueeze(0), noise, t_tensor)
                # # inject clean conditioning channels (u_{t-1}, v_{t-1})
                # xt[:, :model_config['im_channels']-2, :, :] = xt_prev_noisy.to(device)

            else:
                model_in = xt
                
            with autocast('cuda'):
                # Get prediction of noise
                noise_pred = model(model_in, t_tensor)
                
                # Use scheduler to get x0 and xt-1
                # if diffusion_config['conditional']:
                #     xt = scheduler.sample_prev_timestep_partial(xt, noise_pred, t_tensor, n_cond=model_config['im_channels']//2)
                # else:
                xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, t_tensor.squeeze(0))
                
                if test_config['save_image'] or batch_count < 5:

                    # if i % 250 == 0 :
                    if i == 0 :

                        # Save x0
                        ims = torch.clamp(xt, -1., 1.).detach().cpu()


                        # ims = model_in.detach().cpu()

                        # U_arr = ims[:,0,:,:].numpy()
                        # V_arr = ims[:,2,:,:].numpy()

                        ims = xt.detach().cpu()


                        U_arr = ims[:,0,:,:].numpy()
                        vmax_U = np.max(np.abs(U_arr))
                        if not ('mnist' in dataset_config['data_dir'].lower()):
                            V_arr = ims[:,1,:,:].numpy()
                            vmax_V = np.max(np.abs(V_arr))
                    
                        # Loop over the grid
                        if nrows == 1:
                            # Handle the case where there is only a single row of axes
                                if 'mnist' in dataset_config['data_dir'].lower():
                                    plotU = axesU.imshow(np.clip(U_arr[0, :, :], 0, 1), cmap='gray', vmin=0, vmax=1)
                                else:
                                    plotU = axesU.pcolorfast(U_arr[0, :, :], cmap='bwr', vmin=-vmax_U, vmax=vmax_U)
                                # plotU = axesU.contourf(U_arr[0, :, :], cmap='bwr', vmin=-vmax_U, vmax=vmax_U)
                                axesU.axis('off')
                        else:
                            for ax_count, ax in enumerate(axesU.flatten()):
                                # Plot each matrix using the 'bwr' colormap
                                if 'mnist' in dataset_config['data_dir'].lower():
                                    plotU = ax.imshow(np.clip(U_arr[ax_count, :, :], 0, 1), cmap='gray', vmin=0, vmax=1)
                                else:
                                    plotU = ax.pcolorfast(U_arr[ax_count, :, :], cmap='bwr', vmin=-vmax_U, vmax=vmax_U)
                                ax.axis('off')
                            # Adjust spacing between subplots to avoid overlap
                            figU.subplots_adjust(wspace=0.1, hspace=0.1)

                        figU.savefig(os.path.join(train_config['save_dir'], 'samples', run_num, f'{str(batch_count)}_U{i}.jpg'), format='jpg', bbox_inches='tight', pad_inches=0)
                        del plotU

                        if not 'mnist' in dataset_config['data_dir'].lower():

                            # Loop over the grid
                            if nrows == 1:
                                # Handle the case where there is only a single row of axes
                                    plotV = axesV.pcolorfast(V_arr[0, :, :], cmap='bwr', vmin=-vmax_V, vmax=vmax_V)
                                    # plotV = axesV.contourf(V_arr[0, :, :], cmap='bwr', vmin=-vmax_V, vmax=vmax_V)
                                    axesV.axis('off')
                            else:
                                for ax_count, ax in enumerate(axesV.flat):
                                    # Plot each matrix using the 'bwr' colormap
                                    plotV = ax.pcolorfast(V_arr[ax_count,:,:], cmap='bwr', vmin=-vmax_V, vmax=vmax_V)
                                    ax.axis('off')
                                # Adjust spacing between subplots to avoid overlap
                                figV.subplots_adjust(wspace=0.1, hspace=0.1)

                            figV.savefig(os.path.join(train_config['save_dir'], 'samples', run_num, f'{str(batch_count)}_V{i}.jpg'), format='jpg', bbox_inches='tight', pad_inches=0)
                            del plotV

        if diffusion_config['conditional']:
            # Shift batch_cond to remove the last time step and add the new one at the beginning
            # batch_cond: shape [1, (T-1)*C, H, W]
            # Remove last C channels, prepend new C channels from xt
            _, C, H, W = xt.shape
            batch_cond = torch.cat(
                (xt.detach(), batch_cond[:, :-C, :, :].detach()),
                dim=1
            )

            print('batch_cond subtracted shape: ',  batch_cond[:, :-C, :, :].shape)
            print('xt shape: ', xt.shape)
            print('batch_cond shape: ', batch_cond.shape)
            
            xt_final = xt[:, :model_config['pred_channels'], :, :].detach()
        else:
            xt_final = xt.detach()

        if test_config['save_data']:
            
            if 'mnist' in dataset_config['data_dir'].lower():
                pass
            elif dataset_config['normalize']:
                xt_final.mul_(std_tensor).add_(mean_tensor)

            xt_cpu = xt_final.cpu()

            np.save(os.path.join(train_config['save_dir'], 'data', run_num, str(batch_count) + '.npy'), xt_cpu.numpy())


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    dataset_config = config['dataset_params']
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    test_config = config['test_params']
    run_num = args.run_num
    
    # Create model and load checkpoint
    model = Unet(model_config)

    # If your model is fully scriptable, script it first
    # model = torch.jit.script(model)

    # Multi-GPU Setup (DataParallel)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPUs available: {num_gpus}")
        if num_gpus > 1:
            print("Using DataParallel to run on multiple GPUs for inference.")
            model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Create output directories &   ### Saving config file with the model weights
    if train_config['model_collapse']:
        task_name = train_config['task_name'] + '_' + train_config['model_collapse_type'] + '_' + str(train_config['model_collapse_gen'])
    else:
        task_name = train_config['task_name']
    train_config['save_dir'] =  os.path.join('results', task_name)

    # Load weights
    model.load_state_dict(torch.load(os.path.join(train_config['save_dir'],
                                                  train_config['best_ckpt_name']), map_location=device))
    model.eval()
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample_turb(model, scheduler, train_config, test_config, model_config, diffusion_config, dataset_config, run_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for DDPM image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/config.yaml', type=str,
                        help='Path to the configuration file')
    parser.add_argument('--run_num', dest='run_num', 
                        default='1', type=str,
                        help='Run number for the experiment')

    args = parser.parse_args()

    # Call function with the parsed arguments
    infer(args)