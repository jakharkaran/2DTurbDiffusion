import torch
import torchvision
import argparse
import yaml
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if torch.cuda.is_available():
    print("Number of GPUs available:", torch.cuda.device_count())

def sample_turb(model, scheduler, train_config, test_config, model_config, diffusion_config, dataset_config, run_num):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    if dataset_config['normalize']:
        mean_std_data = np.load(dataset_config['data_dir'] + 'mean_std.npz')
        mean = [mean_std_data['U_mean'], mean_std_data['V_mean']]
        std = [mean_std_data['U_std'], mean_std_data['V_std']]

    # saving generated data
    if test_config['save_data']:
        os.makedirs(os.path.join(train_config['task_name'], 'data', run_num), exist_ok=True)
    
    for batch_count in range(0,test_config['num_test_batch']):

        xt = torch.randn((test_config['batch_size'],
                        model_config['im_channels'],
                        model_config['im_size'],
                        model_config['im_size'])).to(device)

        # Saving generated images
        if test_config['save_image'] or batch_count < 5:
            # if not os.path.exists(os.path.join(train_config['task_name'], 'samples', str(batch_count))):
            #     os.mkdir(os.path.join(train_config['task_name'], 'samples', str(batch_count)))
            os.makedirs(os.path.join(train_config['task_name'], 'samples'), exist_ok=True)
            os.makedirs(os.path.join(train_config['task_name'], 'samples', run_num), exist_ok=True)

            nrows = int(np.floor(np.sqrt(test_config['batch_size'])))

            figU, axesU = plt.subplots(nrows=nrows, ncols=nrows, figsize=(15, 15))
            figV, axesV = plt.subplots(nrows=nrows, ncols=nrows, figsize=(15, 15))

        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            # Get prediction of noise
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
            
            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

            if test_config['save_image'] or batch_count < 5:

                if i % 200 == 0 :
                
                    # Save x0
                    ims = torch.clamp(xt, -1., 1.).detach().cpu()

                    U_arr = ims[:,0,:,:].numpy()
                    V_arr = ims[:,1,:,:].numpy()

                    vmax_U = np.max(np.abs(U_arr))
                    vmax_V = np.max(np.abs(V_arr))
                
                    # Loop over the grid
                    for ax_count, ax in enumerate(axesU.flat):
                        # Plot each matrix using the 'bwr' colormap
                        plotU = ax.pcolorfast(U_arr[ax_count,:,:], cmap='bwr', vmin=-vmax_U, vmax=vmax_U)
                        ax.axis('off')
                    # Adjust spacing between subplots to avoid overlap
                    figU.subplots_adjust(wspace=0.1, hspace=0.1)
                    figU.savefig(os.path.join(train_config['task_name'], 'samples', run_num, f'{str(batch_count)}_U{i}.jpg'), format='jpg', bbox_inches='tight', pad_inches=0)

                    # Loop over the grid
                    for ax_count, ax in enumerate(axesV.flat):
                        # Plot each matrix using the 'bwr' colormap
                        plotV = ax.pcolorfast(V_arr[ax_count,:,:], cmap='bwr', vmin=-vmax_V, vmax=vmax_V)
                        ax.axis('off')
                    # Adjust spacing between subplots to avoid overlap
                    figV.subplots_adjust(wspace=0.1, hspace=0.1)
                    figV.savefig(os.path.join(train_config['task_name'], 'samples', run_num, f'{str(batch_count)}_V{i}.jpg'), format='jpg', bbox_inches='tight', pad_inches=0)

                    # figU.clear(), figV.clear()
                    del plotU, plotV

        if test_config['save_data']:
            xt_cpu = xt.detach().cpu()

            if dataset_config['normalize']:
                # Convert to tensors and reshape
                mean  = np.asarray(mean)
                std = np.asarray(std)

                mean_tensor = mean.reshape(mean.shape[0], 1, 1)
                std_tensor = std.reshape(std.shape[0], 1, 1)

                xt_cpu= xt_cpu*std_tensor + mean_tensor

            np.save(os.path.join(train_config['task_name'], 'data', run_num, str(batch_count) + '.npy'), xt_cpu.numpy())


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
    # Multi-GPU Setup (DataParallel)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPUs available: {num_gpus}")
        if num_gpus > 1:
            print("Using DataParallel to run on multiple GPUs for inference.")
            model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Load weights
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))
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