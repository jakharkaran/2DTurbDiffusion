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


# def sample(model, scheduler, train_config, model_config, diffusion_config):
#     r"""
#     Sample stepwise by going backward one timestep at a time.
#     We save the x0 predictions
#     """
#     xt = torch.randn((train_config['num_samples'],
#                       model_config['im_channels'],
#                       model_config['im_size'],
#                       model_config['im_size'])).to(device)
#     for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
#         # Get prediction of noise
#         noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
#         # Use scheduler to get x0 and xt-1
#         xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
#         # Save x0
#         ims = torch.clamp(xt, -1., 1.).detach().cpu()
#         ims = (ims + 1) / 2
#         grid = make_grid(ims, nrow=train_config['num_grid_rows'])
#         img = torchvision.transforms.ToPILImage()(grid)
#         if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
#             os.mkdir(os.path.join(train_config['task_name'], 'samples'))
#         img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
#         img.close()

def sample_turb(model, scheduler, train_config, test_config, model_config, diffusion_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt = torch.randn((test_config['batch_size'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)

    # saving generated data
    if test_config['save_data']:
        if not os.path.exists(os.path.join(train_config['task_name'], 'data')):
            os.mkdir(os.path.join(train_config['task_name'], 'data'))
    
    for batch_count in range(test_config['num_test_batch']):

        # Saving generated images
        if test_config['save_image']:
            if not os.path.exists(os.path.join(train_config['task_name'], 'samples', str(batch_count))):
                os.mkdir(os.path.join(train_config['task_name'], 'samples', str(batch_count)))

            nrows = int(np.floor(np.sqrt(test_config['num_samples'])))

            figU, axesU = plt.subplots(nrows=nrows, ncols=nrows, figsize=(15, 15))
            figV, axesV = plt.subplots(nrows=nrows, ncols=nrows, figsize=(15, 15))

        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            # Get prediction of noise
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
            
            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

            if test_config['save_image']:

                if i % 100 == 0 :
                
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
                    figU.savefig(os.path.join(train_config['task_name'], 'samples', 'U0_{}.jpg'.format(i)), format='jpg', bbox_inches='tight', pad_inches=0)

                    # Loop over the grid
                    for ax_count, ax in enumerate(axesV.flat):
                        # Plot each matrix using the 'bwr' colormap
                        plotV = ax.pcolorfast(V_arr[ax_count,:,:], cmap='bwr', vmin=-vmax_V, vmax=vmax_V)
                        ax.axis('off')
                    # Adjust spacing between subplots to avoid overlap
                    figV.subplots_adjust(wspace=0.1, hspace=0.1)
                    figV.savefig(os.path.join(train_config['task_name'], 'samples', 'V0_{}.jpg'.format(i)), format='jpg', bbox_inches='tight', pad_inches=0)

                    # figU.clear(), figV.clear()
                    del plotU, plotV

        if test_config['save_data']:
            np.save(os.path.join(train_config['task_name'], 'data', str(batch_count) + '.npy'), xt.numpy())



# def sample_turb(model, scheduler, train_config, model_config, diffusion_config):
#     r"""
#     Sample stepwise by going backward one timestep at a time.
#     We save the x0 predictions
#     """
#     xt = torch.randn((train_config['num_samples'],
#                       model_config['im_channels'],
#                       model_config['im_size'],
#                       model_config['im_size'])).to(device)
#     for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
#         # Get prediction of noise
#         noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
#         # Use scheduler to get x0 and xt-1
#         xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
#     # Save x0
#     ims = torch.clamp(xt, -1., 1.).detach().cpu()
#     U_arr = ims[:,0,:,:].numpy()
#     V_arr = ims[:,1,:,:].numpy()

    


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    test_config = config['test_params']
    
    # Load model with checkpoint
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))
    model.eval()
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample_turb(model, scheduler, train_config, test_config, model_config, diffusion_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
