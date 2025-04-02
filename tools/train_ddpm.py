import torch
import yaml
import argparse
import os, sys
import shutil
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
# from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
import wandb
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from dataset.dataloader import CustomMatDataset, CustomMatDatasetEmulator

from py2d.derivative import derivative 
from py2d.initialize import initialize_wavenumbers_rfft2

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
        
    # # Create the dataset
    # mnist = MnistDataset('train', im_path=dataset_config['im_path'])
    # mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)

    
    # Create Dataset and DataLoader
    if dataset_config['data_dir'] == 'Re500_fkx4fky4_r0.1_b20/2D_64_spectral_dt0.02_noise_2500/' or dataset_config['data_dir'] == 'Re500_fkx4fky4_r0.1_b0/2D_64_spectral_dt0.02_noise_2500/':
        dataset = CustomMatDatasetEmulator(data_dir=dataset_config['data_dir'], file_range=dataset_config['file_range'], step_size=dataset_config['step_size'], downsample_factor=dataset_config['downsample_factor'], downsample_procedure=dataset_config['downsample_procedure'], normalize=dataset_config['normalize'])
        print('Using CustomMatDatasetEmulator')
    else:
        dataset = CustomMatDataset(data_dir=dataset_config['data_dir'], file_range=dataset_config['file_range'], step_size=dataset_config['step_size'], downsample_factor=dataset_config['downsample_factor'], downsample_procedure=dataset_config['downsample_procedure'], normalize=dataset_config['normalize'])
        print('Using CustomMatDataset')

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
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        shutil.copy(args.config_path, train_config['task_name'])
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(train_config['task_name'],train_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ckpt_name']), map_location=device))

    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 2D Trubulence System Parameter
    nx, ny = int(model_config['im_size']/dataset_config['downsample_factor']), int(model_config['im_size']/dataset_config['downsample_factor'])
    Lx, Ly = 2 * np.pi, 2 * np.pi
    Kx, Ky, _, _, invKsq = initialize_wavenumbers_rfft2(nx, ny, Lx, Ly, INDEXING='ij')

    mean_std_data = np.load(dataset_config['data_dir'] + 'mean_std.npz')
    mean_data = [mean_std_data['U_mean'], mean_std_data['V_mean']]
    std_data = [mean_std_data['U_std'], mean_std_data['V_std']]

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

            if train_config['divergence_loss']:
                model.eval()
                with torch.inference_mode():

                    # Compute Divergence
                    xt = torch.randn((test_config['batch_size'],
                    model_config['im_channels'],
                    model_config['im_size'],
                    model_config['im_size'])).to(device)


                    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
                        # Get prediction of noise
                        noise_pred  = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
                        
                        # Use scheduler to get x0 and xt-1
                        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

                    ims = torch.clamp(xt, -1., 1.).detach().cpu()

                    # De-normalize the data
                    U_arr = ims[:,0,:,:].numpy()*std_data[0] + mean_data[0]
                    V_arr = ims[:,1,:,:].numpy()*std_data[1] + mean_data[1]

                    # Caculate the divergence
                    Div_arr = []
                    for count in range(test_config['batch_size']):
                        Ux = derivative(U_arr[count,:,:], [1,0], Kx, Ky, spectral=False)
                        Vy = derivative(V_arr[count,:,:], [0,1], Kx, Ky, spectral=False)
                        Div_arr.append(np.mean(np.abs(Ux + Vy)))

                    div = np.mean(Div_arr)
                    loss_div = train_config['divergence_loss_weight'] * div

                model.train()
                loss = loss_mse + loss_div

                del xt, x0_pred, ims, U_arr, V_arr, noise_pred
                torch.cuda.empty_cache()

                log_data["loss_div"] = loss_div # loggin for wandb

            else:
                loss = loss_mse

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            log_data["loss"] = loss
            log_data["best_loss"] = best_loss
            wandb.log(log_data) # Logging all data to wandb


        if best_loss > loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                        train_config['best_ckpt_name']))
            
        print('Finished epoch:{} | Loss (mean/current/best) : {:.4f}/{:.4f}/{:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses), loss, best_loss
        ))
        # Save the model
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
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
