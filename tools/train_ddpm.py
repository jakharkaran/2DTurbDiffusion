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

    if logging_config['log_to_wandb']:
        # Set up wandb
        wandb.login()
        wandb.init(project=logging_config['project'], group=logging_config['group'],
                   name=logging_config['name'], config=config)


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

    # Run training
    best_loss = 1e6 # initialize
    for epoch_idx in range(num_epochs):
        losses = []
        # for im in tqdm(mnist_loader):
        for im in tqdm(turb_dataloader):
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
                loss = criterion(model_out, noise) # Noise is predicted by the model
            elif train_config['loss'] == 'sample':
                loss = criterion(model_out, im) # x0 (denoised sample) is predicted by the diffusion model
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        if best_loss > loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                        train_config['best_ckpt_name']))
            
        print('Finished epoch:{} | Loss (mean/current/best) : {:.4f}/{:.4f}/{:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses), loss, best_loss
        ))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ckpt_name']))
        
        if logging_config['log_to_wandb']:
            wandb.log({
                "epoch": epoch_idx + 1,
                "loss": loss,
                "best_loss": best_loss,
            })
    
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
