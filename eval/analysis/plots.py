import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist

def save_image(xt, time_step, train_config, sample_config, dataset_config, run_num, batch_count):

    device_ID = dist.get_rank() if dist.is_initialized() else 0

    os.makedirs(os.path.join(train_config['save_dir'], 'samples'), exist_ok=True)
    os.makedirs(os.path.join(train_config['save_dir'], 'samples', str(run_num) + '_' + str(device_ID)), exist_ok=True)

    nrows = int(np.floor(np.sqrt(sample_config['sample_batch_size'])))

    figU, axesU = plt.subplots(nrows=nrows, ncols=nrows, figsize=(15, 15))

    if not 'mnist' in dataset_config['data_dir'].lower():
        figV, axesV = plt.subplots(nrows=nrows, ncols=nrows, figsize=(15, 15))

        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()

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

        figU.savefig(os.path.join(train_config['save_dir'], 'samples', str(run_num) + '_' + str(device_ID), f'{str(batch_count)}_U{time_step}.jpg'), format='jpg', bbox_inches='tight', pad_inches=0)
        plt.close(figU) # Close the figure to free up memory

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

            figV.savefig(os.path.join(train_config['save_dir'], 'samples', str(run_num) + '_' + str(device_ID), f'{str(batch_count)}_V{time_step}.jpg'), format='jpg', bbox_inches='tight', pad_inches=0)
            plt.close(figV)
