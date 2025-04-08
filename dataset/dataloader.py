import os, sys
import numpy as np
import torch
import scipy

from torch.utils.data import Dataset

from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.convert import Omega2Psi, Psi2UV, UV2Omega
from py2d.filter import filter2D

class CustomMatDataset(Dataset):
    def __init__(self, dataset_config, train_config, test_config, get_UV=True, get_Psi=False, get_Omega=False):
        """
        Args:
            data_dir (str): Directory with all the .mat files.
            file_range (tuple): Range of file numbers (start, end).
        """
        self.data_dir = dataset_config['data_dir']
        self.file_range = dataset_config['file_range']
        self.file_numbers = range(self.file_range[0], self.file_range[1] + 1)
        self.step_size = dataset_config['step_size']
        files_data = [os.path.join(self.data_dir, 'data', f"{i}.mat") for i in self.file_numbers if (self.file_range[0]-i) % self.step_size == 0]  # include only every step_size-th file
        self.file_list_data = files_data
        self.downsample_factor = dataset_config['downsample_factor']
        self.downsample_procedure = dataset_config['downsample_procedure']
        self.normalize = dataset_config['normalize']
        self.model_collapse = train_config['model_collapse']

        self.get_UV = get_UV
        self.get_Psi = get_Psi
        self.get_Omega = get_Omega

        if train_config['model_collapse']:

            self.model_collapse_gen = train_config['model_collapse_gen']
            self.model_collapse_type = train_config['model_collapse_type']

            self.file_batch_size = test_config['batch_size']
            filenum1 = 0
            filenum2 = len(self.file_list_data)//self.file_batch_size - 1
            self.file_numbers = range(filenum1, filenum2 + 1)

            data_dir_first_gen = os.path.join('results', train_config['task_name'])

            if self.model_collapse_type == 'last_gen':
                if self.model_collapse_gen == 1:
                    data_dir_last_gen = data_dir_first_gen
                elif self.model_collapse_gen > 1:
                    data_dir_last_gen = data_dir_first_gen + '_' + self.model_collapse_type + '_' + str(self.model_collapse_gen-1)

                self.file_list_model_collapse = [os.path.join(data_dir_last_gen, 'data/1/', f"{i}.npy") for i in self.file_numbers]
                print('** filenum **', filenum1, filenum2)

            elif self.model_collapse_type == 'all_gen':
                # Build a unified file list
                self.file_list_model_collapse = []
                for gen in range(1, self.model_collapse_gen+1):
                    if gen == 1:
                        data_dir_gen = data_dir_first_gen
                    elif gen > 1:
                        data_dir_gen = data_dir_first_gen + '_' + self.model_collapse_type + '_' + str(self.model_collapse_gen-1)
                    files = [os.path.join(data_dir_gen, 'data/1/', f"{i}.npy") for i in self.file_numbers]

                    self.file_list_model_collapse.extend(files)


            print('************************** length of file list - Model Collapse:', len(self.file_list_model_collapse), ' files', len(self.file_list_model_collapse)*self.file_batch_size, ' files')

        print('************************** length of file list:', len(self.file_list_data))

    def __len__(self):

        if self.model_collapse:
            if self.model_collapse_type == 'last_gen':
                # Return the total number of files in the model collapse
                return len(self.file_list_model_collapse) * self.file_batch_size
            elif self.model_collapse_type == 'all_gen':
                # Return the total number of files in the model collapse
                # This includes both the original .mat files and the .npy files
                # from all generations.
                return len(self.file_list_data) + len(self.file_list_model_collapse) * self.file_batch_size
        else:
            # Return the total number of files in the dataset
            return len(self.file_list_data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the file (snapshot) to load.
            
        Returns:
            torch.Tensor: Data loaded from the .mat file.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        Omega, Psi, U, V = None, None, None, None

        # Loading correct index of files from .mat or .npy files (model_collapse)
        if not self.model_collapse:
            # Load the .mat file
            file_path = self.file_list_data[idx]
            mat_data = scipy.io.loadmat(file_path)
            Omega = mat_data['Omega']
            nx, ny = Omega.shape

        else:
            if self.model_collapse_type == 'last_gen':
            # Load from last generation
                batch_ind = (idx)//self.file_batch_size
                file_path = self.file_list_model_collapse[batch_ind]
                npy_data = np.load(file_path)
                file_num = (idx) % self.file_batch_size

            elif self.model_collapse_type == 'all_gen':
            # Load from all generations
                if idx < len(self.file_list_data):
                    # load the .mat file
                    file_path = self.file_list_data[idx]
                    mat_data = scipy.io.loadmat(file_path)
                    Omega = mat_data['Omega']
                    nx, ny = Omega.shape
                    npy_data = None
                else:
                    batch_ind = (idx-len(self.file_list_data))//self.file_batch_size
                    file_path = self.file_list_model_collapse[batch_ind]
                    npy_data = np.load(file_path)
                    file_num = (idx-len(self.file_list_data)) % self.file_batch_size

            if npy_data is not None:
                U = npy_data[file_num, 0, :, :]
                V = npy_data[file_num, 1, :, :]
                nx, ny = U.shape

        nx_downsample = nx // self.downsample_factor
        ny_downsample = ny // self.downsample_factor
        Lx, Ly = 2 * np.pi, 2 * np.pi
        Kx, Ky, _, _, invKsq = initialize_wavenumbers_rfft2(nx, ny, Lx, Ly, INDEXING='ij')

        if self.get_Psi or self.get_UV:
            if Omega is None:
                Omega = UV2Omega(U, V, Kx, Ky)
            Psi = Omega2Psi(Omega, invKsq)
        if self.get_UV:
            if U is None or V is None:
                U, V = Psi2UV(Psi, Kx, Ky)
            else:
                pass

        if self.get_UV:
            if self.downsample_procedure == 'physical':
                U_downsampled = self.downsample_array(U, self.downsample_factor, self.downsample_factor)
                V_downsampled = self.downsample_array(V, self.downsample_factor, self.downsample_factor)
            elif self.downsample_procedure == 'spectral':
                U_downsampled = filter2D(U, filterType=None, coarseGrainType='spectral', Delta=None, 
                                         Ngrid=[nx_downsample, ny_downsample], spectral=False)
                V_downsampled = filter2D(V, filterType=None, coarseGrainType='spectral', Delta=None, 
                                         Ngrid=[nx_downsample, ny_downsample], spectral=False)

        if self.get_Psi:
            if self.downsample_procedure == 'physical':
                Psi_downsampled = self.downsample_array(Psi, self.downsample_factor, self.downsample_factor)
            elif self.downsample_procedure == 'spectral':
                Psi_downsampled = filter2D(Psi, filterType=None, coarseGrainType='spectral', Delta=None, 
                                         Ngrid=[nx_downsample, ny_downsample], spectral=False)

        if self.get_Omega:
            if self.downsample_procedure == 'physical':
                Omega_downsampled = self.downsample_array(Omega, self.downsample_factor, self.downsample_factor)
            elif self.downsample_procedure == 'spectral':
                Omega_downsampled = filter2D(Omega, filterType=None, coarseGrainType='spectral', Delta=None, 
                                         Ngrid=[nx_downsample, ny_downsample], spectral=False)

        if self.normalize:
            mean_std_data = np.load(self.data_dir + 'mean_std.npz')

        
        # Combine U, V, Psi, Omega into a single tensor with 2 channels
        if self.get_UV and self.get_Psi and self.get_Omega:
            data_tensor = torch.tensor(np.stack([U_downsampled, V_downsampled, Psi_downsampled, Omega_downsampled], axis=0))
            if self.normalize:
                mean = [mean_std_data['U_mean'], mean_std_data['V_mean'], mean_std_data['Psi_mean'], mean_std_data['Omega_mean']]
                std = [mean_std_data['U_std'], mean_std_data['V_std'], mean_std_data['Psi_std'], mean_std_data['Omega_std']]

        elif self.get_UV and self.get_Psi:
            data_tensor = torch.tensor(np.stack([U_downsampled, V_downsampled, Psi_downsampled], axis=0))
            if self.normalize:
                mean = [mean_std_data['U_mean'], mean_std_data['V_mean'], mean_std_data['Psi_mean']]
                std = [mean_std_data['U_std'], mean_std_data['V_std'], mean_std_data['Psi_std']]

        elif self.get_UV and self.get_Omega:
            data_tensor = torch.tensor(np.stack([U_downsampled, V_downsampled, Omega_downsampled], axis=0))
            if self.normalize:
                mean = [mean_std_data['U_mean'], mean_std_data['V_mean'], mean_std_data['Omega_mean']]
                std = [mean_std_data['U_std'], mean_std_data['V_std'], mean_std_data['Omega_std']]

        elif self.get_Psi and self.get_Omega:
            data_tensor = torch.tensor(np.stack([Psi_downsampled, Omega_downsampled], axis=0))
            if self.normalize:
                mean = [mean_std_data['Psi_mean'], mean_std_data['Omega_mean']]
                std = [mean_std_data['Psi_std'], mean_std_data['Omega_std']]

        elif self.get_UV:
            data_tensor = torch.tensor(np.stack([U_downsampled, V_downsampled], axis=0))
            if self.normalize:
                mean = [mean_std_data['U_mean'], mean_std_data['V_mean']]
                std = [mean_std_data['U_std'], mean_std_data['V_std']]

        elif self.get_Psi:
            data_tensor = torch.tensor(np.stack([Psi_downsampled], axis=0))
            if self.normalize:
                mean = [mean_std_data['Psi_mean']]
                std = [mean_std_data['Psi_std']]

        elif self.get_Omega:
            data_tensor = torch.tensor(np.stack([Omega_downsampled], axis=0))
            if self.normalize:
                mean = [mean_std_data['Omega_mean']]
                std = [mean_std_data['Omega_std']]

        if self.normalize:
            # Convert to tensors and reshape
            mean  = np.asarray(mean)
            std = np.asarray(std)

            mean_tensor = torch.tensor(mean).reshape(mean.shape[0], 1, 1)
            std_tensor = torch.tensor(std).reshape(std.shape[0], 1, 1)
            data_tensor_normalized = (data_tensor - mean_tensor) / std_tensor

            return data_tensor_normalized
        else:
            # print(data_tensor.shape)
            return data_tensor

    def downsample_array(self, arr, x_factor, y_factor):
        """
        Downsamples a 2D array by selecting every x_factor-th element in the x-direction
        and every y_factor-th element in the y-direction.

        Parameters:
            arr (numpy.ndarray): The original 2D array to downsample.
            x_factor (int): Downsampling factor in the x-direction.
            y_factor (int): Downsampling factor in the y-direction.

        Returns:
            numpy.ndarray: The downsampled 2D array.

        Raises:
            ValueError: If x_factor or y_factor is not a divisor of the array's dimensions.
        """
        ny, nx = arr.shape
        if nx % x_factor != 0:
            raise ValueError(f"Downsampling factor x_factor={x_factor} is not a divisor of number of columns nx={nx}.")
        if ny % y_factor != 0:
            raise ValueError(f"Downsampling factor y_factor={y_factor} is not a divisor of number of rows ny={ny}.")
        return arr[::y_factor, ::x_factor]
