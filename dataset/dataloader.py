import os
import numpy as np
import torch
import scipy

from torch.utils.data import Dataset

from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.convert import Omega2Psi, Psi2UV
from py2d.filter import filter2D

class CustomMatDataset(Dataset):
    def __init__(self, data_dir, file_range, downsample_factor=1, downsample_procedure='physical', get_UV=True, get_Psi=False, get_Omega=False, normalize=True):
        """
        Args:
            data_dir (str): Directory with all the .mat files.
            file_range (tuple): Range of file numbers (start, end).
        """
        self.data_dir = data_dir
        self.file_numbers = range(file_range[0], file_range[1] + 1)
        self.file_list = [os.path.join(data_dir, 'data2', f"{i}.mat") for i in self.file_numbers]
        self.downsample_factor = downsample_factor
        self.downsample_procedure = downsample_procedure
        self.get_UV = get_UV
        self.get_Psi = get_Psi
        self.get_Omega = get_Omega
        self.normalize = normalize

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the file to load.
            
        Returns:
            torch.Tensor: Data loaded from the .mat file.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.file_list[idx]
        mat_data = scipy.io.loadmat(file_path)
        Omega = mat_data['Omega']

        nx, ny = Omega.shape
        nx_downsample = nx // self.downsample_factor
        ny_downsample = ny // self.downsample_factor
        Lx, Ly = 2 * np.pi, 2 * np.pi
        Kx, Ky, _, _, invKsq = initialize_wavenumbers_rfft2(nx, ny, Lx, Ly, INDEXING='ij')

        if self.get_Psi or self.get_UV:
            Psi = Omega2Psi(Omega, invKsq)
        if self.get_UV:
            U, V = Psi2UV(Psi, Kx, Ky)

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

