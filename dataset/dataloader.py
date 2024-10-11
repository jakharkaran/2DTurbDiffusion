import os
import numpy as np
import torch
import scipy

from torch.utils.data import Dataset

from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.convert import Omega2Psi, Psi2UV

class CustomMatDataset(Dataset):
    def __init__(self, data_dir, file_range):
        """
        Args:
            data_dir (str): Directory with all the .mat files.
            file_range (tuple): Range of file numbers (start, end).
        """
        self.data_dir = data_dir
        self.file_numbers = range(file_range[0], file_range[1] + 1)
        self.file_list = [os.path.join(data_dir, f"{i}.mat") for i in self.file_numbers]

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
        Lx, Ly = 2 * np.pi, 2 * np.pi
        Kx, Ky, _, _, invKsq = initialize_wavenumbers_rfft2(nx, ny, Lx, Ly, INDEXING='ij')

        Psi = Omega2Psi(Omega, invKsq)
        U, V = Psi2UV(Psi, Kx, Ky)

        # Combine U and V into a single tensor with 2 channels
        data_tensor = torch.tensor(np.stack([U, V]), dtype=torch.float32)

        return data_tensor
