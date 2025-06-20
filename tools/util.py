import torch
import numpy as np

def grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def grad_max(model):
    max_grad = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_max = torch.max(torch.abs(p.grad.detach().data))
        if max_grad < param_max.item():
            max_grad = param_max.item()
    return param_max

def generate_grid(nx, ny, device='cpu'):
    """
    Generate a 2D coordinate grid normalized to [-1, 1] x [-1, 1].
    
    Parameters:
    -----------
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    device : str, optional
        Torch device ('cpu' or 'cuda'). Default is 'cpu'.
    
    Returns:
    --------
    torch.Tensor
        Coordinate grids of shape [1, 2, nx, ny] where the first channel is X 
        and the second channel is Y coordinates.
    """
    
    # Domain size normalized to 1
    Lx_grid, Ly_grid = 1.0, 1.0

    # Calculate the size of the grid spacing
    dx = Lx_grid / nx
    dy = Ly_grid / ny

    # Create an array of x-coordinates, ranging from 0 to (Lx_grid - dx)
    x = np.linspace(0, Lx_grid - dx, num=nx) * 2 - 1  # Scale to range [-1, 1]
    y = np.linspace(0, Ly_grid - dy, num=ny) * 2 - 1  # Scale to range [-1, 1]

    # Create 2D arrays of the x and y-coordinates using a meshgrid.
    X, Y = np.meshgrid(x, y, indexing='ij')

    X = torch.from_numpy(X).float().to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, nx, ny]
    Y = torch.from_numpy(Y).float().to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, nx, ny]
    coord_grids = torch.cat((X, Y), dim=1) # [1, 2, nx, ny]
    
    return coord_grids


class SpectralDifferentiator:
    def __init__(self, nx, ny, Lx=2*torch.pi, Ly=2*torch.pi, device='cpu', dtype=torch.float32):
        """
        Precompute the 2D wavenumber grids for a periodic domain.
        
        Parameters:
        -----------
        nx : int
            Number of grid points in the x-direction.
        ny : int
            Number of grid points in the y-direction.
        Lx : float, optional
            Domain length in the x-direction. Default is 2*pi.
        Ly : float, optional
            Domain length in the y-direction. Default is 2*pi.
        device : str, optional
            Torch device ('cpu' or 'cuda'). Default is 'cpu'.
        dtype : torch.dtype, optional
            Data type for the tensors. Default is torch.float32.
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.device = device
        self.dtype = dtype
        
        # Compute the 1D wavenumbers
        self.kx = 2 * torch.pi * torch.fft.fftfreq(nx, d=Lx/nx, device=device, dtype=dtype)

        ky_full = 2 * torch.pi * torch.fft.fftfreq(ny, d=Ly/ny, device=device, dtype=dtype)
        self.ky = ky_full[:self.ny//2+1]
        
        # Create 2D grids for the wavenumbers (using 'ij' indexing)
        self.Kx, self.Ky = torch.meshgrid(self.kx, self.ky, indexing='ij')
    
    def derivative(self, T, order, spectral=False):
        """
        Compute spatial derivatives for a 2D field using spectral methods.
        
        This method supports both single sample tensors of shape [X, Y] and batched tensors of shape [B, X, Y].
        
        Parameters:
        -----------
        T : torch.Tensor
            Input field. If spectral is False, T is in the physical domain;
            otherwise, it is assumed to be in the spectral domain.
        order : list or tuple of two non-negative integers
            Derivative orders in the x and y dimensions, respectively.
        spectral : bool, optional
            Flag indicating whether T is already in spectral space. Default is False.
        
        Returns:
        --------
        torch.Tensor
            The derivative of the input field. Returns the result in the physical
            domain if spectral is False, otherwise in the spectral domain.
        """
        orderX, orderY = order
        
        # Compute the FFT along the last two dimensions.
        if not spectral:
            T_hat = torch.fft.rfft2(T)
        else:
            T_hat = T
        
        # Apply the Fourier derivative theorem.
        # The wavenumber grids (self.Kx, self.Ky) have shape [nx, ny_rfft] and will be broadcasted.
        factor = (1j * self.Kx) ** orderX * (1j * self.Ky) ** orderY
        T_derivative_hat = T_hat * factor
        
        # If the input was in the physical domain, transform back.
        if not spectral:
            T_derivative = torch.fft.irfft2(T_derivative_hat, s=(self.nx, self.ny))
            return T_derivative
        else:
            return T_derivative_hat
            
    def divergence(self, T, spectral=False):
        """
        Compute the divergence of a 2D vector field.
        
        Parameters:
        -----------
        T : torch.Tensor
            [B, C, X, Y] tensor representing the vector field, where C=2 (u and v components).
        spectral : bool, optional
            Flag indicating whether T is already in spectral space.
            If False (default), T is assumed to be in the physical domain.
        
        Returns:
        --------
        torch.Tensor
            [B, X, Y] tensor representing the divergence of the vector field.
            If spectral is False, the result is returned in the physical domain;
            otherwise, in the spectral domain.
        """
        # If the input is not already in spectral space, transform it using FFT.
        if not spectral:
            T_hat = torch.fft.rfft2(T)
        else:
            T_hat = T

        # Extract the u and v components in spectral space.
        U_hat = T_hat[:, 0, :, :]
        V_hat = T_hat[:, 1, :, :]

        # Compute the spatial derivatives in spectral space:
        # Ux_hat: derivative of the u component in the x direction.
        Ux_hat = self.derivative(U_hat, (1, 0), spectral=True)
        # Vy_hat: derivative of the v component in the y direction.
        Vy_hat = self.derivative(V_hat, (0, 1), spectral=True)

        # Compute the divergence in spectral space.
        divergence_hat = Ux_hat + Vy_hat

        # If the input was in the physical domain, transform the divergence back.
        if not spectral:
            return torch.fft.irfft2(divergence_hat, s=(self.nx, self.ny))
        else:
            return divergence_hat

