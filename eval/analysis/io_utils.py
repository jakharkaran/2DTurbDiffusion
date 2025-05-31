import numpy as np
import scipy.io as sio

from py2d.convert import UV2Omega, Omega2UV

def frame_generator(files, data_type, Kx, Ky, invKsq):
    """
    Yields (U, V, Omega) one snapshot at a time from emulator or truth files.
    - For emulator: each file is a .npy chunk of shape [B, 2, H, W] (channels: [U, V])
    - For truth: each file is a .mat with Omega of shape [W, H]
    """
    for fname in files:
        if data_type == "emulator":
            arr = np.load(fname)  # shape [B, 2, H, W]
            for i in range(arr.shape[0]):
                U = arr[i, 0].T
                V = arr[i, 1].T
                Omega = UV2Omega(U.T, V.T, Kx, Ky, spectral=False).T
                yield U.astype(np.float32), V.astype(np.float32), Omega.astype(np.float32)
        else:  # truth
            mat = sio.loadmat(fname)
            Omega = mat["Omega"].T.astype(np.float32)
            U_t, V_t = Omega2UV(Omega.T, Kx, Ky, invKsq, spectral=False)
            U, V = U_t.T.astype(np.float32), V_t.T.astype(np.float32)
            yield U, V, Omega

def remove_boundaries(arr, n):
    """
    Removes n boundary layers from both axes of a 2D array.

    Parameters:
    -----------
    arr : np.ndarray
        Input 2D array.
    n : int
        Number of layers to remove from each side (top, bottom, left, right).

    Returns:
    --------
    np.ndarray
        Cropped 2D array.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    if arr.shape[0] <= 2*n or arr.shape[1] <= 2*n:
        raise ValueError("n is too large for the array dimensions.")

    return arr[n:-n, n:-n]