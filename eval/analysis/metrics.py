import numpy as np
from scipy.stats import gaussian_kde, pearsonr

from py2d.initialize import initialize_wavenumbers_rfft2, gridgen
from py2d.derivative import derivative
from py2d.convert import Omega2Psi, Psi2UV

def manual_eof(X_demeaned, n_comp=1):
    # Step 1: Demean the data
    # X_demeaned = X - np.mean(X, axis=0)
    # Step 2: Covariance matrix
    C = np.cov(X_demeaned, rowvar=False)
    # Step 3: Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # Step 4: Sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    # First three EOFs and PCs
    EOFs_manual = eigenvectors[:, :n_comp]
    PCs_manual = X_demeaned @ EOFs_manual

    # Calculate the explained variance
    total_variance = np.sum(eigenvalues)
    explained_variance = eigenvalues / total_variance

    return EOFs_manual, PCs_manual, np.flip(explained_variance)[:3]

# Method 2: Manual calculation of EOFs and PCs using SVD
def manual_svd_eof(X_demeaned):

    # Step 2: Apply SVD on the demeaned data
    # U: left singular vectors (PCs)
    # s: singular values
    # Vt: right singular vectors (EOFs, transposed)
    U, s, Vt = np.linalg.svd(X_demeaned, full_matrices=False)
    
    # Step 3: Extract the first three EOFs and PCs
    EOFs_svd = Vt.T    # EOFs and transpose to shape (N, 3)
    PCs_svd = U * s # Scale U by the singular values to get PCs
    
    # Step 4: Calculate explained variance
    total_variance = np.sum(s ** 2)
    explained_variance = (s ** 2) / total_variance
    
    return EOFs_svd, PCs_svd, explained_variance

def divergence(U, V):
    """
    Args:
        U: [X, Y] 
        V: [X, Y]
    Returns:
        div: [X,Y] divergence vs time
    """
   
    Lx, Ly = 2*np.pi, 2*np.pi
    Nx, Ny = U.shape[0], U.shape[1]
    Lx, Ly, X, Y, dx, dy = gridgen(Lx, Ly, Nx, Ny, INDEXING='ij')

    Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_rfft2(Nx, Ny, Lx, Ly, INDEXING='ij')

    Ux_t = derivative(U.T, [1,0], Kx, Ky, spectral=False) #[1,0]
    Vy_t = derivative(V.T, [0,1], Kx, Ky, spectral=False) #[0,1]
    div = Ux_t.T + Vy_t.T

    return div

def PDF_compute(data, bw_factor=1):
    data_arr = np.array(data).flatten()
    del data

    # Calculate mean and standard deviation
    data_mean, data_std = np.mean(data_arr), np.std(data_arr)

    # Define bins within 10 standard deviations from the mean, but also limit them within the range of the data
    bin_max = np.min(np.abs([np.min(data_arr), np.max(data_arr)]))
    bin_min = -bin_max
    bins = np.linspace(bin_min, bin_max, 100)

    print('PDF Calculation')
    print('bin min', bin_min)
    print('bin max', bin_max)
    print('data Shape', data_arr.shape)
    print('data mean', data_mean)
    print('data_std', data_std)
    print('Total nans', np.sum(np.isnan(data_arr)))

    # Compute PDF using Scipy
    bw1 = bw_factor*(data_arr.shape[0])**(-1/5) # custom bw method scott method n**(-1/5)
    kde = gaussian_kde(data_arr, bw_method=bw1)

    # # Define a range over which to evaluate the density
    data_bins = bins
    bw_scott = kde.factor
    # # Evaluate the density over the range
    data_pdf = kde.evaluate(data_bins)

    return data_mean, data_std, data_pdf, data_bins, bw_scott

def percentile_data(data, percentile):
    """
    Calculate error bands and
    return the lower/upper bounds in percentile

    Parameters:
    -----------
    data : np.ndarray
        2D NumPy array of shape (N, samples)
    percentile : float
        A number between 0 and 100 (typically <= 50 for symmetrical bounds)
    
    Returns:
    --------
    means : np.ndarray
        Array of sample means (shape: N)
    lower_bounds: np.ndarray
        Array of lower bounds in percentage relative to the mean (shape: N)
    upper_bounds: np.ndarray
        Array of upper bounds in percentage relative to the mean (shape: N)
    """
    # Mean of each row
    means = np.mean(data, axis=0)
    
    # Lower (p-th) and upper ((100-p)-th) percentiles of each row
    lower_vals = np.percentile(data, percentile, axis=0)
    upper_vals = np.percentile(data, 100 - percentile, axis=0)

    print(data.shape)
    # print(lower_vals.shape, upper_vals.shape)
    # print(lower_vals, upper_vals)
    
    # Calculate percentage difference relative to the mean
    # (m - L)/m * 100 for lower, (U - m)/m * 100 for upper
    # lower_bounds = 100.0 * (means - lower_vals) / means
    # upper_bounds = 100.0 * (upper_vals - means) / means
    
    return means, lower_vals, upper_vals

def std_dev_data(data, std_dev=1):
    """
    Calculate error bands and
    return the lower/upper bounds in percentile

    Parameters:
    -----------
    data : np.ndarray
        2D NumPy array of shape (N, samples)
    std_dev : float
        A number between 0 and 100 (typically <= 50 for symmetrical bounds)
    
    Returns:
    --------
    means : np.ndarray
        Array of sample means (shape: N)
    lower_bounds: np.ndarray
        Array of lower bounds in percentage relative to the mean (shape: N)
    upper_bounds: np.ndarray
        Array of upper bounds in percentage relative to the mean (shape: N)
    """
    # Mean of each row
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    
    # Lower (p-th) and upper ((100-p)-th) percentiles of each row
    lower_vals = means - stds*std_dev
    upper_vals = means + stds*std_dev
    
    return means, lower_vals, upper_vals

def corr_truth_train_model(truth, train, model):
    # Correlation between truth, train, model fields
    corr_truth_train, _ = pearsonr(truth.flatten(), train.flatten())
    corr_truth_model, _ = pearsonr(truth.flatten(), model.flatten())
    corr_train_model, _ = pearsonr(train.flatten(), model.flatten())
    return np.round(corr_truth_train,2), np.round(corr_truth_model,2), np.round(corr_train_model,2)