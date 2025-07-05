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

def return_period_empirical(X, dt=1):
    """
    Calculate the empirical return period for a dataset.

    Args:
        X: 1D array-like, input data (e.g., time series of amplitudes)
        dt: float, time step (default=1)

    Returns:
        return_period: 1D array, empirical return period for each data point (same length as X)
        data_amplitude: 1D array, sorted data amplitudes (ascending order)
    """

    # Sort the data in ascending order
    data_amplitude = np.sort(X)
    n = len(X)
    # Rank order (1-based)
    m = np.arange(1, n + 1)
    # Empirical cumulative distribution function (CDF)
    cdf_empirical = m / (n + 1)
    # Empirical return period formula: T = 1 / (1 - F)
    return_period = 1 / (1 - cdf_empirical)

    # Scale return period by time step
    return return_period * dt, data_amplitude

def return_period_bins(data, dt=1, bins_num=100):
    """
    Calculate return period of data, bin it and interpolate the amplitude values to the bins.

    Args:
        data: 1D array of data [time]
        dt: time step
        bins_num: number of bins for binning the data

    Returns:
        bins: array of bin edges (return periods)
        interp_data_amplitude: interpolated amplitude values at each bin
    """

    # Compute empirical return period and sorted data amplitude
    return_period, data_amplitude = return_period_empirical(data, dt=dt)

    # Set bin range 
    bin_min = np.min(return_period)
    bin_max = np.max(return_period)
    # Create logarithmically spaced bins for return period
    bins = np.logspace(np.log10(bin_min), np.log10(bin_max), num=bins_num)

    # Interpolate the amplitude values to the bins
    interp_data_amplitude = np.interp(bins, return_period, data_amplitude)

    return bins, interp_data_amplitude


def ensemble_return_period_amplitude(data, dt=1, bins_num=100, central_tendency='mean', error_bands='std'):
    '''
    Calculate return period and error band using ensemble of data. The error bands are calculated for data amplitude.

    Args:
        data: 2D array of data [ensemble, time]
        dt: time step
        bins_num: number of bins for binning the data
        central_tendency: 'mean' or 'median'
            Determines the central tendency for the amplitude (default: 'mean')
        error_bands: 'std', '50ci', '95ci', or None
            Determines the method for calculating the error/confidence interval

    Returns:
        bins: array of bin edges (return periods)
        central_data_amplitude_interp: central tendency (mean/median) of amplitude at each bin
        lb_data_amplitude_interp: lower bound of error band at each bin
        ub_data_amplitude_interp: upper bound of error band at each bin
    '''

    # Arrays to store return periods and amplitudes for each ensemble member
    return_period_arr = []
    data_amplitude_arr = []

    number_ensemble = data.shape[0]
    total_data_points = data.shape[1]

    # Compute empirical return period and amplitude for each ensemble member
    for i in range(number_ensemble):
        data_ensemble = data[i, :]
        return_period, data_amplitude = return_period_empirical(data_ensemble, dt=dt)
        return_period_arr.append(return_period)
        data_amplitude_arr.append(data_amplitude)

    # Define bins for return period (logarithmic spacing)
    bin_min = np.min(return_period_arr)
    bin_max = np.max(return_period_arr)
    bins = np.logspace(np.log10(bin_min), np.log10(bin_max), num=bins_num)

    # Interpolate amplitude values to the common bins for each ensemble member
    data_amplitude_interp_arr = []
    for i in range(number_ensemble):
        data_amplitude_interp_arr.append(np.interp(bins, return_period_arr[i], data_amplitude_arr[i]))

    # Compute central tendency (mean or median) across the ensemble
    if central_tendency == 'mean':
        central_data_amplitude_interp = np.mean(data_amplitude_interp_arr, axis=0)
    elif central_tendency == 'median':
        central_data_amplitude_interp = np.median(data_amplitude_interp_arr, axis=0)

    # Compute error bands (confidence intervals or standard deviation)
    if error_bands in ['50ci', '95ci']:
        if error_bands == '50ci':
            confidence_level = 25
        elif error_bands == '95ci':
            confidence_level = 2.5

        # Use percentiles for error bands
        _, lb_data_amplitude_interp, ub_data_amplitude_interp = percentile_data(
            np.asarray(data_amplitude_interp_arr), percentile=confidence_level)

    elif error_bands == 'std':
        # Use standard deviation for error bands
        _, lb_data_amplitude_interp, ub_data_amplitude_interp = std_dev_data(
            np.asarray(data_amplitude_interp_arr), std_dev=1)

    elif error_bands is None:
        # No error bands requested
        return bins, central_data_amplitude_interp, None, None

    return bins, central_data_amplitude_interp, lb_data_amplitude_interp, ub_data_amplitude_interp

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