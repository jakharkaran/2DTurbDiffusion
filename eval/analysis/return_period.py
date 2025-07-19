import numpy as np

from analysis.metrics import percentile_data, std_dev_data

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

# # Ensemble return period calculation with uncertainty in frequency or Frequency exceedance
# The code is needs to be updated to wrok with the rest of the codebase
def return_period_ensemble_freq_freq_exceedance(data, dt=1, bins=50, uncertainty='freq_exceedance',
                           confidence_level_type='percentile', confidence_level=25):
    '''
    Calculate return period using ensemble of data
    data: 2D array of data [ensemble, time]
    dt: time step
    bins: number of bins for binning the data
    uncertainty: 'freq' / 'freq_exceedance': 
        Determines whether uncertainty is calculated based on frequency or frequency exceedance
    confidence_level_type: 'percentile' / 'std'
        Determines the method for calculating the confidence interval
    confidence_level: level (For confidence_level_type='percentile', % e.g.:25, 50, 75;'std', number of standard deviations e.g.: 1, 2, 3)
    '''
    
    number_ensemble = data.shape[0]
    total_data_points = data.shape[1]
    
    bin_min = np.min(data)
    bin_max = np.max(data)
    
    # Initialize lists to store per-ensemble probabilities
    prob_exceedance_ensemble = []
    freq_ensemble = []

    bins_centers = None  # Will be set during the first iteration

    for i in range(number_ensemble):
        data_ensemble = data[i, :]
        freq_sample, bin_edges = np.histogram(data_ensemble, bins=bins, range=(bin_min, bin_max))
        if bins_centers is None:
            bins_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        freq_exceedance_sample = np.cumsum(freq_sample[::-1])[::-1]
        prob_exceedance_sample = freq_exceedance_sample / total_data_points

        freq_ensemble.append(freq_sample)
        prob_exceedance_ensemble.append(prob_exceedance_sample)

    freq_ensemble = np.array(freq_ensemble)
    prob_exceedance_ensemble = np.array(prob_exceedance_ensemble)

    if uncertainty == 'freq' or uncertainty == 'freq_exceedance':
        # Calculate uncertainty based on the selected method
        if uncertainty == 'freq':
            # Average frequencies and compute probabilities from the mean frequencies
            freq_mean = np.mean(freq_ensemble, axis=0)

            # Compute cumulative frequencies
            freq_exceedance_mean = np.cumsum(freq_mean[::-1])[::-1]

            # Calculate exceedance probabilities
            prob_exceedance_mean = freq_exceedance_mean / (total_data_points * number_ensemble)

            # Calculate confidence intervals for frequencies
            lower_percentile = (100 - confidence_level) / 2
            upper_percentile = 100 - lower_percentile

            freq_exceedance_lower = np.percentile(np.cumsum(freq_ensemble[:, ::-1], axis=1)[:, ::-1],
                                                lower_percentile, axis=0)
            freq_exceedance_upper = np.percentile(np.cumsum(freq_ensemble[:, ::-1], axis=1)[:, ::-1],
                                                upper_percentile, axis=0)

            # Calculate exceedance probabilities for confidence intervals
            prob_exceedance_lower = freq_exceedance_lower / (total_data_points)
            prob_exceedance_upper = freq_exceedance_upper / (total_data_points)

        elif uncertainty == 'freq_exceedance':
            # Calculate mean of probabilities directly
            # prob_exceedance_mean, prob_exceedance_lower, prob_exceedance_upper = ci_t_distribution(prob_exceedance_ensemble, confidence_level)
            # prob_exceedance_mean, prob_exceedance_lower, prob_exceedance_upper = ci_normal_distribution(prob_exceedance_ensemble, confidence_level)

            if confidence_level_type == 'percentile':
                prob_exceedance_mean, prob_exceedance_lower, prob_exceedance_upper = percentile_data(prob_exceedance_ensemble, confidence_level)
            elif confidence_level_type == 'std':
                prob_exceedance_mean, prob_exceedance_lower, prob_exceedance_upper = std_dev_data(prob_exceedance_ensemble, confidence_level)
    
        # print(prob_exceedance_mean)
        # print(prob_exceedance_lower)
        # print(prob_exceedance_upper)

        # Ensure probabilities are within valid range
        return_period_mean = np.clip(prob_exceedance_mean, 1e-14, 1)
        prob_exceedance_lower = np.clip(prob_exceedance_lower, 1e-14, 1)
        prob_exceedance_upper = np.clip(prob_exceedance_upper, 1e-14, 1)

        # Calculate return periods
        return_period_mean = dt / prob_exceedance_mean
        return_period_lower = dt / prob_exceedance_upper  # Higher probability, lower return period
        return_period_upper = dt / prob_exceedance_lower  # Lower probability, higher return period

    elif uncertainty == 'return_period':

        prob_exceedance_ensemble = np.clip(prob_exceedance_ensemble, 1e-14, 1)
        return_period_ensemble = dt/prob_exceedance_ensemble

        if confidence_level_type == 'percentile':
            return_period_mean, return_period_lower, return_period_upper = percentile_data(return_period_ensemble, confidence_level)
        elif confidence_level_type == 'std':
            return_period_mean, return_period_lower, return_period_upper = std_dev_data(return_period_ensemble, confidence_level)

    else:
        raise ValueError("Invalid uncertainty method. Choose 'freq' or 'freq_exceedance'.")

    return return_period_mean, return_period_lower, return_period_upper, bins_centers