import os, sys, argparse, yaml, glob
import numpy as np

import scipy.io as sio
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import acf

from py2d.convert import UV2Omega, Omega2UV
from py2d.derivative import derivative
from py2d.initialize import initialize_wavenumbers_rfft2, gridgen
from py2d.spectra import spectrum_zonal_average, spectrum_angled_average
from py2d.filter import filter2D

from analysis.io_utils import frame_generator, remove_boundaries
from analysis.metrics import divergence, PDF_compute, manual_eof

def eval(config):

    long_analysis_config = config['long_analysis_params']
    system_config = config['system_params']
    root_dir = config['root_dir']

    Ngrid = system_config['Ngrid']
    Lx, Ly = 2*np.pi, 2*np.pi
    Lx, Ly, X, Y, dx, dy = gridgen(Lx, Ly, Ngrid, Ngrid, INDEXING='ij')
    Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_rfft2(Ngrid, Ngrid, Lx, Ly, INDEXING='ij')

    if long_analysis_config['data_type'] == 'emulator':
        total_files_to_analyze = long_analysis_config['file_range'][1] - long_analysis_config['file_range'][0] + 1
    else:
        total_files_to_analyze = (long_analysis_config['file_range'][1] - long_analysis_config['file_range'][0] + 1) // long_analysis_config['step_size']


    # Getting list of files to analyze
    if long_analysis_config['data_type'] == 'emulator':
        # Load config of the training
        # Load config of the dataet
        # Find all .yaml files in the root_dir
        yaml_files = glob.glob(os.path.join(root_dir, "*.yaml"))
        if len(yaml_files) == 0:
            raise FileNotFoundError(f"No .yaml files found in {root_dir}")

        # Load the first .yaml file found
        with open(yaml_files[0], 'r') as f:
            emulator_config = yaml.safe_load(f)

        diffusion_config = emulator_config['diffusion_params']
        dataset_config = emulator_config['dataset_params']
        model_config = emulator_config['model_params']
        train_config = emulator_config['train_params']
        logging_config = emulator_config['logging_params']
        sample_config = emulator_config['sample_params']

        # Prepare list of .npy files for analysis based on conditional/unconditional model
        files = []
        if diffusion_config['conditional']:
            # Conditional: go to subfolder with name diffusion_config['ensemble_num'] under 'data'
            
            ensemble_dir = os.path.join(root_dir, 'data', str(long_analysis_config['ensemble_num']))
            if not os.path.exists(ensemble_dir):
                raise FileNotFoundError(f"Ensemble directory {ensemble_dir} does not exist")
            npy_files = [f for f in os.listdir(ensemble_dir) if f.endswith('.npy')]
            # Ensure files are sorted in ascending order by numeric filename
            npy_files_sorted = sorted(npy_files, key=lambda x: int(os.path.splitext(x)[0]))
            files = [os.path.join(ensemble_dir, f) for f in npy_files_sorted]

            print(f"Looking for files in {ensemble_dir} for conditional model")
            
        else:
            # Unconditional: recursively find all .npy files at the same depth in all subdirectories under 'data'
            data_dir = os.path.join(root_dir, 'data')
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"Data directory {data_dir} does not exist")
            # Walk through all subdirectories and collect .npy files at the leaf level
            for dirpath, dirnames, filenames in os.walk(data_dir):
                npy_files = [os.path.join(dirpath, f) for f in filenames if f.endswith('.npy')]
                if npy_files:
                    files.extend(npy_files)

            print(f"Looking for files in {data_dir} for unconditional model")

    else:
        step_size = long_analysis_config['step_size']

        # Identify the last directory in dataset_config['data_dir'] (e.g., dt0005_IC1)
        last_dir = os.path.basename(os.path.normpath(root_dir))
        # Remove the trailing digits after '_IC' to get the base (e.g., dt0005_IC1 -> dt0005_IC)
        if '_IC' in last_dir:
            base_ic_dir = last_dir[:last_dir.rfind('_IC') + 3]
        else:
            raise ValueError("Expected '_IC' in the last directory name of data_dir")
            # Get the parent directory
        parent_dir = os.path.dirname(os.path.normpath(root_dir))

        # Get list of IC numbers to consider
        IC_nums = long_analysis_config.get('IC_num', [1])
        files = []
        # Append files from all ICs together
        for ic in IC_nums:
            ic_folder = f"{base_ic_dir}{ic}"
            ic_data_dir = os.path.join(parent_dir, ic_folder, 'data')
            print(f"Looking for data in {ic_data_dir}")

            if not os.path.exists(ic_data_dir):
                raise FileNotFoundError(f"IC data directory {ic_data_dir} does not exist")
            
            mat_files = [f for f in os.listdir(ic_data_dir) if f.endswith('.mat')]
            # Sort files by numeric filename
            mat_files_sorted = sorted(mat_files, key=lambda x: int(os.path.splitext(x)[0]))

            files.extend([os.path.join(ic_data_dir, f) for f in mat_files_sorted])

            # Subsample files based on the file_range specified in long_analysis_config
            start, end = long_analysis_config['file_range']

            # Extract numeric part from each filename and filter within the range
            def file_in_range(f):
                num = int(os.path.splitext(os.path.basename(f))[0])
                return start <= num <= end

            files = [f for f in files if file_in_range(f)]

            # Use step_size to only get every step_sizeth file
            files = files[::step_size]

    print(f"Found {len(files)} files to analyze.")

    save_dir = config['save_dir']
    if not save_dir:
        save_dir = root_dir

    if long_analysis_config['data_type'] == 'emulator' and diffusion_config['conditional']:
        save_dir = os.path.join(save_dir, 'analysis', long_analysis_config['data_type'], str(long_analysis_config['ensemble_num']))
    else:
        save_dir = os.path.join(save_dir, 'analysis', long_analysis_config['data_type'])
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")



    num_remove_boundary = long_analysis_config['num_remove_boundary']
    U_mean_single, V_mean_single, Omega_mean_single = np.zeros((Ngrid-2*num_remove_boundary, Ngrid-2*num_remove_boundary)), np.zeros((Ngrid-2*num_remove_boundary, Ngrid-2*num_remove_boundary)), np.zeros((Ngrid-2*num_remove_boundary, Ngrid-2*num_remove_boundary))
    U_zonal_mean_arr, V_zonal_mean_arr, Omega_zonal_mean_arr = [], [], []

    spectra_U_angular_avg_arr, spectra_V_angular_avg_arr, spectra_Omega_angular_avg_arr = [], [], []
    spectra_U_zonal_avg_arr, spectra_V_zonal_avg_arr, spectra_Omega_zonal_avg_arr = [], [], []
    spectra_U_arr, spectra_V_arr, spectra_Omega_arr = [], [], []

    U_arr, V_arr, Omega_arr = [], [], []
    div_arr, energy_arr, enstrophy_arr = [], [], []

    U_min_arr, U_max_arr, V_min_arr, V_max_arr, Omega_min_arr, Omega_max_arr  = [], [], [], [], [], []
    U_min_ind_arr, U_max_ind_arr, V_min_ind_arr, V_max_ind_arr, Omega_min_ind_arr, Omega_max_ind_arr = [], [], [], [], [], []
    U_min_2Delta_arr, U_max_2Delta_arr, V_min_2Delta_arr, V_max_2Delta_arr, Omega_min_2Delta_arr, Omega_max_2Delta_arr = [], [], [], [], [], []
    U_min_2Delta_ind_arr, U_max_2Delta_ind_arr, V_min_2Delta_ind_arr, V_max_2Delta_ind_arr, Omega_min_2Delta_ind_arr, Omega_max_2Delta_ind_arr = [], [], [], [], [], []
    U_min_4Delta_arr, U_max_4Delta_arr, V_min_4Delta_arr, V_max_4Delta_arr, Omega_min_4Delta_arr, Omega_max_4Delta_arr = [], [], [], [], [], []
    U_min_4Delta_ind_arr, U_max_4Delta_ind_arr, V_min_4Delta_ind_arr, V_max_4Delta_ind_arr, Omega_min_4Delta_ind_arr, Omega_max_4Delta_ind_arr = [], [], [], [], [], []

    U_max_anom_arr, U_min_anom_arr, V_max_anom_arr, V_min_anom_arr, Omega_max_anom_arr, Omega_min_anom_arr = [], [], [], [], [], []
    U_max_anom_ind_arr, U_min_anom_ind_arr, V_max_anom_ind_arr, V_min_anom_ind_arr, Omega_max_anom_ind_arr, Omega_min_anom_ind_arr = [], [], [], [], [], []
    U_max_2Delta_anom_arr, U_min_2Delta_anom_arr, V_max_2Delta_anom_arr, V_min_2Delta_anom_arr, Omega_max_2Delta_anom_arr, Omega_min_2Delta_anom_arr = [], [], [], [], [], []
    U_max_2Delta_anom_ind_arr, U_min_2Delta_anom_ind_arr, V_max_2Delta_anom_ind_arr, V_min_2Delta_anom_ind_arr, Omega_max_2Delta_anom_ind_arr, Omega_min_2Delta_anom_ind_arr = [], [], [], [], [], []
    U_max_4Delta_anom_arr, U_min_4Delta_anom_arr, V_max_4Delta_anom_arr, V_min_4Delta_anom_arr, Omega_max_4Delta_anom_arr, Omega_min_4Delta_anom_arr = [], [], [], [], [], []
    U_max_4Delta_anom_ind_arr, U_min_4Delta_anom_ind_arr, V_max_4Delta_anom_ind_arr, V_min_4Delta_anom_ind_arr, Omega_max_4Delta_anom_ind_arr, Omega_min_4Delta_anom_ind_arr = [], [], [], [], [], []

    if long_analysis_config["extreme_anomaly"]:
        try:
            data = data = np.load(os.path.join(save_dir, 'temporal_mean.npz'))
            U_sample_mean_climatology = data['U_sample_mean']
            V_sample_mean_climatology = data['V_sample_mean']
            Omega_sample_mean_climatology = data['Omega_sample_mean']
        except FileNotFoundError:
            print("No climatology data found, can't compute extreme anomaly.")

    frame_gen = frame_generator(files,  long_analysis_config['data_type'], Kx, Ky, invKsq)

    # initialize all your accumulators here…
    total_files_analyzed = 0
    for U, V, Omega in frame_gen:

        if num_remove_boundary > 0:
            U = remove_boundaries(U, num_remove_boundary)
            V = remove_boundaries(V, num_remove_boundary)
            Omega = remove_boundaries(Omega, num_remove_boundary)   

        if total_files_analyzed >  (total_files_to_analyze):
            break
        total_files_analyzed = total_files_analyzed+1

        if total_files_analyzed % 100 == 0:
            print(f"File {total_files_analyzed}/{total_files_to_analyze}")

        if long_analysis_config["temporal_mean"]:
            U_mean_single += U
            V_mean_single += V
            Omega_mean_single += Omega

        if long_analysis_config["zonal_mean"] or long_analysis_config["zonal_eof_pc"] or \
              long_analysis_config["zonal_U"] or long_analysis_config["zonal_V"] or long_analysis_config["zonal_Omega"]:
            U_zonal_mean_arr.append(np.mean(U, axis=1))
            V_zonal_mean_arr.append(np.mean(V, axis=1))
            Omega_zonal_mean_arr.append(np.mean(Omega, axis=1))

        if long_analysis_config["spectra"]:

            ## Angular Averaged Spectra
            U_abs_hat = np.sqrt(np.fft.fft2(U)*np.conj(np.fft.fft2(U)))
            V_abs_hat = np.sqrt(np.fft.fft2(V)*np.conj(np.fft.fft2(V)))
            Omega_abs_hat = np.sqrt(np.fft.fft2(Omega)*np.conj(np.fft.fft2(Omega)))

            spectra_U_temp, wavenumber_angular_avg = spectrum_angled_average(U_abs_hat, spectral=True)
            spectra_V_temp, wavenumber_angular_avg = spectrum_angled_average(V_abs_hat, spectral=True)
            spectra_Omega_temp, wavenumber_angular_avg = spectrum_angled_average(Omega_abs_hat, spectral=True)

            spectra_U_angular_avg_arr.append(spectra_U_temp)
            spectra_V_angular_avg_arr.append(spectra_V_temp)
            spectra_Omega_angular_avg_arr.append(spectra_Omega_temp)

            ## Zonal Spectra
            spectra_U_temp, wavenumber_zonal_avg = spectrum_zonal_average(U.T)
            spectra_V_temp, wavenumber_zonal_avg = spectrum_zonal_average(V.T)
            spectra_Omega_temp, wavenumber_zonal_avg = spectrum_zonal_average(Omega.T)

            spectra_U_zonal_avg_arr.append(spectra_U_temp)
            spectra_V_zonal_avg_arr.append(spectra_V_temp)
            spectra_Omega_zonal_avg_arr.append(spectra_Omega_temp)


            spectra_U_temp, wavenumber = spectrum_zonal_average(U)
            spectra_V_temp, wavenumber = spectrum_zonal_average(V)
            spectra_Omega_temp, wavenumber = spectrum_zonal_average(Omega)

            spectra_U_arr.append(spectra_U_temp)
            spectra_V_arr.append(spectra_V_temp)
            spectra_Omega_arr.append(spectra_Omega_temp)

        if long_analysis_config["div"]:
            div_temp = divergence(U, V)
            div_arr.append(np.mean(np.abs(div_temp)))

        if long_analysis_config["energy"]:
            energy_temp = np.mean(0.5 * (U**2 + V**2))
            energy_arr.append(energy_temp)

        if long_analysis_config["enstrophy"]:
            enstrophy_temp = 0.5 * np.mean(Omega**2)
            enstrophy_arr.append(enstrophy_temp)

        if long_analysis_config["extreme"]:

            U_max_arr.append(U.max()), U_max_ind_arr.append(U.argmax())
            U_min_arr.append(U.min()), U_min_ind_arr.append(U.argmin())
            V_max_arr.append(V.max()), V_max_ind_arr.append(V.argmax())
            V_min_arr.append(V.min()), V_min_ind_arr.append(V.argmin())
            Omega_max_arr.append(Omega.max()), Omega_max_ind_arr.append(Omega.argmax())
            Omega_min_arr.append(Omega.min()), Omega_min_ind_arr.append(Omega.argmin())

            if long_analysis_config["extreme_block"]:

                Delta2 = 2 * Lx / Ngrid
                Delta4 = 4 * Lx / Ngrid

                # Applying box filter - Find the max value in average of 3x3 grid points
                U_box_2Delta = filter2D(U, filterType='box', coarseGrainType=None, Delta=Delta2)
                V_box_2Delta = filter2D(V, filterType='box', coarseGrainType=None, Delta=Delta2)
                Omega_box_2Delta = filter2D(Omega, filterType='box', coarseGrainType=None, Delta=Delta2)

                U_box_4Delta = filter2D(U, filterType='box', coarseGrainType=None, Delta=Delta4)
                V_box_4Delta = filter2D(V, filterType='box', coarseGrainType=None, Delta=Delta4)
                Omega_box_4Delta = filter2D(Omega, filterType='box', coarseGrainType=None, Delta=Delta4)

                U_max_2Delta_arr.append(U_box_2Delta.max()), U_max_2Delta_ind_arr.append(U_box_2Delta.argmax())
                U_min_2Delta_arr.append(U_box_2Delta.min()), U_min_2Delta_ind_arr.append(U_box_2Delta.argmin())
                V_max_2Delta_arr.append(V_box_2Delta.max()), V_max_2Delta_ind_arr.append(V_box_2Delta.argmax())
                V_min_2Delta_arr.append(V_box_2Delta.min()), V_min_2Delta_ind_arr.append(V_box_2Delta.argmin())
                Omega_max_2Delta_arr.append(Omega_box_2Delta.max()), Omega_max_2Delta_ind_arr.append(Omega_box_2Delta.argmax())
                Omega_min_2Delta_arr.append(Omega_box_2Delta.min()), Omega_min_2Delta_ind_arr.append(Omega_box_2Delta.argmin())

                U_max_4Delta_arr.append(U_box_4Delta.max()), U_max_4Delta_ind_arr.append(U_box_4Delta.argmax())
                U_min_4Delta_arr.append(U_box_4Delta.min()), U_min_4Delta_ind_arr.append(U_box_4Delta.argmin())
                V_max_4Delta_arr.append(V_box_4Delta.max()), V_max_4Delta_ind_arr.append(V_box_4Delta.argmax())
                V_min_4Delta_arr.append(V_box_4Delta.min()), V_min_4Delta_ind_arr.append(V_box_4Delta.argmin())
                Omega_max_4Delta_arr.append(Omega_box_4Delta.max()), Omega_max_4Delta_ind_arr.append(Omega_box_4Delta.argmax())
                Omega_min_4Delta_arr.append(Omega_box_4Delta.min()), Omega_min_4Delta_ind_arr.append(Omega_box_4Delta.argmin())


        if long_analysis_config["extreme_anomaly"]:

            U_anom = U - U_sample_mean_climatology
            V_anom = V - V_sample_mean_climatology
            Omega_anom = Omega - Omega_sample_mean_climatology

            U_max_anom_arr.append(U_anom.max()), U_max_anom_ind_arr.append(U_anom.argmax())
            U_min_anom_arr.append(U_anom.min()), U_min_anom_ind_arr.append(U_anom.argmin())
            V_max_anom_arr.append(V_anom.max()), V_max_anom_ind_arr.append(V_anom.argmax())
            V_min_anom_arr.append(V_anom.min()), V_min_anom_ind_arr.append(V_anom.argmin())
            Omega_max_anom_arr.append(Omega_anom.max()), Omega_max_anom_ind_arr.append(Omega_anom.argmax())
            Omega_min_anom_arr.append(Omega_anom.min()), Omega_min_anom_ind_arr.append(Omega_anom.argmin())

            if long_analysis_config["extreme_block"]:

                Delta2 = 2 * Lx / Ngrid
                Delta4 = 4 * Lx / Ngrid

                # Applying box filter - Find the max value in average of 3x3 grid points
                U_anom_box_2Delta = filter2D(U_anom, filterType='box', coarseGrainType=None, Delta=Delta2)
                V_anom_box_2Delta = filter2D(V_anom, filterType='box', coarseGrainType=None, Delta=Delta2)
                Omega_anom_box_2Delta = filter2D(Omega_anom, filterType='box', coarseGrainType=None, Delta=Delta2)

                U_anom_box_4Delta = filter2D(U_anom, filterType='box', coarseGrainType=None, Delta=Delta4)
                V_anom_box_4Delta = filter2D(V_anom, filterType='box', coarseGrainType=None, Delta=Delta4)
                Omega_anom_box_4Delta = filter2D(Omega_anom, filterType='box', coarseGrainType=None, Delta=Delta4)

                U_max_2Delta_anom_arr.append(U_anom_box_2Delta.max()), U_max_2Delta_anom_ind_arr.append(U_anom_box_2Delta.argmax())
                U_min_2Delta_anom_arr.append(U_anom_box_2Delta.min()), U_min_2Delta_anom_ind_arr.append(U_anom_box_2Delta.argmin())
                V_max_2Delta_anom_arr.append(V_anom_box_2Delta.max()), V_max_2Delta_anom_ind_arr.append(V_anom_box_2Delta.argmax())
                V_min_2Delta_anom_arr.append(V_anom_box_2Delta.min()), V_min_2Delta_anom_ind_arr.append(V_anom_box_2Delta.argmin())
                Omega_max_2Delta_anom_arr.append(Omega_anom_box_2Delta.max()), Omega_max_2Delta_anom_ind_arr.append(Omega_anom_box_2Delta.argmax())
                Omega_min_2Delta_anom_arr.append(Omega_anom_box_2Delta.min()), Omega_min_2Delta_anom_ind_arr.append(Omega_anom_box_2Delta.argmin())

                U_max_4Delta_anom_arr.append(U_anom_box_4Delta.max()), U_max_4Delta_anom_ind_arr.append(U_anom_box_4Delta.argmax())
                U_min_4Delta_anom_arr.append(U_anom_box_4Delta.min()), U_min_4Delta_anom_ind_arr.append(U_anom_box_4Delta.argmin())
                V_max_4Delta_anom_arr.append(V_anom_box_4Delta.max()), V_max_4Delta_anom_ind_arr.append(V_anom_box_4Delta.argmax())
                V_min_4Delta_anom_arr.append(V_anom_box_4Delta.min()), V_min_4Delta_anom_ind_arr.append(V_anom_box_4Delta.argmin())
                Omega_max_4Delta_anom_arr.append(Omega_anom_box_4Delta.max()), Omega_max_4Delta_anom_ind_arr.append(Omega_anom_box_4Delta.argmax())
                Omega_min_4Delta_anom_arr.append(Omega_anom_box_4Delta.min()), Omega_min_4Delta_anom_ind_arr.append(Omega_anom_box_4Delta.argmin())

        # Subsampling data for PDF analysis
        PDF_num_samples = min(Ngrid**2, int(long_analysis_config["PDF_data_ratio"] * (Ngrid ** 2)))
        PDF_indices = np.random.choice(Ngrid ** 2, PDF_num_samples, replace=False)

        if long_analysis_config["PDF_U"]:
            U_arr.append(U.flatten()[PDF_indices])

        if long_analysis_config["PDF_V"]:
            V_arr.append(V.flatten()[PDF_indices])

        if long_analysis_config["PDF_Omega"]:
            Omega_arr.append(Omega.flatten()[PDF_indices])

    print('Total snapshots analyzed:', total_files_analyzed)

    if long_analysis_config["temporal_mean"]:
        U_mean_single /= total_files_analyzed
        V_mean_single /= total_files_analyzed
        Omega_mean_single /= total_files_analyzed

        np.savez(os.path.join(save_dir, 'temporal_mean.npz'),
                U_sample_mean=U_mean_single,
                V_sample_mean=V_mean_single,
                Omega_sample_mean=Omega_mean_single)
        print("Temporal mean saved.")

    if long_analysis_config["spectra"]:

        spectra_U = np.mean(spectra_U_arr, axis=0)
        spectra_V = np.mean(spectra_V_arr, axis=0)
        spectra_Omega = np.mean(spectra_Omega_arr, axis=0)

        spectra_U_angular_avg = np.mean(spectra_U_angular_avg_arr, axis=0)
        spectra_V_angular_avg = np.mean(spectra_V_angular_avg_arr, axis=0)
        spectra_Omega_angular_avg = np.mean(spectra_Omega_angular_avg_arr, axis=0)

        spectra_U_zonal_avg = np.mean(spectra_U_zonal_avg_arr, axis=0)
        spectra_V_zonal_avg = np.mean(spectra_V_zonal_avg_arr, axis=0)
        spectra_Omega_zonal_avg = np.mean(spectra_Omega_zonal_avg_arr, axis=0)

        np.savez(os.path.join(save_dir, 'spectra.npz'),
                spectra_U=spectra_U,
                spectra_V=spectra_V,
                spectra_Omega=spectra_Omega,
                wavenumber = wavenumber,
                spectra_U_angular_avg=spectra_U_angular_avg,
                spectra_V_angular_avg=spectra_V_angular_avg,
                spectra_Omega_angular_avg=spectra_Omega_angular_avg,
                wavenumber_angular_avg=wavenumber_angular_avg,
                spectra_U_zonal_avg=spectra_U_zonal_avg,
                spectra_V_zonal_avg=spectra_V_zonal_avg,
                spectra_Omega_zonal_avg=spectra_Omega_zonal_avg,
                wavenumber_zonal_avg=wavenumber_zonal_avg)
        print("Spectra saved.")
        print(f"Angular averaged spectra U shape: {spectra_U_angular_avg.shape}, V shape: {spectra_V_angular_avg.shape}, Omega shape: {spectra_Omega_angular_avg.shape}")
        print(f"Zonal averaged spectra U shape: {spectra_U_zonal_avg.shape}, V shape: {spectra_V_zonal_avg.shape}, Omega shape: {spectra_Omega_zonal_avg.shape}")

    if long_analysis_config["zonal_mean"] or long_analysis_config["zonal_eof_pc"]:

        U_zonal_mean = np.mean(U_zonal_mean_arr, axis=0)
        V_zonal_mean = np.mean(V_zonal_mean_arr, axis=0)
        Omega_zonal_mean = np.mean(Omega_zonal_mean_arr, axis=0)

        if long_analysis_config["zonal_mean"]:
            np.savez(os.path.join(save_dir, 'zonal_mean.npz'),
                    U_zonal_mean=U_zonal_mean,
                    V_zonal_mean=V_zonal_mean,
                    Omega_zonal_mean=Omega_zonal_mean)
            print("Zonal mean saved.")
            print(f"U zonal mean shape: {U_zonal_mean.shape}, V zonal mean shape: {V_zonal_mean.shape}, Omega zonal mean shape: {Omega_zonal_mean.shape}")


        if long_analysis_config["zonal_eof_pc"]:

            n_lags = long_analysis_config["PC_autocorr_nlags"]

            U_zonal_anom = np.asarray(U_zonal_mean_arr) - U_zonal_mean
            EOF_U, PC_U, exp_var_U = manual_eof(U_zonal_anom, long_analysis_config["eof_ncomp"])

            PC_acf_U= []
            for i in range(long_analysis_config["eof_ncomp"]):
                acf_i, confint_i = acf(PC_U[:, i], nlags=n_lags, alpha=0.5)
                PC_acf_U.append({"acf": acf_i, "confint": confint_i})

            Omega_zonal_anom = np.array(Omega_zonal_mean_arr) - Omega_zonal_mean
            EOF_Omega, PC_Omega, exp_var_Omega = manual_eof(Omega_zonal_anom, long_analysis_config["eof_ncomp"])

            PC_acf_Omega = []
            for i in range(long_analysis_config["eof_ncomp"]):
                acf_i, confint_i = acf(PC_Omega[:, i], nlags=n_lags, alpha=0.5)
                PC_acf_Omega.append({"acf": acf_i, "confint": confint_i})

            np.savez(os.path.join(save_dir, 'zonal_eof_pc.npz'),
                    U_eofs=EOF_U, U_pc=PC_U, U_expvar=exp_var_U, U_pc_acf=PC_acf_U, 
                    Omega_eofs=EOF_Omega, Omega_PC=PC_Omega, Omega_expvar=exp_var_Omega, Omega_pc_acf=PC_acf_Omega)
            
            print("Zonal EOFs and PCs saved.")

    if long_analysis_config["zonal_U"]:
        np.savez(os.path.join(save_dir, 'zonal_U.npz'),
                U_zonal=np.asarray(U_zonal_mean_arr))
        print("Zonal U saved.")

    if long_analysis_config["zonal_V"]:
        np.savez(os.path.join(save_dir, 'zonal_V.npz'),
                V_zonal=np.asarray(V_zonal_mean_arr))
        print("Zonal V saved.")

    if long_analysis_config["zonal_Omega"]:
        np.savez(os.path.join(save_dir, 'zonal_Omega.npz'),
                Omega_zonal=np.asarray(Omega_zonal_mean_arr))
        print("Zonal Omega saved.")
            
    if long_analysis_config["extreme"]:
        np.savez(os.path.join(save_dir, 'extremes.npz'),\
                U_max_arr=U_max_arr, U_min_arr=U_min_arr, V_max_arr=V_max_arr, V_min_arr=V_min_arr, Omega_max_arr=Omega_max_arr, Omega_min_arr=Omega_min_arr, \
                    U_max_ind_arr=U_max_ind_arr, U_min_ind_arr=U_min_ind_arr, V_max_ind_arr=V_max_ind_arr, V_min_ind_arr=V_min_ind_arr, Omega_max_ind_arr=Omega_max_ind_arr, Omega_min_ind_arr=Omega_min_ind_arr)
        print("Extremes saved.")

        if long_analysis_config["extreme_block"]:
            np.savez(os.path.join(save_dir, 'extremes_block.npz'),\
                U_max_2Delta_arr=U_max_2Delta_arr, U_min_2Delta_arr=U_min_2Delta_arr, V_max_2Delta_arr=V_max_2Delta_arr, V_min_2Delta_arr=V_min_2Delta_arr, Omega_max_2Delta_arr=Omega_max_2Delta_arr, Omega_min_2Delta_arr=Omega_min_2Delta_arr,\
                U_max_4Delta_arr=U_max_4Delta_arr, U_min_4Delta_arr=U_min_4Delta_arr, V_max_4Delta_arr=V_max_4Delta_arr, V_min_4Delta_arr=V_min_4Delta_arr, Omega_max_4Delta_arr=Omega_max_4Delta_arr, Omega_min_4Delta_arr=Omega_min_4Delta_arr,\
                U_max_2Delta_ind_arr=U_max_2Delta_ind_arr, U_min_2Delta_ind_arr=U_min_2Delta_ind_arr, V_max_2Delta_ind_arr=V_max_2Delta_ind_arr, V_min_2Delta_ind_arr=V_min_2Delta_ind_arr, Omega_max_2Delta_ind_arr=Omega_max_2Delta_ind_arr, Omega_min_2Delta_ind_arr=Omega_min_2Delta_ind_arr,\
                U_max_4Delta_ind_arr=U_max_4Delta_ind_arr, U_min_4Delta_ind_arr=U_min_4Delta_ind_arr, V_max_4Delta_ind_arr=V_max_4Delta_ind_arr, V_min_4Delta_ind_arr=V_min_4Delta_ind_arr, Omega_max_4Delta_ind_arr=Omega_max_4Delta_ind_arr, Omega_min_4Delta_ind_arr=Omega_min_4Delta_ind_arr)

    if long_analysis_config["extreme_anomaly"]:
        np.savez(os.path.join(save_dir, 'extremes_anom.npz'),\
                U_max_arr=U_max_anom_arr, U_min_arr=U_min_anom_arr, V_max_arr=V_max_anom_arr, V_min_arr=V_min_anom_arr, Omega_max_arr=Omega_max_anom_arr, Omega_min_arr=Omega_min_anom_arr, \
                    U_max_ind_arr=U_max_anom_ind_arr, U_min_ind_arr=U_min_anom_ind_arr, V_max_ind_arr=V_max_anom_ind_arr, V_min_ind_arr=V_min_anom_ind_arr, Omega_max_ind_arr=Omega_max_anom_ind_arr, Omega_min_ind_arr=Omega_min_anom_ind_arr)
        print("Extreme anomalies saved.")

        if long_analysis_config["extreme_block"]:
            np.savez(os.path.join(save_dir, 'extremes_anom_block.npz'),\
            U_max_2Delta_arr=U_max_2Delta_anom_arr, U_min_2Delta_arr=U_min_2Delta_anom_arr, V_max_2Delta_arr=V_max_2Delta_anom_arr, V_min_2Delta_arr=V_min_2Delta_anom_arr, Omega_max_2Delta_arr=Omega_max_2Delta_anom_arr, Omega_min_2Delta_arr=Omega_min_2Delta_anom_arr,\
            U_max_4Delta_arr=U_max_4Delta_anom_arr, U_min_4Delta_arr=U_min_4Delta_anom_arr, V_max_4Delta_arr=V_max_4Delta_anom_arr, V_min_4Delta_arr=V_min_4Delta_anom_arr, Omega_max_4Delta_arr=Omega_max_4Delta_anom_arr, Omega_min_4Delta_arr=Omega_min_4Delta_anom_arr,\
            U_max_2Delta_ind_arr=U_max_2Delta_anom_ind_arr, U_min_2Delta_ind_arr=U_min_2Delta_anom_ind_arr, V_max_2Delta_ind_arr=V_max_2Delta_anom_ind_arr, V_min_2Delta_ind_arr=V_min_2Delta_anom_ind_arr, Omega_max_2Delta_ind_arr=Omega_max_2Delta_anom_ind_arr, Omega_min_2Delta_ind_arr=Omega_min_2Delta_anom_ind_arr,\
            U_max_4Delta_ind_arr=U_max_4Delta_anom_ind_arr, U_min_4Delta_ind_arr=U_min_4Delta_anom_ind_arr, V_max_4Delta_ind_arr=V_max_4Delta_anom_ind_arr, V_min_4Delta_ind_arr=V_min_4Delta_anom_ind_arr, Omega_max_4Delta_ind_arr=Omega_max_4Delta_anom_ind_arr, Omega_min_4Delta_ind_arr=Omega_min_4Delta_anom_ind_arr)

    if long_analysis_config["div"]:
        np.savez(os.path.join(save_dir, 'div.npz'),
                    div=np.asarray(div_arr))
        print("Divergence saved.")

    if long_analysis_config["energy"]:
        np.savez(os.path.join(save_dir, 'energy.npz'),
                    energy=np.asarray(energy_arr))
        print("Energy saved.")
    
    if long_analysis_config["enstrophy"]:
        np.savez(os.path.join(save_dir, 'enstrophy.npz'),
                    enstrophy=np.asarray(enstrophy_arr))
        print("Enstrophy saved.")

    if long_analysis_config["PDF_U"]:
        U_mean, U_std, U_pdf, U_bins, bw_scott = PDF_compute(np.asarray(U_arr))
        np.savez(os.path.join(save_dir, 'PDF_U.npz'),
                U_mean=U_mean, U_std=U_std, U_pdf=U_pdf, U_bins=U_bins, bw_scott=bw_scott)
        print("PDF U saved.")

    if long_analysis_config["PDF_V"]:
        V_mean, V_std, V_pdf, V_bins, bw_scott = PDF_compute(np.asarray(V_arr))
        np.savez(os.path.join(save_dir, 'PDF_V.npz'),
                V_mean=V_mean, V_std=V_std, V_pdf=V_pdf, V_bins=V_bins, bw_scott=bw_scott)
        print("PDF V saved.")
        
    if long_analysis_config["PDF_Omega"]:
        Omega_mean, Omega_std, Omega_pdf, Omega_bins, bw_scott = PDF_compute(np.asarray(Omega_arr))
        np.savez(os.path.join(save_dir, 'PDF_Omega.npz'),
                Omega_mean=Omega_mean, Omega_std=Omega_std, Omega_pdf=Omega_pdf, Omega_bins=Omega_bins, bw_scott=bw_scott)
        print("PDF Omega saved.")



parser = argparse.ArgumentParser(description='Arguments for analysis')
parser.add_argument('--config', dest='config_path',
                    default='config/config.yaml', type=str,
                    help='Path to the configuration file')
args = parser.parse_args()

with open(args.config_path, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print("Error in configuration file:", exc)
        sys.exit(1)

if config['long_analysis_params']['extreme_anomaly'] == True:
    # compute without extreme_anomaly first
    config['long_analysis_params']['extreme_anomaly'] = False
    print("Running analysis without extreme anomaly first...")
    eval(config)

    # then compute with extreme_anomaly
    config['long_analysis_params']['extreme_anomaly'] = True
    config['long_analysis_params']['temporal_mean'] = False
    config['long_analysis_params']['zonal_mean'] = False
    config['long_analysis_params']['zonal_eof_pc'] = False
    config['long_analysis_params']['spectra'] = False
    config['long_analysis_params']['div'] = False
    config['long_analysis_params']['energy'] = False
    config['long_analysis_params']['enstrophy'] = False
    config['long_analysis_params']['extreme'] = False
    config['long_analysis_params']['zonal_U'] = False
    config['long_analysis_params']['zonal_V'] = False
    config['long_analysis_params']['zonal_Omega'] = False
    config['long_analysis_params']['extreme_block'] = False
    config['long_analysis_params']['PDF_U'] = False
    config['long_analysis_params']['PDF_V'] = False
    config['long_analysis_params']['PDF_Omega'] = False
    print("Running analysis with extreme anomaly...")
    eval(config)
else:
    eval(config)
