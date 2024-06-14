import numpy as np
import matplotlib.pyplot as plt
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta, tau0_from_mass1_mass2, tau3_from_mass1_mass2, mass1_from_mchirp_q, mass2_from_mchirp_q, mass1_from_tau0_tau3, mass2_from_tau0_tau3, q_from_mass1_mass2, eta_from_q
import random
from pycbc.filter import sigmasq
from scipy.linalg import svd
from scipy.interpolate import CubicSpline, CubicHermiteSpline, interp1d, BarycentricInterpolator, krogh_interpolate, PchipInterpolator, Akima1DInterpolator
from scipy import special
from rbf.interpolate import RBFInterpolant
#from rbf.interpolate import _objective
from rbf.poly import mvmonos
from pycbc.detector import Detector
import h5py
import pickle
import matplotlib
from pycbc.types.timeseries import TimeSeries
from pycbc.types.array import complex_same_precision_as
from pycbc.conversions import mchirp_from_mass1_mass2
from pycbc.pnutils import get_final_freq
from pycbc.filter.matchedfilter import get_cutoff_indices
from pycbc.fft import ifft
from pycbc.types import zeros
from pycbc.types.frequencyseries import load_frequencyseries
from pycbc.types.timeseries import load_timeseries
from tqdm import tqdm
from multiprocessing import Process
from pycbc.inference.models.marginalized_gaussian_noise import MarginalizedPhaseGaussianNoise

from gwosc.datasets import event_gps

import statistics
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from pycbc import distributions
from scipy.stats import truncnorm
import copy
from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD_NRTidalv2, IMRPhenomD
from gwfast.signal import GWSignal
from gwfast.network import DetNet
from fisherTools import CovMatr, compute_localization_region, check_covariance, fixParams
import gwfast.gwfastGlobals as glob
from scipy.stats import gaussian_kde
from gen_interpolants import *
from eval_likelihood import *
from data_cleaning import *
import time
import sys
import os
import psutil

# The program will use the first 32 cores 

# p = psutil.Process()
# p.cpu_affinity()
# p.cpu_affinity([i for i in range(32)])


# ------- Filtering the Data ---------------
fmax = 1600
fmin = 20.
filter_order = 8
tc = 1187008882.4
ifos = ['L1', 'H1', 'V1']
ifos_gwfast = ['L1', 'H1', 'Virgo']

analysis_start_time = 340 # Relative to trigger time
analysis_end_time = 28 # Relative to trigger time

if not os.path.exists('filter_data/filtered_L1_8_post_new_data.hdf'):
    raw_data_file = {ifos[i]: f"{ifos[i]}-{ifos[i]}_LOSC_CLN_4_V1-1187007040-2048.gwf" for i in range(ifos)}
    channel_name = {ifos[i]: f'{ifos[i]}:LOSC-STRAIN' for i in range(ifos)}

    data = reading_data(tc, analysis_start_time, analysis_end_time, file_name, channel_name, ifos)
    
    filtered_data = filter_data(data, fmin - 5, fHigh, filter_order)
    psd = psd_from_data(filtered_data, seg_len, fmin - 5)

# ---------defining general parameters ----------
verbose = False
phi = 'ga'
rbf_order = 7
nbasis = 50
eps = 10
approximant = 'IMRPhenomD'
seed = int(sys.argv[1])
data = {'L1': load_timeseries('filter_data/filtered_L1_8_post_new_data.hdf'),\
         'V1': load_timeseries('filter_data/filtered_V1_8_post_new_data.hdf'),\
        'H1': load_timeseries('filter_data/filtered_H1_8_post_new_data.hdf')}

psd = {'L1': load_frequencyseries('PSD/PSD_L1.txt'),\
       'V1': load_frequencyseries('PSD/PSD_V1.txt'),\
       'H1': load_frequencyseries('PSD/PSD_H1.txt')}

Nnodes = 800
nodes_gauss_num = 0.1
size = 10000000

boundary = {'mchirp': np.array([1.19735, 1.19775]), 'mass_ratio': np.array([1.31, 1.45]), 's1z': np.array([0.0035, 0.0085]), 's2z': np.array([0.0035, 0.0085]), 'distance': [10, 60], 'ra': [0, 2*np.pi], 'dec':[-np.pi/2, np.pi/2],'inc':[0, np.pi], 'tc':[tc - 0.12, tc + 0.12]}


# Fiducial point to generate covariance matrix
GW170817 = {'m1': np.array([1.628476219780637]), 'm2': np.array([1.168209671926553]), \
'chi1z': np.array([0.006099936513203831]), 'chi2z': np.array([0.007499201230484316]),\
            'tGPS': np.array([1187008882.4]),\
            'iota':np.array([2.54]),\
            'ra': np.array([3.42]),\
            'dec': np.array([-0.361]),\
            'dL': np.array([39.4/1000]),\
            'psi': np.array([0.]),\
            'Phicoal':np.array([0.])}

psd_file = {'L1': 'PSD/PSD_L1.txt', 'H1': 'PSD/PSD_H1.txt', 'Virgo': 'PSD/PSD_V1.txt'}

center = [1.197436777863998, 1.3939930980840414, 0.006099936513203831, 0.007499201230484316]


# --------- Generate the Covariance matrix ---------
if verbose:
    print("----- Generating Covariance Matrix ------")

start_up_st = time.time()
cov_mat = generate_cov_matrix(ifos_gwfast, GW170817, psd_file, fmax, data['L1'].duration)[0]

if verbose:
    print('Covariance Matrix:\n', cov_mat)




# --------- Spraying Nodes -----------

if verbose:
    print('-------- Generating Nodes ---------')
    

nodesList = generate_nodes(boundary, center, cov_mat, Nnodes, nodes_gauss_num, seed, size)

    

# ------- Generate interpolants --------
if verbose:
    print('------- Generating Interpolants ----------')

hp_hp_interpolant, C_interpolant, basis_vectors, times = RBF_parallel(ifos, data, approximant, nodesList, fmin, fmax, tc, psd, nbasis, phi, rbf_order, eps, 32)

start_up_et = time.time()
print('Start-up stage takes:', (start_up_et - start_up_st), 'seconds')

with open(f'nodes_IMR/fixed_sampling_seed_0/nodes_{seed}.pickle', 'wb') as f:
    pickle.dump(nodesList, f, pickle.HIGHEST_PROTOCOL)
    f.close()


start_up_timing = open('start_up_stage_time_IMR/start_up_timing_samp_seed_0.txt', 'a')
start_up_timing.write(str(seed))
start_up_timing.write(': ')
start_up_timing.write(str((start_up_et - start_up_st)/60.))
start_up_timing.write('\n')
start_up_timing.close()


# ---------- Sampling ------------

det = {}

for ifo in ifos:
    
    det[ifo] = Detector(ifo)




def prior_transform(cube):
        
    cube[0] = ((boundary['mchirp'][1]**2 - boundary['mchirp'][0]**2)*cube[0] + boundary['mchirp'][0]**2)**(1./2)      # chirpmass: uniform prior
    cube[1] = cdfinv_q(boundary['mass_ratio'][0], boundary['mass_ratio'][1], cube[1])              # q: uniform prior
    cube[2] = boundary['s1z'][0] + (boundary['s1z'][1] - boundary['s1z'][0]) * cube[2]                # s1z: uniform prior
    cube[3] = boundary['s2z'][0] + (boundary['s2z'][1] - boundary['s2z'][0]) * cube[3]                # s2z: uniform prior
    cube[4] = (boundary['distance'][0]**3 + (boundary['distance'][1]**3 - boundary['distance'][0]**3) * cube[4])**(1./3)                   # distance: uniform prior
    cube[5] = boundary['tc'][0] + (boundary['tc'][1] - boundary['tc'][0]) * cube[5] # tc: uniform prior
    cube[6] = 2*np.pi*cube[6] 
    
    cube[7] = np.arcsin(2*cube[7] - 1)   
    
    cube[8] = np.arccos(2*cube[8] - 1)
    
    cube[9] = 2*np.pi*cube[9]

    return cube


nWalks = 100
print('********** Sampling starts *********\n')

def RBF_LogL(q):
    return net_llr(q, data['L1'].sample_rate, hp_hp_interpolant = hp_hp_interpolant, C_interpolant = C_interpolant, basis_vectors = basis_vectors, det = det, times = times, f_low = fmin)


nlive_pts = 500
#-- definig dynesty sampler ---
with mp.Pool(32) as pool:
    
    sampler = dynesty.NestedSampler(RBF_LogL,
                                               prior_transform,
                                               10,
                                               sample='rwalk',
                                               pool=pool, nlive = nlive_pts, queue_size = 32, walks = nWalks, rstate = Generator(PCG64(seed = 0)))
    
    
    st = time.time()
    sampler.run_nested(dlogz=0.1)
    et = time.time()
    print('Sampling takes:', (et - st)/60., 'minutes')
    
    
#-- saving pe and timing samples ---
res = sampler.results
print('Evidence:{}'.format(res['logz'][-1]))


timing = open('sampling_time_IMR/sampling_timing_sampling_seed_0.txt', 'a')
timing.write(str(seed))
timing.write(': ')
timing.write(str((et - st)/60.))
timing.write('\n')
timing.close()


file = h5py.File(f'sampling_results_IMR/fixed_sampling_seed_0/samples_data_interpolated_{nlive_pts}_GW170817_800_nwalks_{nWalks}_nodeSeed_{seed}_gauss_frac_{nodes_gauss_num}.hdf5', 'w')
file.create_dataset('mchirp', data=res['samples'][:,0])
file.create_dataset('q', data=res['samples'][:,1])
file.create_dataset('s1z', data=res['samples'][:,2])
file.create_dataset('s2z', data=res['samples'][:,3])
file.create_dataset('distance', data=res['samples'][:,4])
file.create_dataset('tc', data=res['samples'][:,5])
file.create_dataset('ra', data = res['samples'][:, 6])
file.create_dataset('dec', data = res['samples'][:, 7])
file.create_dataset('inc', data = res['samples'][:, 8])
file.create_dataset('pol', data = res['samples'][:, 9])
file.create_dataset('logwt', data=res['logwt'])
file.create_dataset('logz', data=res['logz'])
file.create_dataset('logl', data=res['logl'])
file.create_dataset('logvol', data=res['logvol'])
file.create_dataset('logzerr', data=res['logzerr'])
file.create_dataset('Evidence', data=res['logz'][-1])
file.create_dataset('niter', data=res['niter'])
file.close()