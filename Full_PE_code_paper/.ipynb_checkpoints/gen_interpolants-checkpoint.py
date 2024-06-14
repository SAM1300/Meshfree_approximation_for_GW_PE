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
from pycbc.waveform import get_fd_waveform
from pycbc.filter.matchedfilter import get_cutoff_indices
from pycbc.fft import ifft
from pycbc.types import zeros
from pycbc.types.frequencyseries import load_frequencyseries
from pycbc.types.timeseries import load_timeseries
from tqdm import tqdm
from multiprocessing import Process
from multiprocessing.pool import Pool
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
from fisherTools import dMc_dm1, dMc_dm2
from scipy.optimize import minimize

def mode(params_array, bins = 25):
    # Function to calculate mode of a distribution
    kernel = gaussian_kde(params_array)
    count, val = np.histogram(params_array, bins = bins)
    val_pdf = kernel.pdf(val)
    map_val = val[np.argmax(val_pdf)]
    return map_val


def dq_dm1(m2):
    return (1/m2)

def dq_dm2(m1, m2):
    return -(m1/m2**2)



def Jacobian_mat(m1, m2):
    # Transforms m1, m2, chi1z, chi2z to Mc, q, chi1z, chi2z
    dMc_dm1_jacob = dMc_dm1(m1, m2)
    dMc_dm2_jacob = dMc_dm2(m1, m2)
    # dMc_dchi1_jacob = ((m1**3)/(-2*s1))*dMc_dm1(m1, m2)
    # dMc_dchi2_jacob = ((m2**3)/(-2*s2))*dMc_dm2(m1, m2)
    dMc_dchi1_jacob = 0
    dMc_dchi2_jacob = 0
    
    dq_dm1_jacob = dq_dm1(m2)
    dq_dm2_jacob = dq_dm2(m1, m2)
    # dq_dchi1_jacob = ((m1**3)/(-2*s1))*dq_dm1(m2)
    # dq_dchi2_jacob = ((m2**3)/(-2*s2))*dq_dm2(m1, m2)
    dq_dchi1_jacob = 0
    dq_dchi2_jacob = 0
    
    dchi1_dm1_jacob = 0
    dchi1_dm2_jacob = 0
    dchi1_dchi1_jacob = 1
    dchi1_dchi2_jacob = 0
    
    dchi2_dm1_jacob = 0
    dchi2_dm2_jacob = 0
    dchi2_dchi1_jacob = 0
    dchi2_dchi2_jacob = 1
    
    jacob = np.array([[dMc_dm1_jacob, dMc_dm2_jacob, dMc_dchi1_jacob, dMc_dchi2_jacob], [dq_dm1_jacob, dq_dm2_jacob, dq_dchi1_jacob, dq_dchi2_jacob], [dchi1_dm1_jacob, dchi1_dm2_jacob, dchi1_dchi1_jacob, dchi1_dchi2_jacob], [dchi2_dm1_jacob, dchi2_dm2_jacob, dchi2_dchi1_jacob, dchi2_dchi2_jacob]])
    
    return jacob

def net_snr(params, data, psd):
    ifo = list(data.keys())
    m1, m2, s1, s2 = params
    hp, hc = get_fd_waveform(approximant = 'TaylorF2', mass1 = m1, mass2 = m2, spin1z = s1, spin2z = s2, delta_f = 1/data['L1'].duration, f_lower = 20, f_final = 1600)
    hp.resize(len(data['L1'])//2+1)
    net_snr = 0
    for i in range(len(ifo)):
        
        snr = matched_filter(hp, data[ifo[i]].to_frequencyseries(), psd[ifo[i]], low_frequency_cutoff = 20, high_frequency_cutoff= 1600)
        
        net_snr += np.max(np.array(abs(snr)**2))
        
    net_snr = np.sqrt(net_snr)
    return net_snr



def optimization(params_init, function, bounds):
    min_obj = minimize(function, params_init, bounds = bounds, method = 'SLSQP')
    return min_obj.x


def generate_cov_matrix(ifo, event, psd_file_path, fmax, data_duration, fmin = 20., usem1m2 = True):
    
    
    # This function generates covariance matrix for intrinsic parameters
    # Signal is generated in TaylorF2RestrictedPN waveform
    # ifo ---> List of ifos
    # event ----> dictionary conatining the fiduacial values of parameters
    # psd_file_path -----> Dictionary containing path to psd files  
    
    
    # Constructing detector object
    detectors_obj = copy.deepcopy(glob.detectors)
    LVdetectors = {det:detectors_obj[det] for det in ifo}
    
    # Adding path of a file containing PSDs
    for i in range(len(ifo)):
        LVdetectors[ifo[i]]['psd_path'] = psd_file_path[ifo[i]]
    
    myLVSignals = {}

    for d in LVdetectors.keys():

        myLVSignals[d] = GWSignal(TaylorF2_RestrictedPN(), 
                    psd_path=LVdetectors[d]['psd_path'],
                    detector_shape = LVdetectors[d]['shape'],
                    det_lat= LVdetectors[d]['lat'],
                    det_long=LVdetectors[d]['long'],
                    det_xax=LVdetectors[d]['xax'], 
                    verbose=True,
                    is_ASD = False,
                    useEarthMotion = False,
                    fmin= fmin, fmax = fmax,
                    IntTablePath=None) 

    myLVNet = DetNet(myLVSignals, verbose=False)
    kwargs = {'df': 1./data_duration, 'spacing': 'geom', 'use_m1m2': usem1m2}
    totF = myLVNet.FisherMatr(event, **kwargs)
    
    
    ParNums = TaylorF2_RestrictedPN().ParNums
    fisher_new, params = fixParams(totF, ParNums, ['iota', 'theta', 'phi', 'dL', 'psi', 'Phicoal', 'tcoal'])
    cov_new, inversion_err = CovMatr(fisher_new)
    reshape_cov = cov_new.reshape(4, 4)
    
    jacob_mat = Jacobian_mat(event['m1'][0], event['m2'][0])
    cov_mat_q = np.array(jacob_mat@reshape_cov@jacob_mat.T, dtype = np.float32)
    
    
    return cov_mat_q, inversion_err




def generate_nodes(boundary, mu, cov_mat, Nnodes, nodes_gauss_num, nodes_seed, size):
        
    """
    Function to generate nodes in the intrinsic parameter space
    
    Parameters
    -----------
    
    boundary: dictionary containing the intrinsic parameter boundaries (maxm and min values)
    fLow: seismic cutoff frequency
    Nnodes: no. of nodes
    
    Returns
    ---------
    
    nodes: uniformly sprayed points in intrinsic parameter space
    
    """
    
    # spraying random nodes in intrinsic parameter space
    
    mchirp_min, mchirp_max = boundary['mchirp']
    mass_ratio_min, mass_ratio_max = boundary['mass_ratio']
    s1z_min, s1z_max = boundary['s1z']
    s2z_min, s2z_max = boundary['s2z']
    
    np.random.seed(nodes_seed)
    temp = np.random.multivariate_normal(mu, cov_mat, size=size)
    
    
    idx = np.where((temp[:, 0] > mchirp_min) & (temp[:, 0] < mchirp_max) & (temp[:, 1] > mass_ratio_min) & (temp[:, 1] < mass_ratio_max) & (temp[:, 2] > s1z_min) & (temp[:, 2] < s1z_max) & (temp[:, 3] > s2z_min) & (temp[:, 3] < s2z_max))[0]
    
    
    gauss_nodes = temp[idx]
#     dist = []
#     for i in range(len(gauss_nodes)):
#         d = (gauss_nodes[i] - mu).T @ np.linalg.inv(cov_mat) @  (gauss_nodes[i] - mu)
#         dist.append(d)
    
#     dist = np.array(dist)
#     dist_idx = np.argsort(dist)
    
    gauss_nodes = gauss_nodes[:int(nodes_gauss_num*Nnodes), :]
    
    mchirp_uni = (mchirp_max - mchirp_min)*np.random.rand(int(Nnodes - nodes_gauss_num*Nnodes))+ mchirp_min
    mass_ratio_uni = (mass_ratio_max - mass_ratio_min)*np.random.rand(int(Nnodes - nodes_gauss_num*Nnodes))+ mass_ratio_min
    s1z_uni = (s1z_max - s1z_min)*np.random.rand(int(Nnodes - nodes_gauss_num*Nnodes))+ s1z_min
    s2z_uni = (s2z_max - s2z_min)*np.random.rand(int(Nnodes - nodes_gauss_num*Nnodes))+ s2z_min
    
    if len(gauss_nodes) != 0:
        mchirp = np.append(gauss_nodes[:, 0], mchirp_uni)
        mass_ratio = np.append(gauss_nodes[:, 1], mass_ratio_uni)
        s1z = np.append(gauss_nodes[:, 2], s1z_uni)
        s2z = np.append(gauss_nodes[:, 3], s2z_uni)

        nodes = np.column_stack((mchirp, mass_ratio, s1z, s2z))
    
    if len(gauss_nodes) == 0:
        nodes = np.column_stack((mchirp_uni, mass_ratio_uni, s1z, s2z))
    
    return nodes




def hnormsq_dh_overlap(node, approximant, f_low, f_high, data, tc, psd):
    
    mchirp, mass_ratio, s1, s2 = node
    
    m1 = mass1_from_mchirp_q(mchirp, mass_ratio)
    m2 = mass2_from_mchirp_q(mchirp, mass_ratio)
    
    theta0 = 2*np.pi*f_low*tau0_from_mass1_mass2(m1, m2, f_low)
    theta3 = 2*np.pi*f_low*tau3_from_mass1_mass2(m1, m2, f_low)
    node_theta_space = [theta0, theta3, s1, s2]
    
    time_slice_z_len = len(data.time_slice(tc - 0.15, tc + 0.15))
    
    hp, _ = get_fd_waveform(approximant=approximant, mass1=m1, mass2=m2, spin1z=s1, spin2z=s2, delta_f=1/data.duration, f_lower=f_low, f_final = f_high)

    
    d = data.copy()
    stilde = d.to_frequencyseries()

    N = (len(stilde)-1) * 2
    kmin, kmax = get_cutoff_indices(f_low, f_high, stilde.delta_f, N)

    kmax = min(len(hp), kmax)

    qtilde = zeros(N, dtype=complex_same_precision_as(d))
    _q = zeros(N, dtype=complex_same_precision_as(d))

    qtilde[kmin:kmax] = hp[kmin:kmax].conj()*stilde[kmin:kmax]

    
    p = psd.copy()
    qtilde[kmin:kmax] /= p[kmin:kmax]

    ifft(qtilde, _q) 
    z = 4.0*stilde.delta_f*_q
    z = TimeSeries(z, delta_t = d.delta_t, epoch = d.start_time, dtype = complex_same_precision_as(d))
    
    z = z.time_slice(tc - 0.15, tc + 0.15)
    
    hnormsq = sigmasq(hp, p, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high).real   
    return hnormsq, z, node_theta_space



def createRBFInterpolants(nodes, h0_h0, C_interp, nBasis, phi, order, eps):
  
    """This function creates rbf interpolants for svd coefficients (C) and template norm square (h0_h0) ---
  
    Parameters
    -----------
    nodes: list of nodes 
    
    h0_h0: values of h0_h0 at nodes
    
    C: svd coefficients
    
    nBasis: no. of retained top basis vectors for which the interpolants (C) are generated
    
    phi: rbf kernel
    
    order: order of the monomial terms
  
    Returns
    --------
    h0_h0_interpolant (python object)
    
    C_interpolant (python object)
  
    For more details regarding rbf interpolation package, please visit https://rbf.readthedocs.io/en/latest/interpolate.html"""
  
    # h0_h0 interpolant
    h0_h0_interpolant = RBFInterpolant(nodes, h0_h0, phi=phi, order=order, eps= eps)
    #print('h_h Interpolant generated....')
    
    # svd coefficients interpolant
    C_interpolant = []
    

    for i in tqdm(range(nBasis)):
        
        C_interpolant.append(RBFInterpolant(nodes, C_interp[:,i], phi=phi, order=order, eps= eps))
        
        #print('C Interpolant for %d basis generated....'%(i+1))
    
    C_interpolant = np.array(C_interpolant)# svd coefficients interpolants
    
    
    return h0_h0_interpolant, C_interpolant



# This function generates RBF interpolants
# While using below functions one must ensure that the data from all detectors 
# must have same start time,end time and sampling frequency
def RBFinterpolants(ifo, data, approximant, nodes, f_low, f_high, tc, psd, nbasis, nprocs):
    
    global hnormsq_z_cal_parallel
    
    # data ----> filtered data output from detector
    # Nnodes ----> number of points in parameter space to be used for start-up stage
    # f_low -----> low cutoff frequency
    # f_high ----> high cutoff frequency
    # ifo ----> list of interferometers
    # nbasis ----->  no. of retained top basis vectors for which interpolants are calculated 
    # tc -------> Time of coalescence
    # order ----> order of monomial terms
    # phi -------> Type of RBF
    # eps -------> Shape Parameter
    # fixed_params ------> extrinsic parameters dictionary; fixed_params = {'ra': .., 'dec': .., 'distance': .., 'iota': .., 'polarization': ..}
    
#     if (Nnodes < nbasis):
#         raise Exception('No. of nodes must be larger than number of retained basis')
    
    
    # Calculating SNR and <h0|h0> for brute force calculations
    # z_real_dict = {}
    # mod_z = {}
    # z_img_dict = {}
    # hnormsq_dict = {}
    # C_real = {}
    # C_img = {}
    # basis_img_vectors = {}
    # basis_real_vectors = {}
    # C_real_interpolant = {}
    # C_img_interpolant = {}
    # hp_hp_interpolant = {}
    times = data.time_slice(tc - 0.15, tc + 0.15).sample_times.data
    #params = nodes_gen(params_range, Nnodes, 0)
    #nodes = nodes_gen(params_range, Nnodes, seed = 0)
    
    def hnormsq_z_cal_parallel(node):
        return hnormsq_dh_overlap(node, approximant = approximant, f_low = f_low, f_high = f_high, data = data, tc = tc, psd = psd)
    #print(nodes)
    # for i in range(len(ifo)):
    
    
    with Pool(nprocs) as exe:
        result = tqdm(exe.map(hnormsq_z_cal_parallel, nodes), total = len(nodes))

    z_mat = np.zeros((len(nodes), len(times)), dtype = complex)

    hnormsq_mat = np.zeros((1, len(nodes)))
    nodes_theta = np.zeros((len(nodes), 4))


    for index, i in enumerate(result):
        hnormsq_mat[0, index] = i[0]
        z_mat[index] = i[1]
        nodes_theta[index] = i[2]

    # hnormsq, z_real, z_img, nodes, mod_z, params = hnormsq_z_calc(approximant, 0, params_range, Nnodes, f_low, f_high, data, tc, psd)
    
    u, s, vh = svd(z_mat, full_matrices=False)
    
    sigma = np.diag(s)
    
    C = np.matmul(u, sigma)
    
    basis_vectors = vh[0:nbasis, :]
    
#         for j in range(nodes):
#             z_dict[ifo[i]][j, :] = z_dict[ifo[i]][j, :].time_slice(tc - 0.15, tc + 0.15)
    # hp_hp_interpolant, C_real_interpolant, C_img_interpolant = createRBFInterpolants(nodes, hnormsq[0], C_real, C_img, nbasis, phi, order, eps)
        #print(basis_vectors[ifo[i]].shape)
    return ifo, basis_vectors, times, C, nodes_theta, hnormsq_mat, z_mat, s

def RBF_parallel(ifo, data, approximant, nodes, f_low, f_high, tc, psd, nbasis, phi, order, eps, nprocs):
    global RBF_interpolants_parallel
    def RBF_interpolants_parallel(ifo):
        return RBFinterpolants(ifo, data = data[ifo], approximant = approximant, nodes = nodes, f_low = f_low, f_high = f_high, tc = tc, psd = psd[ifo], nbasis = nbasis, nprocs = nprocs)
    
    
    #start = time.time()
    with ProcessPoolExecutor(max_workers = 1) as exe:
        result = exe.map(RBF_interpolants_parallel, ifo)

    basis_vectors = {}

    C = {}

    z = {}

    nodes_theta = {}
    hnormsq = {}
    hp_hp_interpolant = {}
    C_interpolant = {}

    s = {}


    for i in result:
        ifos = i[0]
        basis_vectors[ifos] = i[1]

        times = i[2] 

        C[ifos] = i[3]


        nodes_theta[ifos] = i[4]
        hnormsq[ifos] = i[5]
        z[ifos] = i[6]

        s[ifos] = i[7]


    for j in ifo:
        hp_hp_interpolant[j], C_interpolant[j] = createRBFInterpolants(nodes_theta[j], hnormsq[j][0], C[j], nbasis, phi, order, eps)
    return hp_hp_interpolant, C_interpolant, basis_vectors, times










