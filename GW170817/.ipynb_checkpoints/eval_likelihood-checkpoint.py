import numpy as np
import matplotlib.pyplot as plt
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta,\
                                tau0_from_mass1_mass2, tau3_from_mass1_mass2,\
mass1_from_mchirp_q, mass2_from_mchirp_q
from pycbc.waveform import get_fd_waveform
from pycbc.waveform.generator import FDomainDetFrameGenerator, FDomainCBCGenerator
from pycbc.frame.frame import read_frame
#import random
from pycbc.filter import sigmasq
from scipy.linalg import svd
from scipy.interpolate import CubicSpline, CubicHermiteSpline, interp1d, BarycentricInterpolator,\
krogh_interpolate, PchipInterpolator, Akima1DInterpolator
from scipy import special
from rbf.interpolate import RBFInterpolant
#from rbf.interpolate import _objective
from rbf.poly import mvmonos
from pycbc.detector import Detector
import h5py
import pickle
#import matplotlib
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
import multiprocessing as mp
# from multiprocessing import Process
# from multiprocessing.pool import ThreadPool
# from concurrent.futures import ProcessPoolExecutor
from pycbc.inference.models.marginalized_gaussian_noise import MarginalizedPhaseGaussianNoise

from functools import partial

from timeit import default_timer as timer
from time import process_time
from time import time
import cProfile
import dynesty
import sys
from scipy.special import hyp2f1
import numpy
from scipy.interpolate import interp1d
from numpy.random import Generator, PCG64


def evaluateRBFList(q, Kqy, Pq, h0_h0_interpolant, rList):
    
    """"
    Function to evaluate interpolated values of template norm square (h0_h0) and SVD coefficient (C) for each basis
    
    Parameters
    -----------
    
    q: query point [mchirp, mass ratio, s1z, s2z]
    
    h0_h0_interpolant: interpolant for template norm square (h0_h0)
    
    rList: list of RBF interpolants for SVD coefficients (assume same observation points)
    
    Returns
    --------
    
    h0_h0: interpolated value of h0_h0 at the query point (q)
    
    res.flatten(): list interpolated values of SVD coefficents (for each basis) at the query point (q)
    
    """
    # r0 = rList_real[0]
    # q = q - r0.center
    # Kqy = r0.phi(q, r0.y, eps=r0.eps, diff=None)
    # Pq = mvmonos(q, r0.order, diff=None)
    # res_real = np.empty((len(rList_real), 1))
    # res_img = np.empty((len(rList_img), 1))
    # real_interp, img_interp = np.array([(Kqy.dot(ri.phi_coeff) + Pq.dot(ri.poly_coeff)) for ri in rList_real]), np.array([(Kqy.dot(ri.phi_coeff) + Pq.dot(ri.poly_coeff)) for ri in rList_img])
    
    #interp = np.column_stack((rList_real, rList_img))
    #start_time = time.time()
    eval_interp = np.empty((len(rList), 2))
#     q = multiprocessing.Queue()
#     jobs = []
#     for i in range(len(rList_real)):
#         p = multiprocessing.Process(target = eval_interp_parallel, args = (interp[i], q, Kqy, Pq))
#         p.start()
#         jobs.append(p)
    
#     for i in range(len(jobs)):
#         jobs[i].join()
        
#     for j, result in enumerate(q.get()):
#         print(result)
        
        
#     pool = ThreadPool()
    
#     results = pool.map(partial(eval_interp_parallel, Kqy = Kqy, Pq = Pq), interp)
    
#     for i, result in enumerate(results):
#         eval_interp['real'][i] = result[0]
#         eval_interp['img'][i] = result[1]

    
    for i in range(len(rList)):
        
        eval_interp[i] = (Kqy.dot(rList[i].phi_coeff) + Pq.dot(rList[i].poly_coeff))
        
    
#     for i, ri in enumerate(rList_img):
        
#         res_img[i] = (Kqy.dot(ri.phi_coeff) + Pq.dot(ri.poly_coeff))
    
    h0_h0 = (Kqy.dot(h0_h0_interpolant.phi_coeff) + Pq.dot(h0_h0_interpolant.poly_coeff))[0]
    
    # end_time = time.time()
    # print('to eval RBF List:', (end_time - start_time))
    # print(end_time - start_time)
     
    return h0_h0, eval_interp

def RBFInterpolatedLikelihood(q, h0_h0_interpolant, C_interpolant, basis_Vecs, nBasis, times, det, fixed_params):
    
    """"
    Function to RBF interpolated likelihood value at a query point (q)
    
    Parameters
    -----------
    
    q: query point; q = [mchirp, mass_ratio, s1z, s2z]
    
    h0_h0_interpolant: interpolant for template norm square (h0_h0)
    
    C_interpolant: list of RBF interpolants for SVD coefficients
    
    basisVecs: basis vectors from SVD
    
    nBasis: no. of retained top basis vectors
    
    times: list of times at for which z time-series is interpolated 
    
    det: python object containing detector information
    
    fixed_params: fixed parameters (including extrinsic parameters e.g, sky location and inclination)
    
    Returns
    --------
    
    llr: interpolated value of log likelihood (marginalized phase) at the query point (q)
        
    """
    
    #-- extract intrinsic and extrinsic parameters from q seperately ---
    
    theta0, theta3, s1z, s2z, tc = q    
#     m1, m2 = mass1_from_mchirp_eta(mchirp, eta), mass2_from_mchirp_eta(mchirp, eta)
#     theta0 = 2*np.pi*fixed_params['fLow']*tau0_from_mass1_mass2(m1, m2, fixed_params['fLow'])
#     theta3 = 2*np.pi*fixed_params['fLow']*tau3_from_mass1_mass2(m1, m2, fixed_params['fLow']) 
    
    q_int = np.array([theta0, theta3, s1z, s2z]).reshape(1,4)
    #print(C_interpolant)
    
    r0 = C_interpolant[list(det.keys())[0]][0]
    #print('r0:', r0)
    
    #q = q_int - r0.center
    
    Kqy = r0.phi(q_int, r0.y, eps=r0.eps, diff=None)
    
    
    Pq = mvmonos((q_int - r0.shift)/r0.scale, r0.order, diff=None)
    
    # r0 = interpolants['R_interpolants'][ifos[0]][0]
    # q = q_int - r0.center
    # Kqy = r0.phi(q, r0.y, eps=r0.eps, diff=None)
    # Pq = mvmonos(q, r0.order, diff=None)
    
#     z_real_prime_1 = {}
#     z_img_prime_1 = {}

    dh_interp = 0j
    hh = 0
    
    #-- calculating interpolated h0_h0 and C_interp at the query point (q) ---
    for i in range(len(list(det.keys()))):
        
        # C_real_interpolant_lis = C_real_interpolant[ifo[i]][0:nBasis]
        # C_img_interpolant_lis = C_img_interpolant[ifo[i]][0:nBasis]
        # basisVecs_real_lis = basis_real_Vecs[ifo[i]]
        # basisVecs_img_lis = basis_img_Vecs[ifo[i]]
        h0_h0, C_interp = evaluateRBFList(q_int, Kqy, Pq, h0_h0_interpolant[list(det.keys())[i]], \
                                          C_interpolant[list(det.keys())[i]][0:nBasis])
        
    #-- calculating quantities dependent on extrinsic parameters ---
        C_real_img = np.zeros(nBasis, dtype = complex)
        for j in range(nBasis):
            C_real_img[j] = complex(C_interp[j, 0], C_interp[j, 1])
            
        # det = Detector(ifo[i])
        del_t = det[list(det.keys())[i]].time_delay_from_earth_center(fixed_params['ra'], fixed_params['dec'], tc)
        fp, fc = det[list(det.keys())[i]].antenna_pattern(fixed_params['ra'], fixed_params['dec'], fixed_params['pol'], tc)
        A = (((1 + np.cos(fixed_params['iota'])**2)/2)*fp - 1j*np.cos(fixed_params['iota'])*fc)/fixed_params['distance']

        #-- index corresponding to proposed tc + del_t (ra, dec, tc) ---

        k = ((tc + del_t) - times[0])*fixed_params['fSamp']

        #-- checking whether tc + del_t coincide with any tc in times 

        if isinstance(k, int):

            #k = int(np.floor(k))
            z_c = np.dot(C_real_img, basis_Vecs[list(det.keys())[i]][:,k]) # A contains the extrinsic dependence
            #dh_img_interp = np.dot(C_img_interp, basis_img_Vecs[ifo[i]][:,k])
            #dh_interp += A.conj() * complex(dh_real_interp, dh_img_interp)
            dh_interp += A.conj() * z_c
            #print('from normal method')
        else:
        #-- otherwise fit a cubic spline on z timeseries values centered at proposed tc + del_t ---

            k = int(np.floor(k))
            val = 4
            z_prime = np.array(np.dot(C_real_img, basis_Vecs[list(det.keys())[i]][:,k-val:k+val]))
            #z_img_prime = np.array(np.dot(C_img_interp, basis_img_Vecs[ifo[i]][:,k-val:k+val]))
            
#             z_real_prime_1[ifo[i]] = np.array(np.dot(C_real_interp, basisVecs_real_lis))
#             z_img_prime_1[ifo[i]] = np.array(np.dot(C_img_interp, basisVecs_img_lis))
            
            # times_diff = gradient(times[k-val:k+val])
            # z_real_prime_diff = gradient(z_real_prime)
            # z_img_prime_diff = gradient(z_img_prime)
            # dy_dx_real = z_real_prime_diff/times_diff
            # dy_dx_img = z_img_prime_diff/times_diff
            z_real_interpolant = CubicSpline(times[k-val:k+val], z_prime.real)
            z_img_interpolant = CubicSpline(times[k-val:k+val], z_prime.imag)
            
            
            #print('cubic spline')
            
            dh_interp += A.conj() * complex(z_real_interpolant(tc + del_t), z_img_interpolant(tc + del_t))

        hh += A*A.conj()*h0_h0
        
    
    llr = np.log(special.i0e(abs(dh_interp))) + abs(dh_interp) - 0.5*hh  # marginalized phase likelihood
            
    return llr.real, dh_interp, hh

def net_llr(q, fSamp, hp_hp_interpolant, C_interpolant, basis_vectors, det, times, f_low):
    
    # hp_hp_interpolant --------> template norm square interpolants
    # C_interpolants ----------> svd coefficients interpolants
    # basis_vectors -----------> retained top basis vectors form svd 
    # ifo ------> interferometers list
    # fSamp ------> Sampling Frequency of data on which it is trained
    # times ------> Observation times
    # fixed_params ------> extrinsic parameters dictionary; fixed_params = {'ra': .., 'dec': .., 'distance': .., 'iota': .., 'polarization': ..}
    
    mchirp, mass_ratio, s1, s2, distance, tc, ra, dec, inc, polarization = q
    #start_ti = time.time()
    m1, m2 = mass1_from_mchirp_q(mchirp, mass_ratio), mass2_from_mchirp_q(mchirp, mass_ratio)
    theta0, theta3 = 2*np.pi*f_low*tau0_from_mass1_mass2(m1, m2, f_low), 2*np.pi*f_low*tau3_from_mass1_mass2(m1, m2, f_low)
    #end_ti = time.time()
    #print('time taken to define m1, m2, theta0, theta3:', (end_ti - start_ti))
    q_new =  [theta0, theta3, s1, s2, tc]
    fixed_params = {'ra' : ra, 'dec' : dec, 'iota' : inc, 'pol' : polarization, 'distance' : distance}
    fixed_params['fSamp'] = fSamp
    #print(basis_real_vectors[ifo[0]].shape, basis_img_vectors[ifo[0]].shape)
    nbasis = basis_vectors[list(det.keys())[0]].shape[0]
#     net_log_likelihood = 0 
#     for i in range(len(ifo)):
#         det = Detector(ifo[i])
#         log_likelihood = RBFInterpolatedLikelihood(q_new, hp_hp_interpolant[ifo[i]], \
#                                 C_interpolant[ifo[i]], basis_vectors[ifo[i]], \
#                                 nbasis, times, det, fixed_params)
#         net_log_likelihood += log_likelihood
    net_lik = RBFInterpolatedLikelihood(q_new, hp_hp_interpolant, \
                                 C_interpolant, basis_vectors, \
                                 nbasis, times, det, fixed_params)[0][0]
    return net_lik



def _cdf_param(value):
        r""">>> from sympy import *
           >>> x = Symbol('x')
           >>> integrate((1+x)**(2/5)/x**(6/5))
           Output:
                             _
                      -0.2  |_  /-0.4, -0.2 |    I*pi\
                -5.0x     |   |           | x*e    |
                           2  1 \   0.8     |        /
        """
        
        return -5. * value**(-1./5) * hyp2f1(-2./5, -1./5, 4./5, -value)

def cdfinv_q(q_min, q_max, value):
    """Return the inverse cdf to map the unit interval to parameter bounds.
    Note that value should be uniform in [0,1]."""

        
    lower_bound = q_min
    upper_bound = q_max
    q_array = numpy.linspace(
        q_min, q_max, num=1000, endpoint=True)
    q_invcdf_interp = interp1d(_cdf_param(q_array),
                               q_array, kind='cubic',
                               bounds_error=True)

    return q_invcdf_interp(
        (_cdf_param(upper_bound) -
         _cdf_param(lower_bound)) * value +
        _cdf_param(lower_bound))
