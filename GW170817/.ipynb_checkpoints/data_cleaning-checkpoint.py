from pycbc.frame.frame import read_frame
from pycbc.filter import matched_filter, highpass, lowpass
import matplotlib.pyplot as plt
from pycbc.psd import interpolate, welch, inverse_spectrum_truncation
from gwosc.datasets import event_gps
import numpy as np
from pycbc.waveform.generator import FDomainDetFrameGenerator, FDomainCBCGenerator
from pycbc.waveform.waveform import get_fd_waveform, get_waveform_filter_length_in_time
from pycbc.catalog import Merger
from scipy import signal 
from pycbc.types.timeseries import TimeSeries
from pycbc.pnutils import get_final_freq
from pycbc import frame
from pycbc.pnutils import f_SchwarzISCO

def reading_data(tc, analysis_start_time, analysis_end_time, file_name, channel_name, ifos):
    data = {}
    for i in range(len(ifos)):
        data[ifo[i]] = read_frame(file_name[ifos[i]],\
                    channel_name[ifos[i]],
                   start_time= tc - analysis_start_time,
                   end_time= tc + analysis_end_time,
                   check_integrity=False)
        
    return data

def filter_data(data, fLow, fHigh, filter_order):
    filtered = {}

    for i in range(len(ifo)):

        filtered[ifo[i]] = highpass(data[ifo[i]], fLow, filter_order)
        filtered[ifo[i]] = lowpass(filtered[ifo[i]], fHigh, filter_order)
        filtered[ifo[i]] = filtered[ifo[i]].crop(4, 4)
        filtered[ifo[i]].save(f'filter_data/filtered_{ifo[i]}_{filter_order}_post_new_data.hdf')
    return filtered

def psd_from_data(filtered_data, seg_len, fLow):

    psd = {}

    for i in range(len(ifo)):

        psd[ifo[i]] = filtered[ifo[i]].psd(seg_len, avg_method = 'median-mean')
        psd[ifo[i]] = interpolate(psd[ifo[i]], filtered[ifo[i]].to_frequencyseries().delta_f)
        psd[ifo[i]] = inverse_spectrum_truncation(psd[ifo[i]],\
                                                   int(seg_len * psd[ifo[i]].sample_rate), low_frequency_cutoff=fLow, trunc_method='hann')
        psd[ifo[i]].save(f'PSD/PSD_{ifo[i]}.txt')

    return psd