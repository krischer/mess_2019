# -*- coding: utf-8 -*-
"""
@authors: A.Maggi 2016 > Orignial code and porting from Matlab
          C. Hibert after 22/05/2017 > addition of spectrogram attributes and other stuffs + comments
          M. Wenner 12/02/2019 > shortened and slightly adapted for use in tutorial for MunichEarthSkienceSchool 2019, for full code contact Clement Hibert

This function computes the attributes of a seismic signal later used to perform identification through machine
learning algorithms.

- Example: from ComputeAttributes_MW import calculate_all_attributes 
        
           all_attributes = calculate_all_attributes(Data,sps,flag)


- Inputs: "Data" is the raw seismic signal of the event (cutted at the onset and at the end of the signal)
          "sps" is the sampling rate of the seismic signal (in samples per second)
          
- Output: "all_attributes" is an array of the attribute values for the input signal, ordered as detailed on lines 69-77

- References: 
    
        Maggi, A., Ferrazzini, V., Hibert, C., Beauducel, F., Boissier, P., & Amemoutou, A. (2017). Implementation of a 
        multistation approach for automated event classification at Piton de la Fournaise volcano. 
        Seismological Research Letters, 88(3), 878-891.
        
        Provost, F., Hibert, C., & Malet, J. P. (2017). Automatic classification of endogenous landslide seismicity 
        using the Random Forest supervised classifier. Geophysical Research Letters, 44(1), 113-120.
        
        Hibert, C., Provost, F., Malet, J. P., Maggi, A., Stumpf, A., & Ferrazzini, V. (2017). Automatic identification 
        of rockfalls and volcano-tectonic earthquakes at the Piton de la Fournaise volcano using a Random Forest 
        algorithm. Journal of Volcanology and Geothermal Research, 340, 130-142.
 
"""

import numpy as np
import scipy
from scipy.signal import hilbert, lfilter, butter, spectrogram
import obspy
from obspy import UTCDateTime


# -----------------------------------#
#            Main Function           #
# -----------------------------------#


def calculate_all_attributes(Data: object, sps: object) -> object:
    """

    :rtype: object
    """
    NATT = 7
           
        
    all_attributes = np.empty((1, NATT), dtype=float)

    env = envelope(Data,sps)
    
    TesMEAN, TesMEDIAN, TesSTD, env_max = get_TesStuff(env)
    
    RappMaxMean, RappMaxMedian = get_RappMaxStuff(TesMEAN, TesMEDIAN)
   
    AsDec, DistDecAmpEnv = get_AsDec(Data, env, sps)
    
    MeanFFT, MedianFFT = get_full_spectrum_stuff(Data, sps)
    

    # waveform
    all_attributes[0, 0] = np.mean(duration(Data,sps))  # 1  Duration of the signal
    all_attributes[0, 1] = np.mean(RappMaxMean)         # 2  Ratio of the Max and the Mean of the normalized envelope
    all_attributes[0, 2] = np.mean(AsDec)               # 3  Ascending time/Decreasing time of the envelope
    all_attributes[0, 3] = np.mean(DistDecAmpEnv)       # 4  Difference bewteen decreasing coda amplitude and straight line
    all_attributes[0, 4] = np.mean(env_max/duration(Data,sps)) # 5  Ratio between max envlelope and duration

    # spectral
    all_attributes[0, 5] = np.mean(MeanFFT)             # 6  Mean FFT
    all_attributes[0, 6] = np.mean(MedianFFT)           # 7  Median FFT

    return all_attributes
    

# -----------------------------------#
#        Secondary Functions         #
# -----------------------------------#
    
def duration(Data,sps):

    dur = len(Data) / sps

    return dur


def envelope(Data,sps):
    
    env = np.abs(hilbert(Data))

    return env


def get_TesStuff(env):

    env_max = np.max(env)
    norm_env = env/env_max
    TesMEAN = np.mean(norm_env)
    TesMEDIAN = np.median(norm_env)
    TesSTD = np.std(norm_env)

    return TesMEAN, TesMEDIAN, TesSTD, env_max


def get_RappMaxStuff(TesMEAN, TesMEDIAN):

    RappMaxMean = 1./TesMEAN
    RappMaxMedian = 1./TesMEDIAN

    return RappMaxMean, RappMaxMedian


def get_AsDec(Data, env, sps):

    strong_filter = np.ones(int(sps)) / float(sps)

    smooth_env = lfilter(strong_filter, 1, env)
    imax = np.argmax(smooth_env)
    
    if float(len(Data) - (imax+1))>0:
        AsDec = (imax+1) / float(len(Data) - (imax+1))
    else:
        AsDec = 0 
    
    dec = Data[imax:]
    lendec = len(dec)
     
    DistDecAmpEnv = np.abs(np.mean(np.abs(hilbert(dec / np.max(Data))) -
            (1 - ((1 / float(lendec)) * (np.arange(lendec)+1)))))

    return AsDec, DistDecAmpEnv


def get_full_spectrum_stuff(Data,sps):

    b = np.ones(300) / 300.0

    data = Data
    npts = 2560
    n = nextpow2(2*npts-1)
    
    FFTdata = 2 * np.abs(np.fft.fft(data, n=n)) / float(npts * npts)
    FFTsmooth = lfilter(b, 1, FFTdata[0:int(len(FFTdata)/2)])
    FFTsmooth_norm = FFTsmooth / max(FFTsmooth)
    
    MeanFFT = np.mean(FFTsmooth_norm)
    MedianFFT = np.median(FFTsmooth_norm)

    return MeanFFT, MedianFFT

def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n

