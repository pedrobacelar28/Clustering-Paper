# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# %%
# FILTRO DERICK:

import scipy.signal as sgn

def remove_baseline_filter(sample_rate):
    fc = 0.8  # [Hz], cutoff frequency
    fst = 0.2  # [Hz], rejection band
    rp = 0.5  # [dB], ripple in passband
    rs = 40  # [dB], attenuation in rejection band
    wn = fc / (sample_rate / 2)
    wst = fst / (sample_rate / 2)

    filterorder, aux = sgn.ellipord(wn, wst, rp, rs)
    sos = sgn.iirfilter(filterorder, wn, rp, rs, btype='high', ftype='ellip', output='sos')
    return sos

# %%
# FILTRO AKIRA:

import BaselineWanderRemoval as bwr

def baseline_wander_removal_filter(ecg):
    ecg_filtrado = np.array(bwr.fix_baseline_wander(ecg, sr=400))
    return ecg_filtrado

# %%
# FUNÇÃO DE FILTRAR UMA LEAD:

def ecg_filtrado(ecg):
    sos = remove_baseline_filter(400)
    ecg_nobaseline = sgn.sosfiltfilt(sos, ecg, padtype='constant', axis=-1)
    ecg_final = baseline_wander_removal_filter(ecg_nobaseline)
    return ecg_final

# %%
# FUNÇÃO QUE FILTRA TODAS AS LEADS DE UM ECG:

def ecg_inteiro_filtrado(ecg):
    ecg_filtrado_completo = np.zeros_like(ecg)
    for i in range(ecg.shape[1]):
        ecg_filtrado_completo[:, i] = ecg_filtrado(ecg[:, i])
    return ecg_filtrado_completo

# %%
# FUNÇÃO QUE FILTRA VARIOS ECGS:

def filtrar_sinais_ecg(sinais):
    sinais_filtrados = []
    for sinal in sinais:
        sinal_filtrado = ecg_inteiro_filtrado(sinal)
        sinais_filtrados.append(sinal_filtrado)
    return np.array(sinais_filtrados)