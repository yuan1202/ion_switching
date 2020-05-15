import gc
import os
from datetime import datetime
import pickle
import psutil
from tqdm import tqdm, tqdm_notebook
from functools import partial, update_wrapper
from itertools import product
from multiprocessing import Pool

import bloscpack as bp

import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression

from YSMLT import utils as g_utils
from YSMLT.series import utils as ts_utils

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy.signal import welch, find_peaks
from scipy import stats
from scipy.special import entr
from scipy.stats import entropy
from tsfresh.feature_extraction import feature_calculators

# ==================================================================
pdf_trn = pd.read_csv('../input/train_clean.csv')
pdf_tst = pd.read_csv('../input/test_clean.csv')

with open('../input/batch_ids_trn.pkl', 'rb') as f:
    batch_id_trn = pickle.load(f)
with open('../input/batch_ids_tst.pkl', 'rb') as f:
    batch_id_tst = pickle.load(f)

# ==================================================================
wndw = 100

def energy_density_peaks(srs): #
    # frequency bounds in which peaks and energy levels are taken
    bounds = (
        (0, 200), (200, 400), (400, 600), (600, 800), (800, 1000),
        (1000, 1200), (1000, 1400), (1400, 1600), (1600, 1800), (1800, 2000),
        (2000, 2200), (2200, 2400), (2400, 2600), (2600, 2800), (2800, 3000),
        (3000, 3200), (3200, 3400), (3400, 3600), (3600, 3800), (3800, 4000),
        (4000, 4200), (4200, 4400), (4400, 4600), (4600, 4800), (4800, 5000),
    )
    # get energy density distribution
    freqs, Es = welch(srs, fs=10000, nperseg=wndw)
    # get the indicese of the maximum energy for each bound
    bins = [np.where((freqs >= bound[0]) & (freqs < bound[1])) for bound in bounds]
    # get corresponding bins of frequencies and energy levels
    freq_bins = [freqs[ndcs] for ndcs in bins]
    E_bins = [Es[ndcs] for ndcs in bins]
    # get max energy level and corresponding frequency for each bin
    max_ndcs = [np.argmax(E_bin) for E_bin in E_bins]
    max_freqs = [freq_bin[ndx] for freq_bin, ndx in zip(freq_bins, max_ndcs)]
    max_Es = [E_bin[ndx] for E_bin, ndx in zip(E_bins, max_ndcs)]
    E_integration = [np.trapz(e, f) for f, e in zip(freq_bins, E_bins)]
    return max_freqs + max_Es + E_integration


feat_func_list = [energy_density_peaks,]

# ---------------------

def binned_entropy_nonan(srs, p):
    srs_nonan = srs.copy().dropna()
    return feature_calculators.binned_entropy(srs_nonan, max_bins=p)
    
for p in [wndw//2, wndw//4]:
    feat_func_list.append(g_utils.named_partial(binned_entropy_nonan, p=p))

for c in [wndw//4, wndw//2]:
    feat_func_list.append(g_utils.named_partial(feature_calculators.time_reversal_asymmetry_statistic, lag=c))

    
# ==================================================================
def runner_v2(ndx, serie, period):
    sr_slice = serie.iloc[ndx:ndx+period]
    # get feature and name
    feat, dat = ts_utils.feature_runner(sr_slice, feat_func_list)
    # collect label
    lbl = trgt.iloc[ndx]
    return feat, dat, lbl, ndx

def runner_v2_tst(ndx, serie, period):
    sr_slice = serie.iloc[ndx:ndx+period]
    # get feature and name
    feat, dat = ts_utils.feature_runner(sr_slice, feat_func_list)
    return feat, dat, ndx

# ==================================================================

start_time = datetime.now()

for i in range(20):
    print('============================')
    print('processing group #{:d}...'.format(i))
    sgnl_ndcs = batch_id_tst[i]
    sgnl = pdf_tst['signal'].iloc[sgnl_ndcs]
    
    sgnl_L = pd.concat([pd.Series([np.nan] * (wndw-1)), sgnl])
    del sgnl
    
    feature_extraction_ndcs = list(
        ts_utils.index_sampler((0, sgnl_L.shape[0]), block_size=wndw, sample_ratio=1)
    )
    
    # ------------------------
    # run left
    with Pool(processes=12) as P:
        extracted = P.map_async(partial(runner_v2_tst, serie=sgnl_L, period=wndw), feature_extraction_ndcs).get()
        
    # post-process
    feat_L = [n + '_L' for n in extracted[0][0]]
    dat_L = np.array([lst[1] for lst in extracted])
    rank_L = np.array([lst[-1] for lst in extracted])
    
    sort_L = np.argsort(rank_L)
    dat_L = dat_L[sort_L]
    
    del extracted

    # ------------------------
    # some checks
    feat = np.array(feat_L)
    dat = dat_L
    
    # ------------------------
    # save and clean up
    bp.pack_ndarray_to_file(feat, '../input/tst_feat_v2_g{:d}_w{:d}.bp'.format(i, wndw))
    bp.pack_ndarray_to_file(dat, '../input/tst_dat_v2_g{:d}_w{:d}.bp'.format(i, wndw))
    
    del feature_extraction_ndcs
    del sgnl_L
    del feat_L, dat_L, rank_L, sort_L
    del feat, dat
    
    gc.collect()
    
    elapsed = (datetime.now() - start_time).seconds
    print('total time elapsed {:d} minutes {:d} seconds.'.format(elapsed//60, elapsed%60))


# ==================================================================

for i in range(10):
    print('============================')
    print('processing group #{:d}...'.format(i))
    sgnl_ndcs = batch_id_trn[i]
    sgnl = pdf_trn['signal'].iloc[sgnl_ndcs]
    trgt = pdf_trn['open_channels'].iloc[sgnl_ndcs]
    
    sgnl_L = pd.concat([pd.Series([np.nan] * (wndw-1)), sgnl])
    del sgnl
    
    feature_extraction_ndcs = list(
        ts_utils.index_sampler((0, sgnl_L.shape[0]), block_size=wndw, sample_ratio=1)
    )
    
    # ------------------------
    # run left
    with Pool(processes=12) as P:
        extracted = P.map_async(partial(runner_v2, serie=sgnl_L, period=wndw), feature_extraction_ndcs).get()
        
    # post-process
    feat_L = [n + '_L' for n in extracted[0][0]]
    dat_L = np.array([lst[1] for lst in extracted])
    lbl_L = np.array([lst[2] for lst in extracted])
    rank_L = np.array([lst[-1] for lst in extracted])
    
    sort_L = np.argsort(rank_L)
    dat_L = dat_L[sort_L]
    lbl_L = lbl_L[sort_L]
    
    del extracted

    # ------------------------
    feat = np.array(feat_L)
    dat = dat_L
    lbl = lbl_L
    
    # ------------------------
    # save and clean up
    bp.pack_ndarray_to_file(feat, '../input/trn_feat_v2_g{:d}_w{:d}.bp'.format(i, wndw))
    bp.pack_ndarray_to_file(dat, '../input/trn_dat_v2_g{:d}_w{:d}.bp'.format(i, wndw))
    bp.pack_ndarray_to_file(lbl, '../input/trn_lbl_v2_g{:d}_w{:d}.bp'.format(i, wndw))
    
    del feature_extraction_ndcs
    del sgnl_L
    del feat_L, dat_L, lbl_L, rank_L, sort_L
    del feat, dat, lbl
    
    gc.collect()
    
    elapsed = (datetime.now() - start_time).seconds
    print('total time elapsed {:d} minutes {:d} seconds.'.format(elapsed//60, elapsed%60))
    