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
wndw = 500

def MA_5th_wndw_avg(srs): return srs.rolling(window=wndw//5).mean().mean(skipna=True)
def MA_5th_wndw_std_avg(srs): return srs.rolling(window=wndw//5).std().mean()
def MA_5th_wndw_BBhigh_avg(srs): return MA_5th_wndw_avg(srs) + 3 * MA_5th_wndw_std_avg(srs)
def MA_5th_wndw_BBlow_avg(srs): return MA_5th_wndw_avg(srs) - 3 * MA_5th_wndw_std_avg(srs)

def MA_2nd_wndw_avg(srs): return srs.rolling(window=wndw//2).mean().mean(skipna=True)
def MA_2nd_wndw_std_avg(srs): return srs.rolling(window=wndw//2).std().mean()
def MA_2nd_wndw_BBhigh_avg(srs): return MA_2nd_wndw_avg(srs) + 3 * MA_2nd_wndw_std_avg(srs)
def MA_2nd_wndw_BBlow_avg(srs): return MA_2nd_wndw_avg(srs) - 3 * MA_2nd_wndw_std_avg(srs)

def MA_10th_wndw_avg(srs): return srs.rolling(window=wndw//10).mean().mean(skipna=True)
def MA_10th_wndw_std_avg(srs): return srs.rolling(window=wndw//10).std().mean()
def MA_10th_wndw_BBhigh_avg(srs): return MA_10th_wndw_avg(srs) + 3 * MA_10th_wndw_std_avg(srs)
def MA_10th_wndw_BBlow_avg(srs): return MA_10th_wndw_avg(srs) - 3 * MA_10th_wndw_std_avg(srs)


feat_func_list = [    
    MA_5th_wndw_std_avg, MA_5th_wndw_BBhigh_avg, MA_5th_wndw_BBlow_avg, 
    MA_2nd_wndw_std_avg, MA_2nd_wndw_BBhigh_avg, MA_2nd_wndw_BBlow_avg, 
    MA_10th_wndw_std_avg, MA_10th_wndw_BBhigh_avg, MA_10th_wndw_BBlow_avg, 
]
    
# def binned_entropy_nonan(srs, p):
#     srs_nonan = srs.copy().dropna()
#     return feature_calculators.binned_entropy(srs_nonan, max_bins=p)
    
# for p in [10, 25, 50, 75, 90]:
#     feat_func_list.append(g_utils.named_partial(binned_entropy_nonan, p=p))

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

for i in range(10):
    print('============================')
    print('processing group #{:d}...'.format(i))
    sgnl_ndcs = batch_id_trn[i]
    sgnl = pdf_trn['signal'].iloc[sgnl_ndcs]
    trgt = pdf_trn['open_channels'].iloc[sgnl_ndcs]
    
    sgnl_L = pd.concat([pd.Series([np.nan] * (wndw-1)), sgnl])
    sgnl_R = pd.concat([sgnl, pd.Series([np.nan] * (wndw-1))])
    del sgnl
    
    feature_extraction_ndcs = list(
        ts_utils.index_sampler((0, sgnl_L.shape[0]), block_size=wndw, sample_ratio=1)
    )
    
    # ------------------------
    # run left
    with Pool(processes=10) as P:
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
    # run right
    with Pool(processes=10) as P:
        extracted = P.map_async(partial(runner_v2, serie=sgnl_R, period=wndw), feature_extraction_ndcs).get()
        
    # post-process
    feat_R = [n + '_R' for n in extracted[0][0]]
    dat_R = np.array([lst[1] for lst in extracted])
    lbl_R = np.array([lst[2] for lst in extracted])
    rank_R = np.array([lst[-1] for lst in extracted])
    
    sort_R = np.argsort(rank_R)
    dat_R = dat_L[sort_R]
    lbl_R = lbl_L[sort_R]
    
    del extracted

    # ------------------------
    # some checks
    assert np.all(lbl_L == lbl_R)
    feat = np.array(feat_L + feat_R)
    dat = np.concatenate([dat_L, dat_R], axis=1)
    lbl = lbl_L
    
    # ------------------------
    # save and clean up
    bp.pack_ndarray_to_file(feat, '../input/trn_feat_g{:d}_w{:d}_fix.bp'.format(i, wndw))
    bp.pack_ndarray_to_file(dat, '../input/trn_dat_g{:d}_w{:d}_fix.bp'.format(i, wndw))
    bp.pack_ndarray_to_file(lbl, '../input/trn_lbl_g{:d}_w{:d}_fix.bp'.format(i, wndw))
    
    del feature_extraction_ndcs
    del sgnl_L, sgnl_R
    del feat_L, dat_L, lbl_L, rank_L, sort_L
    del feat_R, dat_R, lbl_R, rank_R, sort_R
    del feat, dat, lbl
    
    gc.collect()
    
    elapsed = (datetime.now() - start_time).seconds
    print('total time elapsed {:d} minutes {:d} seconds.'.format(elapsed//60, elapsed%60))

    