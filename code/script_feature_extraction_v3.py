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
from scipy.stats import percentileofscore
# ==================================================================
pdf_trn = pd.read_csv('../input/train_clean.csv')
pdf_tst = pd.read_csv('../input/test_clean.csv')

with open('../input/batch_ids_trn.pkl', 'rb') as f:
    batch_id_trn = pickle.load(f)
with open('../input/batch_ids_tst.pkl', 'rb') as f:
    batch_id_tst = pickle.load(f)

# ==================================================================
wndw = 500
    
def neighbour_quantile_L(srs):
    return percentileofscore(srs, srs.iloc[-1], kind='mean')

def neighbour_quantile_R(srs):
    return percentileofscore(srs, srs.iloc[0], kind='mean')
    
# ==================================================================
def runner_v2_L(ndx, serie, period):
    sr_slice = serie.iloc[ndx:ndx+period]
    # get feature and name
    feat, dat = ts_utils.feature_runner(sr_slice, [neighbour_quantile_L])
    # collect label
    lbl = trgt.iloc[ndx]
    return feat, dat, lbl, ndx

def runner_v2_R(ndx, serie, period):
    sr_slice = serie.iloc[ndx:ndx+period]
    # get feature and name
    feat, dat = ts_utils.feature_runner(sr_slice, [neighbour_quantile_R])
    # collect label
    lbl = trgt.iloc[ndx]
    return feat, dat, lbl, ndx

def runner_v2_tst_L(ndx, serie, period):
    sr_slice = serie.iloc[ndx:ndx+period]
    # get feature and name
    feat, dat = ts_utils.feature_runner(sr_slice, [neighbour_quantile_L])
    return feat, dat, ndx

def runner_v2_tst_R(ndx, serie, period):
    sr_slice = serie.iloc[ndx:ndx+period]
    # get feature and name
    feat, dat = ts_utils.feature_runner(sr_slice, [neighbour_quantile_R])
    return feat, dat, ndx

# ==================================================================

start_time = datetime.now()

tst_dat_collection = []

for i in range(20):
    print('============================')
    print('processing group #{:d}...'.format(i))
    sgnl_ndcs = batch_id_tst[i]
    sgnl = pdf_tst['signal'].iloc[sgnl_ndcs]
    
    sgnl_L = pd.concat([pd.Series([np.nan] * (wndw-1)), sgnl])
    sgnl_R = pd.concat([sgnl, pd.Series([np.nan] * (wndw-1))])
    del sgnl
    
    feature_extraction_ndcs = list(
        ts_utils.index_sampler((0, sgnl_L.shape[0]), block_size=wndw, sample_ratio=1)
    )
    
    # ------------------------
    # run left
    with Pool(processes=12) as P:
        extracted = P.map_async(partial(runner_v2_tst_L, serie=sgnl_L, period=wndw), feature_extraction_ndcs).get()
        
    # post-process
    feat_L = [n + '_L' for n in extracted[0][0]]
    dat_L = np.array([lst[1] for lst in extracted])
    rank_L = np.array([lst[-1] for lst in extracted])
    
    sort_L = np.argsort(rank_L)
    dat_L = dat_L[sort_L]
    
    del extracted
    # ------------------------
    # run right
    with Pool(processes=12) as P:
        extracted = P.map_async(partial(runner_v2_tst_R, serie=sgnl_R, period=wndw), feature_extraction_ndcs).get()
        
    # post-process
    feat_R = [n + '_R' for n in extracted[0][0]]
    dat_R = np.array([lst[1] for lst in extracted])
    rank_R = np.array([lst[-1] for lst in extracted])
    
    sort_R = np.argsort(rank_R)
    dat_R = dat_R[sort_R]
    
    del extracted
    # ------------------------
    # some checks
    feat = np.array(feat_L + feat_R)
    tst_dat_collection.append(np.concatenate([dat_L, dat_R], axis=1))
    
    # ------------------------
    # save and clean up
    #bp.pack_ndarray_to_file(feat, '../input/tst_feat_neighbour_quantile_g{:d}_w{:d}.bp'.format(i, wndw))
    #bp.pack_ndarray_to_file(dat, '../input/tst_dat_neighbour_quantile_g{:d}_w{:d}.bp'.format(i, wndw))
    
    del feature_extraction_ndcs
    del sgnl_L, sgnl_R
    del feat_L, dat_L, rank_L, sort_L
    del feat_R, dat_R, rank_R, sort_R
    del feat#, dat
    
    gc.collect()
    
    elapsed = (datetime.now() - start_time).seconds
    print('total time elapsed {:d} minutes {:d} seconds.'.format(elapsed//60, elapsed%60))

bp.pack_ndarray_to_file(np.concatenate(tst_dat_collection, 0), '../input/tst_dat_neighbour_quantile_all_w{:d}.bp'.format(wndw))

# ==================================================================

trn_dat_collection = []

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
    with Pool(processes=12) as P:
        extracted = P.map_async(partial(runner_v2_L, serie=sgnl_L, period=wndw), feature_extraction_ndcs).get()
        
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
    with Pool(processes=12) as P:
        extracted = P.map_async(partial(runner_v2_R, serie=sgnl_R, period=wndw), feature_extraction_ndcs).get()
        
    # post-process
    feat_R = [n + '_R' for n in extracted[0][0]]
    dat_R = np.array([lst[1] for lst in extracted])
    lbl_R = np.array([lst[2] for lst in extracted])
    rank_R = np.array([lst[-1] for lst in extracted])
    
    sort_R = np.argsort(rank_R)
    dat_R = dat_R[sort_R]
    lbl_R = lbl_R[sort_R]
    
    del extracted

    # ------------------------
    # some checks
    assert np.all(lbl_L == lbl_R)
    feat = np.array(feat_L + feat_R)
    trn_dat_collection.append(np.concatenate([dat_L, dat_R], axis=1))
    lbl = lbl_L
    
    # ------------------------
    # save and clean up
    #bp.pack_ndarray_to_file(feat, '../input/trn_feat_neighbour_quantile_g{:d}_w{:d}.bp'.format(i, wndw))
    #bp.pack_ndarray_to_file(dat, '../input/trn_dat_neighbour_quantile_g{:d}_w{:d}.bp'.format(i, wndw))
    #bp.pack_ndarray_to_file(lbl, '../input/tst_lbl_neighbour_quantile_g{:d}_w{:d}.bp'.format(i, wndw))
    
    del feature_extraction_ndcs
    del sgnl_L, sgnl_R
    del feat_L, dat_L, lbl_L, rank_L, sort_L
    del feat_R, dat_R, lbl_R, rank_R, sort_R
    del feat, lbl
    
    gc.collect()
    
    elapsed = (datetime.now() - start_time).seconds
    print('total time elapsed {:d} minutes {:d} seconds.'.format(elapsed//60, elapsed%60))
    
bp.pack_ndarray_to_file(np.concatenate(trn_dat_collection, 0), '../input/trn_dat_neighbour_quantile_all_w{:d}.bp'.format(wndw))