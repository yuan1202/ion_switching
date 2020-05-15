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
wndw = 200

def maximum(srs): return srs.max()
def minimum(srs): return srs.min()
def average(srs): return srs.mean()
def standard_deviation(srs): return srs.std()
def mean_change_abs(srs): return srs.diff().mean()

def change_rate(x):
    if np.any(np.isnan(x)):
        return np.nan
    else:
        change = (np.diff(x) / x[:-1]).values
        change = change[np.nonzero(change)[0]]
        change = change[~np.isnan(change)]
        change = change[change != -np.inf]
        change = change[change != np.inf]
        return np.mean(change)

def std_F50p(srs): return srs[:(srs.shape[0] // 2)].std()
def std_L50p(srs): return srs[-(srs.shape[0] // 2):].std()
def std_F10p(srs): return srs[:(srs.shape[0] // 10)].std()
def std_L10p(srs): return srs[-(srs.shape[0] // 10):].std()

def avg_F50p(srs): return srs[:(srs.shape[0] // 2)].mean()
def avg_L50p(srs): return srs[-(srs.shape[0] // 2):].mean()
def avg_F10p(srs): return srs[:(srs.shape[0] // 10)].mean()
def avg_L10p(srs): return srs[-(srs.shape[0] // 10):].mean()

def min_F50p(srs): return srs[:(srs.shape[0] // 2)].min()
def min_L50p(srs): return srs[-(srs.shape[0] // 2):].min()
def min_F10p(srs): return srs[:(srs.shape[0] // 10)].min()
def min_L10p(srs): return srs[-(srs.shape[0] // 10):].min()

def max_F50p(srs): return srs[:(srs.shape[0] // 2)].max()
def max_L50p(srs): return srs[-(srs.shape[0] // 2):].max()
def max_F10p(srs): return srs[:(srs.shape[0] // 10)].max()
def max_L10p(srs): return srs[-(srs.shape[0] // 10):].max()

def ratio_maxmin(srs): return np.abs(srs.max()) / max(np.abs(srs.min()), 1e-6)
def diff_maxmin(srs): return srs.max() - srs.min()

def total(srs): return srs.sum()
def count_mid(srs): return (srs > .5 * (srs.max() + srs.min())).sum()

def change_rate_F50p(srs): return change_rate(srs[:(srs.shape[0] // 2)])
def change_rate_L50p(srs): return change_rate(srs[-(srs.shape[0] // 2):])
def change_rate_F10p(srs): return change_rate(srs[:(srs.shape[0] // 10)])
def change_rate_L10p(srs): return change_rate(srs[-(srs.shape[0] // 10):])

def q99(srs): return srs.quantile(.99)
def q90(srs): return srs.quantile(.90)
def q75(srs): return srs.quantile(.75)
def q25(srs): return srs.quantile(.25)
def q10(srs): return srs.quantile(.10)
def q01(srs): return srs.quantile(.01)

def abs_q99(srs): return srs.abs().quantile(.99)
def abs_q90(srs): return srs.abs().quantile(.90)
def abs_q75(srs): return srs.abs().quantile(.75)
def abs_q25(srs): return srs.abs().quantile(.25)
def abs_q10(srs): return srs.abs().quantile(.10)
def abs_q01(srs): return srs.abs().quantile(.01)

def abs_average(srs): return srs.abs().mean()
def abs_standard_deviation(srs): return srs.abs().std()

def mad(srs): return srs.mad()
def kurt(srs): return srs.kurtosis()
def skew(srs): return srs.skew()
def med(srs): return srs.median()

def hilbert_mean(srs): return np.abs(hilbert(srs.fillna(0))).mean()
def hann_wndw_mean(srs): return (convolve(srs, hann(150), mode='same') / sum(hann(150))).mean()

def classic_sta_lta(x, length_sta, length_lta):
    
    x_nonan = x.fillna(-5)
    
    sta = np.cumsum(x_nonan ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # Pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    idx = lta < np.finfo(0.0).tiny
    lta[idx] = np.nan

    return sta / lta

def sta_lta_20_4_mean(srs): return np.nanmean(classic_sta_lta(srs, (wndw // 20), (wndw // 4)))
def sta_lta_10_8_mean(srs): return np.nanmean(classic_sta_lta(srs, (wndw // 10), (wndw // 8)))
def sta_lta_20_15_mean(srs): return np.nanmean(classic_sta_lta(srs, (wndw // 20), (wndw // 15)))
def sta_lta_10_5_mean(srs): return np.nanmean(classic_sta_lta(srs, (wndw // 10), (wndw // 5)))
def sta_lta_5_2_mean(srs): return np.nanmean(classic_sta_lta(srs, (wndw // 5), (wndw // 2)))
def sta_lta_10_1_mean(srs): return np.nanmean(classic_sta_lta(srs, (wndw // 10), (wndw // 2)))

# replaced by new functions
def exp_MA_wndw_avg(srs): return srs.ewm(span=wndw).mean().mean(skipna=True)
def exp_MA_halfwndw_avg(srs): return srs.ewm(span=wndw//2).mean().mean(skipna=True)
def exp_MA_quaterwndw_avg(srs): return srs.ewm(span=wndw//4).mean().mean(skipna=True)

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

def iqr(srs): return np.subtract(*np.percentile(srs, [75, 25]))

def q999(srs): return srs.quantile(.999)
def q001(srs): return srs.quantile(.001)
def ave10(srs): return stats.trim_mean(srs, 0.1)
    
def roll_std_avg(srs, window): return srs.rolling(window).std().mean()
def roll_std_std(srs, window): return srs.rolling(window).std().std()
def roll_std_max(srs, window): return srs.rolling(window).std().max()
def roll_std_min(srs, window): return srs.rolling(window).std().min()

# replaced by new functions
def roll_std_q01(srs, window): return srs.rolling(window).std().dropna().quantile(.01)
def roll_std_q05(srs, window): return srs.rolling(window).std().dropna().quantile(.05)
def roll_std_q95(srs, window): return srs.rolling(window).std().dropna().quantile(.95)
def roll_std_q99(srs, window): return srs.rolling(window).std().dropna().quantile(.99)

def roll_std_avg_change(srs, window): return srs.rolling(window).std().diff().mean()
def roll_std_avg_change_rate(srs, window): 
    std_ = srs.rolling(window).std().dropna()
    std_demoniator = std_.values[:-1]
    rate_ = std_.diff().values[1:][std_demoniator!=0] / std_demoniator[std_demoniator!=0]
    return np.nanmean(rate_)

def roll_std_abs_max(srs, window): return srs.rolling(window).std().abs().max()

def roll_avg_avg(srs, window): return srs.rolling(window).mean().mean()
def roll_avg_std(srs, window): return srs.rolling(window).mean().std()
def roll_avg_max(srs, window): return srs.rolling(window).mean().max()
def roll_avg_min(srs, window): return srs.rolling(window).mean().min()

# replaced by new functions
def roll_avg_q01(srs, window): return srs.rolling(window).mean().dropna().quantile(.01)
def roll_avg_q05(srs, window): return srs.rolling(window).mean().dropna().quantile(.05)
def roll_avg_q95(srs, window): return srs.rolling(window).mean().dropna().quantile(.95)
def roll_avg_q99(srs, window): return srs.rolling(window).mean().dropna().quantile(.99)

def roll_avg_avg_change(srs, window): return srs.rolling(window).mean().diff().mean()
def roll_avg_avg_change_rate(srs, window): 
    std_ = srs.rolling(window).mean().dropna()
    std_demoniator = std_.values[:-1]
    rate_ = std_.diff().values[1:][std_demoniator!=0] / std_demoniator[std_demoniator!=0]
    return np.nanmean(rate_)

def roll_avg_abs_max(srs, window): return srs.rolling(window).mean().abs().max()

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

# ---------------------
# new
#def hmean(srs): return stats.hmean(np.abs(srs[np.nonzero(srs)[0]]))
def hmean(srs): return stats.hmean(srs[srs != 0].abs().fillna(0).values)
#def gmean(srs): return stats.gmean(np.abs(srs[np.nonzero(srs)[0]]))
def gmean(srs): return stats.gmean(srs[srs != 0].abs().fillna(0).values)

def std_Fhalf(srs): return srs[:wndw//2].std()
def std_Lhalf(srs): return srs[-wndw//2:].std()
def avg_Fhalf(srs): return srs[:wndw//2].mean()
def avg_Lhalf(srs): return srs[-wndw//2:].mean()
def min_Fhalf(srs): return srs[:wndw//2].min()
def min_Lhalf(srs): return srs[-wndw//2:].min()
def max_Fhalf(srs): return srs[:wndw//2].max()
def max_Lhalf(srs): return srs[-wndw//2:].max()

def hann_wndw_mean_4thwndw(srs): return (convolve(srs, hann(wndw//4), mode='same') / sum(hann(150))).mean()
def hann_wndw_mean_8thwndw(srs): return (convolve(srs, hann(wndw//8), mode='same') / sum(hann(150))).mean()
def hann_wndw_mean_16thwndw(srs): return (convolve(srs, hann(wndw//16), mode='same') / sum(hann(150))).mean()

def iqrl(srs): return np.subtract(*np.percentile(srs, [95, 5]))

def rng_n6n4(srs): return feature_calculators.range_count(srs, -6, -4)
def rng_n4n2(srs): return feature_calculators.range_count(srs, -4, -2)
def rng_n20(srs): return feature_calculators.range_count(srs, -2, 0)
def rng_0p2(srs): return feature_calculators.range_count(srs, 0, 2)
def rng_p2p4(srs): return feature_calculators.range_count(srs, 2, 4)
def rng_p4p6(srs): return feature_calculators.range_count(srs, 4, 6)
def rng_p6p8(srs): return feature_calculators.range_count(srs, 6, 8)
def rng_p8p10(srs): return feature_calculators.range_count(srs, 8, 10)

def num_crossing_mean(srs): return feature_calculators.number_crossing_m(srs, srs.mean())

def roll_std_qntl(srs, window, qntl): return srs.rolling(wndw).std().dropna().quantile(qntl)
def roll_avg_qntl(srs, window, qntl): return srs.rolling(wndw).mean().dropna().quantile(qntl)

feat_func_list = [
    maximum, minimum, standard_deviation, # average, 
    mean_change_abs, # abs_maximum, abs_minimum, abs_std, change_rate, 
    
    std_F50p, std_L50p, std_F10p, std_L10p, 
    avg_F50p, avg_L50p, avg_F10p, avg_L10p, 
    min_F50p, min_L50p, min_F10p, min_L10p, 
    max_F50p, max_L50p, max_F10p, max_L10p,
    
    ratio_maxmin, diff_maxmin, total, count_mid,
    
    change_rate_F50p, change_rate_L50p, change_rate_F10p, change_rate_L10p,
    
    q99, q90, q75, q25, q10, q01, abs_q99, abs_q90, abs_q75, abs_q25, abs_q10, abs_q01,
    
    abs_average, abs_standard_deviation,
    mad, kurt, skew, med,
    hilbert_mean, hann_wndw_mean,
    
    sta_lta_20_4_mean, sta_lta_10_8_mean, sta_lta_20_15_mean, sta_lta_10_5_mean, sta_lta_5_2_mean, sta_lta_10_1_mean,
    
    MA_5th_wndw_std_avg, MA_5th_wndw_BBhigh_avg, MA_5th_wndw_BBlow_avg, 
    MA_2nd_wndw_std_avg, MA_2nd_wndw_BBhigh_avg, MA_2nd_wndw_BBlow_avg, 
    MA_10th_wndw_std_avg, MA_10th_wndw_BBhigh_avg, MA_10th_wndw_BBlow_avg, 
    
    iqr, q999, q001, ave10,

    energy_density_peaks,
    hmean, gmean, 
    std_Fhalf, std_Lhalf, avg_Fhalf, avg_Lhalf, min_Fhalf, min_Lhalf, max_Fhalf, max_Lhalf, 
    hann_wndw_mean_4thwndw, hann_wndw_mean_8thwndw, hann_wndw_mean_16thwndw,
    iqrl, num_crossing_mean,
    rng_n6n4, rng_n4n2, rng_n20, rng_0p2, rng_p2p4, rng_p4p6, rng_p6p8, rng_p8p10,
]

for wd in (wndw//5, wndw//10, wndw//20):
    feat_func_list.append(g_utils.named_partial(roll_std_avg, window=wd))
    feat_func_list.append(g_utils.named_partial(roll_std_std, window=wd))
    feat_func_list.append(g_utils.named_partial(roll_std_max, window=wd))
    feat_func_list.append(g_utils.named_partial(roll_std_min, window=wd))
    feat_func_list.append(g_utils.named_partial(roll_std_avg_change, window=wd))
    feat_func_list.append(g_utils.named_partial(roll_std_avg_change_rate, window=wd))
    feat_func_list.append(g_utils.named_partial(roll_std_abs_max, window=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_avg, window=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_std, window=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_max, window=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_min, window=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_avg_change, window=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_avg_change_rate, window=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_abs_max, window=wd))
    
    for p in [1, 10, 25, 50, 75, 90, 99]:
        feat_func_list.append(g_utils.named_partial(roll_std_qntl, window=wd, qntl=p/100))
        feat_func_list.append(g_utils.named_partial(roll_avg_qntl, window=wd, qntl=p/100))

# ---------------------
for i in range(1, 5):
    feat_func_list.append(g_utils.named_partial(stats.kstat, n=i))
    feat_func_list.append(g_utils.named_partial(stats.moment, moment=i))
    
for i in [1, 2]:
    feat_func_list.append(g_utils.named_partial(stats.kstatvar, n=i))
    
def first_change_rate(srs, length): return change_rate(srs[:length])
def last_change_rate(srs, length): return change_rate(srs[-length:]) 
    
for slice_length in [wndw//20, wndw//10, wndw//5, wndw//2]:
    feat_func_list.append(g_utils.named_partial(first_change_rate, length=slice_length))
    feat_func_list.append(g_utils.named_partial(last_change_rate, length=slice_length))
    
# def pctl(srs, p): return srs.quantile(p)
# def abs_pctl(srs, p): return srs.abs().quantile(p)

# for p in [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]:
#     feat_func_list.append(g_utils.named_partial(pctl, p=p/100))
#     feat_func_list.append(g_utils.named_partial(abs_pctl, p=p/100))
    
def exp_MA_avg(srs, span): return srs.ewm(span=span).mean(skipna=True).mean(skipna=True)
def exp_MA_std(srs, span): return srs.ewm(span=span).mean(skipna=True).std(skipna=True)
def exp_MS_avg(srs, span): return srs.ewm(span=span).std(skipna=True).mean(skipna=True)
def exp_MS_std(srs, span): return srs.ewm(span=span).std(skipna=True).std(skipna=True)

for s in [wndw//20, wndw//10, wndw//5, wndw//2]:
    feat_func_list.append(g_utils.named_partial(exp_MA_avg, span=s))
    feat_func_list.append(g_utils.named_partial(exp_MA_std, span=s))
    feat_func_list.append(g_utils.named_partial(exp_MS_avg, span=s))
    feat_func_list.append(g_utils.named_partial(exp_MS_std, span=s))

borders = np.linspace(-5, 8, 20)
for i, j in zip(borders[:-1], borders[1:]):
    feat_func_list.append(g_utils.named_partial(feature_calculators.range_count, min=i, max=j))
    
for autocorr_lag in [wndw//20, wndw//10, wndw//5, wndw//2]:
    feat_func_list.append(g_utils.named_partial(feature_calculators.autocorrelation, lag=autocorr_lag))
    feat_func_list.append(g_utils.named_partial(feature_calculators.c3, lag=autocorr_lag))
    
def binned_entropy_nonan(srs, p):
    srs_nonan = srs.copy().dropna()
    return feature_calculators.binned_entropy(srs_nonan, max_bins=p)
    
for p in [10, 25, 50, 75, 90]:
    feat_func_list.append(g_utils.named_partial(binned_entropy_nonan, p=p))

for c in [wndw//20, wndw//10, wndw//5, wndw//2]:
    feat_func_list.append(g_utils.named_partial(feature_calculators.time_reversal_asymmetry_statistic, lag=c))

def wrapped_spkt_welch_density(srs, c): return list(feature_calculators.spkt_welch_density(srs, [{'coeff': c}]))[0][1]

for c in range(-50, 60, 10):
    feat_func_list.append(g_utils.named_partial(wrapped_spkt_welch_density, c=c))


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
    sgnl_R = pd.concat([sgnl, pd.Series([np.nan] * (wndw-1))])
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
    # run right
    with Pool(processes=10) as P:
        extracted = P.map_async(partial(runner_v2_tst, serie=sgnl_R, period=wndw), feature_extraction_ndcs).get()
        
    # post-process
    feat_R = [n + '_R' for n in extracted[0][0]]
    dat_R = np.array([lst[1] for lst in extracted])
    rank_R = np.array([lst[-1] for lst in extracted])
    
    sort_R = np.argsort(rank_R)
    dat_R = dat_L[sort_R]
    
    del extracted

    # ------------------------
    # some checks
    feat = np.array(feat_L + feat_R)
    dat = np.concatenate([dat_L, dat_R], axis=1)
    
    # ------------------------
    # save and clean up
    bp.pack_ndarray_to_file(feat, '../input/tst_feat_g{:d}_w{:d}.bp'.format(i, wndw))
    bp.pack_ndarray_to_file(dat, '../input/tst_dat_g{:d}_w{:d}.bp'.format(i, wndw))
    
    del feature_extraction_ndcs
    del sgnl_L, sgnl_R
    del feat_L, dat_L, rank_L, sort_L
    del feat_R, dat_R, rank_R, sort_R
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
    sgnl_R = pd.concat([sgnl, pd.Series([np.nan] * (wndw-1))])
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
    bp.pack_ndarray_to_file(feat, '../input/trn_feat_g{:d}_w{:d}.bp'.format(i, wndw))
    bp.pack_ndarray_to_file(dat, '../input/trn_dat_g{:d}_w{:d}.bp'.format(i, wndw))
    bp.pack_ndarray_to_file(lbl, '../input/trn_lbl_g{:d}_w{:d}.bp'.format(i, wndw))
    
    del feature_extraction_ndcs
    del sgnl_L, sgnl_R
    del feat_L, dat_L, lbl_L, rank_L, sort_L
    del feat_R, dat_R, lbl_R, rank_R, sort_R
    del feat, dat, lbl
    
    gc.collect()
    
    elapsed = (datetime.now() - start_time).seconds
    print('total time elapsed {:d} minutes {:d} seconds.'.format(elapsed//60, elapsed%60))
    