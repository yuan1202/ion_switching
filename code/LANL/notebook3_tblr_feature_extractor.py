import gc
import os
import datetime
import pickle
import psutil
from tqdm import tqdm, tqdm_notebook
from functools import partial, update_wrapper
from itertools import product
from multiprocessing import Pool

import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms


# In[4]:


from YSMLT import utils as g_utils
from YSMLT.series import utils as ts_utils
from YSMLT.series.models_torch import *


# In[5]:


from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy.signal import welch, find_peaks
from scipy import stats
from scipy.special import entr
from scipy.stats import entropy

from tsfresh.feature_extraction import feature_calculators

from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


# In[6]:


print(psutil.cpu_percent())
print(psutil.virtual_memory())


# ### reading train data and some visualisation

# In[7]:


trn_pdf = pd.read_csv('../input/train/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
trn_pdf.rename({"acoustic_data": "data", "time_to_failure": "tminus"}, axis="columns", inplace=True)


# ### reading test data and some visualisation

# In[8]:


tst_path = "../input/test/"
tst_fs = os.listdir(tst_path)


# feature generation functions
def maximum(srs): return srs.max()
def minimum(srs): return srs.min()
def average(srs): return srs.mean()
def standard_deviation(srs): return srs.std()
def mean_change_abs(srs): return srs.diff().mean()

def change_rate(x):
    change = (np.diff(x) / x[:-1]).values
    change = change[np.nonzero(change)[0]]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]
    return np.mean(change)

def abs_maximum(srs): return srs.abs().max()
def abs_minimum(srs): return srs.abs().min()
def abs_std(srs): return srs.abs().std()

def std_F50k(srs): return srs[:50000].std()
def std_L50k(srs): return srs[-50000:].std()
def std_F10k(srs): return srs[:10000].std()
def std_L10k(srs): return srs[-10000:].std()

def avg_F50k(srs): return srs[:50000].mean()
def avg_L50k(srs): return srs[-50000:].mean()
def avg_F10k(srs): return srs[:10000].mean()
def avg_L10k(srs): return srs[-10000:].mean()

def min_F50k(srs): return srs[:50000].min()
def min_L50k(srs): return srs[-50000:].min()
def min_F10k(srs): return srs[:10000].min()
def min_L10k(srs): return srs[-10000:].min()

def max_F50k(srs): return srs[:50000].max()
def max_L50k(srs): return srs[-50000:].max()
def max_F10k(srs): return srs[:10000].max()
def max_L10k(srs): return srs[-10000:].max()

def ratio_maxmin(srs): return srs.max() / max(srs.abs().min(), .001)
def diff_maxmin(srs): return srs.max() - srs.abs().min()

def total(srs): return srs.sum()
def count500(srs): return (srs > 500).sum()

def change_rate_F50k(srs): return change_rate(srs[:50000])
def change_rate_L50k(srs): return change_rate(srs[-50000:])
def change_rate_F10k(srs): return change_rate(srs[:10000])
def change_rate_L10k(srs): return change_rate(srs[-10000:])

def q99(srs): return srs.quantile(.99)
def q95(srs): return srs.quantile(.95)
def q05(srs): return srs.quantile(.05)
def q01(srs): return srs.quantile(.01)

def abs_q99(srs): return srs.abs().quantile(.99)
def abs_q95(srs): return srs.abs().quantile(.95)
def abs_q05(srs): return srs.abs().quantile(.05)
def abs_q01(srs): return srs.abs().quantile(.01)

def trend(srs, abs_values=False):
    ndx = np.array(range(srs.shape[0])).reshape(-1, 1)
    if abs_values:
        arr = srs.abs().values.reshape(-1, 1)
    else:
        arr = srs.values.reshape(-1, 1)
        
    lr = LinearRegression()
    lr.fit(ndx, arr)
    
    return lr.coef_[0][0]

def abs_trend(srs): return trend(srs, True)

def abs_average(srs): return srs.abs().mean()
def abs_standard_deviation(srs): return srs.abs().std()

def mad(srs): return srs.mad()
def kurt(srs): return srs.kurtosis()
def skew(srs): return srs.skew()
def med(srs): return srs.median()

def hilbert_mean(srs): return np.abs(hilbert(srs)).mean()
def hann_wndw_mean(srs): return (convolve(srs, hann(150), mode='same') / sum(hann(150))).mean()

def classic_sta_lta(x, length_sta, length_lta):
    
    sta = np.cumsum(x ** 2)

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

def sta_lta_500_10k_mean(srs): return np.nanmean(classic_sta_lta(srs, 500, 10000))
def sta_lta_5k_100k_mean(srs): return np.nanmean(classic_sta_lta(srs, 5000, 100000))
def sta_lta_3k3_6k6_mean(srs): return np.nanmean(classic_sta_lta(srs, 3333, 6666))
def sta_lta_10k_25k_mean(srs): return np.nanmean(classic_sta_lta(srs, 10000, 25000))
def sta_lta_50_1k_mean(srs): return np.nanmean(classic_sta_lta(srs, 50, 1000))
def sta_lta_100_5k_mean(srs): return np.nanmean(classic_sta_lta(srs, 100, 5000))
def sta_lta_333_666_mean(srs): return np.nanmean(classic_sta_lta(srs, 333, 666))
def sta_lta_4k_10k_mean(srs): return np.nanmean(classic_sta_lta(srs, 4000, 10000))

# replaced by new functions
#def exp_MA_300_avg(srs): return srs.ewm(span=300).mean().mean(skipna=True)
#def exp_MA_3k_avg(srs): return srs.ewm(span=3000).mean().mean(skipna=True)
#def exp_MA_30k_avg(srs): return srs.ewm(span=30000).mean().mean(skipna=True)

def MA_700_avg(srs): return srs.rolling(window=700).mean().mean(skipna=True)
def MA_700_std_avg(srs): return srs.rolling(window=700).std().mean()
def MA_700_BBhigh_avg(srs): return MA_700_avg(srs) + 3 * MA_700_std_avg(srs)
def MA_700_BBlow_avg(srs): return MA_700_avg(srs) - 3 * MA_700_std_avg(srs)

def MA_400_avg(srs): return srs.rolling(window=400).mean().mean(skipna=True)
def MA_400_std_avg(srs): return srs.rolling(window=400).std().mean()
def MA_400_BBhigh_avg(srs): return MA_400_avg(srs) + 3 * MA_400_std_avg(srs)
def MA_400_BBlow_avg(srs): return MA_400_avg(srs) - 3 * MA_400_std_avg(srs)

def MA_1k_std_avg(srs): return srs.rolling(window=1000).std().mean()

def iqr(srs): return np.subtract(*np.percentile(srs, [75, 25]))

def q999(srs): return srs.quantile(.999)
def q001(srs): return srs.quantile(.001)
def ave10(srs): return stats.trim_mean(srs, 0.1)
    
def roll_std_avg(srs, wndw): return srs.rolling(wndw).std().mean()
def roll_std_std(srs, wndw): return srs.rolling(wndw).std().std()
def roll_std_max(srs, wndw): return srs.rolling(wndw).std().max()
def roll_std_min(srs, wndw): return srs.rolling(wndw).std().min()

# replaced by new functions
#def roll_std_q01(srs, wndw): return srs.rolling(wndw).std().dropna().quantile(.01)
#def roll_std_q05(srs, wndw): return srs.rolling(wndw).std().dropna().quantile(.05)
#def roll_std_q95(srs, wndw): return srs.rolling(wndw).std().dropna().quantile(.95)
#def roll_std_q99(srs, wndw): return srs.rolling(wndw).std().dropna().quantile(.99)

def roll_std_avg_change(srs, wndw): return srs.rolling(wndw).std().diff().mean()
def roll_std_avg_change_rate(srs, wndw): 
    std_ = srs.rolling(wndw).std().dropna()
    std_demoniator = std_.values[:-1]
    rate_ = std_.diff().values[1:][std_demoniator!=0] / std_demoniator[std_demoniator!=0]
    return np.nanmean(rate_)

def roll_std_abs_max(srs, wndw): return srs.rolling(wndw).std().abs().max()

def roll_avg_avg(srs, wndw): return srs.rolling(wndw).mean().mean()
def roll_avg_std(srs, wndw): return srs.rolling(wndw).mean().std()
def roll_avg_max(srs, wndw): return srs.rolling(wndw).mean().max()
def roll_avg_min(srs, wndw): return srs.rolling(wndw).mean().min()

# replaced by new functions
#def roll_avg_q01(srs, wndw): return srs.rolling(wndw).mean().dropna().quantile(.01)
#def roll_avg_q05(srs, wndw): return srs.rolling(wndw).mean().dropna().quantile(.05)
#def roll_avg_q95(srs, wndw): return srs.rolling(wndw).mean().dropna().quantile(.95)
#def roll_avg_q99(srs, wndw): return srs.rolling(wndw).mean().dropna().quantile(.99)

def roll_avg_avg_change(srs, wndw): return srs.rolling(wndw).mean().diff().mean()
def roll_avg_avg_change_rate(srs, wndw): 
    std_ = srs.rolling(wndw).mean().dropna()
    std_demoniator = std_.values[:-1]
    rate_ = std_.diff().values[1:][std_demoniator!=0] / std_demoniator[std_demoniator!=0]
    return np.nanmean(rate_)

def roll_avg_abs_max(srs, wndw): return srs.rolling(wndw).mean().abs().max()

def energy_density_peaks(srs): #
    # frequency bounds in which peaks and energy levels are taken
    bounds = (
        (0, 20000), (20000, 40000), (40000, 60000), (60000, 80000), (80000, 100000),
        (100000, 120000), (120000, 140000), (140000, 160000), (160000, 180000), (180000, 200000),
        (200000, 220000), (220000, 240000), (240000, 260000), (260000, 280000), (280000, 300000),
        (300000, 320000), (320000, 340000), (340000, 360000), (360000, 380000), (380000, 400000),
    )
    # get energy density distribution
    freqs, Es = welch(srs, fs=4000000, nperseg=4096)
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
def hmean(srs): return stats.hmean(srs[srs != 0].abs().values)
#def gmean(srs): return stats.gmean(np.abs(srs[np.nonzero(srs)[0]]))
def gmean(srs): return stats.gmean(srs[srs != 0].abs().values)

def std_F1k(srs): return srs[:1000].std()
def std_L1k(srs): return srs[-1000:].std()
def avg_F1k(srs): return srs[:1000].mean()
def avg_L1k(srs): return srs[-1000:].mean()
def min_F1k(srs): return srs[:1000].min()
def min_L1k(srs): return srs[-1000:].min()
def max_F1k(srs): return srs[:1000].max()
def max_L1k(srs): return srs[-1000:].max()

def hann_wndw_mean_50(srs): return (convolve(srs, hann(50), mode='same') / sum(hann(150))).mean()
def hann_wndw_mean_1k5(srs): return (convolve(srs, hann(1500), mode='same') / sum(hann(150))).mean()
def hann_wndw_mean_15k(srs): return (convolve(srs, hann(15000), mode='same') / sum(hann(150))).mean()

def iqrl(srs): return np.subtract(*np.percentile(srs, [95, 5]))

def rng_minf_m4k(srs): return feature_calculators.range_count(srs, -np.inf, -4000)
def rng_p4k_pinf(srs): return feature_calculators.range_count(srs, 4000, np.inf)

def num_crossing_0(srs): return feature_calculators.number_crossing_m(srs, 0)

def roll_std_qntl(srs, wndw, qntl): return srs.rolling(wndw).std().dropna().quantile(qntl)
def roll_avg_qntl(srs, wndw, qntl): return srs.rolling(wndw).mean().dropna().quantile(qntl)


feat_func_list = [
    maximum, minimum, standard_deviation, # average, 
    mean_change_abs, change_rate, abs_maximum, abs_minimum, abs_std,
    std_F50k, std_L50k, std_F10k, std_L10k, # avg_F50k, avg_L50k,
    min_F50k, min_L50k, min_F10k, min_L10k, # avg_F10k, avg_L10k, 
    max_F50k, max_L50k, max_F10k, max_L10k, 
    ratio_maxmin, diff_maxmin, total, count500,
    change_rate_F50k, change_rate_L50k, change_rate_F10k, change_rate_L10k,
    q99, q95, q05, q01, abs_q99, abs_q95, abs_q05, abs_q01,
    trend, abs_trend, abs_average, abs_standard_deviation,
    mad, kurt, skew, med,
    hilbert_mean, hann_wndw_mean,
    sta_lta_500_10k_mean, sta_lta_5k_100k_mean, sta_lta_3k3_6k6_mean,
    sta_lta_10k_25k_mean, sta_lta_50_1k_mean, sta_lta_100_5k_mean,
    sta_lta_333_666_mean, sta_lta_4k_10k_mean,
    #exp_MA_300_avg, exp_MA_3k_avg, exp_MA_30k_avg,
    MA_700_std_avg, MA_700_BBhigh_avg, MA_700_BBlow_avg, # MA_700_avg, 
    MA_400_std_avg, MA_400_BBhigh_avg, MA_400_BBlow_avg, MA_1k_std_avg, # MA_400_avg, 
    iqr, q999, q001, ave10,
    # ------------
    # new
    energy_density_peaks,
    hmean, gmean, std_F1k, std_L1k, min_F1k, min_L1k, max_F1k, max_L1k, # avg_F1k, avg_L1k, 
    hann_wndw_mean_50, hann_wndw_mean_1k5, hann_wndw_mean_15k,
    iqrl, rng_minf_m4k, rng_p4k_pinf, num_crossing_0
]

for wd in (10, 100, 1000):
    feat_func_list.append(g_utils.named_partial(roll_std_avg, wndw=wd))
    feat_func_list.append(g_utils.named_partial(roll_std_std, wndw=wd))
    feat_func_list.append(g_utils.named_partial(roll_std_max, wndw=wd))
    feat_func_list.append(g_utils.named_partial(roll_std_min, wndw=wd))
    #feat_func_list.append(g_utils.named_partial(roll_std_q01, wndw=wd))
    #feat_func_list.append(g_utils.named_partial(roll_std_q05, wndw=wd))
    #feat_func_list.append(g_utils.named_partial(roll_std_q95, wndw=wd))
    #feat_func_list.append(g_utils.named_partial(roll_std_q99, wndw=wd))
    feat_func_list.append(g_utils.named_partial(roll_std_avg_change, wndw=wd))
    feat_func_list.append(g_utils.named_partial(roll_std_avg_change_rate, wndw=wd))
    feat_func_list.append(g_utils.named_partial(roll_std_abs_max, wndw=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_avg, wndw=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_std, wndw=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_max, wndw=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_min, wndw=wd))
    #feat_func_list.append(named_partial(roll_avg_q01, wndw=wd))
    #feat_func_list.append(named_partial(roll_avg_q05, wndw=wd))
    #feat_func_list.append(named_partial(roll_avg_q95, wndw=wd))
    #feat_func_list.append(named_partial(roll_avg_q99, wndw=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_avg_change, wndw=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_avg_change_rate, wndw=wd))
    feat_func_list.append(g_utils.named_partial(roll_avg_abs_max, wndw=wd))
    
    for p in [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]:
        feat_func_list.append(g_utils.named_partial(roll_std_qntl, wndw=wd, qntl=p/100))
        feat_func_list.append(g_utils.named_partial(roll_avg_qntl, wndw=wd, qntl=p/100))

# ---------------------
# new
for i in range(1, 5):
    feat_func_list.append(g_utils.named_partial(stats.kstat, n=i))
    feat_func_list.append(g_utils.named_partial(stats.moment, moment=i))
    
for i in [1, 2]:
    feat_func_list.append(g_utils.named_partial(stats.kstatvar, n=i))
    
def first_change_rate(srs, length): return change_rate(srs[:length])
def last_change_rate(srs, length): return change_rate(srs[-length:]) 
    
for slice_length in [1000, 10000, 50000]:
    feat_func_list.append(g_utils.named_partial(first_change_rate, length=slice_length))
    feat_func_list.append(g_utils.named_partial(last_change_rate, length=slice_length))
    
def pctl(srs, p): return srs.quantile(p)
def abs_pctl(srs, p): return srs.abs().quantile(p)

for p in [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]:
    feat_func_list.append(g_utils.named_partial(pctl, p=p/100))
    feat_func_list.append(g_utils.named_partial(abs_pctl, p=p/100))
    
def exp_MA_avg(srs, span): return srs.ewm(span=span).mean(skipna=True).mean(skipna=True)
def exp_MA_std(srs, span): return srs.ewm(span=span).mean(skipna=True).std(skipna=True)
def exp_MS_avg(srs, span): return srs.ewm(span=span).std(skipna=True).mean(skipna=True)
def exp_MS_std(srs, span): return srs.ewm(span=span).std(skipna=True).std(skipna=True)

for s in [300, 3000, 30000, 50000]:
    feat_func_list.append(g_utils.named_partial(exp_MA_avg, span=s))
    feat_func_list.append(g_utils.named_partial(exp_MA_std, span=s))
    feat_func_list.append(g_utils.named_partial(exp_MS_avg, span=s))
    feat_func_list.append(g_utils.named_partial(exp_MS_std, span=s))

def count_big_threshold(srs, length, threshold): return (np.abs(srs[-length:]) > threshold).sum()
def count_big_less_threshold(srs, length, threshold): return (np.abs(srs[-length:]) < threshold).sum()

for slice_length, threshold in product([50000, 100000, 150000], [5, 10, 20, 50, 100]):
    feat_func_list.append(g_utils.named_partial(count_big_threshold, length=slice_length, threshold=threshold))
    feat_func_list.append(g_utils.named_partial(count_big_less_threshold, length=slice_length, threshold=threshold))
    
borders = list(range(-4000, 4001, 1000))
for i, j in zip(borders, borders[1:]):
    feat_func_list.append(g_utils.named_partial(feature_calculators.range_count, min=i, max=j))
    
for autocorr_lag in [5, 10, 50, 100, 500, 1000, 5000, 10000]:
    feat_func_list.append(g_utils.named_partial(feature_calculators.autocorrelation, lag=autocorr_lag))
    feat_func_list.append(g_utils.named_partial(feature_calculators.c3, lag=autocorr_lag))
    
for p in [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]:
    feat_func_list.append(g_utils.named_partial(feature_calculators.binned_entropy, max_bins=p))
    
for peak in [10, 20, 50, 100]:
    feat_func_list.append(g_utils.named_partial(feature_calculators.number_peaks, n=peak))

def wrapped_spkt_welch_density(srs, c): return list(feature_calculators.spkt_welch_density(srs, [{'coeff': c}]))[0][1]

for c in [1, 5, 10, 50, 100]:
    feat_func_list.append(g_utils.named_partial(feature_calculators.time_reversal_asymmetry_statistic, lag=c))
    
for c in range(-120, 130, 10):
    feat_func_list.append(g_utils.named_partial(wrapped_spkt_welch_density, c=c))

# multi-processing helper
def runner_tst(f_name):
    try:
        serie = pd.read_csv(os.path.join(tst_path, f_name))['acoustic_data']
    except Exception as e:
        return None, None, None
    # get feature and name
    if serie.shape[0] == 150000:
        feat, dat = ts_utils.feature_runner(serie, feat_func_list)
    else:
        feat, dat = None, None
    return feat, dat, f_name.split('.')[0]

# get test feature data
with Pool(processes=4) as P:
    tst_dat_fe = P.map_async(runner_tst, tst_fs, chunksize=1).get()

f_name = 'tst_fe_tblr.pylist'
with open(f_name, 'wb') as f:
    pickle.dump(tst_dat_fe, f)


t_diff = trn_pdf['tminus'].diff()
ndcs = t_diff[t_diff > 0].index.astype('uint32').tolist()
ndcs = [0] + ndcs + [t_diff.index[-1]]
segments = [(a, b) for a, b in zip(ndcs[:-1], ndcs[1:])]
segment_lbl = [2, 0, 2, 0, 2, 1, 1, 2, 1, 0, 0, 0, 0, 0, 2, 0, 0]
max_t = t_diff.loc[t_diff > 0].values.tolist()
max_t = [trn_pdf['tminus'].iloc[0]] + max_t
quake_info = [(a, b[0], b[1]) for a, b in enumerate(zip(segments, max_t))]
quake_info

sub_segments = segments#[segments[ndx] for ndx in (0, 2, 4, 6, 7, 8, 14, 16)] 

# create label data
trn_pdf['label'] = trn_pdf['tminus'].astype('uint16')

kf_splitter = StratifiedKFold(n_splits=3, random_state=42)

# multi-processing helper
def runner_rgr(ndx, period=150000):
    serie = trn_pdf['data'].iloc[ndx:ndx+period]

    # make sure sampling does go out of bound
    if ndx+period > trn_pdf.shape[0]:
        return None, None, None

    # get feature and name
    feat, dat = ts_utils.feature_runner(serie, feat_func_list)

    # collect label
    lbl = trn_pdf['tminus'].iloc[ndx+period-1]
    
    return feat, dat, lbl


data_type = str(torch.get_default_dtype()).split('.')[1]

n_skip_blocks = [0., .2, .4, .6, .8]

ratio = 1/150000

for i in range(len(sub_segments)):
    for j, skip_block in enumerate(n_skip_blocks):
        sub_ndx_trn = list(ts_utils.index_sampler(sub_segments[i], skip_blocks=skip_block, sample_ratio=ratio))
        
        with Pool(processes=4) as P:
            trn_dat_fe = P.map_async(runner_rgr, sub_ndx_trn).get()
            
        f_name = 'quake{}_subsample{}_fe_tblr.pylist'.format(i, j)
        with open(f_name, 'wb') as f:
            pickle.dump(trn_dat_fe, f)


# rebalance switch
#rebalance = False

# sampling ratio
#ratio_trn = 1/150000
#ratio_vld = 1/300000

# under-sampler initialised
#cc = ClusterCentroids(sampling_strategy='auto', n_jobs=4)
#
#for i, (trn_quakes, vld_quakes) in enumerate(kf_splitter.split(sub_segments, segment_lbl)):
#    print('========= fold #{} ========='.format(i))
#    
#    print(trn_quakes, vld_quakes)
#    
#    for j, skip_block in enumerate(n_skip_blocks):
#        print('------ sub-iteraion #{} ------'.format(j))
#            
#        # data sampling
#        sub_ndx_trn = []
#        for ndx in trn_quakes:
#            sub_ndx_trn += list(ts_utils.index_sampler(sub_segments[ndx], skip_blocks=skip_block, sample_ratio=ratio_trn))
#        sub_trn_lbl = trn_pdf['label'].iloc[sub_ndx_trn].values
#            
#        sub_ndx_vld = []
#        for ndx in vld_quakes:
#            sub_ndx_vld += list(ts_utils.index_sampler(sub_segments[ndx], skip_blocks=j, sample_ratio=ratio_vld))
#        sub_vld_lbl = trn_pdf['label'].iloc[sub_ndx_vld].values            
#        
#        # resampling for account for tminus imbalance
#        if rebalance:
#            # resample training indices
#            resample_strategy_trn = {i: 400 for i in np.unique(sub_trn_lbl)}
#            #undersampler = ClusterCentroids(sampling_strategy=resample_strategy_trn, n_jobs=4)
#            undersampler = RandomUnderSampler(sampling_strategy=resample_strategy_trn)
#            sub_ndx_trn, _0 = undersampler.fit_resample(np.array(sub_ndx_trn, dtype=int).reshape(-1, 1), sub_trn_lbl)
#            sub_ndx_trn = sub_ndx_trn.squeeze().astype('uint32').tolist()
#            
#            # resample validation indices
#            resample_strategy_vld = {i: 80 for i in np.unique(sub_vld_lbl)}
#            #undersampler = ClusterCentroids(sampling_strategy=resample_strategy_vld, n_jobs=4)
#            undersampler = RandomUnderSampler(sampling_strategy=resample_strategy_vld)
#            sub_ndx_vld, _1 = undersampler.fit_resample(np.array(sub_ndx_vld, dtype=int).reshape(-1, 1), sub_vld_lbl)
#            sub_ndx_vld = sub_ndx_vld.squeeze().astype('uint32').tolist()
#            
#        print('number of training samples: {}; number of validation samples: {};'.format(len(sub_ndx_trn), len(sub_ndx_vld)))
#        
#        # data pre-processing
#        # create training data
#        with Pool(processes=4) as P:
#            trn_dat_fe = P.map_async(runner_rgr, sub_ndx_trn).get()
#        
#        f_name = 'fold{}_subfold{}_trn_fe_tblr.pylist'.format(i, j)
#        with open(f_name, 'wb') as f:
#            pickle.dump(trn_dat_fe, f)
#        
#        # create validation data
#        with Pool(processes=4) as P:
#            vld_dat_fe = P.map_async(runner_rgr, sub_ndx_vld).get()
#        
#        f_name = 'fold{}_subfold{}_vld_fe_tblr.pylist'.format(i, j)
#        with open(f_name, 'wb') as f:
#            pickle.dump(vld_dat_fe, f)
#
