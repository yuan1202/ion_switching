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
def maximum(arr):
    return np.max(arr)
#     wip = np.max(arr)
#     if wip == 0.:
#         wip = .000001
#     return np.log10(np.abs(wip)) * np.sign(wip)
    
def minimum(arr):
    return np.min(arr)
#     wip = np.min(arr)
#     if wip == 0.:
#         wip = .000001
#     return np.log10(np.abs(wip)) * np.sign(wip)
    
def standard_dev(arr):
    return np.std(arr)
#     wip = arr.copy()
#     wip[wip == 0.] = .000001
#     wip = np.log10(np.abs(wip)) * np.sign(wip)
#     return np.std(wip)

def pctl1(arr):
    return np.percentile(arr, q=1)
#     wip = np.percentile(arr, q=1)
#     if wip == 0.:
#         wip = .000001
#     return np.log10(np.abs(wip)) * np.sign(wip)
    
def pctl10(arr):
    return np.percentile(arr, q=10)
#     wip = np.percentile(arr, q=10)
#     if wip == 0.:
#         wip = .000001
#     return np.log10(np.abs(wip)) * np.sign(wip)
    
def pctl50(arr):
    return np.percentile(arr, q=50)
#     wip = np.percentile(arr, q=50)
#     if wip == 0.:
#         wip = .000001
#     return np.log10(np.abs(wip)) * np.sign(wip)
    
def pctl90(arr):
    return np.percentile(arr, q=90)
#     wip = np.percentile(arr, q=90)
#     if wip == 0.:
#         wip = .000001
#     return np.log10(np.abs(wip)) * np.sign(wip)
    
def pctl99(arr):
    return np.percentile(arr, q=99)
#     wip = np.percentile(arr, q=99)
#     if wip == 0.:
#         wip = .000001
#     return np.log10(np.abs(wip)) * np.sign(wip)
    
def ncrossing_0(arr):
    return feature_calculators.number_crossing_m(arr, 0)
    
def ncrossing_med(arr):
    return feature_calculators.number_crossing_m(arr, np.median(arr))
    
def ncrossing_10(arr):
    return feature_calculators.number_crossing_m(arr, np.percentile(arr, q=10))
    
def ncrossing_90(arr):
    return feature_calculators.number_crossing_m(arr, np.percentile(arr, q=90))
    
def npeaks_10(arr):
    return feature_calculators.number_peaks(arr, n=10)
    
def npeaks_50(arr):
    return feature_calculators.number_peaks(arr, n=50)
    
def npeaks_100(arr):
    return feature_calculators.number_peaks(arr, n=100)
    
def ncwtpeaks_10(arr):
    return feature_calculators.number_cwt_peaks(arr, n=10)
    
def ncwtpeaks_50(arr):
    return feature_calculators.number_cwt_peaks(arr, n=50)
    
def ncwtpeaks_100(arr):
    return feature_calculators.number_cwt_peaks(arr, n=100)
    
def abs_energy(arr):
    return feature_calculators.abs_energy(arr)
    #return np.log10(max(feature_calculators.abs_energy(arr), .000001))

def welchn120(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': -120}]))[0][1]

def welchn110(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': -110}]))[0][1]

def welchn100(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': -100}]))[0][1]

def welchn90(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': -90}]))[0][1]

def welchn80(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': -80}]))[0][1]

def welchn70(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': -70}]))[0][1]

def welchn60(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': -60}]))[0][1]

def welchn50(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': -50}]))[0][1]

def welchn40(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': -40}]))[0][1]

def welchn30(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': -30}]))[0][1]

def welchn20(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': -20}]))[0][1]

def welchn10(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': -10}]))[0][1]

def welch0(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': 0}]))[0][1]

def welchp10(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': 10}]))[0][1]

def welchp20(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': 20}]))[0][1]

def welchp30(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': 30}]))[0][1]

def welchp40(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': 40}]))[0][1]

def welchp50(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': 50}]))[0][1]

def welchp60(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': 60}]))[0][1]

def welchp70(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': 70}]))[0][1]

def welchp80(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': 80}]))[0][1]

def welchp90(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': 90}]))[0][1]

def welchp100(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': 100}]))[0][1]

def welchp110(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': 110}]))[0][1]

def welchp120(arr):
    return list(feature_calculators.spkt_welch_density(arr, [{'coeff': 120}]))[0][1]
    

feat_func_list = [
    g_utils.named_partial(
        ts_utils.serie_stacker,
        step_size=1000,
        block_size=1000,
        feature_extractor=[
            maximum,
            minimum,
            standard_dev,
            np.median,
            pctl1,
            pctl10,
            pctl50,
            pctl90,
            pctl99,
            ncrossing_0,
            ncrossing_med,
            ncrossing_10,
            ncrossing_90,
            npeaks_10,
            npeaks_50,
            npeaks_100,
            ncwtpeaks_10,
            ncwtpeaks_50,
            ncwtpeaks_100,
            abs_energy,
            welchp120,
            welchp110,
            welchp100,
            welchp90,
            welchp80,
            welchp70,
            welchp60,
            welchp50,
            welchp40,
            welchp30,
            welchp20,
            welchp10,
            welch0,
            welchn10,
            welchn20,
            welchn30,
            welchn40,
            welchn50,
            welchn60,
            welchn70,
            welchn80,
            welchn90,
            welchn100,
            welchn110,
            welchn120,
        ],
    )
]
    
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

f_name = 'tst_fe_srs.pylist'
with open(f_name, 'wb') as f:
   pickle.dump(tst_dat_fe, f)


# t_diff = trn_pdf['tminus'].diff()
# ndcs = t_diff[t_diff > 0].index.astype('uint32').tolist()
# ndcs = [0] + ndcs + [t_diff.index[-1]]
# segments = [(arr, b) for arr, b in zip(ndcs[:-1], ndcs[1:])]
# segment_lbl = [2, 0, 2, 0, 2, 1, 1, 2, 1, 0, 0, 0, 0, 0, 2, 0, 0]
# max_t = t_diff.loc[t_diff > 0].values.tolist()
# max_t = [trn_pdf['tminus'].iloc[0]] + max_t
# quake_info = [(arr, b[0], b[1]) for arr, b in enumerate(zip(segments, max_t))]
# quake_info

# sub_segments = segments#[segments[ndx] for ndx in (0, 2, 4, 6, 7, 8, 14, 16)] 

# # create label data
# trn_pdf['label'] = trn_pdf['tminus'].astype('uint16')

# kf_splitter = StratifiedKFold(n_splits=3, random_state=42)

# # multi-processing helper
# def runner_rgr(ndx, period=150000):
#     serie = trn_pdf['data'].iloc[ndx:ndx+period]

#     # make sure sampling does go out of bound
#     if ndx+period > trn_pdf.shape[0]:
#         return None, None, None

#     # get feature and name
#     feat, dat = ts_utils.feature_runner(serie, feat_func_list)

#     # collect label
#     lbl = trn_pdf['tminus'].iloc[ndx+period-1]
    
#     return feat, dat, lbl


# data_type = str(torch.get_default_dtype()).split('.')[1]

# n_skip_blocks = [0., .2, .4, .6, .8]

# # rebalance switch
# rebalance = False

# # sampling ratio
# ratio = 1/150000

# for ndx in range(0, 10):
#     for j, skip_block in enumerate(n_skip_blocks):
#         sub_ndx_trn = list(ts_utils.index_sampler(sub_segments[ndx], skip_blocks=skip_block, sample_ratio=ratio))
        
#         with Pool(processes=4) as P:
#             trn_dat_fe = P.map_async(runner_rgr, sub_ndx_trn).get()

#         f_name = 'quake{}_subsample{}_fe_srs.pylist'.format(ndx, j)
#         with open(f_name, 'wb') as f:
#             pickle.dump(trn_dat_fe, f)
