
import os
import joblib
import math
import warnings
import gc
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pickle

import bloscpack as bp

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import NuSVC
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split, RepeatedStratifiedKFold
from sklearn import metrics

import lightgbm as lgb

with open('../input/batch_ids_trn.pkl', 'rb') as f:
    batch_id_trn = pickle.load(f)
with open('../input/batch_ids_tst.pkl', 'rb') as f:
    batch_id_tst = pickle.load(f)

orig_feats = [
    'grad_1',
    'grad_2',
    'grad_3',
    'grad_4',
    'lowpass_lf_0.0100',
    'lowpass_ff_0.0100',
    'lowpass_lf_0.0154',
    'lowpass_ff_0.0154',
    'lowpass_lf_0.0239',
    'lowpass_ff_0.0239',
    'lowpass_lf_0.0369',
    'lowpass_ff_0.0369',
    'lowpass_lf_0.0570',
    'lowpass_ff_0.0570',
    'lowpass_lf_0.0880',
    'lowpass_ff_0.0880',
    'lowpass_lf_0.1359',
    'lowpass_ff_0.1359',
    'lowpass_lf_0.2100',
    'lowpass_ff_0.2100',
    'lowpass_lf_0.3244',
    'lowpass_ff_0.3244',
    'lowpass_lf_0.5012',
    'lowpass_ff_0.5012',
    'highpass_lf_0.0100',
    'highpass_ff_0.0100',
    'highpass_lf_0.0163',
    'highpass_ff_0.0163',
    'highpass_lf_0.0264',
    'highpass_ff_0.0264',
    'highpass_lf_0.0430',
    'highpass_ff_0.0430',
    'highpass_lf_0.0699',
    'highpass_ff_0.0699',
    'highpass_lf_0.1136',
    'highpass_ff_0.1136',
    'highpass_lf_0.1848',
    'highpass_ff_0.1848',
    'highpass_lf_0.3005',
    'highpass_ff_0.3005',
    'highpass_lf_0.4885',
    'highpass_ff_0.4885',
    'highpass_lf_0.7943',
    'highpass_ff_0.7943',
    'ewm_mean_10',
    'ewm_std_10',
    'ewm_mean_50',
    'ewm_std_50',
    'ewm_mean_100',
    'ewm_std_100',
    'ewm_mean_200',
    'ewm_std_200',
    'ewm_mean_500',
    'ewm_std_500',
    'ewm_mean_1000',
    'ewm_std_1000',
    'ewm_mean_2000',
    'ewm_std_2000',
    'lag_t1',
    'lag_t2',
    'lag_t3',
    'lag_t4',
    'lag_t5',
    'lag_t6',
    'lag_t7',
    'lag_t8',
    'lag_t9',
    'lag_t10',
    'lag_t11',
    'lag_t12',
    'lag_t13',
    'lag_t14',
    'lag_t15',
]
trn_datx = pd.read_pickle('../input/trn_dat_orig_v4_all.pkl').loc[:, orig_feats].values
trn_dat0 = pd.read_csv('../input/train_clean.csv').loc[:, ['signal']].values
trn_dat1 = pd.read_pickle('../input/trn_dat_refresh1_all.pkl').values #entropy
trn_dat2 = pd.read_pickle('../input/train_clean_encoded.pkl').drop(columns=['open_channels', 'time', 'signal']).values #target encoding
trn_dat3 = bp.unpack_ndarray_from_file(os.path.join('../input', 'trn_dat_neighbour_quantile_all.bp')) #quantile data
trn_dat4 = pd.read_pickle('../input/trn_dat_refresh2_all.pkl').values #grouped_rela_pct

trn_dat = np.concatenate([trn_datx, trn_dat0, trn_dat1, trn_dat2, trn_dat3, trn_dat4], axis=1)
del trn_datx, trn_dat0, trn_dat1, trn_dat2, trn_dat3, trn_dat4
print(trn_dat.shape)

trn_lbl = pd.read_pickle('../input/trn_lbl_orig_v3_all.pkl')['open_channels'].values

strat_lbl = pd.qcut(pd.read_pickle('../input/tblr_data_stratification_group.pkl'), 200, labels=False).fillna(256).values
new_lbl = [
    str(a) + '_' + str(b) for a, b, c in zip(
        strat_lbl.astype('uint32'),
        np.concatenate([np.ones(500000).astype('uint32') * i for i in range(10)]),
        trn_lbl.astype('uint32'),
    )
]
unq_l = np.unique(new_lbl)
lbl_map = {str_l: i for str_l, i in zip(unq_l, np.arange(len(unq_l)))}
new_lbl = [lbl_map[s] for s in new_lbl]

params = {
    "boosting": "gbdt",
    'objective': 'multiclass',
    'num_leaves': 96,
    'learning_rate': 0.02,
    'max_depth': 16,
    'reg_alpha': 2.4, # L1
    'reg_lambda': 1.2, # L2
    "bagging_fraction": 0.8,
    "bagging_freq": 8,
    'feature_fraction': 0.8
}


rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=20190819)

for fold, (trn_ndcs, vld_ndcs) in enumerate(rskf.split(trn_dat, new_lbl)):
    if fold == 0:
        continue
    
    print('----------------- training/validation on fold {} -----------------'.format(fold))
    
    x_trn, x_vld = trn_dat[trn_ndcs], trn_dat[vld_ndcs]
    y_trn, y_vld = trn_lbl[trn_ndcs], trn_lbl[vld_ndcs]

    model = lgb.LGBMClassifier(**params, n_estimators=10000, n_jobs=14)
    model.fit(X=x_trn, y=y_trn, eval_set=[(x_vld, y_vld)], eval_metric='logloss', verbose=False, early_stopping_rounds=100)
    #model = NuSVC()
    #model.fit(X=x_trn, y=y_trn)

    joblib.dump(model, './saved_models/lgbm_cls_feats_refresh_newstratv2_fld{:d}.pkl'.format(fold))

    vld_pred = model.predict(x_vld, num_iteration=model.best_iteration_)
    #vld_pred = model.predict(x_vld)
    
    f1 = metrics.f1_score(y_vld.astype(int), vld_pred.astype(int), average = 'macro')
    print('validation f1 score: {}'.format(f1))
    
