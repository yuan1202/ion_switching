
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

# from sklearn.svm import NuSVC
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split, RepeatedStratifiedKFold
from sklearn import metrics

import lightgbm as lgb

pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)

with open('../input/batch_ids_trn.pkl', 'rb') as f:
    batch_id_trn = pickle.load(f)
with open('../input/batch_ids_tst.pkl', 'rb') as f:
    batch_id_tst = pickle.load(f)

trn_dat_orig = pd.read_pickle('../input/feats_tblr/trn_dat_orig_v2_all.pkl').drop(columns=['open_channels', 'time', 'signal']).values
trn_dat_ent = pd.read_pickle('../input/trn_dat_refresh1_all.pkl').values
trn_dat_trgt_enc = pd.read_pickle('../input/train_clean_encoded.pkl').drop(columns=['open_channels', 'time', 'signal']).values
#trn_dat_grouped_rela_pct = pd.read_pickle('../input/trn_dat_refresh2_all.pkl').values
trn_dat = np.concatenate([trn_dat_orig, trn_dat_ent, trn_dat_trgt_enc], axis=1)
del trn_dat_orig, trn_dat_ent, trn_dat_trgt_enc

trn_lbl = pd.read_pickle('../input/trn_lbl_orig_v3_all.pkl')['open_channels'].values

strat_lbl = pd.qcut(pd.read_pickle('../input/tblr_data_stratification_group.pkl'), 200, labels=False).fillna(256).values
new_lbl = [str(a) + '_' + str(b) for a, b in zip(strat_lbl.astype('uint32'), np.concatenate([np.ones(500000).astype('uint32') * i for i in range(10)]))]
unq_l = np.unique(new_lbl)
lbl_map = {str_l: i for str_l, i in zip(unq_l, np.arange(len(unq_l)))}
new_lbl = [lbl_map[s] for s in new_lbl]

params = {
    "boosting": "gbdt",
    'objective': 'multiclass',
    'random_state': 236,
    'num_leaves': 192,
    'learning_rate': 0.02,
    'max_depth': 24,
    'reg_alpha': 2.959759088169741, # L1
    'reg_lambda': 1.331172832164913, # L2
    "bagging_fraction": 0.9,
    "bagging_freq": 7,
    'feature_fraction': 0.6867118652742716
}


rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=20190819)

for fold, (trn_ndcs, vld_ndcs) in enumerate(rskf.split(trn_dat, new_lbl)):
    if fold > 0:
        break
    
    print('----------------- training/validation on fold {} -----------------'.format(fold))
    
    x_trn, x_vld = trn_dat[trn_ndcs], trn_dat[vld_ndcs]
    y_trn, y_vld = trn_lbl[trn_ndcs], trn_lbl[vld_ndcs]

    model = lgb.LGBMClassifier(**params, n_estimators=10000, n_jobs=12)
    model.fit(X=x_trn, y=y_trn, eval_set=[(x_vld, y_vld)], eval_metric='logloss', verbose=False, early_stopping_rounds=100)

    joblib.dump(model, './saved_models/lgbm_cls_feats_origv2_ent_trgtenc_newstrat_fld{:d}.pkl'.format(fold))

    vld_pred = model.predict(x_vld, num_iteration=model.best_iteration_)
    #vld_pred = vld_pred.argmax(1)
    f1 = metrics.f1_score(y_vld.astype(int), vld_pred.astype(int), average = 'macro')
    print('validation f1 score: {}'.format(f1))
