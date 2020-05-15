
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



trn_dat = bp.unpack_ndarray_from_file('../input/feats_tblr/trn_dat_all_origv2_w500.bp')
trn_lbl = pd.read_pickle('../input/feats_tblr/trn_lbl_orig_v2_all.pkl')['open_channels'].values

# tst_dat = bp.unpack_ndarray_from_file('../input/feats_tblr/tst_dat_all_origv2_w500.bp')


new_lbl = [str(a) + '_' + str(b) for a, b in zip(trn_lbl.astype('uint32'), np.concatenate([np.ones(500000).astype('uint32') * i for i in range(10)]))]
unq_l = np.unique(new_lbl)
lbl_map = {str_l: i for str_l, i in zip(unq_l, np.arange(len(unq_l)))}
new_lbl = [lbl_map[s] for s in new_lbl]


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


for fold, (trn_ndcs, vld_ndcs) in enumerate(kf.split(trn_dat, new_lbl)):
    if fold == 2:
        x_trn, x_vld = trn_dat[trn_ndcs], trn_dat[vld_ndcs]
        y_trn, y_vld = trn_lbl[trn_ndcs], trn_lbl[vld_ndcs]
        del trn_dat, trn_lbl
        break


params = {
    "boosting": "gbdt",
    'objective': 'multiclass',
    'random_state': 236,
    'num_leaves': 280,
    'learning_rate': 0.026623466966581126,
    'max_depth': 80,
    'reg_alpha': 2.959759088169741, # L1
    'reg_lambda': 1.331172832164913, # L2
    "bagging_fraction": 0.9655406551472153,
    "bagging_freq": 9,
    'colsample_bytree': 0.6867118652742716
}


model = lgb.LGBMClassifier(**params, n_estimators=10000, n_jobs=14)
model.fit(X=x_trn, y=y_trn, eval_set=[(x_vld, y_vld)], eval_metric='logloss', verbose=50, early_stopping_rounds=100)

joblib.dump(model, './saved_models/lgbm_cls_feats_origv2_myw500_fld{:d}.pkl'.format(fold))


vld_pred = model.predict(x_vld, num_iteration=model.best_iteration_)
f1 = metrics.f1_score(y_vld.astype(int), vld_pred.astype(int), average = 'macro')
print(f1)


