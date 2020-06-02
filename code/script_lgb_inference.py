
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

import lightgbm as lgb

with open('../input/batch_ids_trn.pkl', 'rb') as f:
    batch_id_trn = pickle.load(f)
with open('../input/batch_ids_tst.pkl', 'rb') as f:
    batch_id_tst = pickle.load(f)
    
# ====================================================================
# ------------------------
# origv2 + w500
tst_origv2_w500 = bp.unpack_ndarray_from_file('../input/feats_tblr/tst_dat_all_origv2_w500.bp')
# ------------------------
tst_origv2 = pd.read_pickle('../input/feats_tblr/tst_dat_orig_v2_all.pkl').drop(columns=['open_channels', 'time', 'signal']).values
tst_entro = pd.read_pickle('../input/tst_dat_refresh1_all.pkl').values
tst_trgtenc = pd.read_pickle('../input/test_clean_encoded.pkl').drop(columns=['time', 'signal']).values
# origv2 + trgt enc
tst_origv2_trgtenc = np.concatenate([tst_origv2, tst_trgtenc], axis=1)
# origv2 + entropy + trgt enc
tst_origv2_ent_trgtenc = np.concatenate([tst_origv2, tst_entro, tst_trgtenc], axis=1)

del tst_origv2, tst_entro, tst_trgtenc
# ------------------------
# origv4 + entropy + target encoding + neighbour quantile
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
tst_datx = pd.read_pickle('../input/tst_dat_orig_v4_all.pkl').loc[:, orig_feats].values
tst_dat0 = pd.read_csv('../input/test_clean.csv').loc[:, ['signal']].values
tst_dat1 = pd.read_pickle('../input/tst_dat_refresh1_all.pkl').values #entropy
tst_dat2 = pd.read_pickle('../input/test_clean_encoded.pkl').drop(columns=['time', 'signal']).values #target encoding
tst_dat3 = bp.unpack_ndarray_from_file(os.path.join('../input', 'tst_dat_neighbour_quantile_all.bp')) #quantile data
tst_dat4 = pd.read_pickle('../input/tst_dat_refresh2_all.pkl').values #grouped_rela_pct

tst_last = np.concatenate([tst_datx, tst_dat0, tst_dat1, tst_dat2, tst_dat3, tst_dat4], axis=1)
del tst_datx, tst_dat0, tst_dat1, tst_dat2, tst_dat3, tst_dat4

# ====================================================================

data_match = {
    'origv2_myw500': tst_origv2_w500,
    'origv2_ent_trgtenc_newstrat': tst_origv2_ent_trgtenc,
    'origv2_trgtenc': tst_origv2_trgtenc,
    'refresh_newstratv2': tst_last,
}

# mdls = [f for f in os.listdir('./saved_models') if ('lgbm' in f) and (tag in f)]
mdls = sorted([f for f in os.listdir('./saved_models') if 'lgbm' in f])
print(mdls)

submission = pd.read_csv('../input/sample_submission.csv', dtype={'time': str, 'open_channels': 'Int64'})
submission_pred = np.zeros(shape=(submission.shape[0], 11))

for i in tqdm(range(len(mdls))) :
    print('-------- model {} --------'.format(mdls[i]))
    
    mdl = joblib.load('./saved_models/{}'.format(mdls[i]))
    
    if 'origv2_myw500' in mdls[i]:
        submission_pred += mdl.predict_proba(data_match['origv2_myw500'])
    elif 'origv2_ent_trgtenc_newstrat' in mdls[i]:
        submission_pred += mdl.predict_proba(data_match['origv2_ent_trgtenc_newstrat'])
    elif 'origv2_trgtenc' in mdls[i]:
        submission_pred += mdl.predict_proba(data_match['origv2_trgtenc'])
    elif 'refresh_newstratv2' in mdls[i]:
        submission_pred += mdl.predict_proba(data_match['refresh_newstratv2'])
    else:
        print('cannot find matching data for model {};'.format(mdls[i]))

np.save('../input/lgbm_test_predictions.pkl', submission_pred/(i+1))

submission['open_channels'] = submission_pred.argmax(1)
submission.to_csv("../submissions/sub_lgb_bagged.csv", index=False)