
import os
import gc
import random

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

import bloscpack as bp

from tsfresh.feature_extraction import feature_calculators

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GroupKFold
from sklearn.metrics import f1_score

from NNs import WaveTRSFM_Classifier, WaveTRSFM_Classifier_shallow, Wave_Classifier, WaveRNN_Classifier, RNN_Classifier, RnnTRSFM_Classifier


DEVICE = 'cuda:0'
EPOCHS = 128
BATCHSIZE = 24
SEED = 19550423
LR = 0.0005
#unit = 1000
step = 500
wndw = 500


series_dat_all = bp.unpack_ndarray_from_file(os.path.join('../input/feats_srs', 'trn_srs_dat_all_s{}_w{}_feat_target_encoded.bp'.format(step, wndw)))
print(series_dat_all.shape)
series_lbl_all = bp.unpack_ndarray_from_file(os.path.join('../input/feats_srs', 'trn_srs_lbl_all_s{}_w{}_feat_target_encoded.bp'.format(step, wndw)))
print(series_lbl_all.shape)

series_bch_all = np.concatenate(
    [np.ones(shape=(series_lbl_all.shape[0] // 10,)) * i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
    axis=0
).astype(int)
print(np.unique(series_bch_all, return_counts=True))

avg = series_dat_all[:, :, 0].mean()
std = series_dat_all[:, :, 0].std()
series_dat_all[:, :, 0] = (series_dat_all[:, :, 0] - avg) / std


bin_ent = [feature_calculators.binned_entropy(lst, max_bins=20) for lst in series_dat_all[:, :, 0].tolist()]
bin_ent = pd.qcut(pd.Series(bin_ent), q=10, duplicates='drop')

skf_trgt = [str(a) + '_' + str(b) for a, b in zip(series_bch_all, bin_ent)]
us = np.unique(skf_trgt)
umap = {u: i for u, i in zip(us, range(len(us)))}
skf_trgt = [umap[u] for u in skf_trgt]


class Waveset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        
        if self.labels is None:
            return data.astype(np.float32)
        else:
            labels = self.labels[idx]
            return (data.astype(np.float32), labels.astype(int))


def fold_train_validate(
    model, optimizer, criterion, scheduler,
    training_loader, validation_loaders, fold_number,
    save_path='./saved_models/wavenet_model_fold{:03d}_checkpoint.pth',
    early_stopping=15,
):
    assert isinstance(validation_loaders, dict)

    trn_losses = [np.nan]
    vld_losses = [np.nan]
    vld_f1s = [np.nan]
    
    last_best = 0

    for epc in range(EPOCHS):
        print('===========================================================')

        epoch_trn_losses = []
        epoch_trn_lbls = []
        epoch_trn_prds = []
        
        # ------ training ------
        model.train()
        for i, (trn_batch_dat, trn_batch_lbl) in enumerate(training_loader):
            trn_batch_dat, trn_batch_lbl = trn_batch_dat.to(DEVICE), trn_batch_lbl.to(DEVICE)

            optimizer.zero_grad()
            trn_batch_prd = model(trn_batch_dat)
            trn_batch_prd = trn_batch_prd.view(-1, trn_batch_prd.size(-1))
            trn_batch_lbl = trn_batch_lbl.view(-1)
            loss = criterion(trn_batch_prd, trn_batch_lbl)
            loss.backward()
            optimizer.step()

            epoch_trn_losses.append(loss.item())
            epoch_trn_lbls.append(trn_batch_lbl.detach().cpu().numpy())
            epoch_trn_prds.append(trn_batch_prd.detach().cpu().numpy())

            print(
               'Epoch {:03d}/{:03d} - Training batch {:04d}/{:04d}: Training loss {:.6f};'.format(
                   epc + 1, EPOCHS, i + 1, len(training_loader), epoch_trn_losses[-1],
               ), 
               end='\r'
            )

        # ------ validation ------
        model.eval()
        
        grp_vld_metrics = {}
        epoch_vld_losses = []
        epoch_vld_lbls = []
        epoch_vld_prds = []

        with torch.no_grad():
            for i, (grp, ldr) in enumerate(validation_loaders.items()):
                
                epoch_grp_vld_losses = []
                epoch_grp_vld_lbls = []
                epoch_grp_vld_prds = []

                for j, (vld_batch_dat, vld_batch_lbl) in enumerate(ldr):
                    vld_batch_dat, vld_batch_lbl = vld_batch_dat.to(DEVICE), vld_batch_lbl.to(DEVICE)

                    vld_batch_prd = model(vld_batch_dat)
                    vld_batch_prd = vld_batch_prd.view(-1, vld_batch_prd.size(-1))
                    vld_batch_lbl = vld_batch_lbl.view(-1)
                    loss = criterion(vld_batch_prd, vld_batch_lbl)

                    epoch_grp_vld_losses.append(loss.item())
                    epoch_grp_vld_lbls.append(vld_batch_lbl.detach().cpu().numpy())
                    epoch_grp_vld_prds.append(vld_batch_prd.detach().cpu().numpy())
                    if grp == 'vld':
                        epoch_vld_losses.append(epoch_grp_vld_losses[-1])
                        epoch_vld_lbls.append(epoch_grp_vld_lbls[-1])
                        epoch_vld_prds.append(epoch_grp_vld_prds[-1])
                    
                epoch_grp_vld_lbls = np.concatenate(epoch_grp_vld_lbls, axis=0)
                epoch_grp_vld_prds = np.concatenate(epoch_grp_vld_prds, axis=0).argmax(1)
                
                grp_f1_vld = f1_score(
                    epoch_grp_vld_lbls, 
                    epoch_grp_vld_prds,
                    #labels=list(range(11)), 
                    average='macro'
                )
                
                grp_vld_metrics.update({grp: {'f1': grp_f1_vld, 'loss': np.mean(epoch_grp_vld_losses)}})

                print('Validation progress: {:03d}/{:03d} group done;'.format(i + 1, len(validation_loaders)), end='\r')

        # ------ epoch end ------
        epoch_trn_lbls = np.concatenate(epoch_trn_lbls, axis=0)
        epoch_trn_prds = np.concatenate(epoch_trn_prds, axis=0).argmax(1)

        f1_trn = f1_score(
            epoch_trn_lbls, 
            epoch_trn_prds,
            #labels=list(range(11)), 
            average='macro'
        )
        
        epoch_vld_lbls = np.concatenate(epoch_vld_lbls, axis=0)
        epoch_vld_prds = np.concatenate(epoch_vld_prds, axis=0).argmax(1)

        f1_vld = f1_score(
            epoch_vld_lbls, 
            epoch_vld_prds,
            #labels=list(range(11)), 
            average='macro'
        )


        print(
            'Epoch {:03d}/{:03d} - Mean training loss {:.6f}; Mean training F1 {:.6f}; Mean validation loss {:.6f}; Mean validation F1 {:.6f}; Learning rate {:.6f};'.format(
                epc + 1, EPOCHS, np.mean(epoch_trn_losses), f1_trn, np.mean(epoch_vld_losses), f1_vld, scheduler.get_lr()[0],
            )
        )
        
        print('Validation metrics:')
        for g, m in grp_vld_metrics.items():
            print('Group {}: f1 - {:.6f}; loss - {:.6f};'.format(g, m['f1'], m['loss']))
        
        if f1_vld > np.nanmax(vld_f1s):
            torch.save(
                {
                    'epoch': epc + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'f1': f1_vld,
                    'loss': np.mean(epoch_vld_losses),
                }, 
                save_path.format(fold_number)
            )
            
            last_best = epc
            
        if epc - last_best > early_stopping:
            continue

        vld_f1s.append(f1_vld)

        scheduler.step()


N_FOLDS = 5
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)


for fld, (ndcs_trn, ndcs_vld) in enumerate(rskf.split(series_dat_all, skf_trgt)):
        
    print('################################################################')
    print('Training/validation for fold {:d}/{:d};'.format(fld+1, N_FOLDS))
    
    # setup fold data
    dat_trn, lbl_trn = series_dat_all[ndcs_trn], series_lbl_all[ndcs_trn]
    dat_vld, lbl_vld = series_dat_all[ndcs_vld], series_lbl_all[ndcs_vld]
    
    waveset_trn = Waveset(dat_trn, lbl_trn)
    waveset_vld = Waveset(dat_vld, lbl_vld)

    loader_trn = DataLoader(waveset_trn, BATCHSIZE, shuffle=True, num_workers=2, pin_memory=True)
    loader_vld = DataLoader(waveset_vld, BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    vld_loaders = {
        'vld': loader_vld,
        #'g0': DataLoader(Waveset(series_dat_all[0*unit:1*unit], series_lbl_all[0*unit:1*unit]), BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True),
        #'g1': DataLoader(Waveset(series_dat_all[1*unit:2*unit], series_lbl_all[1*unit:2*unit]), BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True),
        #'g2': DataLoader(Waveset(series_dat_all[2*unit:3*unit], series_lbl_all[2*unit:3*unit]), BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True),
        #'g3': DataLoader(Waveset(series_dat_all[3*unit:4*unit], series_lbl_all[3*unit:4*unit]), BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True),
        #'g4': DataLoader(Waveset(series_dat_all[4*unit:5*unit], series_lbl_all[4*unit:5*unit]), BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True),
        #'g5': DataLoader(Waveset(series_dat_all[5*unit:6*unit], series_lbl_all[5*unit:6*unit]), BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True),
        #'g6': DataLoader(Waveset(series_dat_all[6*unit:7*unit], series_lbl_all[6*unit:7*unit]), BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True),
        #'g7': DataLoader(Waveset(series_dat_all[7*unit:8*unit], series_lbl_all[7*unit:8*unit]), BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True),
        #'g8': DataLoader(Waveset(series_dat_all[8*unit:9*unit], series_lbl_all[8*unit:9*unit]), BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True),
        #'g9': DataLoader(Waveset(series_dat_all[9*unit:10*unit], series_lbl_all[9*unit:10*unit]), BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True),
    }
    
    # setup fold model
    mdl = RnnTRSFM_Classifier(series_dat_all.shape[-1]).to(DEVICE)
    critrn = nn.CrossEntropyLoss()
    optimzr = torch.optim.AdamW(mdl.parameters(), lr=LR)
    schdlr = torch.optim.lr_scheduler.CosineAnnealingLR(optimzr, T_max=EPOCHS, eta_min=LR/100)
    
    # run
    fold_train_validate(
        model=mdl, optimizer=optimzr, criterion=critrn,
        scheduler=schdlr, training_loader=loader_trn, validation_loaders=vld_loaders,
        fold_number=fld,
        save_path='./saved_models/RNN_TRSFM_shallow_model_new_encoded_feats_s{}_w{}_fold{:03d}_checkpoint.pth'.format(step, wndw, fld),
        early_stopping=50
    )
    


# submission = pd.read_csv('../input/sample_submission.csv', dtype={'time': str, 'open_channels': 'Int64'})
# submission_pred = np.zeros(shape=(submission.shape[0], 11))

# waveset_tst = Waveset(series_tst)
# loader_tst = DataLoader(waveset_tst, BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True)

for fld in range(15):
    print('-------- fold {:d} --------'.format(fld))
    fld_weight = torch.load('./saved_models/RNN_TRSFM_shallow_model_new_encoded_feats_s{}_w{}_fold{:03d}_checkpoint.pth'.format(step, wndw, fld))
    print('model validation loss: {:.3f}; validation f1: {:.3f};'.format(fld_weight['loss'], fld_weight['f1']))
#     mdl = WaveTRSFM_Classifier(series_trn.shape[-1]).to(DEVICE)
#     mdl.load_state_dict(fld_weight['model'])
#     mdl.eval()
#     with torch.no_grad():
#         tst_fold_prd = []
#         for tst_batch_dat in loader_tst:
#             tst_batch_prd = mdl(tst_batch_dat.to(DEVICE))
#             tst_batch_prd = tst_batch_prd.view(-1, tst_batch_prd.size(-1)).detach().cpu().numpy()
#             tst_fold_prd.append(tst_batch_prd)
            
#         submission_pred += np.concatenate(tst_fold_prd, 0)

# submission['open_channels'] = submission_pred.argmax(1)
# submission.to_csv("../submissions/sub0_waveTRSFM_basicwithnew2_cvbyentropy_meanstdnorm.csv", index=False)wavenet + 