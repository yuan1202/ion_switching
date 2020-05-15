import numpy as np
import torch

def fold_train_validate(
    model, optimizer, criterion, scheduler, scorer,
    training_loader, validation_loaders, fold_id,
    num_epochs,
    save_path='./checkpoint.pth',
    early_stopping=15,
    instant_progress=False, mode='classification',
    DEVICE='cuda:0',
):
    assert isinstance(validation_loaders, dict)
    
    save_path = save_path.split('.')
    name = '.'.join(save_path[:-1])
    ext = save_path[-1]
    save_path = name + '_fold{}.'.format(fold_id) + ext

    trn_losses = [np.nan]
    vld_losses = [np.nan]
    vld_f1s = [np.nan]
    
    last_best = 0

    for epc in range(num_epochs):
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
            if mode == 'classification':
                trn_batch_prd = trn_batch_prd.view(-1, trn_batch_prd.size(-1))
            else:
                trn_batch_prd = trn_batch_prd.view(-1)
            trn_batch_lbl = trn_batch_lbl.view(-1)
            loss = criterion(trn_batch_prd, trn_batch_lbl)
            loss.backward()
            optimizer.step()

            epoch_trn_losses.append(loss.item())
            epoch_trn_lbls.append(trn_batch_lbl.detach().cpu().numpy())
            if mode == 'classification':
                epoch_trn_prds.append(trn_batch_prd.detach().cpu().numpy())
            else:
                epoch_trn_prds.append(
                    np.round(
                        np.clip(trn_batch_prd.detach().cpu().numpy().squeeze(), 0, 10)
                    ).astype(int)
                )
            
            if instant_progress:
                print(
                    'Epoch {:03d}/{:03d} - Training batch {:04d}/{:04d}: Training loss {:.6f};'.format(
                        epc + 1, num_epochs, i + 1, len(training_loader), epoch_trn_losses[-1],
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
                    if mode == 'classification':
                        vld_batch_prd = vld_batch_prd.view(-1, vld_batch_prd.size(-1))
                    else:
                        vld_batch_prd = vld_batch_prd.view(-1)
                    vld_batch_lbl = vld_batch_lbl.view(-1)
                    loss = criterion(vld_batch_prd, vld_batch_lbl)

                    epoch_grp_vld_losses.append(loss.item())
                    epoch_grp_vld_lbls.append(vld_batch_lbl.detach().cpu().numpy())
                    if mode == 'classification':
                        epoch_grp_vld_prds.append(
                            vld_batch_prd.detach().cpu().numpy().argmax(1)
                        )
                    else:
                        epoch_grp_vld_prds.append(
                            np.round(
                                np.clip(vld_batch_prd.detach().cpu().numpy().squeeze(), 0, 10)
                            ).astype(int)
                        )
                    
                    epoch_vld_losses.append(epoch_grp_vld_losses[-1])
                    epoch_vld_lbls.append(epoch_grp_vld_lbls[-1])
                    epoch_vld_prds.append(epoch_grp_vld_prds[-1])
                    
                epoch_grp_vld_lbls = np.concatenate(epoch_grp_vld_lbls, axis=0)
                epoch_grp_vld_prds = np.concatenate(epoch_grp_vld_prds, axis=0)
                
                grp_f1_vld = scorer(
                    epoch_grp_vld_lbls.astype(int), 
                    epoch_grp_vld_prds,
                    labels=list(range(int(np.min(epoch_grp_vld_lbls)), int(np.max(epoch_grp_vld_lbls)))), 
                    average='macro'
                )
                
                grp_vld_metrics.update({grp: {scorer.__name__: grp_f1_vld, 'loss': np.mean(epoch_grp_vld_losses)}})
                
                if instant_progress:
                    print('Validation progress: {:03d}/{:03d} group done;'.format(i + 1, len(validation_loaders)), end='\r')

        # ------ epoch end ------
        f1_trn = scorer(
            np.concatenate(epoch_trn_lbls, axis=0).astype(int), 
            np.concatenate(epoch_trn_prds, axis=0),
            labels=list(range(11)), 
            average='macro'
        )
        f1_vld = scorer(
            np.concatenate(epoch_vld_lbls, axis=0).astype(int), 
            np.concatenate(epoch_vld_prds, axis=0),
            labels=list(range(11)), 
            average='macro'
        )


        print(
            'Epoch {:03d}/{:03d} - Mean training loss {:.6f}; Mean training {} {:.6f}; Mean validation loss {:.6f}; Mean validation {} {:.6f}; Learning rate {:.6f};'.format(
                epc + 1, num_epochs, np.mean(epoch_trn_losses), scorer.__name__, f1_trn, np.mean(epoch_vld_losses), scorer.__name__, f1_vld, scheduler.get_lr()[0],
            )
        )
        
        print('Validation metrics:')
        for g, m in grp_vld_metrics.items():
            print('Group {}: {} - {:.6f}; loss - {:.6f};'.format(g, scorer.__name__, m[scorer.__name__], m['loss']))
        
        if f1_vld > np.nanmax(vld_f1s):
            torch.save(
                {
                    'epoch': epc + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    scorer.__name__: f1_vld,
                    'loss': np.mean(epoch_vld_losses),
                }, 
                save_path
            )
            
            last_best = epc
            
        if epc - last_best > early_stopping:
            break

        vld_f1s.append(f1_vld)

        scheduler.step()