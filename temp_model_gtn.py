import numpy as np
import os
import pandas as pd

import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from models.gtn.config import Config
from models.gtn.transformer import Transformer
from utils.loader import DatasetWithPadding

# from models.simmtm.masking import data_transform_masked4cl
# from models.simmtm.model import TFC, target_classifier

from utils.model_size import get_model_size
from utils.path_utils import project_root

from utils.pretrain_utils.data import get_pretrain_finetune_test_datasets, csv_to_pt, Load_Dataset
from utils.pretrain_utils.get_args import get_args

import torch.nn.functional as F

from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, f1_score, recall_score)


def model_pretrain(model, model_optimizer, model_scheduler, train_loader, configs, args, device):
    total_loss = []
    total_cl_loss = []
    total_rb_loss = []

    model.to(device)
    model.train()
    for batch_idx, (data, labels) in tqdm.tqdm(enumerate(train_loader), desc="Pre-training model",
                                               total=len(train_loader)):
        model_optimizer.zero_grad()

        data_masked_m, mask = data_transform_masked4cl(data, args.masking_ratio, args.lm, args.positive_nums)
        data_masked_om = torch.cat([data, data_masked_m], 0)

        del data_masked_m, mask, data, labels  # To save space

        # data, labels, data_masked_om = data.float().to(device), labels.float().to(device), data_masked_om.float().to(
        #     device)

        data_masked_om = data_masked_om.float().to(device)

        # Produce embeddings of original and masked samples
        loss, loss_cl, loss_rb = model(stage='train', x_in_t=data_masked_om, pre_train=True)

        loss.backward()
        model_optimizer.step()

        total_loss.append(loss.item())
        total_cl_loss.append(loss_cl.item())
        total_rb_loss.append(loss_rb.item())

    total_loss = torch.tensor(total_loss).mean()
    total_cl_loss = torch.tensor(total_cl_loss).mean()
    total_rb_loss = torch.tensor(total_rb_loss).mean()

    model_scheduler.step()

    return total_loss, total_cl_loss, total_rb_loss


def build_model(args, lr, configs, device='cuda', chkpoint=None):
    # DONE
    model = Transformer(d_model=config.d_model, d_input=config.d_input,
                        d_channel=config.d_channel, d_output=config.d_output,
                        d_hidden=config.d_hidden, q=config.q, v=config.v,
                        h=config.h, N=config.N, device=config.device,
                        dropout=config.dropout, pe=config.pe, mask=config.mask)

    pretrained_dict = chkpoint
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(configs.beta1, configs.beta2), weight_decay=0)
    classifier_optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                            betas=(configs.beta1, configs.beta2),
                                            weight_decay=0)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=args.finetune_epoch)

    return model, model_optimizer, classifier_optimizer, model_scheduler


def model_finetune(model, val_dl, device, model_optimizer, model_scheduler, classifier=None, classifier_optimizer=None):
    model.train()  # Not freezing the pretrained layers
    # model.eval() # Freezing the pretrained layers
    classifier.train()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    for idx, (data, labels) in tqdm.tqdm(enumerate(val_dl), desc="Fine-tuning model", total=len(val_dl)):
        model_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        data, labels = data.float().to(device), labels.long().to(device)

        # Produce embeddings
        h, z = model(stage='train', x_in_t=data, pre_train=False)

        # Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test
        fea_concat = h

        predictions = classifier(fea_concat)
        fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
        loss = criterion(predictions, labels)  # torch.Size([32, 2]) torch.Size([32, 1]) <- ERROR

        acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
        onehot_label = F.one_hot(labels)
        pred_numpy = predictions.detach().cpu().numpy()

        try:
            auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr")
        except:
            auc_bs = 0.0

        try:
            prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)
        except:
            prc_bs = 0.0

        total_acc.append(acc_bs)

        if auc_bs != 0:
            total_auc.append(auc_bs)
        if prc_bs != 0:
            total_prc.append(prc_bs)
        total_loss.append(loss.item())

        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

        pred = predictions.max(1, keepdim=True)[1]
        outs = np.append(outs, pred.cpu().numpy())
        trgs = np.append(trgs, labels.data.cpu().numpy())

    labels_numpy = labels.detach().cpu().numpy()
    pred_numpy = np.argmax(pred_numpy, axis=1)
    F1 = f1_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))

    total_loss = torch.tensor(total_loss).mean()  # average loss
    total_acc = torch.tensor(total_acc).mean()  # average acc
    total_auc = torch.tensor(total_auc).mean()  # average auc
    total_prc = torch.tensor(total_prc).mean()

    # model_scheduler.step(total_loss)
    model_scheduler.step()

    return model, classifier, total_loss, total_acc, total_auc, total_prc, fea_concat_flat, trgs, F1


def train(model, args, config, train_loader):
    params_group = [{'params': model.parameters()}]
    model_optimizer = torch.optim.Adam(params_group, lr=args.pretrain_lr,
                                       betas=(config.beta1, config.beta2),
                                       weight_decay=0)

    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=args.pretrain_epoch)

    experiment_log_dir = os.path.join(project_root(), 'results', 'simmtm')
    os.makedirs(os.path.join(experiment_log_dir, f"saved_models"), exist_ok=True)

    best_performance = None
    for epoch in range(1, config.pretrain_epoch + 1):
        total_loss, total_cl_loss, total_rb_loss = model_pretrain(model=model, model_optimizer=model_optimizer,
                                                                  model_scheduler=model_scheduler,
                                                                  train_loader=train_loader,
                                                                  configs=config, args=args, device='cuda')
        print(f'Pre-training Epoch: {epoch + 1}\t Train Loss: {total_loss:.4f}\t '
              f'CL Loss: {total_cl_loss:.4f}\t RB Loss: {total_rb_loss:.4f}\n')

        chkpoint = {'seed': args.seed, 'epoch': epoch, 'train_loss': total_loss, 'model_state_dict': model.state_dict()}
        torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models/", f'ckp_ep{epoch}.pt'))


def finetune(finetune_loader, args, config, chkpoint):

    ft_model, ft_classifier, ft_model_optimizer, ft_classifier_optimizer, ft_scheduler = build_model(
        args, args.lr, config, device='cuda', chkpoint=chkpoint)

    experiment_log_dir = os.path.join(project_root(), 'results', 'simmtm')
    os.makedirs(os.path.join(experiment_log_dir, f"saved_models"), exist_ok=True)

    for ep in range(1, config.finetune_epoch + 1):
        ft_model, classifier, valid_loss, valid_acc, valid_auc, valid_prc, emb_finetune, label_finetune, F1 = model_finetune(
            ft_model, finetune_loader, 'cuda', ft_model_optimizer, ft_scheduler, classifier=ft_classifier,
            classifier_optimizer=ft_classifier_optimizer)

        print("Fine-tuning ended ....")
        print(f"epoch: {ep}")
        print("=" * 100)
        print(f"valid_auc: {valid_auc} valid_prc: {valid_prc} F1: {F1}")
        print(f"valid_loss: {valid_loss} valid_acc: {valid_acc}")
        print("=" * 100)

        # Saving feature encoder and classifier after finetuning for testing.
        chkpoint = {'seed': args.seed, 'epoch': ep, 'train_loss': valid_loss, 'model_state_dict': ft_model.state_dict(),
                    'classifier': classifier.state_dict()}
        torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models/", f'finetune_ep{ep}.pt'))


if __name__ == '__main__':

    pretrain_exp = True

    # Gathering args and configs
    config = Config()
    args = get_args()

    # Model
    model = Transformer(d_model=config.d_model, d_input=config.d_input,
                        d_channel=config.d_channel, d_output=config.d_output,
                        d_hidden=config.d_hidden, q=config.q, v=config.v,
                        h=config.h, N=config.N, device=config.device,
                        dropout=config.dropout, pe=config.pe, mask=config.mask)

    # Model size
    get_model_size(model)

    # Get pretrain, finetune datasets from .../tl_datasets/pretrain and .../tl_datasets/finetune
    # Pre-training
    pt_pickle = pd.read_pickle(os.path.join(project_root(), 'data', 'tl_datasets', 'final_dataset_pretrain_A.pickle'))

    pt_files = []
    for pdata in tqdm.tqdm(pt_pickle, desc='Preparing pretraining dataset', total=len(pt_pickle)):
        pt_files.append(pdata)

    pt_lengths = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'lengths_pretrain_A.txt'),
                             header=None).values.squeeze()
    pt_sepsis = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'is_sepsis_pretrain_A.txt'),
                            header=None).values.squeeze()

    print(f"Pre-training files: {len(pt_files), len(pt_sepsis), len(pt_lengths)}")

    # Fine-tuning
    ft_files = torch.load(os.path.join(project_root(), 'data', 'tl_datasets', 'finetune', 'finetune.pt'))['samples']
    ft_lengths = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'finetune', 'lengths.txt'),
                             header=None).values.squeeze()
    ft_sepsis = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'finetune', 'is_sepsis.txt'),
                            header=None).values.squeeze()

    if pretrain_exp:

        # Train set
        pt_train = DatasetWithPadding(training_examples_list=pt_files, lengths_list=pt_lengths,
                                      is_sepsis=pt_sepsis)
        train_loader = DataLoader(pt_train, batch_size=config.batch_size, shuffle=False,
                                  num_workers=config.num_workers)

        # Training
        train(model, args, config, train_loader)

    else:
        # Handle data correctly. ft_files is already a tensor of (4000, 336, 40).
        # Convert into loader with batch size and finetune.

        # Gathering dataset
        ft = DatasetWithPadding(training_examples_list=ft_files, lengths_list=ft_lengths,
                                is_sepsis=ft_sepsis)
        finetune_loader = DataLoader(ft, batch_size=config.batch_size, shuffle=False,
                                     num_workers=config.num_workers)

        # Fine tuning
        chkpoint = torch.load(os.path.join(project_root(), 'results', 'simmtm', 'saved_models', 'ckp_ep9.pt'))['model_state_dict']
        finetune(finetune_loader, args, config, chkpoint)
