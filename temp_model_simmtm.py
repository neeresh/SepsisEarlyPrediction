import numpy as np
import os

import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from models.simmtm.config import Config
from models.simmtm.masking import data_transform_masked4cl
from models.simmtm.model_mlp import TFC, target_classifier

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
    model = TFC(configs, args).to(device)

    pretrained_dict = chkpoint
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    classifier = target_classifier(configs).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(configs.beta1, configs.beta2), weight_decay=0)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=lr,
                                            betas=(configs.beta1, configs.beta2),
                                            weight_decay=0)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=args.finetune_epoch)

    return model, classifier, model_optimizer, classifier_optimizer, model_scheduler


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
    os.makedirs(os.path.join(experiment_log_dir, f"gtn_mlp"), exist_ok=True)

    best_performance = None
    for epoch in range(1, config.pretrain_epoch + 1):
        total_loss, total_cl_loss, total_rb_loss = model_pretrain(model=model, model_optimizer=model_optimizer,
                                                                  model_scheduler=model_scheduler,
                                                                  train_loader=train_loader,
                                                                  configs=config, args=args, device='cuda')
        print(f'Pre-training Epoch: {epoch + 1}\t Train Loss: {total_loss:.4f}\t '
              f'CL Loss: {total_cl_loss:.4f}\t RB Loss: {total_rb_loss:.4f}\n')

        if epoch % 2 == 0:
            chkpoint = {'seed': args.seed, 'epoch': epoch, 'train_loss': total_loss,
                        'model_state_dict': model.state_dict()}
            torch.save(chkpoint,
                       os.path.join(experiment_log_dir, f"gtn_mlp/train_on_finetune/", f'ckp_ep{epoch}.pt'))


def finetune(finetune_loader, args, config, chkpoint):
    ft_model, ft_classifier, ft_model_optimizer, ft_classifier_optimizer, ft_scheduler = build_model(
        args, args.lr, config, device='cuda', chkpoint=chkpoint)

    experiment_log_dir = os.path.join(project_root(), 'results', 'simmtm')
    os.makedirs(os.path.join(experiment_log_dir, f"gtn_mlp"), exist_ok=True)

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

        if ep % 2 == 0:
            # Saving feature encoder and classifier after finetuning for testing.
            chkpoint = {'seed': args.seed, 'epoch': ep, 'train_loss': valid_loss,
                        'model_state_dict': ft_model.state_dict(),
                        'classifier': classifier.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, f"gtn_mlp/", f'finetune_ep{ep}.pt'))


if __name__ == '__main__':

    pretrain_exp = True

    # Gathering args and configs
    args, unknown = get_args()
    config = Config()

    # Model
    model = TFC(configs=config, args=args)

    # Model size
    get_model_size(model)

    # Gathering dataset (test psv files are also saved here)
    # pt_train, ft, test = get_pretrain_finetune_test_datasets()
    ptpath = os.path.join(project_root(), 'data', 'tl_datasets', 'pretrain', 'pretrain.pt')
    ftpath = os.path.join(project_root(), 'data', 'tl_datasets', 'finetune', 'finetune.pt')
    testpath = os.path.join(project_root(), 'data', 'tl_datasets', 'test', 'psv_files')

    pt_train = torch.load(ptpath)
    ft = torch.load(ftpath)

    # ft_dataset = Load_Dataset(ft, TSlength_aligned=336, training_mode='pretrain')
    # train_loader = DataLoader(dataset=ft_dataset, batch_size=config.batch_size, shuffle=True,
    #                           drop_last=True, num_workers=4)
    # train(model, args, config, train_loader)

    if pretrain_exp:

        # Train set
        pt_dataset = Load_Dataset(pt_train, TSlength_aligned=336, training_mode='pretrain')
        train_loader = DataLoader(dataset=pt_dataset, batch_size=config.batch_size, shuffle=True,
                                  drop_last=True, num_workers=4)

        # Training
        train(model, args, config, train_loader)

        # # Saving test set (for evaluation)
        # destination_path = os.path.join(project_root(), 'data', 'test_data', 'simmtm')
        # torch.save(test, destination_path + '/test.pt')

    else:

        # Gathering dataset
        finetune_dataset = Load_Dataset(ft, TSlength_aligned=336, training_mode='finetune')
        finetune_loader = DataLoader(finetune_dataset, batch_size=config.batch_size, shuffle=True,
                                     drop_last=True, num_workers=4)

        # Fine tuning
        chkpoint = torch.load(os.path.join(project_root(), 'results', 'simmtm', 'gtn_mlp', 'ckp_ep9.pt'))[
            'model_state_dict']
        finetune(finetune_loader, args, config, chkpoint)

"""
Utility Score: 0.26
class Config(object):
    def __init__(self):

        # GTN
        self.d_model = 512
        self.d_hidden = 1024
        self.q = 8
        self.v = 8
        self.h = 8
        self.N = 8
        self.dropout = 0.2
        self.pe = True
        self.mask = True
        self.lr = 1e-4
        self.batch_size = 16
        self.num_epochs = 20

        self.device = 'cuda'

        # Dataset params
        self.d_input = 336
        self.d_channel = 40
        self.d_output = 2

        # pre-train configs
        self.pretrain_epoch = 10
        self.finetune_epoch = 20

        # fine-tune configs
        self.num_classes_target = 2

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-8  # 3e-4
        self.lr_f = self.lr

        # data parameters
        self.drop_last = True
        self.batch_size = 32

        # self.target_batch_size = 32  # the size of target dataset (the # of samples used to fine-tune).

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
"""
