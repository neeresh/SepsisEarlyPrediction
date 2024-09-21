import numpy as np
import os

import torch
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from torch import nn
import torch.nn.functional as F

from models.tfc.config import Config
from models.tfc.dataloader import generate_freq, data_generator
from models.tfc.loss import NTXentLoss_poly
from models.tfc.model import TFC, target_classifier
from utils.model_size import get_model_size
from utils.path_utils import project_root
from utils.pretrain_utils.get_args import get_args


def model_pretrain(model, model_optimizer, train_loader, config, device):
    total_loss = []
    model.train()
    model.to(device)

    global loss, loss_t, loss_f, l_TF, loss_c, data_test, data_f_test

    model_optimizer.zero_grad()

    for batch_idx, (data, labels, aug1, data_f, aug1_f) in tqdm.tqdm(enumerate(train_loader), desc='Pre-training model',
                                                                     total=len(train_loader)):
        data, labels = data.float().to(device), labels.long().to(device)
        aug1 = aug1.float().to(device)
        data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)

        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f, stage='train')
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f, stage='train')

        """Compute Pre-train loss"""
        """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
        nt_xent_criterion = NTXentLoss_poly(device, config.batch_size, config.Context_Cont.temperature,
                                            config.Context_Cont.use_cosine_similarity)  # device, 128, 0.2, True

        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        l_TF = nt_xent_criterion(z_t, z_f)  # this is the initial version of TF loss

        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug,
                                                                                                            z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        lam = 0.2
        loss = lam * (loss_t + loss_f) + l_TF

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

    print('Pretraining: overall loss:{}, l_t: {}, l_f:{}, l_c:{}'.format(loss, loss_t, loss_f, l_TF))

    ave_loss = torch.tensor(total_loss).mean()

    return ave_loss


def train(model, args, config, train_loader, device='cuda'):
    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                       weight_decay=config.weight_decay)

    experiment_log_dir = os.path.join(project_root(), 'results', 'tfc')
    os.makedirs(os.path.join(experiment_log_dir, f"saved_models"), exist_ok=True)

    log_file_path = 'pretrain_tfc.txt'
    with open(log_file_path, 'a') as log_file:
        for epoch in range(1, config.pretrain_epoch + 1):
            train_loss = model_pretrain(model, model_optimizer, train_loader, config, device)
            log_text = f'Pre-training Epoch: {epoch}\t Train Loss: {train_loss:.4f}\t'

            print(log_text)
            log_file.write(log_text)

            chkpoint = {'epoch': epoch, 'train_loss': train_loss, 'model_state_dict': model.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_ep{epoch}.pt'))


def build_model(args, lr, configs, device='cuda', chkpoint=None):

    model = TFC(configs).to(device)

    pretrained_dict = chkpoint
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    classifier = target_classifier(configs).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                       weight_decay=configs.weight_decay)
    classifier_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr,
                                            betas=(configs.beta1, configs.beta2), weight_decay=configs.weight_decay)
    model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    return model, classifier, model_optimizer, classifier_optimizer, model_scheduler


def model_finetune(model, model_optimizer, val_dl, config, classifier=None, classifier_optimizer=None, device='cuda'):

    global labels, pred_numpy, fea_concat_flat

    model.train()
    classifier.train()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    
    feas = np.array([])

    for data, labels, aug1, data_f, aug1_f in tqdm.tqdm(val_dl, desc="Fine-tuning model", total=len(val_dl)):

        # print('Fine-tuning: {} of target samples'.format(labels.shape[0]))
        data, labels = data.float().to(device), labels.long().to(device)
        data_f = data_f.float().to(device)
        aug1 = aug1.float().to(device)
        aug1_f = aug1_f.float().to(device)
        
        """if random initialization:"""
        model_optimizer.zero_grad()  # The gradients are zero, but the parameters are still randomly initialized.
        classifier_optimizer.zero_grad()  # the classifier is newly added and randomly initialized
        
        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f, stage='train')
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f, stage='train')
        nt_xent_criterion = NTXentLoss_poly(device, config.batch_size, config.Context_Cont.temperature,
                                            config.Context_Cont.use_cosine_similarity)
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        l_TF = nt_xent_criterion(z_t, z_f)

        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), \
            nt_xent_criterion(z_t_aug, z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)  #

        """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test."""
        fea_concat = torch.cat((z_t, z_f), dim=1)
        predictions = classifier(fea_concat)

        fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
        loss_p = criterion(predictions, labels)

        lam = 0.1
        loss = loss_p + l_TF + lam * (loss_t + loss_f)

        acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
        onehot_label = F.one_hot(labels)
        pred_numpy = predictions.detach().cpu().numpy()

        try:
            auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr")
        except:
            auc_bs = np.float32(0)

        try:
            prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)
        except:
            prc_bs = 0.0

        total_acc.append(acc_bs)
        total_auc.append(auc_bs)
        total_prc.append(prc_bs)
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

        # training_mode != 'pre_train'
        pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
        outs = np.append(outs, pred.cpu().numpy())
        trgs = np.append(trgs, labels.data.cpu().numpy())
        feas = np.append(feas, fea_concat_flat.data.cpu().numpy())

    feas = feas.reshape([len(trgs), -1])  # produce the learned embeddings

    labels_numpy = labels.detach().cpu().numpy()
    pred_numpy = np.argmax(pred_numpy, axis=1)
    precision = precision_score(labels_numpy, pred_numpy, average='macro', )
    recall = recall_score(labels_numpy, pred_numpy, average='macro', )

    F1 = f1_score(labels_numpy, pred_numpy, average='macro', )

    ave_loss = torch.tensor(total_loss).mean()
    ave_acc = torch.tensor(total_acc).mean()
    ave_auc = torch.tensor(total_auc).mean()
    ave_prc = torch.tensor(total_prc).mean()

    print(' Finetune: loss = %.4f| Acc=%.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f| AUROC=%.4f | AUPRC = %.4f'
          % (ave_loss, ave_acc * 100, precision * 100, recall * 100, F1 * 100, ave_auc * 100, ave_prc * 100))

    return model, classifier, ave_loss, ave_acc, ave_auc, ave_prc, feas, trgs, F1


def finetune(finetune_loader, args, config, chkpoint):

    ft_model, ft_classifier, ft_model_optimizer, ft_classifier_optimizer, ft_scheduler = build_model(
        args, args.lr, config, device='cuda', chkpoint=chkpoint)

    experiment_log_dir = os.path.join(project_root(), 'results', 'tfc')
    os.makedirs(os.path.join(experiment_log_dir, f"saved_models"), exist_ok=True)

    log_file_path = 'finetune_tfc.txt'
    with open(log_file_path, 'a') as log_file:
        for epoch in range(1, config.finetune_epoch + 1):
            model, classifier, ave_loss, ave_acc, ave_auc, ave_prc, feas, trgs, F1 = model_finetune(
                ft_model, ft_model_optimizer, finetune_loader, config, classifier=ft_classifier,
                classifier_optimizer=ft_classifier_optimizer)

            ft_scheduler.step(ave_loss)

            log_text = (f"Fine-tuning ended ....\n"
                        f"{'=' * 100}\n"
                        f"epoch: {epoch}\n"
                        f"valid_auc: {ave_auc} valid_prc: {ave_prc} F1: {F1}\n"
                        f"valid_loss: {ave_loss} valid_acc: {ave_acc}\n"
                        f"{'=' * 100}\n"
                        )

            print(log_text)
            log_file.write(log_text)

            # Saving feature encoder and classifier after finetuning for testing.
            chkpoint = {'seed': args.seed, 'epoch': epoch, 'train_loss': ave_loss, 'model_state_dict': model.state_dict(),
                        'classifier': ft_classifier.state_dict()}

            torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models/", f'finetune_ep{epoch}.pt'))


if __name__ == '__main__':

    pretrain_exp = False

    # Gathering args and configs
    args, unknown = get_args()
    configs = Config()

    # Model
    model = TFC(configs=configs)
    
    # Model size
    get_model_size(model)

    # Gather datasets
    tl_datasets = os.path.join(project_root(), 'data', 'tl_datasets')
    
    if pretrain_exp:
        pretrain = torch.load(os.path.join(tl_datasets, 'pretrain', 'pretrain.pt'))
        train_loader = data_generator(pretrain, configs)

        train(model, args, configs, train_loader)

    else:
        """Fine-tuning and Test"""
        finetune_dataset = torch.load(os.path.join(tl_datasets, 'finetune', 'finetune.pt'))
        finetune_loader = data_generator(finetune_dataset, configs)

        chkpoint = torch.load(os.path.join(project_root(), 'results', 'tfc', 'saved_models', 'ckp_ep20.pt'))[
            'model_state_dict']
        
        finetune(finetune_loader, args, configs, chkpoint)
