import logging

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.gtn.loss import Myloss
from models.gtn.transformer import Transformer
from utils.config import pretrain_params
from utils.path_utils import project_root
from utils.pretrain_utils.dataset_processes import MyDataset

from time import time
import os

from utils.pretrain_utils.random_seed import setup_seed

setup_seed(30)
reslut_figure_path = 'result_figure'
save_model_path = os.path.join(project_root(), 'data', 'pretrain_datasets', 'saved_models')

# path = '/localscratch/neeresh/data/physionet2019/data/pretrain_datasets/AUSLAN/AUSLAN.mat'  # length=1140  input=136 channel=22 output=95
# path = '/localscratch/neeresh/data/physionet2019/data/pretrain_datasets/CharacterTrajectories/CharacterTrajectories.mat'  # length=300 input=205 channel=3 output=20
# path = '/localscratch/neeresh/data/physionet2019/data/pretrain_datasets/CMUsubject16/CMUsubject16.mat'  # length=29  input=580 channel=62 output=2
# path = '/localscratch/neeresh/data/physionet2019/data/pretrain_datasets/ECG/ECG.mat'  # length=100  input=152 channel=2 output=2
# path = '/localscratch/neeresh/data/physionet2019/data/pretrain_datasets/JapaneseVowels/JapaneseVowels.mat'  # length=270  input=29 channel=12 output=9
# path = '/localscratch/neeresh/data/physionet2019/data/pretrain_datasets/Libras/Libras.mat'  # length=180  input=45 channel=2 output=15
# path = '/localscratch/neeresh/data/physionet2019/data/pretrain_datasets/UWave/UWave.mat'  # length=4278  input=315 channel=3 output=8
# path = '/localscratch/neeresh/data/physionet2019/data/pretrain_datasets/KickvsPunch/KickvsPunch.mat'  # length=10  input=841 channel=62 output=2
# path = '/localscratch/neeresh/data/physionet2019/data/pretrain_datasets/NetFlow/NetFlow.mat'  # length=803  input=997 channel=4 output=只有1和13
path = '/localscratch/neeresh/data/physionet2019/data/pretrain_datasets/ArabicDigits/ArabicDigits.mat'  # length=6600  input=93 channel=13 output=10

# path = '/localscratch/neeresh/data/physionet2019/data/pretrain_datasets/PEMS/PEMS.mat'  # length=267, input=144, channel=963, output=7
# path = '/localscratch/neeresh/data/physionet2019/data/pretrain_datasets/Wafer/Wafer.mat'  # length=298 input=198 channel=6, output=2
# path = '/localscratch/neeresh/data/physionet2019/data/pretrain_datasets/WalkvsRun/WalkvsRun.mat'  # length=28  input=1918 channel=62 output=2

save_model_name = os.path.basename(path)
save_model_name = os.path.splitext(save_model_name)[0]
save_model_name = save_model_name.lower()

test_interval = 5
draw_key = 1
file_name = path.split('\\')[-1][0:path.split('\\')[-1].index('.')]

EPOCH = pretrain_params['num_epochs']
BATCH_SIZE = pretrain_params['batch_size']
LR = pretrain_params['lr']
DEVICE = 'cuda'
print(f'use device: {DEVICE}')

d_model = pretrain_params['d_model']
d_hidden = pretrain_params['d_hidden']
q = pretrain_params['q']
v = pretrain_params['v']
h = pretrain_params['h']
N = pretrain_params['N']
dropout = pretrain_params['dropout']
pe = pretrain_params['pe']
mask = pretrain_params['mask']
optimizer_name = 'Adagrad'

train_dataset = MyDataset(path, 'train')
test_dataset = MyDataset(path, 'test')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

DATA_LEN = train_dataset.train_len
d_input = train_dataset.input_len
d_channel = train_dataset.channel_len
d_output = train_dataset.output_len

print('data structure: [lines, timesteps, features]')
print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'mytest data size: [{train_dataset.test_len, d_input, d_channel}]')
print(f'Number of classes: {d_output}')

net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)

loss_function = Myloss()
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)

correct_on_train = []
correct_on_test = []

loss_list = []
time_cost = 0


def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
        if flag == 'test_set':
            correct_on_test.append(round((100 * correct / total), 2))
        elif flag == 'train_set':
            correct_on_train.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))

        return round((100 * correct / total), 2)


def train():
    net.train()
    max_accuracy = 0
    pbar = tqdm(total=EPOCH)
    for index in range(EPOCH):
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')

            loss = loss_function(y_pre, y.to(DEVICE))

            # print(f'Epoch:{index + 1}:\t\tloss:{loss.item()}')
            loss_list.append(loss.item())

            loss.backward()

            optimizer.step()

        if ((index + 1) % test_interval) == 0:
            current_accuracy = test(test_dataloader)
            test(train_dataloader, 'train_set')
            print(f'Current maximum accuracy\tTest:{max(correct_on_test)}%\t Training set:{max(correct_on_train)}%')

            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                print(save_model_path)
                torch.save(net.state_dict(), f'{save_model_path}/{save_model_name}.pkl')

        pbar.update()

    os.rename(f'{save_model_path}/{save_model_name}.pkl',
              f'{save_model_path}/{save_model_name}_{int(max_accuracy)}.pkl')


if __name__ == '__main__':
    train()
