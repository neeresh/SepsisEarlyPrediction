from matplotlib import pyplot as plt

import os

import pandas as pd
import torch
import tqdm

from utils.path_utils import project_root


def save_data_as_pt():
    def csv_to_pt(patient_files, lengths, is_sepsis, pretrain):

        all_patients = {'samples': [], 'labels': []}

        max_time_step = 336
        print(len(patient_files), len(lengths), len(is_sepsis))
        for idx, (file, length, sepsis) in tqdm.tqdm(enumerate(zip(patient_files, lengths, is_sepsis)),
                                                     desc=f"Pre-trainng ({pretrain})",
                                                     total=len(patient_files)):

            pad_width = ((0, max_time_step - len(file)), (0, 0))
            file = np.pad(file, pad_width=pad_width, mode='constant').astype(np.float32)

            if len(file) == max_time_step:
                all_patients['samples'].append(torch.from_numpy(file).unsqueeze(0))
                all_patients['labels'].append(torch.tensor(sepsis, dtype=torch.float32).unsqueeze(0))
            else:
                raise ValueError(f"Length {length} does not match length of patient {idx} with length {len(file)}")

        all_patients['samples'] = torch.cat(all_patients['samples'], dim=0)
        all_patients['labels'] = torch.cat(all_patients['labels'], dim=0)

        # Saving pre-train files as .pt
        if pretrain:
            save_path = os.path.join(project_root(), 'data', 'tl_datasets', 'pretrain', 'pretrain.pt')
        else:
            save_path = os.path.join(project_root(), 'data', 'tl_datasets', 'finetune', 'finetune.pt')

        torch.save(all_patients, save_path)

        return save_path

    pt_files = pd.read_pickle(os.path.join(project_root(), 'data', 'tl_datasets', 'final_dataset_pretrain_A.pickle'))
    pt_lengths = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'lengths_pretrain_A.txt'),
                             header=None).values
    pt_sepsis = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'is_sepsis_pretrain_A.txt'),
                            header=None).values

    pretrain_files = []
    for pdata, length in tqdm.tqdm(zip(pt_files, pt_lengths), desc="Checking files",
                                   total=len(pt_files)):
        plength = len(pdata)
        assert plength == length[0], f"{plength} doesn't match {length}"
        pretrain_files.append(pdata.drop(['PatientID', 'SepsisLabel'], axis=1))

    # Save data as .pt
    saved_path = csv_to_pt(pretrain_files, pt_lengths, pt_sepsis, pretrain=True)

    return saved_path


def plot(X_train, x_data_f, sample=20334, channel=0, num_steps=40):

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X_train[sample][channel][:num_steps].detach().numpy())

    plt.subplot(1, 2, 2)
    plt.plot(x_data_f[sample][channel][:num_steps].detach().numpy())

    plt.show()
    plt.close()

def get_model_size(model):
    def convert_to_gigabytes(input_megabyte):
        gigabyte = 1.0 / 1024
        convert_gb = gigabyte * input_megabyte
        return convert_gb

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2

    print('model size: {:.3f} GB'.format(convert_to_gigabytes(size_all_mb)))
