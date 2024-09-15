import argparse

import os


def get_args():

    parser = argparse.ArgumentParser()

    home_dir = os.getcwd()
    parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')
    parser.add_argument('--seed', default=2024, type=int, help='seed value')

    parser.add_argument('--training_mode', default='pre_train', type=str, help='pre_train, fine_tune')
    parser.add_argument('--pretrain_dataset', default='SleepEEG', type=str,
                        help='Dataset of choice: SleepEEG, FD_A, HAR, ECG')
    parser.add_argument('--target_dataset', default='Epilepsy', type=str,
                        help='Dataset of choice: Epilepsy, FD_B, Gesture, EMG')

    parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
    parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
    parser.add_argument('--subset', action='store_true', default=False, help='use the subset of datasets')
    parser.add_argument('--log_epoch', default=5, type=int, help='print loss and metrix')
    parser.add_argument('--draw_similar_matrix', default=10, type=int, help='draw similarity matrix')
    parser.add_argument('--pretrain_lr', default=0.0001, type=float, help='pretrain learning rate')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--use_pretrain_epoch_dir', default=None, type=str,
                        help='choose the pretrain checkpoint to finetune')
    parser.add_argument('--pretrain_epoch', default=10, type=int, help='pretrain epochs')
    parser.add_argument('--finetune_epoch', default=300, type=int, help='finetune epochs')

    parser.add_argument('--masking_ratio', default=0.5, type=float, help='masking ratio')
    parser.add_argument('--positive_nums', default=3, type=int, help='positive series numbers')
    parser.add_argument('--lm', default=3, type=int, help='average masked lenght')

    parser.add_argument('--finetune_result_file_name', default="finetune_result.json", type=str,
                        help='finetune result json name')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature')

    args, unknown = parser.parse_known_args()

    return args, unknown
