import datetime
import logging
import os.path
import shutil

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from tensorboardX import SummaryWriter

from utils.normalized_utility_score import normalized_utility_score
from utils.path_utils import project_root
from utils.config import nn_config
from models.lgbm_classifier import LGBMClassifier, lgb_classifier_params


# save_features_importance, save_model

class TrainModel:
    def __init__(self):
        pass

    def initialize_experiment(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        destination_path = self._setup_destination(current_time)

        data_file = "training_ffill_bfill_zeros.pickle"
        self._log(message="Datafile used: {}".format(data_file))

        training_examples = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', data_file))

        with open(os.path.join(project_root(), 'data', 'processed', 'lengths.txt')) as f:
            lengths_list = [int(length) for length in f.read().splitlines()]

        with open(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt')) as f:
            is_sepsis = [int(is_sep) for is_sep in f.read().splitlines()]

        writer = SummaryWriter(log_dir=os.path.join(project_root(), 'data', 'logs', current_time), comment='')

        # Shifting labels and generating windows
        training_examples = self.shift_labels(training_examples)
        training_examples = self.generate_windows(training_examples)

        return training_examples, lengths_list, is_sepsis, writer, destination_path

    def get_train_test_splits(self, ind_train, ind_test, training_examples, lengths_list, is_sepsis):
        x_train = [t for i, t in enumerate(training_examples) if i in ind_train]
        x_train_lens = [t for i, t in enumerate(lengths_list) if i in ind_train]
        is_sepsis_train = [t for i, t in enumerate(is_sepsis) if i in ind_train]

        x_test = [t for i, t in enumerate(training_examples) if i in ind_test]
        x_test_lens = [t for i, t in enumerate(lengths_list) if i in ind_test]
        is_sepsis_test = [t for i, t in enumerate(is_sepsis) if i in ind_test]

        # print(len(x_train), len(x_train_lens), len(is_sepsis_train), len(x_test), len(x_test), len(is_sepsis_test))

        return x_train, x_train_lens, is_sepsis_train, x_test, x_test_lens, is_sepsis_test

    def run_experiment(self, training_examples, lengths_list, is_sepsis, writer, destination_path):
        skf = StratifiedKFold(n_splits=5)
        train_scores, test_scores, y_preds_test, inds_test = [], [], [], []
        self._log(message="Config={}", value=nn_config)
        self._log(message="Config={}", value=lgb_classifier_params)
        train_scores_limit = 1000
        self._log(message="train_score_limit={}", value=train_scores_limit)

        for i, (ind_train, ind_test) in tqdm.tqdm(enumerate(skf.split(training_examples, is_sepsis)),
                                                  desc="Training Folds", total=5):
            # Getting splits
            x_train, x_train_lens, is_sepsis_train, x_test, x_test_lens, is_sepsis_test = self.get_train_test_splits(
                ind_train, ind_test, training_examples, lengths_list, is_sepsis
            )

            # Initializing the models
            model = LGBMClassifier(config=nn_config, writer=writer, eval_set=[(x_test, is_sepsis_test),
                                                                              (x_train, is_sepsis_train)])
            model.fit(x_train, x_train_lens, is_sepsis_train)
            y_pred_train, y_train = model.predict(x_train[:train_scores_limit])  # Train Predictions
            y_pred_test, y_test = model.predict(x_test, search_threshold=True)  # Test Predictions

            train_score, _, train_f_score = normalized_utility_score(targets=y_train[:train_scores_limit],
                                                                     predictions=y_pred_train[:train_scores_limit])
            test_score, _, test_f_score = normalized_utility_score(targets=y_test, predictions=y_pred_test)

            test_scores.append(test_score)
            train_scores.append(train_score)

            y_pred_test.extend(y_pred_test)
            inds_test.extend(list(ind_test))

            self._log(message="Train score: {}", value=train_score)
            self._log(message="Test score: {}", value=test_score)
            self._log(message="Train f-score: {}", value=train_f_score)
            self._log(message="Test f-score: {}", value=test_f_score)

            model.save_feature_importance(model.feature_importances_, x_train[0].columns.values,
                                          os.path.join(destination_path, 'feature_importance.png'))
            model.save_model(model, path=os.path.join(destination_path, 'lgbm_{}.bim'.format(i)))

        self._log(message="\n\nMean train MAE: {}", value=np.mean(train_scores))
        self._log(message="Mean test MAE: {}", value=np.mean(test_scores))
        self._log(message="Std train MAE: {}", value=np.std(train_scores))
        self._log(message="Std test MAE: {}", value=np.std(test_scores))

    def _log(self, message: str = '{}', value: any = None):
        print(message.format(value))
        logging.info(message.format(value))

    def _setup_destination(self, current_time):
        log_path = os.path.join(project_root(), 'data', 'logs', current_time)
        os.mkdir(log_path)
        logging.basicConfig(filename=os.path.join(log_path, current_time + '.log'), level=logging.DEBUG)
        # shutil.copy(os.path.join(project_root(), 'pytorch_classifier.py'), log_path)
        # shutil.copy(os.path.join(project_root(), 'train.py'), log_path)

        return log_path

    def shift_labels(self, examples, hours_ahead=6):
        for example in tqdm.tqdm(examples, desc='Shifting labels', total=len(examples)):
            example['SepsisLabel'] = example['SepsisLabel'].shift(hours_ahead, fill_value=0)  # Filling NaNs 0

        return examples

    def generate_windows(self, examples, window_size=6):
        windowed_examples = []

        for example in tqdm.tqdm(examples, desc='Generating windows', total=len(examples)):
            for start in range(0, len(example), window_size + 1):
                windowed_example = example.iloc[start: start + window_size].copy()
            windowed_example['SepsisLabel'] = example['SepsisLabel'].shift(-window_size, fill_value=0)  # Filling NaNs 0
            windowed_examples.append(windowed_example)

        return windowed_examples


if __name__ == '__main__':
    model = TrainModel()
    training_examples, lengths_list, is_sepsis, writer, destination_path = model.initialize_experiment()
    model.run_experiment(training_examples, lengths_list, is_sepsis, writer, destination_path)
