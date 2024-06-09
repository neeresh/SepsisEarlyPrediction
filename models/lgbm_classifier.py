import logging

import joblib
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

from utils.config import lgb_classifier_params
import lightgbm as lgb

from utils.normalized_utility_score import normalized_utility_score


class LGBMClassifier:
    def __init__(self, config, writer, eval_set: list):
        self.eval_set = eval_set
        self.feature_importances_ = None
        self.model = lgb.LGBMClassifier(**lgb_classifier_params, n_estimators=20000, n_jobs=-1)

    def fit(self, examples, length, sepsis):
        examples = pd.concat(examples)

        y_train = examples['SepsisLabel'].values
        x_train = examples.drop(['SepsisLabel'], axis=1).values

        eval_set = pd.concat(self.eval_set[0][0])  # x_test
        y_eval = eval_set['SepsisLabel'].values
        x_eval = eval_set.drop(['SepsisLabel'], axis=1).values

        self.model.fit(x_train, y_train, sample_weight=None, eval_set=[(x_eval, y_eval)]
                       # eval_metric=eval_metric
                       )
        self.feature_importances_ = self.model.feature_importances_

    def threshold_search(self, y_true, y_proba):
        # from utils import normalized_utility_score
        best_threshold = 0
        best_score = 0
        print(self.threshold_search)
        for threshold in tqdm.tqdm([i * 0.05 for i in range(20)]):
            score, _, _ = normalized_utility_score(y_true, [y > threshold for y in y_proba])
            if score > best_score:
                best_threshold = threshold
                best_score = score

        search_result = {'threshold': best_threshold, 'f1': best_score}

        return search_result

    def predict(self, examples, num_iterations=None, search_threshold=False):
        references, predictions, probas = [], [], []
        for example in examples:
            reference = example['SepsisLabel'].values
            x = example.drop(['SepsisLabel'], axis=1).values

            if num_iterations is None:
                num_iterations = self.model.best_iteration_
            proba = self.model.predict_proba(x, num_iterations=num_iterations)[:, 1]

            references.append(reference)
            probas.append(proba)

        if search_threshold:
            search_result = self.threshold_search(references, probas)
            thr = search_result['threshold']
        else:
            thr = 0.5

        logging.info("thr: {}".format(thr))
        print(f"Threshold: {thr}")

        for proba in probas:
            prediction = np.where(proba > thr, 1, 0)
            predictions.append(prediction)

        return predictions, references

    def save_model(self, model, path):
        joblib.dump(model.model, path)

    def load_model(self, path):
        return joblib.load(path)

    def save_feature_importance(self, features_importance, features_names, path):
        features_importance = pd.DataFrame(sorted(zip(features_importance, features_names)),
                                           columns=['Value', 'Feature'])

        plt.figure(figsize=(20, 10))
        sns.barplot(x="Value", y="Feature", data=features_importance.sort_values(by="Value", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(path)

    def eval_metric(self, labels, preds):
        labels = labels.reshape(1, -1)
        preds = np.where(preds > 0.5, 1, 0)
        preds = preds.reshape(1, -1)

        return 'normalized_utility_score', normalized_utility_score(labels, preds)[0], True
