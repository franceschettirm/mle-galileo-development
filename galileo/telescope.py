"""docstring"""
from typing import List, Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from lightgbm import LGBMClassifier
from category_encoders import CatBoostEncoder
import warnings

warnings.filterwarnings("ignore")


class FeatureSelectionConstructor:
    pass


class IterativeFeatureSelection:
    """docstring"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        test_size: float,
        target_name: str,
        sample_size: Union[int, float],
        threshold: float,
        metric: str,
    ) -> None:
        self.dataframe = dataframe
        self.test_size = test_size
        self.target_name = target_name
        self.sample_size = sample_size
        self.threshold = threshold
        self.metric = metric

    def _get_features(self) -> List[str]:
        features = list(
            filter(lambda x: x != self.target_name, self.dataframe.columns.values)
        )
        return features

    def _make_sample(self):
        if type(self.sample_size) == int:
            print(
                f"Sampling data with a sample size of {self.sample_size} observations"
            )
            sample = self.dataframe.sample(n=self.sample_size, random_state=42)
            return sample
        else:
            if self.sample_size >= 1:
                raise ValueError("Argument sample_size must be between 0 and 1")
            print(f"Sampling data with a sample size of {self.sample_size}%")
            sample = self.dataframe.sample(frac=self.sample_size, random_state=42)
            return sample

    def _split_sample_data(
        self, sample: pd.DataFrame, features: List[str], test_size: float
    ) -> Union[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        print(f"Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            sample[features],
            sample[self.target_name],
            test_size=test_size,
            stratify=sample[self.target_name],
        )

        return X_train, X_test, y_train, y_test

    def _check_has_categorical(self) -> int:
        return len(self.dataframe.select_dtypes(include="object").columns)

    def _build_pipeline(self):
        if self._check_has_categorical() > 0:
            return Pipeline(
                [
                    ("encoder", CatBoostEncoder(random_state=42)),
                    ("estimator", LGBMClassifier(random_state=42, verbosity=-1)),
                ]
            )
        else:
            return LGBMClassifier(random_state=42, verbosity=-1)

    @staticmethod
    def _get_feature_sample(features: List[str], size: float):
        return np.random.choice(features, size=size, replace=False)

    @staticmethod
    def _return_proba_from_model(
        model: Pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> List[float]:
        model.fit(X_train, y_train)
        y_scores = model.predict_proba(X_test)[:, 1]

        return y_scores

    @staticmethod
    def _get_metric_score(y_test: pd.Series, y_scores: pd.Series, metric: str) -> float:
        if metric == "roc_auc":
            return roc_auc_score(y_test, y_scores)

    @staticmethod
    def _get_fractions() -> List[float]:
        fractions = list(range(1, 101))
        fractions = list(map(lambda x: x / 100, fractions))

        return fractions

    def _calculate_score_from_model(self) -> List[tuple]:
        fraction_score_pair = list()
        sample = self._make_sample()
        features = self._get_features()

        fractions = IterativeFeatureSelection._get_fractions()
        features = self._get_features()
        X_train, X_test, y_train, y_test = self._split_sample_data(
            sample, features, self.test_size
        )
        print("Building score-feature tradeoff curve")
        print("Fitting model...")
        for fraction in tqdm(fractions):
            pipe = self._build_pipeline()
            size = round(len(features) * fraction)
            sample_features = IterativeFeatureSelection._get_feature_sample(
                features=features, size=size
            )
            y_scores = IterativeFeatureSelection._return_proba_from_model(
                model=pipe,
                X_train=X_train[sample_features],
                X_test=X_test[sample_features],
                y_train=y_train,
            )

            score = IterativeFeatureSelection._get_metric_score(
                y_test, y_scores, metric="roc_auc"
            )
            fraction_score_pair.append((fraction, score))
        print("Done!")

        return fraction_score_pair

    def _calculate_best_feature_number(self) -> float:
        fraction_score_tuple = self._calculate_score_from_model()
        reverse_scores = fraction_score_tuple[::-1]
        deltas = [
            round((reverse_scores[0][1] - score[1]) / reverse_scores[0][1], 4)
            for score in reverse_scores[1:]
        ]
        least_than_threshold = list(filter(lambda x: x <= self.threshold, deltas))
        best_value = max(least_than_threshold)
        idx = deltas.index(best_value)

        #  Tupla do tipo (fraction, score)
        best_fraction_score_tuple: tuple = fraction_score_tuple[:-1][idx]
        print(f"Optimal % of features is: {best_fraction_score_tuple[0]*100}")
        print(
            f"Score for metric {self.metric}: {round(best_fraction_score_tuple[1], 3)}"
        )

        return best_fraction_score_tuple, fraction_score_tuple

    def _plot(
        self,
        fractions: List[str],
        scores: List[float],
        best_fraction: float,
        ymin: float,
        ymax: float,
    ):

        plt.figure(figsize=(15, 5))
        sns.lineplot(x=fractions, y=scores, color="black")
        plt.vlines(
            x=best_fraction,
            ymin=ymin,
            ymax=ymax,
            color="red",
            label=f"best cut is {round(best_fraction*100)}% of features",
            linestyles="dashed",
        )
        plt.legend()
        plt.title(
            f"Feature selection based on given cutoff (THRESHOLD={self.threshold})"
        )
        plt.ylabel(f"{self.metric.upper()} score")
        plt.xlabel(f"% of features")
        plt.show()

    def build_selection_curve(self):
        (
            best_fraction_score_tuple,
            fraction_score_pair,
        ) = self._calculate_best_feature_number()

        fractions, scores = zip(*fraction_score_pair)
        best_fraction = best_fraction_score_tuple[0]
        # best_score = best_fraction_score_tuple[1]
        print(scores[0])
        print(scores[-1])

        self._plot(
            fractions, scores, best_fraction, scores[0], best_fraction_score_tuple[1]
        )
