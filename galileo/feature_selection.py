"""
This file contains the implementation of feature selection techniques used by Galileo.

Author: Rafael Franceschetti
"""

from typing import Callable, List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm
import warnings

import galileo.utils as utils

from abc import ABC, abstractmethod

warnings.filterwarnings("ignore")


def return_proba_from_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> List[float]:
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:, 1]

    return y_scores


class BaseFeatureSelection(ABC):
    def __init__(
        self,
        model: Callable,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        features: List[str],
        target_name: str,
        metric: str,
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.features = features
        self.target_name = target_name
        self.metric = metric

        @abstractmethod
        def select_features(self):
            pass


class IterativeFeatureSelection(BaseFeatureSelection):
    """
    Executes feature selection iterating over fractions of the total number of features within
    the interval [0.01, 1] and returns the selected features based on the chosen threshold, which
    is the maximum difference allowed between the best score and the score computed from the selected features.

    Attributes:
    ------------
    model : Callable
        Model used to select the features.
    X_train : pd.DataFrame
        The dataframe used for model training.
    y_train : pd.Series
        Target values for the training dataset
    X_test : pd.DataFrame
        The dataframe used for model evaluation
    y_series : pd.Series
        Target values for the testing dataset
    target_name : string
        The name of the target column in the dataframe
    threshold : float
        Maximum difference allowed between the best score and the score computed in each step
    interval : list
        Range of values for the iterative process of feature selection
        default = [0.01, 1.0]
    step : float
        Step to increment each value of the interval
    metric : string
        The metric chosen for model evaluation

    Methods:
    ------------
    select_features():
        Execute the iterative feature selection process and returns the selected features.

    """

    def __init__(
        self,
        model: Callable,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        features: List[str],
        target_name: str,
        threshold: float,
        interval: List[float] = [0.01, 1.0],
        step: float = 0.01,
        metric: str = "roc_auc",
    ) -> None:
        super().__init__(
            model, X_train, y_train, X_test, y_test, features, target_name, metric
        )
        self.threshold = threshold
        self.interval = interval
        self.step = step

    def _get_interval(self) -> List[float]:
        fractions = np.arange(self.interval[0], self.interval[1] + self.step, self.step)
        return fractions

    @staticmethod
    def _get_feature_sample(features: List[str], size: float):
        return np.random.choice(features, size=size, replace=False)

    def _calculate_score_from_model(self) -> List[tuple]:
        fraction_score_feature_set = list()
        fractions = self._get_interval()
        feature_array_length = len(self.features)

        print("Building score-feature tradeoff curve")
        print("Fitting model...")
        for fraction in tqdm(fractions):
            size = round(feature_array_length * fraction)
            sample_features = IterativeFeatureSelection._get_feature_sample(
                features=self.features, size=size
            )
            y_scores = utils.return_proba_from_model(
                model=self.model,
                X_train=self.X_train[sample_features],
                X_test=self.X_test[sample_features],
                y_train=self.y_train,
            )

            score = utils.get_metric_score(self.y_test, y_scores, metric=self.metric)
            fraction_score_feature_set.append((fraction, score, sample_features))
        print("Done!")

        return fraction_score_feature_set

    def _calculate_best_feature_number(self) -> float:
        fraction_score_feature_set = self._calculate_score_from_model()
        # reverse_scores = fraction_score_feature_set[::-1]
        deltas = [
            round(
                (fraction_score_feature_set[-1][1] - score[1])
                / fraction_score_feature_set[-1][1],
                4,
            )
            for score in fraction_score_feature_set[:-1]
        ]

        least_than_threshold = list(filter(lambda x: x <= self.threshold, deltas))
        if len(least_than_threshold) == 0:
            best_selection = fraction_score_feature_set[-1]
            print(f"No optimal feature selection found for threshold: {self.threshold}")
            return best_selection, fraction_score_feature_set
        else:
            best_value = max(least_than_threshold)
            print(f"Found a score with a delta of {best_value}")
            print(f"Highest score is {fraction_score_feature_set[-1][1]}")
            idx = deltas.index(best_value)

            #  Tupla do tipo (fraction, score, features)
            best_selection: tuple = fraction_score_feature_set[idx]
            total_number_of_features = round(best_selection[0] * len(self.features))
            print(
                f"Optimal % of features is: {round(best_selection[0]*100)} ({total_number_of_features} features)"
            )
            print(f"Score for metric {self.metric}: {best_selection[1]}")

            return best_selection, fraction_score_feature_set

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

    def select_features(self):
        (
            best_selection,
            fraction_score_feature_set,
        ) = self._calculate_best_feature_number()

        fractions, scores, features = zip(*fraction_score_feature_set)
        best_fraction = best_selection[0]
        selected_features = best_selection[2]

        self._plot(fractions, scores, best_fraction, min(scores), best_selection[1])

        return selected_features


class OneShotFeatureSelection(BaseFeatureSelection):
    """
    Executes a one-time feature selection process using a machine learning model.
    The result comes from the builtin feature importance method from the estimator.

    Attributes:
    ------------
    model : Callable
        Model used to select the features.
    X_train : pd.DataFrame
        The dataframe used for model training.
    y_train : pd.Series
        Target values for the training dataset
    X_test : pd.DataFrame
        The dataframe used for model evaluation
    y_series : pd.Series
        Target values for the testing dataset
    target_name : string
        The name of the target column in the dataframe
    threshold : float
        Maximum difference allowed between the best score and the score computed in each step
    interval : list
        Range of values for the iterative process of feature selection
        default = [0.01, 1.0]
    step : float
        Step to increment each value of the interval
    metric : string
        The metric chosen for model evaluation
    importance_getter : str
        The method that the estimator uses to get its feature importance estimations
        default = "named_steps.estimator.feature_importances_"

    Methods:
    ------------
    select_features():
        Execute the iterative feature selection process and returns the selected features.

    """

    def __init__(
        self,
        model: Callable,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        features: List[str],
        target_name: str,
        metric: str = "roc_auc",
        importance_getter: str = "named_steps.estimator.feature_importances_",
    ) -> None:
        super().__init__(
            model, X_train, y_train, X_test, y_test, features, target_name, metric
        )
        if isinstance(model, Pipeline):
            self.importance_getter = importance_getter
        else:
            self.importance_getter = "auto"

    def _select_features_from_model(self):
        selection = SelectFromModel(
            self.model, importance_getter=self.importance_getter
        )
        selection.fit(self.X_train[self.features], self.y_train)

        selected_features = (
            self.X_train[self.features].columns[(selection.get_support())].values
        )
        print(f"Number of initial features selected: {len(selected_features)}")
        print(
            f"{round(len(selected_features)/len(self.features), 2)*100}% of total features."
        )

        return selected_features

    def select_features(self):
        selected_features = self._select_features_from_model()
        y_scores_all_features = utils.return_proba_from_model(
            model=self.model,
            X_train=self.X_train[self.features],
            X_test=self.X_test[self.features],
            y_train=self.y_train,
        )
        y_scores_selected_features = utils.return_proba_from_model(
            model=self.model,
            X_train=self.X_train[selected_features],
            X_test=self.X_test[selected_features],
            y_train=self.y_train,
        )

        score_all_features = utils.get_metric_score(
            self.y_test, y_scores_all_features, metric=self.metric
        )

        score_from_selection = utils.get_metric_score(
            self.y_test, y_scores_selected_features, metric=self.metric
        )
        if score_all_features > score_from_selection:
            difference = (
                score_all_features - score_from_selection
            ) / score_all_features
            print(f"Scores with all features: {round(score_all_features, 3)}")
            print(f"Scores with selected features: {round(score_from_selection, 3)}")
            print(f"Loss is {round(difference, 3)*100}%")

        elif score_all_features == score_from_selection:
            print(
                "Found no difference in score between fit with all features and with selected features"
            )
        else:
            difference = (
                score_from_selection - score_all_features
            ) / score_from_selection
            print(f"Scores with all features: {round(score_all_features, 3)}")
            print(f"Scores with selected features: {round(score_from_selection, 3)}")
            print(f"Gain is {round(difference, 3)*100}%")

        return selected_features
