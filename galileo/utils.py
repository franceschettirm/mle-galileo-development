"""
"""
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


def return_proba_from_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> List[float]:
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:, 1]

    return y_scores


def build_pipeline():
    if check_has_categorical() > 0:
        return Pipeline(
            [
                ("encoder", CatBoostEncoder(random_state=42)),
                ("estimator", LGBMClassifier(random_state=42, verbosity=-1)),
            ]
        )
    else:
        return LGBMClassifier(random_state=42, verbosity=-1)


def get_metric_score(y_test: pd.Series, y_scores: pd.Series, metric: str) -> float:
    if metric == "roc_auc":
        return roc_auc_score(y_test, y_scores)


def check_has_categorical(dataframe) -> int:
    return len(dataframe.select_dtypes(include="object").columns)
