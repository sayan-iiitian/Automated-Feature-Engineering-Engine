import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

class FeatureScorer:
    def __init__(self, task="classification"):
        self.task = task

    def score(self, X, y):
        X = X.fillna(0)

        if self.task == "classification":
            scores = mutual_info_classif(X, y)
        else:
            scores = mutual_info_regression(X, y)

        return pd.Series(scores, index=X.columns).sort_values(ascending=False)
