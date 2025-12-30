from feature_generation import FeatureGenerator
from feature_scoring import FeatureScorer

class AutoFeatureEngineer:
    def __init__(self, k=20, task="classification"):
        self.k = k
        self.generator = FeatureGenerator()
        self.scorer = FeatureScorer(task)
        self.selected_features = None

    def fit(self, X, y):
        X_gen = self.generator.generate(X, y, fit=True)
        scores = self.scorer.score(X_gen, y)
        self.selected_features = scores.head(self.k).index.tolist()
        return self

    def transform(self, X, y=None):
        X_gen = self.generator.generate(X, y, fit=False)
        return X_gen[self.selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)
