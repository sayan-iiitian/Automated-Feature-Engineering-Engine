import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer

class FeatureGenerator:
    def __init__(self, poly_degree=2, n_bins=5):
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', 
                                       quantile_method='averaged_inverted_cdf')
        self.target_encodings = {}

    def polynomial_features(self, X, fit=False):
        numeric = X.select_dtypes(include=np.number)
        if fit:
            poly_features = self.poly.fit_transform(numeric)
        else:
            poly_features = self.poly.transform(numeric)
        poly_cols = self.poly.get_feature_names_out(numeric.columns)
        return pd.DataFrame(poly_features, columns=poly_cols)

    def bin_features(self, X, fit=False):
        numeric = X.select_dtypes(include=np.number)
        if fit:
            binned = self.binner.fit_transform(numeric)
        else:
            binned = self.binner.transform(numeric)
        binned_cols = [f"{c}_bin" for c in numeric.columns]
        return pd.DataFrame(binned, columns=binned_cols)

    def target_encode(self, X, y=None, fit=False):
        cat_cols = X.select_dtypes(include='object').columns
        X_encoded = X.copy()

        for col in cat_cols:
            if fit and y is not None:
                # Fit: compute and store encodings
                means = y.groupby(X[col]).mean()
                self.target_encodings[col] = means
                X_encoded[col] = X[col].map(means)
            elif col in self.target_encodings:
                # Transform: use stored encodings
                X_encoded[col] = X[col].map(self.target_encodings[col]).fillna(
                    self.target_encodings[col].mean()
                )
            else:
                # No encoding available, keep original
                pass

        return X_encoded

    def generate(self, X, y=None, fit=False):
        X_te = self.target_encode(X, y, fit=fit)
        poly = self.polynomial_features(X_te, fit=fit)
        bins = self.bin_features(X_te, fit=fit)
        return pd.concat([X_te.reset_index(drop=True),
                          poly.reset_index(drop=True),
                          bins.reset_index(drop=True)], axis=1)