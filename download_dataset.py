"""
Script to download the breast cancer dataset and save it as CSV.
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')
df = pd.concat([X, y], axis=1)
df.to_csv('breast_cancer.csv', index=False)
print(f"Dataset saved to 'breast_cancer.csv'")
print(f"Shape: {df.shape}")
print(f"Features: {len(data.feature_names)}")
print(f"Target classes: {data.target_names}")