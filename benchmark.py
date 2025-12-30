from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

from auto_feature_engineer import AutoFeatureEngineer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Raw model
raw_model = RandomForestClassifier()
raw_model.fit(X_train, y_train)
raw_acc = accuracy_score(y_test, raw_model.predict(X_test))

# Engineered model
afe = AutoFeatureEngineer(k=30)
X_train_fe = afe.fit_transform(X_train, y_train)
X_test_fe = afe.transform(X_test, y_test)

fe_model = RandomForestClassifier()
fe_model.fit(X_train_fe, y_train)
fe_acc = accuracy_score(y_test, fe_model.predict(X_test_fe))

print(f"Raw Accuracy: {raw_acc:.4f}")
print(f"Engineered Accuracy: {fe_acc:.4f}")
