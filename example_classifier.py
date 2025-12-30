import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from auto_feature_engineer import AutoFeatureEngineer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
afe = AutoFeatureEngineer(k=25)
X_train_fe = afe.fit_transform(X_train, y_train)
X_test_fe = afe.transform(X_test, y_test)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_fe, y_train)
preds = clf.predict(X_test_fe)
print("Accuracy with engineered features:", accuracy_score(y_test, preds))