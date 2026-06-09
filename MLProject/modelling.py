import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

mlflow.sklearn.autolog()

BASE_DIR = os.path.dirname(__file__)

X_train = pd.read_csv(os.path.join(BASE_DIR, "dataset_preprocessing/X_train.csv"))
X_test  = pd.read_csv(os.path.join(BASE_DIR, "dataset_preprocessing/X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE_DIR, "dataset_preprocessing/y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(BASE_DIR, "dataset_preprocessing/y_test.csv")).values.ravel()

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    mlflow.log_metric("accuracy", accuracy)

print("Training selesai")