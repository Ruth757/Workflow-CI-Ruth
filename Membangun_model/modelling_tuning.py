import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# Load Dataset
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# Hyperparameter Tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None]
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Prediction
y_pred = best_model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# MLflow Tracking
mlflow.set_experiment("RandomForest_Tuning")

with mlflow.start_run():

    # Log Best Parameters
    mlflow.log_params(grid.best_params_)

    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Confusion Matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
        ax=ax
    )
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ROC Curve
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
        ax=ax
    )
    plt.savefig("roc_curve.png")
    plt.close()

    # Precision Recall Curve
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
        ax=ax
    )
    plt.savefig("precision_recall_curve.png")
    plt.close()

    # Log Artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("roc_curve.png")
    mlflow.log_artifact("precision_recall_curve.png")

    # Log Model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

print("Training selesai dan tercatat di MLflow.")