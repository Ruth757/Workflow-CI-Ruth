import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

import matplotlib.pyplot as plt

mlflow.set_experiment("Telco_Churn")

mlflow.sklearn.autolog()

# Load dataset
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

with mlflow.start_run():

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Training
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy : {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # ==========================
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    # ==========================
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # ==========================
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")
    plt.close()
    # ==========================
    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.savefig("precision_recall_curve.png")
    mlflow.log_artifact("precision_recall_curve.png")
    plt.close()

print("Training selesai dan tercatat di MLflow")