import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Aktifkan autolog MLflow
mlflow.set_experiment("Telco_Churn")
mlflow.sklearn.autolog()

# Load dataset
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# Ubah target menjadi array
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Mulai MLflow Run
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

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy : {accuracy:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

print("Training selesai dan tercatat di MLflow")