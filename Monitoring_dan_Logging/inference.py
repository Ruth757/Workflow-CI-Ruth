from fastapi import FastAPI
from fastapi import Body
from typing import List

import mlflow.pyfunc
import pandas as pd
import uvicorn
import time

from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import start_http_server

app = FastAPI(title="Telco Churn Prediction API")

# Jalankan Prometheus metrics di port 8000
start_http_server(8000)

# Load model MLflow
MODEL_PATH = "../Membangun_model/mlruns/2/models/m-81c262a6f9804db4b926ec496db74d1c/artifacts"

model = mlflow.pyfunc.load_model(MODEL_PATH)

# Counter jumlah request
prediction_requests = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

# Histogram waktu prediksi
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)


@app.get("/")
def home():
    return {
        "message": "Telco Churn API is running"
    }


@app.post("/predict")
def predict(data: List[float] = Body(...)):

    prediction_requests.inc()

    start_time = time.time()

    # Model kamu punya 19 fitur
    columns = [str(i) for i in range(19)]

    df = pd.DataFrame(
        [data],
        columns=columns
    )

    prediction = model.predict(df)

    prediction_latency.observe(
        time.time() - start_time
    )

    return {
        "prediction": int(prediction[0])
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001
    )