# prometheus_exporter.py

from prometheus_client import start_http_server
from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import Gauge
import random
import time

# Total request
prediction_requests = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

# Waktu inference
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

# Akurasi dummy
model_accuracy = Gauge(
    "model_accuracy",
    "Current model accuracy"
)

def simulate_prediction():

    prediction_requests.inc()

    start = time.time()

    time.sleep(random.uniform(0.1, 0.5))

    prediction_latency.observe(
        time.time() - start
    )

    model_accuracy.set(0.82)

if __name__ == "__main__":

    start_http_server(8000)

    while True:
        simulate_prediction()
        time.sleep(2)