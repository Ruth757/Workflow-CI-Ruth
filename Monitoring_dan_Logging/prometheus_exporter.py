from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import start_http_server
import time

prediction_requests = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

start_http_server(8000)

while True:
    time.sleep(1)