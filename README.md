# Workflow-CI-Ruth

## Workflow CI

GitHub Actions digunakan untuk menjalankan training model secara otomatis setiap kali terjadi push ke branch `main`.

Workflow akan:
1. Checkout repository
2. Setup Python environment
3. Install dependencies
4. Menjalankan training model menggunakan MLflow Project

## Docker Image

Docker image tersedia di Docker Hub dan dapat dijalankan dengan:

```bash
docker pull ruth757/telco-churn-ruth:latest
```

Menjalankan container:

```bash
docker run --name churn-container ruth757/telco-churn-ruth:latest
```

## Monitoring dan Logging

Monitoring dan logging dilakukan menggunakan MLflow.

Fitur yang dicatat oleh MLflow:
- Parameter model
- Accuracy model
- Training metrics
- Artifacts model
- Confusion Matrix
- Precision Recall Curve
- ROC Curve

Hasil tracking tersimpan pada:
- `mlruns/`
- `mlflow.db`

## Repository

GitHub Repository:
https://github.com/Ruth757/Workflow-CI-Ruth

Docker Hub:
https://hub.docker.com/r/ruth757/telco-churn-ruth
