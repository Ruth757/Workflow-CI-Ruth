# Workflow-CI-Ruth

## Workflow CI

GitHub Actions digunakan untuk menjalankan training model secara otomatis setiap kali terjadi push ke branch main.

## Docker Image

```bash
docker pull ruth757/telco-churn-ruth:latest
```

## Monitoring dan Logging

Monitoring dan logging dilakukan menggunakan MLflow.

Hasil tracking tersimpan pada:
- mlruns/
- mlflow.db