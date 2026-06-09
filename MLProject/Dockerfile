FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install --default-timeout=1000 pandas scikit-learn mlflow

CMD ["python", "modelling.py"]