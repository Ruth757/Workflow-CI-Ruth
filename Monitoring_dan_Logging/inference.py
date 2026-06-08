from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import uvicorn

app = FastAPI(title="Telco Churn Prediction API")

# Sesuaikan path model dengan model MLflow milikmu
MODEL_PATH = "../Membangun_model/mlruns/2/models/m-81c262a6f9804db4b926ec496db74d1c/artifacts"

model = mlflow.pyfunc.load_model(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Telco Churn API is running"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0])
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)