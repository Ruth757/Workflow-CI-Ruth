import mlflow.pyfunc

MODEL_PATH = "../Membangun_model/mlruns/2/models/m-81c262a6f9804db4b926ec496db74d1c/artifacts"

model = mlflow.pyfunc.load_model(MODEL_PATH)

print(model.metadata.signature)