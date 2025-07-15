import os
import joblib
import pandas as pd

def init():
    global model
    # Load the model from the model directory
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)

def run(mini_batch):
    # mini_batch is a list of input file paths
    results = []
    for input_path in mini_batch:
        data = pd.read_csv(input_path)
        predictions = model.predict(data)
        results.append(predictions.tolist())
    return results
