from mlflow import MlflowClient
from azureml.core import Workspace, Experiment, Run
import mlflow
import os


# MLflow model and run details
model_name = "Optimal_channel_prediction_XGBoost_optuna"


# Connect to Azure ML Workspace
ws = Workspace.from_config()
# Get latest experiment by last update time
experiments = list(ws.experiments.values())
experiments.sort(key=lambda exp: exp._last_update_time, reverse=True)
latest_experiment = experiments[0]
print(f"Using latest experiment: {latest_experiment.name}")
experiment = latest_experiment

# Find best run by F1-Score (can change to precision/recall if needed)
client = MlflowClient()
best_run = None
best_f1 = -1
for run in client.search_runs(experiment.experiment_id, "", order_by=["metrics.f1_score DESC"]):
    f1 = run.data.metrics.get("f1_score", None)
    if f1 is not None and f1 > best_f1:
        best_f1 = f1
        best_run = run

if best_run is None:
    raise Exception("No run found with F1-Score metric.")
run_id = best_run.info.run_id
print(f"Best run ID by F1-Score: {run_id} (F1: {best_f1})")


# Check existing registered models
existing_models = [rm.name for rm in client.search_registered_models()]
if model_name not in existing_models:
    client.create_registered_model(model_name)
    print(f"Created registered model '{model_name}'")
else:
    print(f"Model '{model_name}' already exists.")

# Get MLflow run artifact URI
artifact_uri = mlflow.get_run(run_id).info.artifact_uri
model_artifact_path = "model/model.xgb"
model_source = f"{artifact_uri}/{model_artifact_path}"

# Register new model version
result = client.create_model_version(
    name=model_name,
    source=model_source,
    run_id=run_id
)

print(f"Registered model version: {result.version}")

