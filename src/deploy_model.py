import xgboost as xgb
from azureml.core import Workspace, Environment
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import BatchEndpoint, BatchDeployment
from processed import processed_pipeline

def get_latest_two_model_versions(ml_client, model_name):
    models = ml_client.models.list(name=model_name)
    models_sorted = sorted(models, key=lambda m: m.created_on, reverse=True)
    if len(models_sorted) < 2:
        raise ValueError("Not enough registered model versions to compare.")
    return models_sorted[0], models_sorted[1]  # candidate, champion
def compare_and_deploy_models():
    # Load Azure ML workspace config
    ws = Workspace.from_config()
    subscription_id = ws.subscription_id
    resource_group = ws.resource_group
    workspace_name = ws.name
    # Create MLClient with DefaultAzureCredential
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)
    # Model details
    model_name = "Optimal_channel_prediction_XGBoost_optuna"
    # Register or get environment
    env_name = "batch-env"
    env_file_path = "./env.yaml"
    env = Environment.from_conda_specification(name=env_name, file_path=env_file_path)
    env.register(workspace=ws)
    # Create or update batch endpoint
    batch_endpoint_name = "optimal-channel-prediction"
    batch_endpoint = BatchEndpoint(
        name=batch_endpoint_name,
        description="Optimal Channel Prediction Batch scoring endpoint"
    )
    poller = ml_client.batch_endpoints.begin_create_or_update(batch_endpoint)
    endpoint = poller.result()
    # Load processed data
    X_train, X_test, y_train, y_test, label_encoders, target = processed_pipeline()
    # Fetch the latest two registered model versions
    candidate_model, champion_model = get_latest_two_model_versions(ml_client, model_name)
    # Load models from their paths
    candidate_xgb = xgb.Booster()
    candidate_xgb.load_model(candidate_model.path)
    champion_xgb = xgb.Booster()
    champion_xgb.load_model(champion_model.path)
    # Evaluate both models
    candidate_pred = candidate_xgb.predict(xgb.DMatrix(X_test))
    champion_pred = champion_xgb.predict(xgb.DMatrix(X_test))
    from sklearn.metrics import accuracy_score, classification_report
    candidate_accuracy = accuracy_score(y_test, candidate_pred)
    champion_accuracy = accuracy_score(y_test, champion_pred)
    print("\n--- Model Comparison ---")
    print(f"Champion Model Accuracy: {champion_accuracy}")
    print(f"Candidate Model Accuracy: {candidate_accuracy}")
    print("\nCandidate Classification Report:")
    print(classification_report(y_test, candidate_pred, target_names=label_encoders[target].classes_))
    print("\nChampion Classification Report:")
    print(classification_report(y_test, champion_pred, target_names=label_encoders[target].classes_))
    # Deploy candidate if it is better
    deployment_name = "default"
    if candidate_accuracy > champion_accuracy:
        print("Candidate model outperforms champion. Deploying candidate as new champion...")
        batch_deployment = BatchDeployment(
            name=deployment_name,
            endpoint_name=batch_endpoint_name,
            model=candidate_model,
            environment=env_name,
            code_configuration={
                "code": "./",
                "scoring_script": "score.py"
            },
            compute="your-compute-cluster-name",
            description="Batch deployment for optimal channel prediction",
            max_concurrency_per_instance=2,
            mini_batch_size=10
        )
        poller_deploy = ml_client.batch_deployments.begin_create_or_update(batch_deployment)
        deployment = poller_deploy.result()
        print(f"Candidate model deployed as new champion: {deployment.name}")
    else:
        print("Champion model outperforms candidate. Retaining current champion deployment.")
if __name__ == "__main__":
    compare_and_deploy_models()
