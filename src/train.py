import logging
import mlflow
from matplotlib import pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import os
from azure.identity import ClientSecretCredential
from azure.storage.filedatalake import DataLakeServiceClient
import pandas as pd
from io import StringIO

# Configure logging
logging.basicConfig(filename='logs/model_training.log', level=logging.INFO)

# App Registration credentials from environment variables
tenant_id = os.environ.get("AZURE_TENANT_ID")
client_id = os.environ.get("AZURE_CLIENT_ID")
client_secret = os.environ.get("AZURE_CLIENT_SECRET")
storage_account_name = "optimalchanneldata"
container_name = "optimalchannel"

credential = ClientSecretCredential(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret
)
service_client = DataLakeServiceClient(
    account_url=f"https://{storage_account_name}.dfs.core.windows.net",
    credential=credential
)
def read_processed_data(service_client, container_name, file_name):
    file_system_client = service_client.get_file_system_client(file_system=container_name)
    file_client = file_system_client.get_file_client(file_name)
    download = file_client.download_file()
    file_content = download.readall()
    try:
        file_content_decoded = file_content.decode("utf-8")
    except UnicodeDecodeError:
        file_content_decoded = file_content.decode("ISO-8859-1")
    df = pd.read_csv(StringIO(file_content_decoded))
    return df
def prepare_data_for_model(df, target_column):
    # Drop non-feature columns if needed
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    return X, y
# Step 1: Train the model and log using MLflow
def train_and_log_model(X_train, y_train, X_test, y_test, target_names):
    """
    Train the model, log parameters, metrics, and the model using MLflow.
    """
    # Start MLflow experiment
    # Log the experiment using MLflow's tracking feature
    with mlflow.start_run():  # Sequential run naming system
    
        # Log hyperparameters for XGBoost
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("random_state", 42)

        # Convert the dataset into DMatrix format for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
    
        # Specify XGBoost parameters
        params = {
        'objective': 'multi:softmax',  # Multi-class classification
        'num_class': len(set(y_train)),  # Number of classes in target
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100
        }
    
        # Initialize and train the XGBoost model
        xgb_model = xgb.train(params, dtrain, num_boost_round=100)
        # Log model signature
        signature = mlflow.models.signature.infer_signature(X_train, y_train)
        # Log the model
        mlflow.xgboost.log_model(xgb_model, "model", signature=signature)
        # Making predictions using the trained model
        y_pred = xgb_model.predict(dtest)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, target_names=target_names)
        # Log metrics: accuracy, precision, recall, F1 score
        mlflow.log_metric("accuracy", accuracy)
        # For precision, recall, and f1_score, parsing the classification report manually
        # and logging them as individual metrics
        # Log precision, recall, f1 for each class
        class_report = classification_rep.split('\n')[1:-3]
        for idx, class_name in enumerate(target_names):
            for line in class_report:
                if class_name in line:
                    precision, recall, f1_score, _ = line.split()[1:5]
                    mlflow.log_metric(f"precision_{class_name}", float(precision))
                    mlflow.log_metric(f"recall_{class_name}", float(recall))
                    mlflow.log_metric(f"f1_score_{class_name}", float(f1_score))                
        # Log the classification report as a text artifact
        mlflow.log_text(classification_rep, "classification_report.txt")
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        # Create a heatmap for the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Heatmap')
        # Save the heatmap as an image file
        conf_matrix_filename = "confusion_matrix.png"
        plt.savefig(conf_matrix_filename)
        plt.close()
        # Log confusion matrix heatmap as an artifact
        mlflow.log_artifact(conf_matrix_filename)
        # Print the results
        print("Accuracy Score:", accuracy)
        print(classification_rep)
        # End of MLflow run
        print("Run has been logged in MLflow.")
        logging.info(f"Model training completed with accuracy: {accuracy}. Model logged in MLflow.")
# Step 4: Main function for model training pipeline
def training_pipeline():
    # Load processed data from ADLS
    df = read_processed_data(service_client, container_name, "processed_data.csv")
    # Specify target column
    target_column = "Optimal_Channel"  # Change if needed
    # Prepare features and target
    X, y = prepare_data_for_model(df, target_column)
    # Encode target labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    target_names = le.classes_
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    # Train and log model
    train_and_log_model(X_train, y_train, X_test, y_test, target_names)
if __name__ == "__main__":
    training_pipeline()


