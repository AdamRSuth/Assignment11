import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature
import subprocess
import json
import requests

# Set the MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Load the Iris dataset and train the model
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model hyperparameters
params = {"solver": "lbfgs", "max_iter": 1000, "multi_class": "auto", "random_state": 8888}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate accuracy as a target loss metric
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# MLflow experiment
mlflow.set_experiment("iris-experiment")

# run to MLflow
with mlflow.start_run() as run:
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag to describe this run
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model to MLflow
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="basic_lr_iris_model",
    )

    print(f"Model logged to MLflow with ID: {run.info.run_id}")

# Deploy the Model Locally
model_uri = f"models:/basic_lr_iris_model/1"  # Update the version as necessary
serve_command = [
    "mlflow", "models", "serve",
    "-m", model_uri,
    "-p", "5000",
    "--enable-mlserver"
]
print(f"Starting MLflow model serving for URI: {model_uri}")
subprocess.Popen(serve_command)

# Send Data to the Inference Endpoint
payload = json.dumps({
    "inputs": {
        "feature_1": X_test[0, 0],
        "feature_2": X_test[0, 1],
        "feature_3": X_test[0, 2],
        "feature_4": X_test[0, 3],
    }
})

response = requests.post(
    url="http://127.0.0.1:5000/invocations",
    data=payload,
    headers={"Content-Type": "application/json"}
)

print("Prediction response from model:")
print(response.json())
