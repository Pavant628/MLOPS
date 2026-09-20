import pandas as pd
import yaml
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load hyperparameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

rf_params = params["random_forest"]


# Load dataset
data = pd.read_csv("data/heart-disease.csv")

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=rf_params["n_estimators"],
    max_depth=rf_params["max_depth"],
    random_state=rf_params["random_state"]
)


# Train model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)


# Print metrics
print("Random Forest Model")
print("-------------------")
print(f"Accuracy : {accuracy:.5f}")
print(f"Precision: {precision:.5f}")
print(f"Recall   : {recall:.5f}")
print(f"F1-score : {f1:.5f}")


# Create output directories
import os

os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)


# Save model
joblib.dump(model, "models/random_forest.pkl")


# Save metrics
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open("metrics/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved to: models/random_forest.pkl")
print("Metrics saved to: metrics/metrics.json")