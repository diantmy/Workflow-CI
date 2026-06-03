import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# MLflow Configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Telco_Churn_RF")

# Load Dataset
df = pd.read_csv("telco_preprocessed.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Training Model
model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Logging Parameter
mlflow.log_param("random_state", 42)

# Logging Metrics
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.close()

# Classification Report
report = classification_report(
    y_test,
    y_pred
)

with open("classification_report.txt", "w") as f:
    f.write(report)

# Save Model
joblib.dump(model, "model.pkl")

# Log Artifacts
mlflow.log_artifact("confusion_matrix.png")
mlflow.log_artifact("classification_report.txt")
mlflow.log_artifact("model.pkl")

# Log Model
mlflow.sklearn.log_model(
    sk_model=model,
    name="random_forest_model"
)

print("\n===== HASIL MODEL =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
