# Import libraries
import argparse
import glob
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import os
import json


# get parameters
parser = argparse.ArgumentParser("train")
parser.add_argument("--training_data", type=str, help="Path to training data")
parser.add_argument("--reg_rate", type=float, default=0.01)
parser.add_argument("--model_output_logistic_reg", type=str, help="Path of output model")
parser.add_argument("--metrics_output", type=str, help="Metrics of model performance")

args = parser.parse_args()

training_data = args.training_data

# load the prepared data file in the training folder
print("Loading Data...")
data_path = args.training_data
all_files = glob.glob(data_path + "/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)

# Separate features and labels
X, y = (
    df[
        [
            "Pregnancies",
            "PlasmaGlucose",
            "DiastolicBloodPressure",
            "TricepsThickness",
            "SerumInsulin",
            "BMI",
            "DiabetesPedigree",
            "Age",
        ]
    ].values,
    df["Diabetic"].values,
)

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0
)

mlflow.start_run()
# Train a logistic regression model
print('Training a logistic regression model...')
model = LogisticRegression(C=1 / args.reg_rate, solver="liblinear").fit(
    X_train, y_train
)

# Calculate accuracy
y_pred = model.predict(X_test)
acc = np.average(y_pred == y_test)
print("Accuracy:", acc)
mlflow.log_metric("Accuracy", np.float(acc))

# Calculate AUC
y_pred_proba = model.predict_proba(X_test)
auc = roc_auc_score(y_test, y_pred_proba[:, 1])
print("AUC: " + str(auc))
mlflow.log_metric("AUC", np.float(auc))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
fig = plt.figure(figsize=(6, 4))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], "k--")
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("ROCcurve.png")
mlflow.log_artifact("ROCcurve.png")

# Create confusion matrix
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(
            x=j, y=i, s=conf_matrix[i, j], va="center", ha="center",
            size="xx-large"
        )

plt.xlabel("Predictions", fontsize=18)
plt.ylabel("Actuals", fontsize=18)
plt.title("Confusion Matrix", fontsize=18)
plt.savefig("ConfusionMatrix.png")
mlflow.log_artifact("ConfusionMatrix.png")

output_dir = Path(args.metrics_output)
save_path = os.path.join(output_dir, "models/")
mlflow.sklearn.save_model(
    sk_model=model,
    path=save_path,
)

# Save metrics to JSON file
metrics = {
    "accuracy": acc,
    "auc": auc
}
metrics_output_path = os.path.join(output_dir, "metrics_logistic_regression_model.json")
with open(metrics_output_path, "w") as f:
    json.dump(metrics, f)


mlflow.end_run()