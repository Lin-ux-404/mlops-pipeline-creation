# Import libraries
import argparse
import glob
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
from mldesigner import command_component, Input, Output
import subprocess


@command_component(
    environment="../env.yaml",
    name="train_logistic_regression_classifier_model",
    display_name="Train Logistic Regression Classifier Model",
)
def train_logistic_regression_classifier_model(
    training_data: Input(type="uri_folder"),
    model_output_logistic_reg: Output(type="uri_folder"),
    metrics_output: Output(type="uri_file"),
    regularization_rate: float = 0.01,
):
    
    # Load training data
    print("Loading data...")
    all_files = glob.glob(os.path.join(training_data, "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)

    # Separate features and labels
    X = df[
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
    ].values
    y = df["Diabetic"].values

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0
    )

    # Train logistic regression model
    print("Training logistic regression model...")
    model = LogisticRegression(
        C=1 / regularization_rate, solver="liblinear"
    ).fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"Accuracy: {acc}")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC: {auc}")
    # Save ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("ROCcurve.png")

    # Save confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(
                x=j, y=i, s=conf_matrix[i, j], ha="center", va="center"
            )
    plt.xlabel("Predictions")
    plt.ylabel("Actuals")
    plt.title("Confusion Matrix")
    plt.savefig("ConfusionMatrix.png")

    print("Saving model and metrics...")
    # Save model
    model_dir = Path(metrics_output)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(model_dir, "models/")
    mlflow.sklearn.save_model(model, path=save_path)

    # Save metrics to JSON
    metrics = {
        "accuracy": acc,
        "auc": auc,
    }
    metrics_output_path = os.path.join(model_dir, "metrics_logistic_regression_model.json")
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f)