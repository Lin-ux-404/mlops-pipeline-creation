import argparse
import json
import os
from pathlib import Path


def determine_better_model(model1_metrics, model2_metrics, model1_savepath, model2_savepath):
    """
    Compares two models based on their AUC and Accuracy and determines the better model.

    Args:
        model1_metrics (dict): Metrics for Model 1.
        model2_metrics (dict): Metrics for Model 2.
        model1_savepath (str): Path to save Model 1.
        model2_savepath (str): Path to save Model 2.

    Returns:
        The save path of the better model.
    """
    model1_auc = model1_metrics['auc']
    model1_accuracy = model1_metrics['accuracy']
    model2_auc = model2_metrics['auc']
    model2_accuracy = model2_metrics['accuracy']

    if model1_auc > model2_auc:
        return model1_savepath
    elif model2_auc > model1_auc:
        return model2_savepath
    else:  # If AUCs are equal, use accuracy as a tiebreaker
        if model1_accuracy >= model2_accuracy:
            return model1_savepath
        else:
            return model2_savepath

# Argument parser setup
parser = argparse.ArgumentParser(description="Compare two models and determine the better one.")
parser.add_argument("--model1", type=str, required=True, help="Path to saved Model 1")
parser.add_argument("--model1_metrics", type=str, required=True, help="Path to JSON file with metrics for Model 1")
parser.add_argument("--model2", type=str, required=True, help="Path to saved Model 2")
parser.add_argument("--model2_metrics", type=str, required=True, help="Path to JSON file with metrics for Model 2")
parser.add_argument("--better_model", type=str, required=True, help="Path to save the better model's path")

args = parser.parse_args()

# Resolve the actual metrics file paths
model1_metrics_path = Path(args.model1_metrics)
if model1_metrics_path.is_dir():
    model1_metrics_path = next(model1_metrics_path.glob("*.json"))

model2_metrics_path = Path(args.model2_metrics)
if model2_metrics_path.is_dir():
    model2_metrics_path = next(model2_metrics_path.glob("*.json"))

# Load metrics from JSON files
with open(model1_metrics_path, "r") as f:
    model1_metrics = json.load(f)

with open(model2_metrics_path, "r") as f:
    model2_metrics = json.load(f)

# Determine the better model
better_model_path = determine_better_model(model1_metrics, model2_metrics, args.model1, args.model2)