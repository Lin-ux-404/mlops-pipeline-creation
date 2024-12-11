import argparse
import json
from pathlib import Path
from distutils.dir_util import copy_tree
from mldesigner import command_component, Input, Output
import subprocess

@command_component(
        environment = '../env.yaml',
        name = "compare_two_models",
        display_name = "Comparing Two Models")

def compare_two_models(
    model1: Input(type="uri_folder", description="Path to the first model"),
    model1_metrics: Input(type="uri_file", description="JSON file containing AUC and accuracy for the first model"),
    model2: Input(type="uri_folder", description="Path to the second model"),
    model2_metrics: Input(type="uri_file", description="JSON file containing AUC and accuracy for the second model"),
    better_model: Output(type="uri_folder", description="Path of the better model based on comparison")
):

    # Resolve the actual metrics file paths
    model1_metrics_path = Path(model1_metrics)
    if model1_metrics_path.is_dir():
        model1_metrics_path = next(model1_metrics_path.glob("*.json"))

    model2_metrics_path = Path(model2_metrics)
    if model2_metrics_path.is_dir():
        model2_metrics_path = next(model2_metrics_path.glob("*.json"))

    # Load metrics from JSON files
    with open(model1_metrics_path, "r") as f:
        model1_metrics = json.load(f)

    with open(model2_metrics_path, "r") as f:
        model2_metrics = json.load(f)

    # Determine the better model
    better_model_path = determine_better_model(model1_metrics, model2_metrics, model1, model2)
    print(f"The better model is located at: {better_model_path}")
    print(f"Metrics were here: {model1_metrics_path}")


    # Save the better model to the specified output directory
    better_model_output_path = Path(better_model)
    better_model_output_path.mkdir(parents=True, exist_ok=True)

    # Copy the entire directory tree to the output directory
    better_model_source_path = Path(better_model_path)
    if not better_model_source_path.exists():
        raise FileNotFoundError(f"The determined better model path does not exist: {better_model_path}")

    # Use distutils to copy the contents of the directory
    copy_tree(str(better_model_source_path), str(better_model_output_path))


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
