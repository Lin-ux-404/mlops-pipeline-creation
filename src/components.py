from mldesigner import command_component, Input, Output
import subprocess
from pathlib import Path
import os
# import libraries
import argparse
import glob
import pandas as pd
import mlflow
from sklearn.preprocessing import MinMaxScaler

@command_component(
        environment = './env.yaml',
        name = "remove_empty_rows",
        display_name = "Remove Empty Rows")

def remove_empty_rows(input_data: Input(type="uri_folder"), output_data: Output(type="uri_folder")):

    # load the data (passed as an input dataset)
    data_path = input_data
    all_files = glob.glob(data_path + "/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)

    # remove nulls
    df = df.dropna()

    # set the processed data as output
    df.to_csv((Path(output_data) / "output_data.csv"))



@command_component(
        environment = './env.yaml',
        name = "normailize_data",
        display_name = "Normalize Data")

def normalize_data(input_data: Input(type="uri_folder"), output_data: Output(type="uri_folder")):
    """Normalize numerical columns in the dataset."""
    # load the data (passed as an input dataset)
    print("files in input_data path: ")
    arr = os.listdir(input_data)
    print(arr)

    for filename in arr:
        print("reading file: %s ..." % filename)
        with open(os.path.join(input_data, filename), "r") as handle:
            print(handle.read())

    data_path = input_data
    all_files = glob.glob(data_path + "/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)
        

    # normalize the numeric columns
    scaler = MinMaxScaler()
    num_cols = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree']
    df[num_cols] = scaler.fit_transform(df[num_cols])


    # set the processed data as output
    df.to_csv((Path(output_data) / "output_data.csv"))

@command_component(
        environment = './env.yaml',
        name = "train_decision_tree_classifier_model",
        display_name = "Train Decision Tree Classifier Model")

def train_decision_tree_classifier_model(
    training_data: Input(type="uri_folder"),
    model_output_decision_tree: Output(type="uri_folder"),
    metrics_output: Output(type="uri_file")
):
    """Train a decision tree classifier model."""

    # Run the corresponding script
    subprocess.run([
        "python", "train_decision_tree.py",
        "--training_data", str(training_data),
        "--model_output_decision_tree", str(model_output_decision_tree),
        "--metrics_output", str(metrics_output)
    ])



@command_component(
        environment = './env.yaml',
        name = "train_logistic_regression_classifier_model",
        display_name = "Train logistic regression classifier model")

def train_logistic_regression_classifier_model(
    training_data: Input(type="uri_folder"),
    model_output_logistic_reg: Output(type="uri_folder"),
    metrics_output: Output(type="uri_file"),
    regularization_rate: float = 0.01,
):
    """Train a logistic regression classifier model."""

    # Run the corresponding script
    subprocess.run([
        "python", "train_logistic_regression.py",
        "--training_data", str(training_data),
        "--reg_rate", str(regularization_rate),
        "--model_output_logistic_reg", str(model_output_logistic_reg),
        "--metrics_output", str(metrics_output)
    ])


@command_component(
        environment = './env.yaml',
        name = "compare_two_models",
        display_name = "Comparing Two Models")

def compare_two_models(
    model1: Input(type="uri_folder", description="Path to the first model"),
    model1_metrics: Input(type="uri_file", description="JSON file containing AUC and accuracy for the first model"),
    model2: Input(type="uri_folder", description="Path to the second model"),
    model2_metrics: Input(type="uri_file", description="JSON file containing AUC and accuracy for the second model"),
    better_model: Output(type="uri_folder", description="Path of the better model based on comparison")
):
    """Compare two models based on their metrics and output the better model."""
    
    # Run the script to compare models
    subprocess.run([
        "python", "compare-models.py",
        "--model1", str(model1),
        "--model1_metrics", str(model1_metrics),
        "--model2", str(model2),
        "--model2_metrics", str(model2_metrics),
        "--better_model", str(better_model)
    ])
