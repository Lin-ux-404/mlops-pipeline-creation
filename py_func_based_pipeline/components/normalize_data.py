from mldesigner import command_component, Input, Output
from pathlib import Path
import os
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

@command_component(
        environment = '../env.yaml',
        name = "normalize_data",
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