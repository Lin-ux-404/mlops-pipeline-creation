from mldesigner import command_component, Input, Output
from pathlib import Path
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

@command_component(
        environment = '../env.yaml',
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


