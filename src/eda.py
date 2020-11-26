"""
This script reads combined data and generates images and tables to be used in further analysis.

Usage: eda.py -i=<input> -o=<output> [-v]

Options:
-i <input>, --input <input>     Local raw data csv filename and path
-o <output>, --output <output>  Local output directory for created png
[-v]                            Report verbose output of dataset retrieval process
"""

# example to run: python src/eda.py -i "data/processed/clean_characters.csv" -o "img"

import sys
import os
import numpy as np
import pandas as pd
#import altair as alt
import matplotlib.pyplot as plt
#import seaborn as sns
#from pathlib import Path
from docopt import docopt
#from pylab import savefig
#from selenium import webdriver
import matplotlib.pyplot as plt
from render_table import render_table
print("start")
args = docopt(__doc__)
#from webdriver_manager.chrome import ChromeDriverManager

#from altair import pipe, limit_rows, to_values
#from pandas_profiling.report.presentation.core import Table, Container

#t = lambda data: pipe(data, limit_rows(max_rows=20000), to_values)
#alt.data_transformers.register('custom', t)
#alt.data_transformers.enable('custom')

def read_input_file(input_file_path):
    """
    Validates input file path.
    Parameters:
    -----------
    input_file_path : str
        input path to be verified
        
    Returns:
    -----------
    pandas.DataFrame
        if path is valid and verified
    """
    if not os.path.isfile(input_file_path):
        print("Input file does not exist.")
        sys.exit()

    try:
        data_frame = pd.read_csv(input_file_path, index_col=0)
        print('Path is valid')
    except:
        print(input_file_path + 'Path is not valid. Please check ')
        sys.exit()

    combined_columns = ['name', 'id', 'align', 'eye', 'hair', 'sex', 'gsm','appearances', 'first_appearance', 'year', 'publisher']

    if not all([item in data_frame.columns for item in combined_columns]):
        print(input_file_path + " should contain these columns: " + str(combined_columns))
        sys.exit()

    print('Returning data frame')
    return data_frame

def generate_dataset_overview(data_frame, output_folder, file_name):
    """
    Generates an overview of the dataset.
    Also saves resulting table as file in given output folder.
    Parameters:
    -----------
    data_frame : pandas.DataFrame
        input path to be verified
    output_folder : str
        output folder path to save the chart
    file_name : str
        file name for generated chart image
        
    Returns:
    -----------
    str
        saved file path
    """
    data_overview = [
            {"Dataset": "Number of features", "Value": len(data_frame.columns)},
            {"Dataset": "Number of characters", "Value": len(data_frame)},
            {"Dataset": "Number of Missing cells", "Value": (data_frame.isnull()).sum().sum()},
            {"Dataset": "Percentage of Missing cells", "Value": round((data_frame.isnull()).sum().sum()/data_frame.size*100, 2)}
        ]
    overview_frame = pd.DataFrame(data_overview)
    fig_1, ax_1 = render_table(overview_frame, header_columns=0, col_width=5)
    fig_1.savefig(output_folder +"/"+ file_name)

    return overview_frame

def generate_feature_overview(data_frame, output_folder, file_name):
    """
    Generates an overview of the features in dataset.
    Also saves resulting table as file in given output folder.
    Parameters:
    -----------
    data_frame : pandas.DataFrame
        input path to be verified
    output_folder : str
        output folder path to save the chart
    file_name : str
        file name for generated chart image
        
    Returns:
    -----------
    object
        saved .png file
    """
    distinct_class = dict()
    nonnull_count = dict()

    for col in data_frame.columns:
        nonnull_count[col]=len(data_frame)-data_frame[col].isnull().sum()
        distinct_class[col]=len(list(data_frame[col].unique()))
    features_frame=pd.DataFrame([distinct_class, nonnull_count]).T.reset_index()
    features_frame.columns=["Features","Dictinct Class", "Non-Null Count"]
    features_frame["Missing Percentage"]=round((len(data_frame) - features_frame["Non-Null Count"])/len(data_frame)*100,2)

    fig_2, ax_2 = render_table(features_frame, header_columns=0, col_width=3)
    fig_2.savefig(output_folder +"/"+ file_name)

    return features_frame



def main(input_file_path, output_folder_path):
    print(input_file_path)
    data_frame = read_input_file(input_file_path)
    print("generating")
    generate_dataset_overview(data_frame, output_folder_path, "Dataset Overview")
    generate_feature_overview(data_frame, output_folder_path, "Feature Overview")
    print("done")


if __name__ == "__main__":
    print("101")
    
    input_file = args["--input"]
    output_dir = args["--output"]
    verbose = args["-v"]
    print("102")
    main(input_file, output_dir)