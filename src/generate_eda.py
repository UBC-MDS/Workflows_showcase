"""
This script reads combined data and generates images and tables to be used in further analysis.

Usage: generate_eda.py -i=<input> -o=<output> [-v]

Options:
-i <input>, --input <input>     Local raw data csv filename and path
-o <output>, --output <output>  Local output directory for created png
[-v]                            Report verbose output of dataset retrieval process
"""

# example to run: python src/generate_eda.py -i "data/processed/clean_characters.csv" -o "results"

import sys
import os
import numpy as np
import pandas as pd
import altair as alt
import pickle
from docopt import docopt
from render_table import render_table
args = docopt(__doc__)

def validate_inputs(input_file_path, output_dir_path):
    if not os.path.isfile(input_file_path):
        print(f"Cannot locate input file: {input_file_path}")
        sys.exit()

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    if not os.path.exists(output_dir_path + "/figures"):
        os.makedirs(output_dir_path + "/figures")
    if not os.path.exists(output_dir_path + "/tables"):
        os.makedirs(output_dir_path + "/tables")
    assert os.path.exists(output_dir_path), f"Invalid output path: {output_dir_path}"
    assert os.path.exists(output_dir_path + "/figures"), f"Invalid output path: {output_dir_path}/figures"
    assert os.path.exists(output_dir_path + "/tables"), f"Invalid output path: {output_dir_path}/tables"


def read_input_file(input_file_path):
    """
    Validates input file path and reads cleaned data.
    Parameters:
    -----------
    input_file_path : str
        input path to be verified
        
    Returns:
    -----------
    pandas.DataFrame
        if path is valid and verified
    """
    try:
        data_frame = pd.read_csv(input_file_path, index_col=0)
        if verbose: print('Input filename path is valid.')
    except:
        print(input_file_path + 'Input filename path is not valid. Please check!')
        sys.exit()

    # TODO possibly move this to a config or test script to remove magic values
    combined_columns = ['name', 'id', 'align', 'eye', 'hair', 'sex', 'gsm','appearances', 'first_appearance', 'year', 'publisher']

    if not all([item in data_frame.columns for item in combined_columns]):
        print(input_file_path + " should contain these columns: " + str(combined_columns))
        sys.exit()

    if verbose: print('Creating and returning EDA data frame.')
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
    None
    """
    data_overview = [
            {"Dataset": "Number of features", "Value": len(data_frame.columns)},
            {"Dataset": "Number of characters", "Value": len(data_frame)},
            {"Dataset": "Number of Missing cells", "Value": (data_frame.isnull()).sum().sum()},
            {"Dataset": "Percentage of Missing cells", "Value": round((data_frame.isnull()).sum().sum()/data_frame.size*100, 2)}
        ]
    overview_frame = pd.DataFrame(data_overview)

    overview_frame.to_pickle(output_folder + "/tables/" + file_name + ".pkl")
    fig_1, ax_1 = render_table(overview_frame, header_columns=0, col_width=5)
    fig_1.savefig(output_folder + "/figures/" + file_name)

    if verbose: print("Saved EDA dataset_overview as " + 
                      output_folder + "/tables/" + file_name + ".pkl and " +
                      output_folder + "/figures/" + file_name + ".png")


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
    None
    """
    distinct_class = dict()
    nonnull_count = dict()

    for col in data_frame.columns:
        nonnull_count[col]=len(data_frame)-data_frame[col].isnull().sum()
        distinct_class[col]=len(list(data_frame[col].unique()))

    features_frame=pd.DataFrame([distinct_class, nonnull_count]).T.reset_index()
    features_frame.columns=["Features","Dictinct Class", "Non-Null Count"]
    features_frame["Missing Percentage"]=round((len(data_frame) - features_frame["Non-Null Count"])/len(data_frame)*100,2)

    features_frame.to_pickle(output_folder + "/tables/" + file_name + ".pkl")
    fig_2, ax_2 = render_table(features_frame, header_columns=0, col_width=3)
    fig_2.savefig(output_folder +"/figures/"+ file_name)

    if verbose: print("Saved EDA feature_overview as " + 
                      output_folder + "/tables/" + file_name + ".pkl and " +
                      output_folder + "/figures/" + file_name + ".png")


def generate_align_vs_features(data_frame, output_folder, file_name):
    """
    Generates a chart of the relation between align and other features in dataset.
    Also saves resulting image as file in given output folder.
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
    None
    """
    features = ['id', 'eye', 'hair', 'sex', 'gsm', 'publisher']
    align_vs_features = (alt.Chart(data_frame).mark_circle().encode(
        alt.Y(alt.repeat(), type='ordinal'),
        alt.X('count()', title = "Character Count"),
        size =alt.Size('count()', legend=alt.Legend(title="Characters")),
        color = alt.Color("align", legend=alt.Legend(title="Alignment"))
        ).properties(height=300, width=200).repeat(repeat=features, columns=3))

    align_vs_features.save(output_folder +"/figures/" + file_name + '.png')
    if verbose: print("Alignment vs features chart created, saved to " + 
                      output_folder + 
                      "/figures/" + 
                      file_name + 
                      '.png')


def generate_align_vs_year(data_frame, output_folder, file_name):
    """
    Generates a chart of the relation between align and year in dataset.
    Also saves resulting image as file in given output folder.
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
    None
    """
    align_vs_year = (alt.Chart(data_frame, title = "Alignment over Time").mark_line().encode(
        alt.X('year:T', title = 'Year(1935-2013)'),
        y = alt.Y('count()', title = "Character Count"),
        color = alt.Color("align", title="Alignment"),
        tooltip = 'year'
        ).properties(height=300, width=500)).interactive()

    align_vs_year.save(output_folder +"/figures/" + file_name + '.png')
    if verbose: print("Alignment vs year chart created, saved to " + 
                      output_folder + 
                      "/figures/" + 
                      file_name + 
                      '.png')


def generate_align_vs_appearances(data_frame, output_folder, file_name):
    """
    Generates a chart of the relation between align and appearances in dataset.
    Also saves resulting image as file in given output folder.
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
    None
    """
    align_vs_appearances = (
        alt.Chart(
            data_frame.dropna(), title="Character Appearances by Alignment"
            ).mark_boxplot().encode(
                alt.X('appearances:Q', title = 'Appearances'),
                y = alt.Y('align:O', title = "Alignment"),
                color = alt.Color("align", title = "Alignment"),
                size='count()'
                ).properties(height=300, width=500)).interactive()

    align_vs_appearances.save(output_folder +"/figures/" + file_name + '.png')
    if verbose: print("Alignment vs appearances chart created, saved to " + 
                      output_folder + 
                      "/figures/" + 
                      file_name + 
                      '.png')


def main(input_file_path, output_folder_path):
    print("\n\n##### EDA: Generating EDA Data!")
    if verbose: print(f"Running eda script with arguments: \n {args}")
    validate_inputs(input_file, output_dir)
    data_frame = read_input_file(input_file_path)
    generate_dataset_overview(data_frame, output_folder_path, "dataset_overview")
    generate_feature_overview(data_frame, output_folder_path, "feature_overview")
    generate_align_vs_features(data_frame, output_folder_path, "alignment_vs_features")
    generate_align_vs_year(data_frame, output_folder_path, "alignment_over_time")
    generate_align_vs_appearances(data_frame, output_folder_path, "appearances_by_alignment")
    print("\n##### EDA: EDA Data Generation Completed!")


if __name__ == "__main__":    
    input_file = args["--input"]
    output_dir = args["--output"]
    verbose = args["-v"]
    main(input_file, output_dir)