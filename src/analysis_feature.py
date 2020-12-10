"""
Author: Zeliha Ural Merpez

Date: Dec 04, 2020

This script reads optimized model found from analysis.py and train data produced by feature_engineer.py.

Usage: analysis_feature.py -i=<input_1> -j=<input_2> -o=<output> [-v]

Options:
-i <input_1>, --input_1 <input_1>     Local raw data pkl filename and path
-j <input_2>, --input_2 <input_2>     Local raw data csv filename and path
-o <output>, --output <output>  Local output directory for created pngs
[-v]                            Report verbose output of dataset retrieval process
"""

# example to run: python src/analysis_feature.py -i results/models/optimized_model.pkl -j data/processed/character_features_train.csv -o results

import sys
import os
import numpy as np
import pandas as pd
import pickle
from docopt import docopt
from render_table import render_table
from sklearn import datasets
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_transformer,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import export_graphviz
from subprocess import call

args = docopt(__doc__)

def main(input_model, input_csv, output_folder_path):
    print("\n\n##### Feature Importances")
    if verbose: print(f"Running analysis_feature script with arguments: \n {args}")
    
    # Validates input argument paths by validate_input(input_file_path_1, input_file_path_2, output_dir_path)
    validate_inputs(input_model, input_csv, output_folder_path)
    
    # Reads input file paths and reads optimized model and processed train data
    model, data_frame = read_input_file(input_model, input_csv)    
    
    # Processes the data, trains the model, and returns a dataframe showing the top 15 feature importances in descending order
    result_df = fit_best_model(model, data_frame, output_folder_path, "importance")
  
    print("\n##### Feature Importances Completed!")


def validate_inputs(input_file_path_1, input_file_path_2, output_dir_path):
    """
    Validates input argument paths.
    Parameters:
    -----------
    input_file_path_1 : str
        input path to be verified

    input_file_path_2 : str
        input path to be verified

    output_file_path : str
        output path to be verified
        
    Returns:
    -----------
    None
    """
    if not os.path.isfile(input_file_path_1):
        print(f"Cannot locate input file: {input_file_path_1}")
        sys.exit()
    if not os.path.isfile(input_file_path_2):
        print(f"Cannot locate input file: {input_file_path_2}")
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


def read_input_file(input_model, input_csv):
    """
    Reads input file paths, reads optimized model and processed train data .
    Parameters:
    -----------
    input_model : str
        input path to be verified
    
    input_csv : str
        input path to be verified
        
    Returns:
    -----------
    sklearn.pipeline.Pipeline, pd.DataFrame
        best model from analysis.py optimized model and train data frame
    """
    try:
        with open(input_model, "rb") as pickle_file:
            model = pd.read_pickle(pickle_file)
        if verbose: print('Input filename path is valid.')
        data_frame = pd.read_csv(input_csv)

    except:
        print(input_file_path + 'Input filename path is not valid. Please check!')
        sys.exit()

    if verbose: print('Returning optimized model and train data frame.')
    
    return model, data_frame


def fit_best_model(model, data_frame, output_folder, file_name):
    """
    Processes the data, trains the model, and returns a dataframe showing the top 15 feature importances in descending order

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Best model to be used in fitting

    data_frame : pd.Dataframe
        Dataframe to be used in model

    output_folder : str
        output folder path
    
    file_name : str
        generated png output file name 

    Returns
    ----------
        None
    """
    X_train = data_frame.drop(columns=["align"])
    y_train = data_frame["align"]

    numeric_features = ['appearances', 'year', 'name_len', 'appear_per_yr']
    categorical_features = ['id', 'eye', 'hair', 'publisher', 'sex']
    drop_features = ['name', 'gsm', 'first_appearance']
    binary_features = ['is_common', 'has_last_name']

    model.fit(X_train, y_train)
   
    ohe_columns_cat = list(
        model.named_steps["preprocessor"].named_transformers_["pipeline-2"]
        .named_steps["onehot"]
        .get_feature_names(categorical_features)
    )

    features_list = numeric_features + ohe_columns_cat + binary_features
    
    result_df = pd.DataFrame(
        {
            "Features": features_list,
            "Importance Coefficient": model.named_steps[
                "LGBMC"
            ].feature_importances_,
            "Importance Type": model.named_steps[
                "LGBMC"
            ].importance_type,
        }
    )

    result_df = result_df.sort_values(by='Importance Coefficient', ascending=False)
    if verbose: print("Feature importances extracted from best model as a data frame.")

    fig_1, ax_1 = render_table(result_df[:15], header_columns=0, col_width=5)
    fig_1.savefig(output_folder + "/figures/" + file_name)
    result_df[:15].to_pickle(output_folder + "/tables/" + file_name)

    if verbose: print("Saved separate feature coefficients as " + 
                      output_folder + "/tables/" + file_name + ".pkl and " +
                      output_folder + "/figures/" + file_name + ".png")
    return 


if __name__ == "__main__":    
    input_model_file_path = args["--input_1"]
    input_csv_file_path = args["--input_2"]
    output_dir = args["--output"]
    verbose = args["-v"]
    main(input_model_file_path, input_csv_file_path, output_dir)