"""
Author: Zeliha Ural Merpez

Date: Dec 04, 2020

This script reads optimized model found from analysis.py.

Usage: analysis_feature.py -i=<input> -o=<output> [-v]

Options:
-i <input>, --input <input>     Local raw data pkl filename and path
-o <output>, --output <output>  Local output directory for created pngs
[-v]                            Report verbose output of dataset retrieval process
"""

# example to run: python src/analysis_feature.py -i results/tables/optimized_model.pkl -o results

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

def main(input_file_path, output_folder_path):
    print("\n\n##### Feature Importances")
    if verbose: print(f"Running eda script with arguments: \n {args}")
    
    # Validates input argument paths by validate_input(input_file_path, output_dir_path)
    validate_inputs(input_file, output_dir)
    
    # Reads input file path and reads optimized model
    best_depth = read_input_file(input_file)    
    
    data_frame = pd.read_csv("data/processed/character_features_train.csv")
    
    # Processes the data, trains the model, and returns a dataframe showing the feature importances in descending order
    result_df = fit_best_model(data_frame, best_depth)
    
    # Separates feature coefficients of the optimized model results, and saves resulting table
    separate_feature_coefficients(result_df, output_folder_path, "importance")   
    print("\n##### Separating Feature Importances Completed!")

def validate_inputs(input_file_path, output_dir_path):
    """
    Validates input argument paths.
    Parameters:
    -----------
    input_file_path : str
        input path to be verified

    output_file_path : str
        output path to be verified
        
    Returns:
    -----------
    None
    """
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
    Reads input file path and reads optimized model.
    Parameters:
    -----------
    input_file_path : str
        input path to be verified
        
    Returns:
    -----------
    float
        The depth of bests model.
    """
    try:
        with open(input_file_path, "rb") as pickle_file:
            data_frame = pd.read_pickle(pickle_file)
        if verbose: print('Input filename path is valid.')
    except:
        print(input_file_path + 'Input filename path is not valid. Please check!')
        sys.exit()

    best_depth = data_frame.loc[0, 'param_LGBMC__max_depth']
    if verbose: print('Creating and returning optimized model depth.')
    return best_depth

def fit_best_model(data_frame, best_depth):
    """
    Processes the data, trains the model, and returns a dataframe showing the feature importances in descending order

    Parameters
    ----------
    train_df : pd.Dataframe
        Dataframe to be used in model

    models : dict
        models to be trained

    param_grid : dict
        hyperparameters for model

    Returns
    ----------
        pandas.Dataframe

    """
    X_train = data_frame.drop(columns=["align"])
    y_train = data_frame["align"]

    numeric_features = ["appearances"]
    categorical_features = ["id", "eye", "hair", "publisher", "sex"]
    ordinal_features = ["year"]
    drop_features = ["name", "gsm", "first_appearance"]
    year_levels = set(data_frame["year"].unique())
    year_levels = list(year_levels)
    # Cutting out NaN value
    year_levels = year_levels[1:]

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    categorical_transformer = categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse=False),
    )

    ordinal_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder(categories=[year_levels], dtype=float),
    )

    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features),
        (ordinal_transformer, ordinal_features),
    )

    model = make_pipeline(preprocessor, RandomForestClassifier(max_depth=best_depth))
    model.fit(X_train, y_train)

    ohe_columns_cat = list(
        preprocessor.named_transformers_["pipeline-2"]
        .named_steps["onehotencoder"]
        .get_feature_names(categorical_features)
    )
    features_list = numeric_features + ohe_columns_cat + ordinal_features
    result_df = pd.DataFrame(
        {
            "features": features_list,
            "Importance_coefficient": model.named_steps[
                "randomforestclassifier"
            ].feature_importances_,
        }
    )
    result_df = result_df.sort_values(by='Importance_coefficient', ascending=False)
    if verbose: print("Feature importances extracted from best model as a data frame.")
    return result_df

def separate_feature_coefficients(data_frame, output_folder, file_name):
    """
    Separates feature coefficients of the optimized model results.
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
    result_df = data_frame
    result_df[['Feature','Sub-feature']] = result_df.features.str.split('_', expand=True) 
    result_df = result_df[['Feature', 'Sub-feature', 'Importance_coefficient']]
    #separate each feature
    separated_dict = dict(tuple(result_df.groupby('Feature')))
    for key in separated_dict.keys():
        separated_dict[key].to_pickle(output_folder + "/tables/" + file_name + "_of_"+ key + ".pkl")
        fig_1, ax_1 = render_table(separated_dict[key], header_columns=0, col_width=5)
        fig_1.savefig(output_folder + "/figures/" + file_name +  "_of_" +key)

    if verbose: print("Saved separate feature coefficients as " + 
                      output_folder + "/tables/" + file_name + ".pkl and " +
                      output_folder + "/figures/" + file_name + ".png")
    return


if __name__ == "__main__":    
    input_file = args["--input"]
    output_dir = args["--output"]
    verbose = args["-v"]
    main(input_file, output_dir)