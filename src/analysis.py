"""
Author: Aidan Mattrick

Date: Nov 28, 2020

This script trains models and output results in the form of a figure to be used for further analysis.

Usage: analysis.py -i=<input> -o=<output> [-v]

Options:
-i <input>, --input <input>     Local processed training data csv file in directory
-o <output>, --output <output>  Local output directory for created pngs
[-v]                            Report verbose output of dataset retrieval process
"""

import os
import re
import sys
from hashlib import sha1
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_transformer,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, SVR

import altair as alt
from vega_datasets import data

from docopt import docopt
from pylab import savefig
from render_table import render_table
args = docopt(__doc__)

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
    if not os.path.exists(output_dir_path + "/models"):
        os.makedirs(output_dir_path + "/models")
    assert os.path.exists(output_dir_path), f"Invalid output path: {output_dir_path}"
    assert os.path.exists(output_dir_path + "/figures"), f"Invalid output path: {output_dir_path}/figures"
    assert os.path.exists(output_dir_path + "/tables"), f"Invalid output path: {output_dir_path}/tables"
    assert os.path.exists(output_dir_path + "/tables"), f"Invalid output path: {output_dir_path}/tables"


def read_input_file(input_file_path):
    """
    Reads input file path and reads cleaned data.
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

#This function is adapted from Varada's lecture code in DSCI571
def store_cross_val_results(model, scores, results):
    """
    Stores mean scores from cross_validate in results_dict for
    the given model model_name.

    Parameters
    ----------
    model :
        scikit-learn classification model
    scores : dict
        object return by `cross_validate`
    results_dict: dict
        dictionary to store results

    Returns
    ----------
        None

    """
    results[model] = {
        "mean_validation_accuracy": "{:0.4f}".format(np.mean(scores["test_score"])),
        "mean_train_accuracy": "{:0.4f}".format(np.mean(scores["train_score"])),
        "mean_fit_time (s)": "{:0.4f}".format(np.mean(scores["fit_time"])),
        "mean_score_time (s)": "{:0.4f}".format(np.mean(scores["score_time"])),
        "std_train_score": "{:0.4f}".format(scores["train_score"].std()),
        "std_test_score": "{:0.4f}".format(scores["test_score"].std()),
    }



def train_models(train_df, models, param_grid=None, output_dir=""):
    """
    Processes the data, trains the model, and returns a dataframe showing the results

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
    X_train = train_df.drop(columns=['align'])
    y_train = train_df['align']

    numeric_features = ['appearances']
    categorical_features = ['name', 'id', 'eye', 'hair', 'publisher', 'sex']
    ordinal_features = ['year']
    drop_features = ['gsm', 'first_appearance']

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("onehot", OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    #Fixing year levels for ordinal encoding
    year_levels = set(train_df['year'].unique())
    year_levels = list(year_levels)
    #Cutting out NaN value
    year_levels = year_levels[1:]

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("ordinal", OrdinalEncoder(categories=[year_levels], dtype=float))
        ]
    )

    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features),
        (ordinal_transformer, ordinal_features)
    )

    results_df = {}

    if param_grid:
        for key, value in models.items():
            random_forest_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (key, value),
                    ]
            )

            random_search = RandomizedSearchCV(random_forest_pipeline, 
                                               param_distributions=param_grid, 
                                               cv=5, 
                                               n_jobs=-1, 
                                               n_iter=20, 
                                               return_train_score=True)
            random_search.fit(X_train, y_train)
            if output_dir:
                pickle.dump(random_search.best_estimator_, 
                open(output_dir + "\models\optimized_model.pkl", 'wb'))

        results = pd.DataFrame(random_search.cv_results_).set_index("rank_test_score").sort_index()
        results.reset_index(inplace=True)
        results = results.rename(columns = {'index':'Ranked Test Scores'})
        return results


    for key, value in models.items():
        pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (key, value)
                ]
            )
        scores = cross_validate(pipeline, X_train, y_train, return_train_score=True, n_jobs=-1, verbose=1)
        store_cross_val_results(key, scores, results_df)

    results = pd.DataFrame(results_df)
    results.reset_index(inplace=True)
    results = results.rename(columns = {'index':'Scores'})
    return results


def save_img(data_frame, output_folder, file_name):
    data_frame.to_pickle(output_folder + "/tables/" + file_name + ".pkl")
    fig_1, ax_1 = render_table(data_frame, header_columns=0, col_width=5)
    fig_1.savefig(output_folder + "/figures/" + file_name)

def save_img_large(data_frame, output_folder, file_name):
    data_frame.to_pickle(output_folder + "/tables/" + file_name + ".pkl")
    fig_1, ax_1 = render_table(data_frame, header_columns=0, col_width=8)
    fig_1.savefig(output_folder + "/figures/" + file_name)

def main(input_file_path, output_folder_path):
    print("\n\n##### Analysis: Training Models!")
    if verbose: print(f"Running analysis script with arguments: \n {args}")
    validate_inputs(input_file, output_dir)
    data_frame = read_input_file(input_file_path)

    #train_data
    if verbose: print("Generating training data")
    train_df = data_frame

    #Run multiple models to select the best one
    if verbose: print("Training model(s)")
    models = {
        "Dummy Classifier": DummyClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "SVM": SVC(),
        "OVR LogisticRegression": OneVsRestClassifier(LogisticRegression()),
        "LogisticRegression": LogisticRegression(),
    }
    model_df = train_models(train_df, models)
    if verbose: print("Trained model(s)")
    save_img(model_df, output_folder_path, "model_comparison")

    #Performing hyperparameter optimization on best one
    if verbose: print("Performing hyperparameter optimization on best model")
    models = {
        "Random Forest Classifier": RandomForestClassifier(),
    }
    param_grid = {"Random Forest Classifier__max_depth": 10.0 ** np.arange(-10, 10)}
    model_df = train_models(train_df, models, param_grid, output_dir)
    if verbose: print("Trained model(s)")
    save_img_large(model_df, output_folder_path, "optimized_model")
    print("\n\n##### Analysis: Training Models Complete!")


if __name__ == "__main__":
    verbose = args["-v"]
    input_file = args["--input"]
    output_dir = args["--output"]
    main(input_file, output_dir)