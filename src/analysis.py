"""
Author: Aidan Mattrick

Date: Nov 28, 2020

This script trains models and output results in the form of a figure to be used for further analysis.

Usage: analysis.py -i=<input> -o=<output> [-f <filename>] [-v]

Options:
-i <input>, --input <input>                 Local processed training data csv file in directory
-o <output>, --output <output>              Local output directory for created pngs
-f <filename>, --filename <filename>        Add a prefix to the saved filename
[-v]                                        Report verbose output of dataset retrieval process
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

from xgboost import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from catboost import CatBoostClassifier

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
    combined_columns = ['name', 'id', 'align', 'eye', 'hair', 'sex', 'gsm','appearances', 'first_appearance', 'year', 'publisher', 'first_name', 'last_name', 'is_common', 'name_len', 'has_last_name', 'appear_per_yr']

    if not all([item in data_frame.columns for item in combined_columns]):
        print(input_file_path + " should contain these columns: " + str(combined_columns))
        sys.exit()

    if verbose: print('Creating and returning EDA data frame.')
    return data_frame

#This function is adapted from Varada's lecture code in DSCI571


def store_cross_val_results(model, X_train, y_train,
                           scoring_metric = "accuracy"):
    """
    Returns mean and std of cross validation.

    Parameters
    ----------
    model :
        scikit-learn classification model
    X_train : DataFrame
        X Training data, indepedent variables
    y_train : DataFrame
        Training data, dependent variables
    scroing_metric: string
        Metric to use for scoring

    Returns
    ----------
        Dict
    """
    scores = cross_validate(model,
                            X_train, y_train,
                            return_train_score=True,
                            n_jobs=-1,
                            scoring=scoring_metric)
    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data = out_col, index = mean_scores.index)


def train_models(train_df, models, output_dir="", fileprefix=""):
    """
    Processes the data, trains the model, and returns a dataframe showing the results

    Parameters
    ----------
    train_df : pd.Dataframe
        Dataframe to be used in model

    models : dict
        Models to be trained

    ouput_dir : string
        The directory where the output of the model will be generated

    fileprefix : string
        Prefix to be added to the filename of the output

    Returns
    ----------
        pandas.Dataframe

    """
    X_train = train_df.drop(columns=['align'])
    y_train = train_df['align']

    numeric_features = ['appearances', 'year', 'name_len', 'appear_per_yr']
    categorical_features = ['id', 'eye', 'hair', 'publisher', 'sex']
    drop_features = ['name', 'gsm', 'first_appearance', 'first_name', 'last_name']
    binary_features = ['is_common', 'has_last_name']

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

    binary_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown='error', drop='if_binary', dtype=int)),
        ]
    )

    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features),
        (binary_transformer, binary_features),
    )

    results_df = {}

    for key, value in models.items():
        pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (key, value)
                ]
            )
        results_df[key] = store_cross_val_results(pipeline, X_train, y_train)

    results = pd.DataFrame(results_df)
    results.reset_index(inplace=True)
    results = results.rename(columns = {'index':'Scores'})
    return results

def optimize_model(train_df, param_grid=None, output_dir="", fileprefix=""):
    """
    Processes the data, performs hyperparameter optimization on the model, and returns that model

    Parameters
    ----------
    train_df : pd.Dataframe
        Dataframe to be used in model

    param_grid : dict
        hyperparameters for model

    ouput_dir : string
        The directory where the output of the model will be generated

    fileprefix : string
        Prefix to be added to the filename of the output

    Returns
    ----------
        pandas.Dataframe

    """
    X_train = train_df.drop(columns=['align'])
    y_train = train_df['align']

    numeric_features = ['appearances', 'year', 'name_len', 'appear_per_yr']
    categorical_features = ['id', 'eye', 'hair', 'publisher', 'sex']
    drop_features = ['name', 'gsm', 'first_appearance', 'first_name', 'last_name']
    binary_features = ['is_common', 'has_last_name']

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

    binary_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown='error', drop='if_binary', dtype=int)),
        ]
    )

    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features),
        (binary_transformer, binary_features),
    )

    results_df = {}

    boosted_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("LGBMC", LGBMClassifier()),
                    ]
            )

    random_search = RandomizedSearchCV(boosted_forest_pipeline,
                                            param_distributions=param_grid,
                                            cv=5,
                                            n_jobs=-1,
                                            n_iter=20,
                                            return_train_score=True)
    random_search.fit(X_train, y_train)
    if output_dir:
        pickle.dump(random_search.best_estimator_,
        open(output_dir + "/models/optimized_model.pkl", 'wb'))

    results = pd.DataFrame(random_search.cv_results_).set_index("rank_test_score").sort_index()
    results.reset_index(inplace=True)
    results = results.rename(columns = {'index':'Ranked Test Scores'})
    return results

def save_img(data_frame, output_folder, file_name, filename_prefix = ""):
    data_frame.to_pickle(output_folder + "/tables/" + filename_prefix + file_name + ".pkl")
    fig_1, ax_1 = render_table(data_frame, header_columns=0, col_width=5)
    fig_1.savefig(output_folder + "/figures/" + filename_prefix + file_name)

def save_img_large(data_frame, output_folder, file_name, filename_prefix = ""):
    data_frame.to_pickle(output_folder + "/tables/" + filename_prefix + file_name + ".pkl")
    fig_1, ax_1 = render_table(data_frame, header_columns=0, col_width=9)
    fig_1.savefig(output_folder + "/figures/" + filename_prefix + file_name)

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

    #Run multiple Forest models to select the best one
    if verbose: print("Training model(s)")
    models = {
        "Random Forest Classifier": RandomForestClassifier(),
        "XGBClassifier": XGBClassifier(),
        "LGBMClassifier": LGBMClassifier(),
        "CatBoostClassifier": CatBoostClassifier(verbose=0)
    }
    model_df = train_models(train_df, models)
    if verbose: print("Trained Forest model(s)")
    if filename_prefix:
        save_img(model_df, output_folder_path, "forest_model_comparison", filename_prefix)
    else:
        save_img(model_df, output_folder_path, "forest_model_comparison")

    #Performing hyperparameter optimization on LightGBM
    if verbose: print("Performing hyperparameter optimization on best model")

    param_grid = {'LGBMC__n_estimators' : [5, 100, 500, 700, 1000, 1500, 4000],
              'LGBMC__learning_rate' : [0.01, 0.1, 1],
              'LGBMC__max_depth' : [1, 3, 5, 6, 10],
              'LGBMC__subsample' : [0.15, 0.25, 0.5, 0.75, 1]
             }

    model_df = optimize_model(train_df, param_grid, output_dir)
    if verbose: print("Optimized model")

    if filename_prefix:
        save_img_large(model_df, output_folder_path, "optimized_model", filename_prefix)
    else:
        save_img_large(model_df, output_folder_path, "optimized_model")

    print("\n\n##### Analysis: Training Models Complete!")


if __name__ == "__main__":
    print(args)
    verbose = args["-v"]
    input_file = args["--input"]
    output_dir = args["--output"]
    filename_prefix = args["--filename"]
    main(input_file, output_dir)