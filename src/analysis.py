"""
This script trains models and output results in the form of a figure to be used for further analysis.

Usage: analysis.py -i=<input> -o=<output> [-v]

Options:
-i <input>, --input <input>     Local raw data csv file directory
-o <output>, --output <output>  Local output filename and path for preprocessed csv
[-v]                            Report verbose output of dataset retrieval process
"""

import os
import re
import sys
from hashlib import sha1

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

print("start")
args = docopt(__doc__)

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
        data_frame = pd.read_csv(input_file_path, index_col=0, parse_dates = ['first_appearance'])
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



def train_model(dataframe):
    """
    Processes the data, trains the model, and returns a dataframe showing the results

    Parameters
    ----------
    dataframe : pd.Dataframe
        Dataframe to be used in model

    Returns
    ----------
        pandas.Dataframe

    """
    train_df = dataframe
    X_train = train_df.drop(columns=['align'])
    y_train = train_df['align']

    numeric_features = ['appearances']
    categorical_features = ['name', 'id', 'eye', 'hair', 'year', 'publisher', 'sex']
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

    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features),
    )

    results_df = {}

    models = {
        "Dummy Classifier": DummyClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "SVM": SVC(),
        "OVR LogisticRegression": OneVsRestClassifier(LogisticRegression()),
        "LogisticRegression": LogisticRegression(),
    }

    for key, value in models.items():
        pipeline = Pipeline(
                steps=[("preprocessor", preprocessor),
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
    fig_1, ax_1 = render_table(data_frame, header_columns=0, col_width=5)
    fig_1.savefig(output_folder + "/" + file_name)

def main(input_file_path, output_folder_path):
    print(input_file_path)
    data_frame = read_input_file(input_file_path)
    print("Generating")
    #process
    # preprocessor = process_data(data_frame)
    # print("processing data for model")
    #train
    model_df = train_model(data_frame)
    print("Trained model")
    save_img(model_df, output_folder_path, "Table_03")
    print("Done")


if __name__ == "__main__":
    print("101")
    input_file = args["--input"]
    output_dir = args["--output"]
    verbose = args["-v"]
    print("102")
    main(input_file, output_dir)