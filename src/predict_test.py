"""
Author: Aidan Mattrick

Date: Dec 9, 2020

This script takes in a trained model and runs it against the test set and returns a confusion matrix outline the results.

Usage: predict_test.py -i=<input> -o=<output> [-f <filename>] [-v]

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
from sklearn.metrics import plot_confusion_matrix


from lightgbm.sklearn import LGBMClassifier
from catboost import CatBoostClassifier

import altair as alt
from vega_datasets import data

from docopt import docopt
from pylab import savefig
from render_table import render_table
args = docopt(__doc__)

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
        data_frame = pd.read_csv(input_file_path)
        if verbose: print('Input filename path is valid.')
    except:
        print(input_file_path + 'Input filename path is not valid. Please check!')
        sys.exit()

    # TODO possibly move this to a config or test script to remove magic values
    combined_columns = ['name',
                        'id',
                        'align',
                        'eye',
                        'hair',
                        'sex',
                        'gsm',
                        'appearances',
                        'first_appearance',
                        'year',
                        'publisher',
                        'is_common',
                        'name_len',
                        'has_last_name',
                        'appear_per_yr']

    if not all([item in data_frame.columns for item in combined_columns]):
        print(input_file_path + " should contain these columns: " + str(combined_columns))
        sys.exit()

    if verbose: print('Creating and returning test data frame.')
    return data_frame


def save_matrix(matrix, output_folder, file_name, filename_prefix = ""):
    matrix.figure_.savefig(output_folder + "/figures/" + filename_prefix + file_name)

def main(input_file_path, output_folder_path):
    model = pd.read_pickle('results/models/optimized_model.pkl')
    print("\n\n##### Model imported!")

    test_df =  read_input_file(input_file_path)

    print("\n\n##### Testing Model!")
    X_test, y_test = test_df.drop(columns=['align']), test_df['align']

    score = model.score(X_test, y_test)
    print("\n\n##### Model Score:")
    print(f'{score}')

    confusion_matrix = plot_confusion_matrix(model, X_test, y_test, display_labels=["Bad", "Neutral", "Good"],
                      values_format="d", cmap=plt.cm.Blues);

    save_matrix(confusion_matrix, output_folder_path, 'confusion_matrix')


if __name__ == "__main__":
    print(args)
    verbose = args["-v"]
    input_file = args["--input"]
    output_dir = args["--output"]
    filename_prefix = args["--filename"]
    main(input_file, output_dir)


