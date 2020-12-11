"""
Author: Aidan Mattrick

Date: Nov 28, 2020

This script trains models and output results in the form of a figure to be used for further analysis.

Usage: model_selection.py -i=<input> -o=<output> [-f <filename>] [-v]

Options:
-i <input>, --input <input>                 Local processed training data csv file in directory
-o <output>, --output <output>              Local output directory for created pngs
-f <filename>, --filename <filename>        Add a prefix to the saved filename
[-v]                                        Report verbose output of dataset retrieval process
"""

import os
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from character_utils import *
from predictive_model import CharacterPredictiveModel

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from lightgbm.sklearn import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from docopt import docopt

args = docopt(__doc__)

def main(input_file, output_dir):
    print("\n\n##### Model Selection: Comparing Models!")
    if verbose: print(f"Running analysis script with arguments: \n {args}")
    validate_inputs(input_file, output_dir)

    # Instantiate the parent character model class object
    character_model = initialize_character_model()

    # Run multiple model types to help select the best
    compare_model_types(character_model)

    # Run multiple Forest decision tree models to select the best one
    compare_decision_tree_forests(character_model)

    print("\n\n##### Model Selection: Training Models Complete!")


def initialize_character_model():
    """
    Reads in data and instantiates character_model
    Parameters:
    -----------
    None

    Returns:
    -----------
    CharacterPredictiveModel
    """
    input_df = read_input_file(input_file, verbose)

    X_train = input_df.drop(columns=['align'])
    y_train = input_df['align']

    # Instantiate predictive model object
    character_model = CharacterPredictiveModel(verbose)
    character_model.fit(X_train, y_train)
    return character_model

def compare_model_types(character_model):
    """
    Compares a sampling of classification models
    Parameters:
    -----------
    model: CharacterPredictiveModel
        Fitted character model for testing

    Returns:
    -----------
    None
    """
    if verbose: print("Comparing model type(s)")

    # List of model types to compare
    models = {
        "Dummy Classifier": DummyClassifier(strategy='stratified'),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "SVM": SVC(),
        "OVR LogisticRegression": OneVsRestClassifier(LogisticRegression()),
        "LogisticRegression": LogisticRegression(),
    }
    comparison_df = character_model.model_compare(models)
    save_img(comparison_df, output_dir, "model_type_comparison")

    if verbose: print("Model type comparison(s) complete")
    return

def compare_decision_tree_forests(character_model):
    """
    Compares a sampling of random forest decision tree type models
    Parameters:
    -----------
    model: CharacterPredictiveModel
        Fitted character model for testing

    Returns:
    -----------
    None
    """
    if verbose: print("Comparing random forest type model(s)")

    # List of decision tree types to compare
    models = {
        "Random Forest Classifier": RandomForestClassifier(random_state = 123),
        "XGBClassifier": XGBClassifier(eval_metric = "mlogloss", random_state = 123),
        "LGBMClassifier": LGBMClassifier(num_leaves = 31, random_state = 123),
        "CatBoostClassifier": CatBoostClassifier(random_state = 123, verbose=0)
    }
    rf_models_df = character_model.model_compare(models)

    save_img(rf_models_df, output_dir, "forest_model_comparison", filename_prefix)
    if verbose: print("Trained Forest model(s)")
    return


if __name__ == "__main__":
    print(args)
    verbose = args["-v"]
    input_file = args["--input"]
    output_dir = args["--output"]
    filename_prefix = args["--filename"]
    if filename_prefix == None: filename_prefix = ""
    main(input_file, output_dir)