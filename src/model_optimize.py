"""
Author: Aidan Mattrick

Date: Nov 28, 2020

This script trains an optimized model for character classification.

Usage: model_optimize.py -i=<input> -o=<output> [-f <filename>] [-v]

Options:
-i <input>, --input <input>                 Local processed training data csv file in directory
-o <output>, --output <output>              Local output directory for created artifacts
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

from lightgbm.sklearn import LGBMClassifier

from docopt import docopt

args = docopt(__doc__)

def main(input_file, output_dir):
    print("\n\n##### Model Optimization: Optimizing Model!")
    if verbose: print(f"Running optimization script with arguments: \n {args}")
    validate_inputs(input_file, output_dir)

    # Instantiate the parent character model class object
    character_model = initialize_character_model()
    
    optimization_df = optimize_model(character_model)

    # Export optimized model for prediction and analysis
    optimized_model = character_model.get_model()
    pickle.dump(optimized_model,
                open(output_dir + "/models/" + filename_prefix + "optimized_model.pkl", 'wb'))

    print("\n\n##### Model Optimization: Optimizing Models Complete!")

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

    # Create character model prediction class
    character_model = CharacterPredictiveModel(verbose)
    character_model.fit(X_train, y_train)
    return character_model


def optimize_model(character_model):
    """
    Optimizes a classification model with the given param_grid
    Parameters:
    -----------
    model: CharacterPredictiveModel
        Fitted character model for optimizing

    Returns:
    -----------
    pandas DataFrame
        A dataframe with optimization results
    """
    if verbose: print("Performing hyperparameter optimization on best model")

    #Performing hyperparameter optimization on LightGBM
    model = {"best_model": LGBMClassifier(random_state = 123)}
    param_grid = {'best_model__n_estimators'  : [5, 100, 500, 700, 1000, 1500, 4000],
                  'best_model__learning_rate' : [0.01, 0.1, 1],
                  'best_model__max_depth'     : [1, 3, 5, 6, 10],
                  'best_model__subsample'     : [0.15, 0.25, 0.5, 0.75, 1],
                  'best_model__num_leaves'    : [31, 64, 128]
                 }

    model_df = character_model.optimize_model(model, param_grid)

    save_img_large(model_df, output_dir, "optimized_model", filename_prefix)

    if verbose: print("Model optimization complete!")
    return character_model


if __name__ == "__main__":
    print(args)
    verbose = args["-v"]
    input_file = args["--input"]
    output_dir = args["--output"]
    filename_prefix = args["--filename"]
    if filename_prefix == None: filename_prefix = ""
    main(input_file, output_dir)