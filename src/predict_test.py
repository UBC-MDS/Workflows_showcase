"""
Author: Aidan Mattrick

Date: Dec 9, 2020

This script takes in a trained model and runs it against the test set and returns a confusion matrix outline the results.

Usage: predict_test.py -i=<input> -o=<output> -m=<model> [-f <filename>] [-v]

Options:
-i <input>, --input <input>                 Local processed training data csv file in directory
-o <output>, --output <output>              Local output directory for created pngs
-m <model>, --model <model>                 Model to be used for testing
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

from sklearn.metrics import plot_confusion_matrix

from character_utils import *

from lightgbm.sklearn import LGBMClassifier
from catboost import CatBoostClassifier

from docopt import docopt
from pylab import savefig
args = docopt(__doc__)


def main(input_file_path, output_folder_path, model, filename_prefix=""):
    model = pd.read_pickle(model)
    if model: print("\n\n##### Model imported!")

    test_df =  read_input_file(input_file_path, verbose=True)

    print("\n\n##### Testing Model!")
    X_test, y_test = test_df.drop(columns=['align']), test_df['align']

    score = model.score(X_test, y_test)
    print("\n\n##### Model Score:")
    print(f'{score}')

    if input_file_path == "data/processed/character_features_test.csv":
        confusion_matrix = plot_confusion_matrix(model, X_test, y_test, display_labels=["Bad", "Neutral", "Good"],
                      values_format="d", cmap=plt.cm.Blues);

    else:
        confusion_matrix = plot_confusion_matrix(model, X_test, y_test, display_labels=["Bad", "Good"],
                      values_format="d", cmap=plt.cm.Blues);

    save_matrix(confusion_matrix, output_folder_path, "confusion_matrix", filename_prefix)


if __name__ == "__main__":
    print(args)
    verbose = args["-v"]
    input_file = args["--input"]
    output_dir = args["--output"]
    model = args["--model"]
    filename_prefix = args["--filename"]
    if filename_prefix:
        main(input_file, output_dir, model, filename_prefix)
    else:
        main(input_file, output_dir, model)


