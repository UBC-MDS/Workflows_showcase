"""
Author: Craig McLaughlin

Date: Dec 2, 2020

Perform feature engineering transformations and split train/test/deploy data

Usage: feature_engineer.py -i=<input> -o=<output> [-v]

Options:
-i <input>, --input <input>     Local processed clean data csv file
-o <output>, --output <output>  Local output filename and path for processed data csv
[-v]                            Report verbose output of dataset retrieval process
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from docopt import docopt
args = docopt(__doc__)


def feature_engineer_data():
    # Starting dataset preprocessing
    print("\n\n##### feature_engineer: Processing dataset")
    if verbose: print(f"Running feature_engineer with arguments: \n {args}")

    assert os.path.isfile(input_filename), "Invalid input filename path provided"
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    assert os.path.exists(os.path.dirname(output_filename)), "Invalid output path provided"

    output_file_slug = output_filename.split(".")[0]
    output_file_ext = output_filename.split(".")[-1]

    clean_df = pd.read_csv(input_filename)

    # Creating deployment data file from rows missing target values")
    deploy_df = clean_df[clean_df['align'].isnull()]
    deploy_df.to_csv(output_file_slug + "_deploy." + output_file_ext)
    clean_df = clean_df[clean_df['align'].notna()]

    # Split train and test data
    train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=123)
    train_df.to_csv(output_file_slug + "_train." + output_file_ext)
    test_df.to_csv(output_file_slug + "_test." + output_file_ext)

    if verbose:
        print(f"Wrote deployment data output file: {output_file_slug}_deploy.{output_file_ext}")
    if verbose:
        print(f"Wrote train data output file: {output_file_slug}_train.{output_file_ext}")
    if verbose:
        print(f"Wrote test data output file: {output_file_slug}_test.{output_file_ext}")

    print("\n##### feature_engineer: Finished processing features!")


if __name__ == "__main__":
    input_filename = args["--input"]
    output_filename = args["--output"]
    verbose = args["-v"]
    feature_engineer_data()
