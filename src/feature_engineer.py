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
import requests
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

    # Read in cleaned data
    clean_df = pd.read_csv(input_filename)

    # Apply feature engineering to generate new features
    common_names = build_common_name_list()
    feature_df = clean_df.assign(is_common = clean_df.first_name.isin(common_names[["Name"]].Name))
    feature_df = feature_df.assign(name_len = feature_df["first_name"].apply(get_name_length) + 
                                              feature_df["last_name"].apply(get_name_length))
    feature_df = feature_df.assign(has_last_name = feature_df["last_name"].apply(has_last_name))
    feature_df["appear_per_yr"] = feature_df.apply(lambda x: norm(x['appearances'], x['year']), axis=1)

    # Remove intermediate name columns
    feature_df = feature_df.drop(["first_name",
                                  "last_name"], axis=1)

    # Create a set of data without neutral targets
    feature_polarized_df = feature_df[feature_df['align'] != 'Neutral']

    # Creating deployment data file from rows missing target values")
    deploy_df = feature_df[feature_df['align'].isnull()]
    deploy_df.to_csv(output_file_slug + "_deploy." + output_file_ext)
    feature_df = feature_df[feature_df['align'].notna()]

    deploy_polarized_df = feature_polarized_df[feature_polarized_df['align'].isnull()]
    deploy_polarized_df.to_csv(output_file_slug + "_polar_deploy." + output_file_ext)
    feature_polarized_df = feature_polarized_df[feature_polarized_df['align'].notna()]

    # Split train and test data
    train_df, test_df = train_test_split(feature_df, test_size=0.2, random_state=123)
    train_df.to_csv(output_file_slug + "_train." + output_file_ext, index = False)
    test_df.to_csv(output_file_slug + "_test." + output_file_ext, index = False)

    train_polar_df, test_polar_df = train_test_split(feature_polarized_df, test_size=0.2, random_state=123)
    train_polar_df.to_csv(output_file_slug + "_polar_train." + output_file_ext, index = False)
    test_polar_df.to_csv(output_file_slug + "_polar_test." + output_file_ext, index = False)

    if verbose:
        print(f"Wrote deployment data output file: {output_file_slug}_deploy.{output_file_ext}")
        print(f"Wrote polarized deployment data output file: {output_file_slug}_polar_deploy.{output_file_ext}")
    if verbose:
        print(f"Wrote train data output file: {output_file_slug}_train.{output_file_ext}")
        print(f"Wrote polarized train data output file: {output_file_slug}_polar_train.{output_file_ext}")
    if verbose:
        print(f"Wrote test data output file: {output_file_slug}_test.{output_file_ext}")
        print(f"Wrote polarized test data output file: {output_file_slug}_polar_test.{output_file_ext}")

    print("\n##### feature_engineer: Finished processing features!")


def build_common_name_list():
    """
    Returns a table of common names.

    Parameters:
    ------

    Returns:
    -------
    table of common names: DataFrame

    """
    # URLs with common names from census for male and female
    url_male = "https://namecensus.com/male_names_alpha.htm"
    url_female = "https://namecensus.com/female_names_alpha.htm"

    male_names_page_1 = requests.get(url_male)
    top_male_first_name = pd.read_html(male_names_page_1.text)

    female_names_page_1 = requests.get(url_female)
    top_female_first_name = pd.read_html(female_names_page_1.text)

    top_names = pd.concat([top_male_first_name[0], top_female_first_name[0]], ignore_index=True)
    return pd.DataFrame(top_names["Name"].str.lower())

def get_name_length(name):
    """
    Returns the length of first name.

    Parameters:
    ------
    name: (str)
    the input name

    Returns:
    -------
    length of name: (float)

    """
    if name == name:
        return len(name)
    else:
        return 0

def has_last_name(name):
    """
    Returns whether the character has a last name.
    
    Parameters:
    ------
    name: (str)
    the input name
    
    Returns:
    ------
    True/False of whether the last name exists (boolean)
    
    """
    return name == name
    
def norm(appearances, year, MAX_YR=2013):
    """
    Returns number of appearances per year.
    
    Parameters:
    ------
    appearances: (float)
    the number of appearances of the character
    
    year: (float)
    the year of its first appearance
    
    Keyword arguments:
    ------
    MAX_YR: (float)
    year boundary for the source data
    
    Returns:
    ------
    appearances_per_yr: (float)
    the number of appearances per year for each character
    
    Examples:
    ------
    norm(10, 2004) = 1
    """
    if year == year and appearances == appearances:
        appearances_per_yr = appearances/(MAX_YR - year + 1)
        return appearances_per_yr
    else:
        return 


if __name__ == "__main__":
    input_filename = args["--input"]
    output_filename = args["--output"]
    verbose = args["-v"]
    feature_engineer_data()
