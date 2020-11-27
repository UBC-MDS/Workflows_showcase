"""
Preprocess raw comic character datasets for analysis

Usage: preprocess_data.py -i=<input> -o=<output> [-v]

Options:
-i <input>, --input <input>     Local raw data csv file directory
-o <output>, --output <output>  Local output filename and path for preprocessed csv
[-v]                            Report verbose output of dataset retrieval process
"""
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from docopt import docopt
args = docopt(__doc__)


def preprocess_data():
    # Setup script argument parsing
    input_dir = args["--input"]
    output_filename = args["--output"]
    verbose = args["-v"]

    assert os.path.exists(input_dir), "Invalid input directory path provided"
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    assert os.path.exists(os.path.dirname(output_filename)), "Invalid output path provided"

    output_file_slug = output_filename.split(".")[0]
    output_file_ext = output_filename.split(".")[-1]

    # Starting dataset retrieval
    print("##### preprocess_data: Preprocessing datasets")
    if verbose: print(f"Running preprocess_data with arguments: \n {args}")

    raw_csv_files = glob.glob(f"{input_dir}/*.csv")

    publisher_dfs = []
    for csv_file in raw_csv_files:
        if verbose: print(f"Processing {csv_file}")
        publisher = os.path.basename(csv_file).split("-")[0]
        publisher_df = pd.read_csv(csv_file)
        publisher_df.columns = publisher_df.columns.str.lower().str.replace(' ', '_')
        publisher_df = publisher_df.drop(["page_id",
                                          "urlslug",
                                          "alive"], axis=1)
        publisher_df['publisher'] = publisher

        if publisher == "dc":
            dateFormat='%Y, %B'
        else:
            dateFormat='%b-%y'

        publisher_df['first_appearance'] = pd.to_datetime(
            publisher_df['first_appearance'],
            format=dateFormat,
            errors='coerce')
    
        # The datetime parser sees years < 1970 as 2000's...Convert back
        publisher_df['first_appearance'] = np.where(
            (publisher_df['first_appearance'].dt.year > datetime.now().year),
            publisher_df['first_appearance'] - pd.DateOffset(years=100),
            publisher_df['first_appearance'])

        publisher_dfs.append(publisher_df)

    characters_data = pd.concat(publisher_dfs, ignore_index=True)

    # There is some weird comic book multiverse stuff going on with this... We
    # can't remove the bracket aliases without creating duplicate entries
    #characters_data['name'] = characters_data['name'].str.split('(').str[0]
    characters_data['align'] = characters_data['align'].replace(regex ="Reformed", value = "Good")
    characters_data['align'] = (characters_data['align'].str.split(' ').str[0]).astype("category")
    characters_data['eye'] = (characters_data['eye'].str.split(' ').str[0]).astype("category")
    characters_data['hair'] = (characters_data['hair'].str.split(' ').str[0]).astype("category")
    characters_data['sex'] = (characters_data['sex'].str.split(' ').str[0]).astype("category")
    characters_data['gsm'] = (characters_data['gsm'].str.split(' ').str[0]).astype("category")

    characters_data['year'] = pd.to_datetime(characters_data['year'],
                                             format="%Y",
                                             errors='coerce').dt.year
    characters_data['publisher'] = characters_data['publisher'].astype("category")

    if verbose: print(f"Writing full data output file: {output_filename}")
    characters_data.to_csv(output_filename)
    if verbose: print("\nOutput data summary:")
    if verbose: print(f"{characters_data.info()}")

    # Creating deployment data file from rows missing target values")
    deploy_df = characters_data[characters_data['align'].isnull()]
    deploy_df.to_csv(output_file_slug + "_deploy." + output_file_ext)
    characters_data = characters_data[characters_data['align'].notna()]

    # Split train and test data
    train_df, test_df = train_test_split(characters_data, test_size=0.2, random_state=123)
    train_df.to_csv(output_file_slug + "_train." + output_file_ext)
    test_df.to_csv(output_file_slug + "_test." + output_file_ext)

    if verbose:
        print(f"Wrote deployment data output file: {output_file_slug}_deploy.{output_file_ext}")
    if verbose:
        print(f"Wrote train data output file: {output_file_slug}_train.{output_file_ext}")
    if verbose:
        print(f"Wrote test data output file: {output_file_slug}_test.{output_file_ext}")

    print("##### preprocess_data: Finished preprocessing")


if __name__ == "__main__":
    preprocess_data()
