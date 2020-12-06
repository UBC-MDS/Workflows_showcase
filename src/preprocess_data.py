"""
Author: Craig McLaughlin

Date: Nov 25, 2020

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
from datetime import datetime
from docopt import docopt
args = docopt(__doc__)


def preprocess_data():
    # Starting dataset preprocessing
    print("\n\n##### preprocess_data: Preprocessing datasets")
    if verbose: print(f"Running preprocess_data with arguments: \n {args}")

    assert os.path.exists(input_dir), "Invalid input directory path provided"
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    assert os.path.exists(os.path.dirname(output_filename)), "Invalid output path provided"

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

    character_name = characters_data['name'].str.split('(').str[0].str.strip()
    characters_data['first_name'] = character_name.str.split(" ").str[0].str.lower()
    characters_data['last_name'] = character_name.str.split(" ", n=1).str[1].str.lower()
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
    characters_data.to_csv(output_filename, index = False)
    if verbose: print("\nOutput data summary:")
    if verbose: print(f"{characters_data.info()}")

    print("\n##### preprocess_data: Finished preprocessing")


if __name__ == "__main__":
    input_dir = args["--input"]
    output_filename = args["--output"]
    verbose = args["-v"]
    preprocess_data()
