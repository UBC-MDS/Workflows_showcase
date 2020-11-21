"""
Retrieve datasets from a URL, file of URLs, or GitHub directory URL

Usage: preprocess_data.py -i=<input> -o=<output> [-v]

Options:
-i <input>, --input <input>     Local raw data csv file directory
-o <output>, --output <output>  Local output filename and path for preprocessed csv
[-v]                            Report verbose output of dataset retrieval process
"""
import os
import glob
import pandas as pd
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
        publisher_dfs.append(publisher_df)

    characters_data = pd.concat(publisher_dfs, ignore_index=True)

    # There is some weird comic book multiverse stuff going on with this... We
    # can't remove the bracket aliases without creating duplicate entries
    #characters_data['name'] = characters_data['name'].str.split('(').str[0]
    characters_data['align'] = (characters_data['align'].str.split(' ').str[0]).astype("category")
    characters_data['eye'] = (characters_data['eye'].str.split(' ').str[0]).astype("category")
    characters_data['hair'] = (characters_data['hair'].str.split(' ').str[0]).astype("category")
    characters_data['sex'] = (characters_data['sex'].str.split(' ').str[0]).astype("category")
    characters_data['gsm'] = (characters_data['gsm'].str.split(' ').str[0]).astype("category")
    characters_data['first_appearance' ]= pd.to_datetime(characters_data['first_appearance'],
                                                        format='%b-%y',
                                                        errors='coerce')
    characters_data['year'] = pd.to_datetime(characters_data['year'],
                                            format="%Y",
                                            errors='coerce')
    characters_data['publisher'] = characters_data['publisher'].astype("category")

    characters_data.to_csv(output_filename)
    if verbose: print("\nOutput data summary:")
    if verbose: print(f"{characters_data.info()}")

    print("##### preprocess_data: Finished preprocessing")


if __name__ == "__main__":
    preprocess_data()
