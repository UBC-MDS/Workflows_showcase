def preprocess_data():
    """
    Clean the raw local csv files before processing

    Parameters
    ----------
    -i/--input  : string
        Local raw data csv file directory
    -o/--output : string
        Local output filename and path for preprocessed csv

    Options
    -v/--verbose    : boolean
        Report verbose output of dataset retrieval process

    Examples
    --------
    python preprocess_data.py -i ../data/raw -o ../data/processed/clean_characters.csv -v
    """
    import os
    import glob
    import argparse
    import pandas as pd

    # Setup script argument parsing
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", required=True, 
        help="input raw csv directory path")
    ap.add_argument("-o", "--output", required=True, 
        help="output csv filename and path")
    ap.add_argument("-v", "--verbose", required=False, action='store_true',
        help="Print script output (Optional)")
    args = vars(ap.parse_args())

    input_dir = args["input"]
    output_filename = args["output"]
    verbose = args["verbose"]

    assert os.path.exists(input_dir), "Invalid input directory path provided"
    assert os.path.exists(os.path.dirname(output_filename)), \
        "Invalid output path provided"

    # Starting dataset retrieval
    print("##### preprocess_data: Preprocessing datasets")
    if verbose: print(f"Running preprocess_data with arguments: \n {args}")

    raw_files = glob.glob(f"{input_dir}/*.csv")

    characters = pd.concat([pd.read_csv(file) for file in raw_files],
        ignore_index=True)

    characters.to_csv(output_filename)

    print("##### preprocess_data: Finished preprocessing")


if __name__ == "__main__":
    preprocess_data()
