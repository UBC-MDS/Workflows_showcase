def get_datasets():
    """
    Retrieve datasets from a URL, file of URLs, or GitHub directory URL

    Parameters
    ----------
    -i/--input  : string
        Input url path
    -o/--output : string
        The local output location to transfer the datasets

    Options
    -------
    -u/--url        : boolean
        (Default) Retrieve the input given as a file URL
    -f/--file       : boolean
        Retrieve the list of file URL specified in the input file
    -g/--github_url : boolean
        Retrieve the entire file contents of a github directory
    -v/--verbose    : boolean
        Report verbose output of dataset retrieval process

    Examples
    --------
    python get_datasets.py -i \
        https://raw.githubusercontent.com/rudeboybert/fivethirtyeight/master/data-raw/comic-characters/dc-wikia-data.csv \
        -o ../data/raw -u -v

    python get_datasets.py -i \
        ../data/my_dataset_file_urls.txt \
        -o ../data/raw -f -v

    python get_datasets.py -i \
        https://github.com/rudeboybert/fivethirtyeight/tree/master/data-raw/comic-characters \
        -o ../data/raw -g -v
    """
    import os
    import sys
    import json
    import urllib.request
    import argparse

    # Setup script argument parsing
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", required=True, help="input url or file path")
    ap.add_argument("-o", "--output", required=True, help="output path")
    ap.add_argument("-u", "--url", required=False, action='store_true',
        help="Read the input url from the input argument (Default) (Optional)")
    ap.add_argument("-f", "--file", required=False, action='store_true',
        help="Read the input urls from a the given input file (Optional)")
    ap.add_argument("-g", "--git_directory", required=False, action='store_true',
        help="Read the input url from the input files from a git directory (Optional)")
    ap.add_argument("-v", "--verbose", required=False, action='store_true',
        help="Print script output (Optional)")
    args = vars(ap.parse_args())

    input = args["input"]
    output_path = args["output"]
    verbose = args["verbose"]

    assert input, "Empty input argument provided"
    assert os.path.exists(output_path), "Invalid output path provided"

    # Starting dataset retrieval
    print("##### get_datasets: Retrieving datasets")
    if verbose: print(f"Running get_dataset with arguments: \n {args}")

    download_urls = []
    if args['git_directory'] is True: 
        # If this is a github repo, construct GET api request to 
        # retrieve all repo directory datafiles
        github_api_url = "https://api.github.com/repos/"
        repo_url = args["input"].split(os.path.sep)
        # These magic numbers parse out the unnecessary github url branch info
        github_api_url = github_api_url + \
            (os.path.sep).join(repo_url[3:5] + ["contents"] + repo_url[7:])

        try:
            if verbose: print(f"Attempting to connect to: {github_api_url}")
            response = urllib.request.urlopen(github_api_url).read()
        except ConnectionError:
            print(f"Failed to connect to: {github_api_url}. \nExiting")
            sys.exit(os.EX_NOHOST)
      
        if verbose: print("Connection Success!")
        git_files = json.loads(response.decode('utf-8'))
        for file in git_files:
            download_urls.append(file["download_url"])
    elif args['file'] is True:
        assert os.path.exists(input), "Input file is not valid"
        input_fh = open(input, "r")
        for line in input_fh:
            download_urls.append(line.strip())
    else:
        download_urls.append(input)

    # Download all files requested
    for file_url in download_urls:
        output_file = output_path + os.path.sep + os.path.basename(file_url)
        try:
            if verbose: print(f"Attempting to retrieve {file_url}")
            transfer = urllib.request.urlretrieve(file_url, output_file)
            if verbose: print(f"Successfully transferred to {output_file}")
        except ConnectionError:
            print(f"Unable to retrieve: {file_url}, continuing")

    print("##### get_datasets: Finished retrieval")


if __name__ == "__main__":
    get_datasets()
