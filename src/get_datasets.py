"""
Retrieve datasets from a URL, file of URLs, or GitHub directory URL

Usage: get_datasets.py -i=<input> -o=<output> [-u] [-f] [-g] [-v]

Options:
-i <input>, --input <input>     Input URL
-o <output>, --output <output>  The local output location to transfer the datasets
[-u]                            (Default) Retrieve the input file from the given URL
[-f]                            Retrieve the list of file URLs specified in the input file
[-g]                            Retrieve the entire file contents of a github directory
[-v]                            Report verbose output of dataset retrieval process
"""
import os
import sys
import json
import urllib.request
from docopt import docopt
args = docopt(__doc__)


def get_datasets():
    # Starting dataset retrieval
    print("\n\n##### get_datasets: Retrieving datasets")
    if verbose: print(f"Running get_dataset with arguments: \n {args}")
    
    assert input, "Empty input argument provided"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    assert os.path.exists(output_path), "Invalid output path provided"

    download_urls = []
    if args['-g'] is True: 
        # If this is a github repo, construct GET api request to 
        # retrieve all repo directory datafiles
        github_api_url = "https://api.github.com/repos/"
        repo_url = args["--input"].split(os.path.sep)
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
    elif args['-f'] is True:
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

    print("\n##### get_datasets: Finished retrieval")


if __name__ == "__main__":
    input = args["--input"]
    output_path = args["--output"]
    verbose = args["-v"]
    get_datasets()
