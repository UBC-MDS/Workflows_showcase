# Workflows_showcase
Group repository for group 25 in DSCI 522

Collect datasets from source:
`
python src/get_datasets.py -i https://github.com/rudeboybert/fivethirtyeight/tree/master/data-raw/comic-characters -o data/raw -g -v
`

Preprocess raw datasets:
`
python src/preprocess_data.py -i ../data/raw -o data/processed/clean_characters.csv -v
`

