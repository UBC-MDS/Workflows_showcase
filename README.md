# Workflows_showcase
## Comic Book Character Align Predictor
This is Group 25's project for DSCI 522(Data Science Workflows); which is a course in the MDS(Master of Data Science) program at the University of British Columbia.
### Authors

- Aidan Mattrick
- Craig McLaughlin
- Zeliha Ural Merpez
- Ivy Zhang

### Overview
For this project, we will be using data containing [Comic Book Characters](https://github.com/rudeboybert/fivethirtyeight/tree/master/data-raw/comic-characters). The data comes from [Marvel Wikia](https://marvel.fandom.com/wiki/Marvel_Database) and [DC Wikia](https://dc.fandom.com/wiki/DC_Comics_Database). We are interested in the following questions:

1. Predictive: What are the most important features in determining a comic book characters' alignment?

2. Inferential: Are comic book characters' physical features and sexual orientations, becoming more diverse with alignment? Is there a change happening this relationship over time?


The data is split into two files, for DC and Marvel, respectively: dc-wikia-data.csv and marvel-wikia-data.csv. Each file has the following variables:

`page_id, name, urlslug, ID, ALIGN, EYE, HAIR, SEX, GSM, ALIVE, APPEARANCES, FIRST APPEARANCE, YEAR`

### Usage and Flowchart

Collect datasets from source:
`
python src/get_datasets.py -i https://github.com/rudeboybert/fivethirtyeight/tree/master/data-raw/comic-characters -o data/raw -g -v
`

Preprocess raw datasets:
`
python src/preprocess_data.py -i ../data/raw -o data/processed/clean_characters.csv -v
`

![Flow Chart](img/flow_chart00.png)
### Dependencies

### References

