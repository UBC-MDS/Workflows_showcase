# Workflows_showcase
## Comic Book Character Align Predictor
This is Group 25's project for DSCI 522(Data Science Workflows); which is a course in the MDS(Master of Data Science) program at the University of British Columbia.

### Authors

- Aidan Mattrick
- Craig McLaughlin
- Zeliha Ural Merpez
- Ivy Zhang

### Overview

For this project, we will be using data containing [Comic Book Characters](https://github.com/rudeboybert/fivethirtyeight/tree/master/data-raw/comic-characters). The data comes from [Marvel Wikia](https://marvel.fandom.com/wiki/Marvel_Database) and [DC Wikia](https://dc.fandom.com/wiki/DC_Comics_Database). We are interested in the following question:

> Predictive: What are the most important features in determining a comic book characters' alignment? How are they changing over time?

Our plan is to see how the alignment target changes over time and identify which feature categories correlate with the alignment. Some methodologies may be considered to answer this question, such as logistic regression, decision tree, and etc..

The data is split into two files, for DC and Marvel, respectively: dc-wikia-data.csv and marvel-wikia-data.csv. Each file has the following variables:

`page_id, name, urlslug, ID, ALIGN, EYE, HAIR, SEX, GSM, ALIVE, APPEARANCES, FIRST APPEARANCE, YEAR`

![Project Overview](img/project_overview_00.png)

### Releases

[Milestone 1: 0.0.1](https://github.com/UBC-MDS/Workflows_showcase/releases/tag/0.0.1)

### Usage and Flowchart

The datasets are collected from the source below.
`
python src/get_datasets.py -i https://github.com/rudeboybert/fivethirtyeight/tree/master/data-raw/comic-characters -o data/raw -g -v
`

The row datasets can be processed by using the following command:
`
python src/preprocess_data.py -i data/raw -o data/processed/clean_characters.csv -v
`

![Flow Chart](img/flow_chart00.png)

### Dependencies

To reproduce, please see dependencies in [environment file](https://github.com/UBC-MDS/Workflows_showcase/blob/main/env/env.yaml).

### References

1. [Comic Book Characters](https://github.com/rudeboybert/fivethirtyeight/tree/master/data-raw/comic-characters) 
