# Makefile
# Craig McLaughlin, Dec 2, 2020

# This makefile script executes the retrieval, cleaning, exploratory analysis,
# feature transformation, machine learning training, machine learning 
# analysis/prediction, and final reporting analysis of a dataset of DC and 
# Marvel comic book characters and their traits. There are no input arguments.

# example usage:
# make clean
# make all

all : report/summary_report.md

# Retrieve datasets from web repo
data/raw/dc-wikia-data.csv data/raw/marvel-wikia-data.csv : \
    src/get_datasets.py
	    python src/get_datasets.py \
	    -i https://github.com/rudeboybert/fivethirtyeight/tree/master/data-raw/comic-characters \
	    -o data/raw -g -v

# Preprocess and clean the csv files into a single output csv
data/processed/clean_characters.csv : \
    data/raw/dc-wikia-data.csv \
    data/raw/marvel-wikia-data.csv \
    src/preprocess_data.py
		python src/preprocess_data.py \
	    -i data/raw \
	    -o data/processed/clean_characters.csv -v

# Generate EDA tables and figures
results/figures/alignment_over_time.png \
results/figures/alignment_vs_features.png \
results/figures/appearances_by_alignment.png \
results/tables/dataset_overview.pkl \
results/tables/feature_overview.pkl : \
    data/processed/clean_characters.csv \
    src/generate_eda.py
	    python src/generate_eda.py \
	    -i data/processed/clean_characters.csv \
	    -o results -v

# Perform feature engineering on the clean data
data/processed/character_features_train.csv \
data/processed/character_features_test.csv \
data/processed/character_features_deploy.csv : \
    data/processed/clean_characters.csv \
    src/feature_engineer.py
		python src/feature_engineer.py \
	    -i data/processed/clean_characters.csv \
	    -o data/processed/character_features.csv -v

# Machine learning modelling
results/figures/optimized_model.png \
results/figures/model_comparison.png \
results/tables/optimized_model.pkl \
results/tables/model_comparison.pkl \
results/models/optimized_model.pkl : \
    data/processed/character_features_train.csv \
	src/analysis.py
	    python src/analysis.py \
		    -i data/processed/character_features_train.csv \
			-o results -v

# Feature importance analysis
results/figures/importance_of_appearances.png \
results/figures/importance_of_eye.png \
results/figures/importance_of_hair.png \
results/figures/importance_of_id.png \
results/figures/importance_of_publisher \
results/figures/importance_of_publisher.png \
results/figures/importance_of_sex.png \
results/figures/importance_of_year.png \
results/tables/importance_of_appearances.pkl \
results/tables/importance_of_eye.pkl \
results/tables/importance_of_hair.pkl \
results/tables/importance_of_id.pkl \
results/tables/importance_of_publisher.pkl \
results/tables/importance_of_sex.pkl \
results/tables/importance_of_year.pkl \
results/tables/importance_of_year.pkl : \
    results/tables/optimized_model.pkl \
	src/analysis_feature.py
	    -i results/tables/optimized_model.pkl \
		-o results

# Generate summary markdown report
report/summary_report.md : \
    results/figures/alignment_over_time.png \
    results/figures/alignment_vs_features.png \
	results/figures/appearances_by_alignment.png \
	results/tables/dataset_overview.pkl \
	results/tables/feature_overview.pkl \
	results/figures/optimized_model.png \
    results/figures/model_comparison.png \
    results/tables/optimized_model.pkl \
    results/tables/model_comparison.pkl \
	results/models/optimized_model.pkl
	    jupyter nbconvert --to html report/summary_report.ipynb --no-input

clean :
	rm -f data/raw/*
	rm -rf data/processed
	rm -rf results/figures
	rm -rf results/tables
	rm -rf results/models
	rm -f report/eda_profile_report.html
	rm -f report/summary_report.html
	rm -rf report/summary_report_files/