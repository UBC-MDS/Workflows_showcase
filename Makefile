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

# Machine learning model selection
results/tables/model_type_comparison.pkl \
results/tables/forest_model_comparison.pkl \
results/figures/forest_model_comparison.png \
results/figures/model_type_comparison.png : \
    data/processed/character_features_train.csv \
	src/model_selection.py
	    python src/model_selection.py \
		    -i data/processed/character_features_train.csv \
			-o results -v

# Machine learning model optimization
results/tables/optimized_model.pkl \
results/models/optimized_model.pkl : \
    results/figures/model_comparison.png \
	src/model_optimize.py
	    python src/model_optimize.py \
		    -i data/processed/character_features_train.csv \
			-o results -v

# Machine learning model optimization no neutrals
results/figures/polarized_optimized_model.png \
results/tables/polarized_optimized_model.pkl \
results/models/polarized_optimized_model.pkl : \
    results/models/optimized_model.pkl \
	src/model_optimize.py
	    python src/model_optimize.py \
		    -i data/processed/character_features_polar_train.csv \
			-o results -f polarized_ -v

# Machine learning model test
results/tables/confusion_matrix.png : \
    results/tables/polarized_optimized_model.pkl \
	src/predict_test.py
	    python src/predict_test.py \
		    -i data/processed/character_features_test.csv \
			-o results \
			-m results/models/optimized_model.pkl \
			-v

# Machine learning model test no neutrals
results/tables/polarized_confusion_matrix.png : \
    results/tables/confusion_matrix.png \
		src/predict_test.py
	    python src/predict_test.py \
		    -i data/processed/character_features_polarized_test.csv \
			-o results -f polarized_ \
			-m results/models/polarized_optimized_model.pkl \
			-v

# Feature importance analysis
results/figures/importance.png : \
    results/tables/polarized_confusion_matrix.png \
	src/analysis_feature.py
		python src/analysis_feature.py \
	    	-i results/models/optimized_model.pkl \
			-j data/processed/character_features_train.csv \
			-o results -v

# Generate summary markdown report
report/summary_report.md : \
    results/figures/alignment_over_time.png \
	results/figures/feature_overview.png \
	results/figures/model_type_comparison.png \
	results/figures/forest_model_comparison.png \
	results/tables/polarized_optimized_model.pkl \
    results/figures/optimized_model.png \
	results/tables/confusion_matrix.png \
	results/figures/importance.png
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
