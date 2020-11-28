all : report/summary_report.md

# Retrieve datasets from repo
data/raw/dc-wikia-data.csv data/raw/marvel-wikia-data.csv : \
    src/get_datasets.py
	    python src/get_datasets.py \
	    -i https://github.com/rudeboybert/fivethirtyeight/tree/master/data-raw/comic-characters \
	    -o data/raw -g -v

# Preprocess and clean the data
data/processed/clean_characters.csv \
data/processed/clean_characters_train.csv \
data/processed/clean_characters_test.csv \
data/processed/clean_characters_deploy.csv : \
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
	    -i "data/processed/clean_characters.csv" \
	    -o "results" -v

# Modelling

# Generate summary markdown report
report/summary_report.md : \
    results/figures/alignment_over_time.png \
    results/figures/alignment_vs_features.png \
	results/figures/appearances_by_alignment.png \
	results/tables/dataset_overview.pkl \
	results/tables/feature_overview.pkl
	    jupyter nbconvert --to markdown report/summary_report.ipynb

clean :
	rm -f data/raw/README.md
	rm -f data/raw/dc-wikia-data.csv
	rm -f data/raw/marvel-wikia-data.csv
	rm -f data/processed/clean_characters.csv
	rm -f data/processed/clean_characters_train.csv
	rm -f data/processed/clean_characters_test.csv
	rm -f data/processed/clean_characters_deploy.csv
	rm -f results/figures/alignment_over_time.png
	rm -f results/figures/alignment_vs_features.png
	rm -f results/figures/appearances_by_alignment.png
	rm -f results/tables/dataset_overview.pkl
	rm -f results/tables/feature_overview.pkl
	rm -f report/eda_profile_report.html