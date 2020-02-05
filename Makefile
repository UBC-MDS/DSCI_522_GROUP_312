all: html_report

# loading data from URL

load_file: scripts/load_data.py
	python scripts/load_data.py --url="https://github.com/ageron/handson-ml/blob/master/datasets/housing/housing.csv?raw=true" --file_path="data/raw_data.csv"

# Wrangling data (depends on load_data script)
wrangle: load_file scripts/wrangle-and-split-data.R 
	Rscript scripts/wrangle-and-split-data.R --filepath_in='data/raw_data.csv' --filepath_out_train='data/train.csv' --filepath_out_test='data/test.csv'

# Make EDA plots and files (depends on wrangle-and-split-data script)
EDA: wrangle scripts/eda_v2.py
	python scripts/eda_v2.py --train_path='data/train.csv' --out_folder_path='results/eda_charts/' 

# Performs Machine learning analysis and saves the results as plots and tables	
model:  scripts/ML_analysis_v2.py
	python scripts/ML_analysis_v2.py --training_input_path='data/train.csv' --testing_input_path='data/test.csv' --output_path='results/ml_results/'

# Excute the report notebook to address any changes made to to the analysis 
# Requires results from EDA and model rules
report: EDA model results/california_housing_predict_report.ipynb 
	jupyter nbconvert --to notebook --execute results/california_housing_predict_report.ipynb
	
# Converts the fnal report from notebook to html file
html_report: report results/california_housing_predict_report.ipynb 
	jupyter nbconvert --to html --template basic results/california_housing_predict_report.ipynb

# cleans all the file produced by the rules except the california_housing_predict_report.ipynb file
clean : 
	rm -f data/raw_data.csv
	rm -f data/test.csv
	rm -f data/train.csv
	rm -f results/eda_charts/*.png 
	rm -f results/eda_charts/*.csv 
	rm -f results/figure/*.png
	rm -f results/ml_results/*.png
	rm -f results/ml_results/*.csv
