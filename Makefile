all: html_report

load_file: scripts/load_data.py
	python scripts/load_data.py --url="https://github.com/ageron/handson-ml/blob/master/datasets/housing/housing.csv?raw=true" --file_path="data/raw_data.csv"

wrangle: load_file scripts/wrangle-and-split-data.R 
	Rscript scripts/wrangle-and-split-data.R --filepath_in='data/raw_data.csv' --filepath_out_train='data/train.csv' --filepath_out_test='data/test.csv'

EDA: wrangle scripts/eda_v2.py
	python3 scripts/eda_v2.py --train_path='data/train.csv' --out_folder_path='results/eda_charts/' 
	
model:  scripts/ML_analysis_v2.py
	python scripts/ML_analysis_v2.py --training_input_path='data/train.csv' --testing_input_path='data/test.csv' --output_path='results/ml_results/'
	
report: EDA model results/california_housing_predict_report.ipynb 
	jupyter nbconvert --to notebook --execute results/california_housing_predict_report.ipynb

html_report: report results/california_housing_predict_report.ipynb 
	jupyter nbconvert --to html --template basic results/california_housing_predict_report.ipynb

clean : 
	rm -f data/raw_data.csv
	rm -f data/test.csv
	rm -f data/train.csv
	rm -f results/eda_charts/*.png 
	rm -f results/eda_charts/*.csv 
	rm -f results/figure/*.png
	rm -f results/ml_results/*.png
	rm -f results/ml_results/*.csv