# DSCI 522 - Group 312

## About

This repository was created as part of an assignment in [UBC's Master of Data Science](https://masterdatascience.ubc.ca/) program for the Data Science Workflows Course.

This project focuses on predicting median housing prices in the difference [census blocks](https://www.census.gov/newsroom/blogs/random-samplings/2011/07/what-are-census-blocks.html) in California based on [data collected during the 1990 Census](https://www.kaggle.com/camnugent/california-housing-prices/kernels). The independent variables cover different attributes such as location, demographics, and attributes of the homes. The dataset that we analysed had an additional categorical variable added by [Aurélien Geron](https://github.com/ageron/handson-ml/tree/master/datasets/housing).

Since this dataset is publicly available, there have been many analyses. We have chosen to focus on [Eric Chen;s analysis](https://www.kaggle.com/ericfeng84), who had obtained a model score of 0.60 with linear regression. We aimed to expand on this analysis and look into the role of multicollinearity on Chen's results. The results report can be found below - we concluded that the best model was KNN with a score of approximately 0.75.

## Report

The written report can be found [here](results/california_housing_predict_report.ipynb)

## Usage

To replicate this analysis and load all relevant packages, ensure that the dependencies below are installed, and clone this repo locally. Run the following command in terminal from the root of this project(note, the complete analysis takes up to 15 minutes):

`make all`

To reset the repository, run:

`make clean`

## Dependencies

Python 3.7.3 and Python packages:
* docopt==0.6.2
* requests==2.22.0
* pandas==0.24.2
* scikit-learn==0.22.1
* numpy==1.18.1
* seaborn==0.9.0
* statsmodels==0.11.0
* selenium==3.141.0
* altair==3.2.0

R version 3.6.1 and R packages:
* knitr==1.26
* tidyverse==1.2.1
* readr==1.3.1
* tidyr==1.0.2
* caret==6.0-85
* checkmate==1.9.4
* testthat==2.3.1

GNU make 4.2.1

## References

Feng, E. (2017, September 29). Housing Price Prediction. Retrieved from https://www.kaggle.com/ericfeng84/housing-price-prediction

Nugent, C. (2017, November 24). California Housing Prices. Retrieved from https://www.kaggle.com/camnugent/california-housing-prices

US Census Bureau. (2016, December 27). What are census blocks? Retrieved from https://www.census.gov/newsroom/blogs/random-samplings/2011/07/what-are-census-blocks.html
