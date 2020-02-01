# DSCI 522 - Group 312

## About
This repository was created as part of an assignment in [UBC's Master of Data Science](https://masterdatascience.ubc.ca/) program.

## Usage
To replicate this analysis and load all relevant packages, ensure that the dependencies below are installed, and clone this repo locally. Run the following command in terminal from the root of this project:

`make all`

To reset the repository, run:

`make clean`

## Dependencies

> Python 3.7.3 and Python packages:
* docopt==0.6.2
* requests==2.22.0
* pandas==0.24.2
* scikit-learn==0.22.1
* numpy==1.18.1
* seaborn==0.9.0
* statsmodels==0.11.0
* selenium==3.141.0
* altair==3.2.0

> R version 3.6.1 and R packages:
* knitr==1.26
* tidyverse==1.2.1
* readr==1.3.1
* tidyr==1.0.2
* caret==6.0-85
* checkmate==1.9.4
* testthat==2.3.1

> GNU make 4.2.1

## Proposal

#### Dataset
This dataset is a modified version of [The California Housing Dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html), with [additional columns added by Aurélien Geron](https://github.com/ageron/handson-ml). This dataset contains information about median California house values (including related factors) as sourced from the 1990 US Census.

#### Research Question
As discussed in a special report by The Economist on January 16, 2020 ['Housing is at the root of many of the rich world’s problems'](https://www.economist.com/special-report/2020/01/16/housing-is-at-the-root-of-many-of-the-rich-worlds-problems) and nowhere is this problem more acute than in the richest and most populous American state where Bloomberg has attempted to explain ['How California Became America’s Housing Market Nightmare'](https://www.bloomberg.com/graphics/2019-california-housing-crisis/).

Although finding solutions for this deep-seated socioeconomic problem is beyond the scope of this research project, we propose to build a machine learning model using this dataset to identify factors which can be used to predict housing prices and help to better inform stakeholders in this increasingly unaffordable market. Based on the latest report [California Housing Affordability Update - Q3 - 2019](https://www.car.org/marketdata/data/haitraditional) from the California Association of Realtors, only 31% of California residents can afford to purchase the median price home in their region.

Our research focus is to predict the median housing price per census block in California based on the nine given variables:
##### Location
* longitude: longitude of the census block
* latitude: latitude of the census block
* ocean_proximity: categorical variable (5 categories) 

##### Home Characteristics
* median age: median house age per census block in years
* total_rooms: total rooms in the census block
* total_bedrooms: total bedrooms in the census block

##### Demographics
* median_income: median income of the census block
* population: total population of the census block
* households: total number of households in the census block

##### Target
* median_house_value

Our goal is to build a model that will predict median house value with a higher model score than the 0.60 achieved by [Eric Chen](https://www.kaggle.com/ericfeng84), the author of [The California House Price](https://www.kaggle.com/ericfeng84/the-california-housing-price) Kaggle page from which the dataset was obtained.

#### How we will analyze the Data
Since this dataset is publicly available on [Kaggle](https://www.kaggle.com/camnugent/california-housing-prices/kernels), there have been many analyses completed. The analysis that we focused on was [completed by Aurélien Geron](https://www.kaggle.com/artemiosgeromitsos/housing-super-duper-kernel/notebook). He concluded that a linear regressor was the most appropriate model. The focus of our analysis is to attempt to improve on his success by taking into consideration multicollinearity (relationships that exist between independent variables), tuning hyperparameters (hyperparameters will be chosen based on GridSearchCV, which uses the given estimator's default scoring method), and trying different models that were not included in his analysis.

We will use Linear Regression, K-Nearest Neighbour Trees, and a Random Forest Regressor to predict the median house value given the independent variables.

#### Exploratory Data Analysis Figures and Tables to be Used
Before diving into the figures, it is key to understand that each record (row) of data represents a [census block](https://www.census.gov/newsroom/blogs/random-samplings/2011/07/what-are-census-blocks.html) in California, and some attributes describe a census block (like the population of the given census block), while others describe medians within a census block (median income in the census block).

The first set of figures produced shows each of the quantitative explanatory variables plotted against the median house value. These 6 plots were chosen as an initial analysis because in Aurélien Geron's version of this analysis, he concluded that linear regression was the best estimation method. We chose to start by attempting to see which (if any) of the explanatory variables had patterns or variances compared to median house price.

From basic relationships, the next 2 items are used to look at multicollinearity. We produced the variance inflation factors (VIFs) of the quantitative explanatory variables, and many of them have significant multicollinearity. This can also be seen in the plot comparing rooms against bedrooms, and as expected, bedrooms has an extremely high VIF of 34. We will continue to explore the role of multicollinearity in the models.

The final item is a heatmap showing correlation between both independent and dependent variables. The variables with strong positive correlations are population, total rooms, and total bedrooms. Most other variables had positive correlation.

#### Sharing Results
To share the results of our analysis, we plan to produce figures showing how model performance (i.e. model error) varies with choice of model and hyperparameter settings both for a general model covering data for the entire state of California, and where appropriate also for region specific models for which the optimized model weights are significantly different than those for the general model.
