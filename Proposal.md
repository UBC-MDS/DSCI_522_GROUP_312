{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data\n",
    "As discussed in a special report by The Economist on January 16, 2020 ['Housing is at the root of many of the rich world’s problems'](https://www.economist.com/special-report/2020/01/16/housing-is-at-the-root-of-many-of-the-rich-worlds-problems) and nowhere is this problem more acute than in the richest and most populous American state where Bloomberg has attempted to explain [How California Became America’s Housing Market Nightmare](https://www.bloomberg.com/graphics/2019-california-housing-crisis/).\n",
    "\n",
    "Although finding solutions for this deep-seated socioeconomic problem is beyond the scope of this research project, we have identified a publicly available data set [(The California Housing Dataset)](https://www.kaggle.com/ericfeng84/the-california-housing-price) with which we propose to build a machine learning model that will identify factors which can be used to predict housing prices to better inform stakeholders in this increasingly unaffordable market. Based on the latest report [(California Housing Affordability Update - Q3 - 2019)](https://www.car.org/marketdata/data/haitraditional) from the California Association of Realtors, only 31% of California residents can afford to purchase the median price home in their region.\n",
    "\n",
    "The main predictive research question we will attempt to answer is which machine learning model and model coefficients are the best predictors of median house prices in California based on the following 10 features from the data set which contains 20,640 examples:\n",
    "\n",
    "#### Location\n",
    "* longitude  \n",
    "* latitude \n",
    "* ocean_proximity\n",
    "\n",
    "#### Home Characteristics\n",
    "* median age \n",
    "* total_rooms\n",
    "* total_bedrooms\n",
    "\n",
    "#### Demographics\n",
    "* median_income\n",
    "* population\n",
    "* households\n",
    "\n",
    "#### Target\n",
    "* median_house_value\n",
    "\n",
    "### Action Plan\n",
    "After cleaning, preprocessing and splitting the data into train (80%) and test sets (20%), we plan to explore the use of different models including (but not necessarily limited to) linear regression and KNN in conjunction with k-fold cross-validation to find the best predictor. Our goal is to build a model with significantly higher accuracy than the 0.60 achieved by [Eric Chen](https://www.kaggle.com/ericfeng84), the author of [The California House Price](https://www.kaggle.com/ericfeng84/the-california-housing-price).\n",
    "\n",
    "### Exploratory Data Analysis\n",
    "As part of our EDA (on the train data only) we will produce a `df.describe()` table to view summary statistices, central tendency, dispersion, and shape of the data set, a correlation matrix plot to observe the relationships between the dependent variables and identify potential multicollinearity, and histograms to visualize distributions of the features.\n",
    "\n",
    "### Sharing Results\n",
    "To share the results of our analysis, we plan to produce figures showing how model performance varies with choice of model and hyperparameter settings\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
