# author: Group 312
# date: 2020-1-23

""" 
This script creates three sets of exploratory data charts and one table 
to help users better understand the dataset. The files are saved in png and
csv formats respectively to the specified paths/filenames which are taken as
arguments together with a path/filename to the data.

Arguments
---------
1. Path/filename pointing to the dataset 
2. Path/filename indictating location and name of charts and table to 
be saved (e.g. results/this_eda.png, results/this_eda.csv) as png or csv file

Usage: eda.py --X_train_path=<X_train_path> --y_train_path=<y_train_path> 
--chart1_path<chart1_path> --chart2_path=<chart2_path> --chart3_path=<chart3_path> 
--table_path=<table_path>   

Options:
--X_train_path=<X_train_path> Path (including filename) to find wrangled and split data
--y_train_path=<y_train_path> Path (including filename) to find wrangled and split data
--chart1_path<chart1_path> Path (including filename) to write first chart
--chart2_path=<chart2_path> Path (including filename) to write second chart
--chart3_path=<chart3_path> Path (including filename) to write third chart
--table_path=<table_path> Path (including filename) to write table
"""

# Dependencies
import requests
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import altair as alt
import selenium
from docopt import docopt

opt = docopt(__doc__)

def main(X_train_path, y_train_path, chart1_path, chart2_path, chart3_path, table_path):

    # Allow Altair to make plots using more than 5000 rows
    alt.data_transformers.disable_max_rows()

    # Read data from file path
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    
    # Scatterplots of various explanatory variables
    train_df = pd.concat([X_train, y_train], axis=1) 
    expl_var = list(train_df.columns[0:7])

    eda1 = alt.Chart(train_df).mark_point().encode(
        alt.X(alt.repeat("column"), 
            type='quantitative'),
        alt.Y(alt.repeat("row"), 
            type='quantitative')).properties(
        width=300,
        height=300).repeat(
        column=['median_house_value'],
        row=expl_var)
    eda1.save('eda1.png')
    os.rename(r'eda1.png', r'chart1_path')


    eda2 = alt.Chart(train_df).mark_point().encode(
        x='total_bedrooms:Q',
        y='total_rooms:Q').properties(
        width=300,
        height=300)
    eda2.save('eda2.png')
    os.rename(r'eda2.png', r'chart2_path')


    # Plot correlation matrix for all variables
    viz_data = pd.merge(X_train, 
                        y_train, 
                        left_index=True, 
                        right_index=True)
    corrMatrix = viz_data.corr()
    corrMatrix['names'] = corrMatrix.columns
    corrMatrix = corrMatrix.melt(id_vars = 'names',
                value_vars = corrMatrix['names'])

    eda3 = alt.Chart(corrMatrix).mark_rect().encode(
                            x = alt.X('names:O', 
                            title = None),
                            y = alt.Y('variable:O', 
                            title = None),
                            color= alt.Color('value:Q', 
                            title = 'Correlation Value')).properties(
                                width = 400, 
                                height = 300, 
                                title = "Correlation Heatmap")
   
    eda3.save('eda3.png')
    os.rename(r'eda3.png', r'chart3_path')

    # Create Variance Inflation Factor (VIF) table

    # Drop `ocean_proximity` column
    mc_data = pd.DataFrame.drop(X_train, 
        columns=['ocean_proximity', 
                'longitude', 
                'latitude'])
    mc_data['intercept'] = 1

    # Create and return dataframe
    mc_data.shape[1]
    vif = pd.DataFrame()
    vif['variable'] = mc_data.columns
    vif['vif_val'] = [variance_inflation_factor(mc_data.values, i) for i in range(mc_data.shape[1])]
    vif.to_csv(table_path, index=False) 

    # Sources:
    # https://campus.datacamp.com/courses/generalized-linear-models-in-python/multivariable-logistic-regression?ex=4

    return

if __name__ == "__main__": 
    main(X_train_path = opt["--X_train_path"], y_train_path = opt["--y_train_path"], chart1_path = opt["--chart1_path"], 
    chart2_path = opt["--chart2_path"], chart3_path = opt["--chart3_path"], table_path = opt["--table_path"])