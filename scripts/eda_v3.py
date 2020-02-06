# author: Group 312
# date: 2020-1-23

""" 
Complete Exploratory Data Analysis on training portion of data and save figures and outputs

Usage: 
scripts/eda_v3.py --train_path=<train_path> --out_folder_path=<out_folder_path>

Options:
--train_path=<train_path> Path (including filename) to find wrangled and split data
--out_folder_path=<out_folder_path> Path to a folder to write outputs to
"""

# Dependencies
import requests
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import altair as alt
import selenium
from docopt import docopt

from string import ascii_letters
import seaborn as sns
import matplotlib.pyplot as plt


opt = docopt(__doc__)

def main(train_path, out_folder_path):

    # Allow Altair to make plots using more than 5000 rows
    alt.data_transformers.disable_max_rows()

    # Read data from file path
    train = pd.read_csv(train_path)
    columns = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 
    'households', 'median_income', 'ocean_proximity', 'latitude', 'longitude']
    X_train = train[columns]
    y_train = train['median_house_value']
    
    
    def make_chart(x, x_title):
        """
        Creates an altair scatterplot with the input on the x-axis and 
        the median house value on the y-axis
    
        Parameters
        ----------
        x: string
            the column name of the x-axis column to be created
        x_title: string
            the title of the x-axis (and to be used in the chart title)
    
        Returns
        ----------
        altair chart object
            scatterplot of defined x compared to median house value
        """
        chart = alt.Chart(train).mark_point(opacity=0.2).encode(
            alt.X(x, title=x_title),
            alt.Y('median_house_value:Q', title="Median House Value")
        ).properties(
            width=300,
            height=250,
            title="Median House Value per " + x_title
        )
        return chart
        
        
    # Make charts and save them
    make_chart('housing_median_age', "House Median Age").save(out_folder_path + 'median-age_scatterplot.png')
    make_chart('total_rooms', "Total Rooms").save(out_folder_path + 'total-rooms_scatterplot.png')
    make_chart('total_bedrooms', "Total Bedrooms").save(out_folder_path + 'total-bedrooms_scatterplot.png')
    make_chart('population', "Population").save(out_folder_path + 'population_scatterplot.png')
    make_chart('households', "Households").save(out_folder_path + 'households_scatterplot.png')
    make_chart('median_income', "Median Income").save(out_folder_path + 'median-income_scatterplot.png')

    # Look at the relationship between total_rooms and total_bedrooms
    alt.Chart(X_train).mark_point(opacity=0.2).encode(
        alt.X('total_bedrooms', title="Total Bedrooms"),
        alt.Y('total_rooms', title="Total Rooms")
    ).properties(
        width=300,
        height=250,
        title="Relationship between Bedroom and Room Counts"
    ).save(out_folder_path + 'total-rooms_total-bedrooms.png')
    
    
    # Visualize Correlation Matrix between variables
    # Rename correlation matrix titles
    corrmatrix_titles = {"housing_median_age":"Median House Age", "total_rooms":"Total Rooms",
                    "total_bedrooms":"Total Bedrooms", "population":"Population", "households":"Households",
                    "median_income":"Median Income", "latitude":"Latitude", "longitude":"Longitude", "median_house_value":"Median House Value"}
    corrMatrix = train.corr()
    corrMatrix = corrMatrix.rename(columns = corrmatrix_titles)
    corrMatrix = corrMatrix.rename(index = corrmatrix_titles)
    
    # Create mask for upper triangle
    upper_mask = np.triu(np.ones_like(corrMatrix, dtype=np.bool))

    # Create matplotlib figure
    corr_plot, ax = plt.subplots(figsize=(12, 12))

    # Create heatmap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Overlay mask on heat map
    sns.heatmap(corrMatrix,
            mask=upper_mask,
            cmap=cmap,
            vmax=0.5,
            center=0,
            square=True, 
            linewidths=.8,
            cbar_kws={"shrink": 0.5})
    plt.title('Correlation Matrix')

    # Save heatmap to file
    corr_plot.savefig(out_folder_path + 'correlation_heatmap.png')
    
    
    # Create Variance Inflation Factor Table
    # Drop `ocean_proximity`, `latitude` and `longitude` columns
    mc_data = pd.DataFrame.drop(X_train, 
        columns=['ocean_proximity', 
                'longitude', 
                'latitude'])
    mc_data['intercept'] = 1

    # Create and return dataframe with VIFs
    vif = pd.DataFrame()
    vif['variable'] = mc_data.columns
    vif['VIF'] = [variance_inflation_factor(mc_data.values, i) for i in range(mc_data.shape[1])]
    vif.to_csv(out_folder_path + 'vif_table.csv', index=False)

    # Sources:
    # https://campus.datacamp.com/courses/generalized-linear-models-in-python/multivariable-logistic-regression?ex=4


    assert os.path.isfile(out_folder_path + 'correlation_heatmap.png')
    assert os.path.isfile(out_folder_path + 'households_scatterplot.png')
    assert os.path.isfile(out_folder_path + 'median-age_scatterplot.png')
    assert os.path.isfile(out_folder_path + 'median-income_scatterplot.png')
    assert os.path.isfile(out_folder_path + 'population_scatterplot.png')
    assert os.path.isfile(out_folder_path + 'total-bedrooms_scatterplot.png')
    assert os.path.isfile(out_folder_path + 'total-rooms_scatterplot.png')
    assert os.path.isfile(out_folder_path + 'total-rooms_total-bedrooms.png')
    assert os.path.isfile(out_folder_path + 'vif_table.csv')
    
if __name__ == "__main__": 
    main(train_path = opt["--train_path"], out_folder_path = opt["--out_folder_path"])
