# Author: Group 312

# Imports
import numpy as np
import pandas as pd
import altair as alt
import selenium
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Allow Altair to make plots using more than 5000 rows
alt.data_transformers.disable_max_rows()

# load the data
data = pd.read_csv('https://github.com/ageron/handson-ml/blob/master/datasets/housing/housing.csv?raw=true')

# Wrangling
# drop rows with NA values
data = data.dropna()
# change median_income to tens of thousands of dollars
data['median_income'] = data['median_income']*10000
# move median_house_value to last column for formatting
columns = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
           'households', 'median_income', 'ocean_proximity']
X = data[columns]
y = data['median_house_value']

# Sources:
# https://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=522)

# For visualizations, combine X_train and y_train
viz_data = pd.merge(X_train, y_train, left_index=True, right_index=True)


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
    chart = alt.Chart(viz_data).mark_point(opacity=0.2).encode(
        alt.X(x, title=x_title),
        alt.Y('median_house_value:Q', title="Median House Value")
    ).properties(
        width=200,
        height=200,
        title="Median House Value per " + x_title
    )
    
    return chart

# Make charts and save them
make_chart('housing_median_age', "House Median Age").save('median-age_scatterplot.png')
make_chart('total_rooms', "Total Rooms").save('total-rooms_scatterplot.png')
make_chart('total_bedrooms', "Total Bedrooms").save('total-bedrooms_scatterplot.png')
make_chart('population', "Population").save('population_scatterplot.png')
make_chart('households', "Households").save('households_scatterplot.png')
make_chart('median_income', "Median Income").save('median-income_scatterplot.png')

# Look at the relationship between total_rooms and total_bedrooms
alt.Chart(viz_data).mark_point(opacity=0.2).encode(
    alt.X('total_bedrooms', title="Total Bedrooms"),
    alt.Y('total_rooms', title="Total Rooms")
).properties(
    width=350,
    height=300,
    title="Relationship between Bedroom and Room Counts"
).save('total-rooms_total-bedrooms.png')

