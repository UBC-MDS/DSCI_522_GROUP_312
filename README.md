# DSCI 522 - Group 312

## About
This repository was created as part of an assignment in [UBC's Master of Data Science](https://masterdatascience.ubc.ca/) program.

## Proposal
#### Dataset
This dataset is a modified version of [The California Housing Dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html), with [additional columns added by Aurélien Geron](https://github.com/ageron/handson-ml). This dataset contains information about median California house values (including related factors) as sourced from the 1990 US Census.

The script to load the California Housing Dataset into python can be found here **(add link!!!)**

#### Research Question
**George**

#### How we will analyze the Data
**Eithar**

#### Exploratory Data Analysis Figures and Tables to be Used
The full Exploratory Data Analysis can be found [here](analysis/EDA.ipynb).

Before diving into the figures, it is key to understand that each record (row) of data represents a [census block](https://www.census.gov/newsroom/blogs/random-samplings/2011/07/what-are-census-blocks.html) in California, and some attributes describe a census block (like the population of the given census block), while others describe medians within a census block (median income in the census block).

The first set of figures produced shows each of the quantitative explanatory variables plotted against the median house value. These 6 plots were chosen as an initial analysis because in Aurélien Geron's version of this analysis, he concluded that linear regression was the best estimation method. We chose to start by attempting to see which (if any) of the explanatory variables had patterns or variances compared to median house price.

From basic relationships, the next 2 items are used to look at multicollinearity. We produced the variance inflation factors (VIFs) of the quantitative explanatory variables, and many of them have significant multicollinearity. This can also be seen in the plot comparing rooms against bedrooms, and as expected, bedrooms has an extremely high VIF of 3. By both diagnosing multicollinearity and examining whether or not there are nested models, we may be able to expand on this analysis.

#### Sharing Results
**George**

## Contributing Policy
Please note that all contributions are subject to the [Code of Conduct](CODE_OF_CONDUCT.md).

We welcome contributions to this project! If you find a bug, have a feature request, or have general suggestions to improve this repository, please submit an issue, or feel free to fork this repository and submit a pull request.