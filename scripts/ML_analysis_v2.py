# author: Group 312
# date: 2020-1-23

"""
Fits models (Linear Regressor, KNN, and Random Forest Regressor) and saves relevant results to specified path.

Usage: 
scripts/ML_analysis_v2.py --training_input_path=<training_input_path> --testing_input_path=<testing_input_path>  --output_path=<output_path> 

Options:
--training_input_path=<training_input_path>  Path (including filename) to the training dataset csv file.
--testing_input_path=<testing_input_path>  Path (including filename) to the testing dataset csv file.
--output_path=<output_path>  Path to the folder for results to be created in
"""

# Import The nessaccery libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from docopt import docopt
import numpy as np
import altair as alt
import os.path

opt = docopt(__doc__)

def main(training_input_path, testing_input_path, output_path):

    # LOAD DATA
    train = pd.read_csv(training_input_path, header = 0)
    test = pd.read_csv(testing_input_path, header = 0)
    
    # PREPROCESSING
    le = LabelEncoder()
    train["ocean_proximity"] = le.fit_transform(train["ocean_proximity"])
    test["ocean_proximity"] = le.transform(test["ocean_proximity"])
    
    # SPLIT TRAINING AND TESTING DATA INTO X AND Y
    X_train = train.drop(columns = "median_house_value")
    y_train = train['median_house_value']
    X_test = test.drop(columns = "median_house_value")
    y_test = test['median_house_value']
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # LINEAR REGRESSION WITH FEATURE SELECTION

    lr_response = {'n_features_to_select':[], 'train_error':[], 'test_error':[]}

    for i in list(range(1, X_train.shape[1]+1, 1)):
        lr_response['n_features_to_select'].append(i)
    
        rfe_lr = RFE(LinearRegression(), n_features_to_select=i)
        rfe_lr.fit(X_train, y_train)
        lr_response['train_error'].append(round(1 - rfe_lr.score(X_train, y_train), 3))
        lr_response['test_error'].append(round(1 - rfe_lr.score(X_test, y_test), 3))
    pd.DataFrame(lr_response).to_csv(output_path + 'lr_rfe_results_table.csv', index=False)

    # Plotting LR performance
    data = pd.DataFrame(lr_response).melt(id_vars = 'n_features_to_select' ,value_vars= ['train_error', 'test_error'])
    plot = alt.Chart(data).mark_line().encode(
    x = alt.X('n_features_to_select:Q', title="Number of Features Selected"),
    y = alt.Y('value:Q', title="Error"),
    color = alt.Color('variable:N', title="Data Split")
    ).properties(title="Recursive Feature Elimination Linear Regression Error")
    plot.save(output_path + 'LR_performace.png')

    # KNN WITH VARYING N_NEIGHBOR VALUES
    knn_response = {'n_neighbours':[], 'train_error':[], 'test_error':[]}

    for i in list(range(1, 20, 1)):
        knn_response['n_neighbours'].append(i)
    
        knn = KNeighborsRegressor(n_neighbors=i)
        knn.fit(X_train, y_train)
        knn_response['train_error'].append(round(1 - knn.score(X_train, y_train), 3))
        knn_response['test_error'].append(round(1 - knn.score(X_test, y_test), 3))
    pd.DataFrame(knn_response).to_csv(output_path + 'knn_results_table.csv', index=False)

    # ploting KNN performance
    data = pd.DataFrame(knn_response).melt(id_vars = 'n_neighbours' ,value_vars= ['train_error', 'test_error'])
    plot = alt.Chart(data).mark_line().encode(
    x = alt.X('n_neighbours:Q', title="Number of Nearest Neighbours"),
    y = alt.Y('value:Q', title="Error"),
    color = alt.Color('variable:N', title="Data Split")
    ).properties(title="K-Nearest Neighbour Error when Varying K")
    plot.save(output_path + 'KNN_performace.png')

    # RANDOM FOREST REGRESSOR
    rfr = RandomForestRegressor(random_state = 522)
    gs = GridSearchCV(rfr, param_grid = {"max_depth": np.arange(5,10,1), "min_samples_leaf": np.arange(1,4,1)})
    gs.fit(X_train, y_train)
    rfr = gs.best_estimator_
    rfr_response = {'type':['Random Forest Regressor'], 
                'train_error':[round(1 - rfr.score(X_train, y_train), 3)], 
                'test_error':[round(1 - rfr.score(X_test, y_test), 3)]}
    pd.DataFrame(rfr_response).to_csv(output_path + 'rfr_results_table.csv', index=False)

    # testing
    assert os.path.isfile(output_path + 'rfr_results_table.csv')
    assert os.path.isfile(output_path + 'KNN_performace.png')
    assert os.path.isfile(output_path + 'lr_rfe_results_table.csv')
    assert os.path.isfile(output_path + 'LR_performace.png')
    assert os.path.isfile(output_path + 'rfr_results_table.csv')


if __name__ == "__main__":
    main(training_input_path = opt["--training_input_path"], testing_input_path = opt["--testing_input_path"], output_path = opt["--output_path"])







