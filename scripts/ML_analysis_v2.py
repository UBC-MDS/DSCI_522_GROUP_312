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
    
    # CREATE A DF THAT EXCLUDES LATITUDE AND LONGITUDE
    X_train_featexc = X_train.drop(columns=["latitude", "longitude"])
    X_test_featexc = X_test.drop(columns=["latitude", "longitude"])
    
    # CREATE A DF THAT EXCLUDES LATITUDE, LONGITUDE, AND TOTAL BEDROOMS
    X_train_featexc_2 = X_train.drop(columns=["latitude", "longitude", "total_bedrooms"])
    X_test_featexc_2 = X_test.drop(columns=["latitude", "longitude", "total_bedrooms"])
    
    # APPLY SCALER
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_featexc = scaler.fit_transform(X_train_featexc)
    X_test_featexc = scaler.transform(X_test_featexc)
    X_train_featexc_2 = scaler.fit_transform(X_train_featexc_2)
    X_test_featexc_2 = scaler.transform(X_test_featexc_2)
    
    
    
    # LINEAR REGRESSION WITH FEATURE SELECTION - ALL FEATURES AVAILABLE
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
    ).properties(
        title="Recursive Feature Elimination Linear Regression Error",
        width=250,
        height=200
    )
    plot.save(output_path + 'LR_performace.png')
    
    
    
    # LINEAR REGRESSION WITH FEATURE SELECTION - EXCLUDING LATITUDE AND LONGITUDE
    lr_response_exc = {'n_features_to_select':[], 'train_error':[], 'test_error':[]}

    for i in list(range(1, X_train_featexc.shape[1]+1, 1)):
        lr_response_exc['n_features_to_select'].append(i)
    
        rfe_lr = RFE(LinearRegression(), n_features_to_select=i)
        rfe_lr.fit(X_train_featexc, y_train)
        lr_response_exc['train_error'].append(round(1 - rfe_lr.score(X_train_featexc, y_train), 3))
        lr_response_exc['test_error'].append(round(1 - rfe_lr.score(X_test_featexc, y_test), 3))
    pd.DataFrame(lr_response_exc).to_csv(output_path + 'lr_rfe_results_table_exc_feats.csv', index=False)

    # Plotting LR performance excluding latitude and longitude
    data = pd.DataFrame(lr_response_exc).melt(id_vars = 'n_features_to_select' ,value_vars= ['train_error', 'test_error'])
    plot = alt.Chart(data).mark_line().encode(
    x = alt.X('n_features_to_select:Q', title="Number of Features Selected"),
    y = alt.Y('value:Q', title="Error"),
    color = alt.Color('variable:N', title="Data Split")
    ).properties(
        title="Recursive Feature Elimination Linear Regression Error Excluding Latitude and Longitude",
        width=250,
        height=200
    )
    plot.save(output_path + 'LR_performace_exc_feats.png')
    
    
    
    # LINEAR REGRESSION WITH FEATURE SELECTION - EXCLUDING LATITUDE, LONGITUDE, AND TOTAL BEDROOMS
    lr_response_exc_2 = {'n_features_to_select':[], 'train_error':[], 'test_error':[]}

    for i in list(range(1, X_train_featexc_2.shape[1]+1, 1)):
        lr_response_exc_2['n_features_to_select'].append(i)
    
        rfe_lr = RFE(LinearRegression(), n_features_to_select=i)
        rfe_lr.fit(X_train_featexc_2, y_train)
        lr_response_exc_2['train_error'].append(round(1 - rfe_lr.score(X_train_featexc_2, y_train), 3))
        lr_response_exc_2['test_error'].append(round(1 - rfe_lr.score(X_test_featexc_2, y_test), 3))
    pd.DataFrame(lr_response_exc_2).to_csv(output_path + 'lr_rfe_results_table_exc_feats_2.csv', index=False)

    # Plotting LR performance excluding latitude and longitude
    data = pd.DataFrame(lr_response_exc_2).melt(id_vars = 'n_features_to_select' ,value_vars= ['train_error', 'test_error'])
    plot = alt.Chart(data).mark_line().encode(
    x = alt.X('n_features_to_select:Q', title="Number of Features Selected"),
    y = alt.Y('value:Q', title="Error"),
    color = alt.Color('variable:N', title="Data Split")
    ).properties(
        title="Recursive Feature Elimination Linear Regression Error Excluding Latitude, Longitude, and Total Bedrooms",
        width=250,
        height=200
    )
    plot.save(output_path + 'LR_performace_exc_feats_2.png')
    
    

    # KNN WITH VARYING N_NEIGHBOR VALUES WITH FULL DATA INCLUSION
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
    ).properties(
        title="K-Nearest Neighbour Error when Varying K",
        width=250,
        height=200
    )
    plot.save(output_path + 'KNN_performace.png')
    
    
    
    # KNN WITH VARYING N_NEIGHBOR VALUES WITH LATITUDE AND LONGITUDE EXCLUSION
    knn_response_exc = {'n_neighbours':[], 'train_error':[], 'test_error':[]}

    for i in list(range(1, 20, 1)):
        knn_response_exc['n_neighbours'].append(i)
    
        knn_exc = KNeighborsRegressor(n_neighbors=i)
        knn_exc.fit(X_train_featexc, y_train)
        knn_response_exc['train_error'].append(round(1 - knn_exc.score(X_train_featexc, y_train), 3))
        knn_response_exc['test_error'].append(round(1 - knn_exc.score(X_test_featexc, y_test), 3))
    pd.DataFrame(knn_response_exc).to_csv(output_path + 'knn_results_table_exc_feats.csv', index=False)

    # ploting KNN performance
    data = pd.DataFrame(knn_response_exc).melt(id_vars = 'n_neighbours' ,value_vars= ['train_error', 'test_error'])
    plot = alt.Chart(data).mark_line().encode(
    x = alt.X('n_neighbours:Q', title="Number of Nearest Neighbours"),
    y = alt.Y('value:Q', title="Error"),
    color = alt.Color('variable:N', title="Data Split")
    ).properties(
        title="K-Nearest Neighbour Error when Varying K and Excluding Latitude and Longitude",
        width=250,
        height=200
    )
    plot.save(output_path + 'KNN_performace_exc_feats.png')
    


    # RANDOM FOREST REGRESSOR
    rfr = RandomForestRegressor(random_state = 522)
    gs = GridSearchCV(rfr, param_grid = {"max_depth": np.arange(5,10,1), "min_samples_leaf": np.arange(1,4,1)})
    gs.fit(X_train, y_train)
    rfr = gs.best_estimator_
    rfr_response = {'type':['Random Forest Regressor'], 
                'train_error':[round(1 - rfr.score(X_train, y_train), 3)], 
                'test_error':[round(1 - rfr.score(X_test, y_test), 3)]}
    pd.DataFrame(rfr_response).to_csv(output_path + 'rfr_results_table.csv', index=False)

    # TESTING
    assert os.path.isfile(output_path + 'rfr_results_table.csv')
    assert os.path.isfile(output_path + 'KNN_performace.png')
    assert os.path.isfile(output_path + 'lr_rfe_results_table.csv')
    assert os.path.isfile(output_path + 'LR_performace.png')
    assert os.path.isfile(output_path + 'rfr_results_table.csv')
    assert os.path.isfile(output_path + 'knn_results_table_exc_feats.csv')
    assert os.path.isfile(output_path + 'KNN_performace_exc_feats.png')
    assert os.path.isfile(output_path + 'lr_rfe_results_table_exc_feats.csv')
    assert os.path.isfile(output_path + 'LR_performace_exc_feats.png')
    assert os.path.isfile(output_path + 'lr_rfe_results_table_exc_feats_2.csv')
    assert os.path.isfile(output_path + 'LR_performace_exc_feats_2.png')


if __name__ == "__main__":
    main(training_input_path = opt["--training_input_path"], testing_input_path = opt["--testing_input_path"], output_path = opt["--output_path"])







