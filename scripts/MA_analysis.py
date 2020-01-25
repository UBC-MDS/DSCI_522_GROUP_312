# author: Group 312
# date: 2020-1-23

"""
Performs data analysis and saves the results in the data folder.

Usage: MA_analysis.py --training_input_path=<training_input_path> --testing_input_path=<testing_input_path>  --output_path=<output_path> 

Options:
--training_input_path=<training_input_path>  Path (including filename) to the training dataset csv file.
--testing_input_path=<testing_input_path>  Path (including filename) to the testing dataset csv file.
--output_path=<output_path>  Path (including filename) to the csv file.
"""

# Import The nessaccery libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from docopt import docopt

opt = docopt(__doc__)

def main(training_input_path, testing_input_path, output_path):

# Read in the training dataset 
    train = pd.read_csv(training_input_path, header = 0)

    # Read in the testing dataset 
    test = pd.read_csv(testing_input_path, header = 0)

    le = LabelEncoder()
    train["ocean_proximity"] = le.fit_transform(train["ocean_proximity"])
    test["ocean_proximity"] = le.transform(test["ocean_proximity"])

    X_train = train.drop(columns = "median_house_value")
    y_train = train['median_house_value']
    X_test = test.drop(columns = "median_house_value")
    y_test = test['median_house_value']

    # Results table
    results = {'Model' : ["Random Forest With 10 features", "Random Forest With 5 Featutes", "SVR With 10 features", "SVR With 5 features"], 
           "Training R squared" : [], 
           "testing R squared": []}
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rfr_before = RandomForestRegressor().fit(X_train, y_train)

    results["Training R squared"].append(round(rfr_before.score(X_train, y_train), 3))

    results["testing R squared"].append(round(rfr_before.score(X_test, y_test), 3))

    # Getting the feature importance To see which features are relevent
    features = {"feature" : ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
       'households', 'median_income', 'ocean_proximity', 'latitude',
       'longitude'], 
        "importance" : list(rfr_before.feature_importances_)}
    features = pd.DataFrame(data = features)

    # Feature Selection based on the feature importance accquired from the RandomForest model. 
    lr = LinearRegression()
    # Choosing 5 features
    rfe = RFE(estimator = lr, n_features_to_select = 5)
    rfe.fit(X_train, y_train)

    X_train_sel = X_train[:, rfe.support_]
    X_test_sel = X_test[:, rfe.support_]
    rfr_after = RandomForestRegressor().fit(X_train_sel, y_train)
    results["Training R squared"].append(round(rfr_after.score(X_train_sel, y_train), 3))
    results["testing R squared"].append(round(rfr_after.score(X_test_sel, y_test), 3))

    # Training a Support vector regressor
    svr_model = SVR().fit(X_train, y_train)
    results["Training R squared"].append(round(svr_model.score(X_train, y_train), 3))
    results["testing R squared"].append(round(svr_model.score(X_test, y_test), 3))
    svr_model_after = SVR().fit(X_train_sel, y_train)

    results["Training R squared"].append(round(svr_model_after.score(X_train_sel, y_train), 3))
    results["testing R squared"].append(round(svr_model_after.score(X_test_sel, y_test), 3))
    
    # Put results in a dataframe
    df = pd.DataFrame(data = results)
    # Saving to a csv file
    df.to_csv(output_path, index = False)

    return

if __name__ == "__main__":

    main(training_input_path = opt["--training_input_path"], testing_input_path = opt["--testing_input_path"], output_path = opt["--output_path"])







