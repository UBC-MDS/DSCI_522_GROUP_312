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
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
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

    X_train = train.drop(columns = "median_income")
    y_train = train['median_income']
    X_test = test.drop(columns = "median_income")
    y_test = test['median_income']

    # Results table
    results = {'Model' : ["Random Forest"], "Training R squared" : [], "testing R squared": []}

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rfr = RandomForestRegressor().fit(X_train, y_train)

    results["Training R squared"].append(round(rfr.score(X_train, y_train), 3))

    results["testing R squared"].append(round(rfr.score(X_test, y_test), 3))

    # Put results in a dataframe
    df = pd.DataFrame(data = results)

    df.to_csv(output_path, index = False)

    return

if __name__ == "__main__":

    main(training_input_path = opt["--training_input_path"], testing_input_path = opt["--testing_input_path"], output_path = opt["--output_path"])







