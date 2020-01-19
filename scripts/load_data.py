# author: Group 312
# date: 2020-01-18

# This file loads The California Housing Price 
# csv dataset into a dataframe

import pandas as pd

def main():
  # read in data
  data = pd.read_csv('https://github.com/ageron/handson-ml/blob/master/datasets/housing/housing.csv?raw=true')
  data.to_csv('loaded_raw_data.csv')


if __name__ == "__main__":
    main()