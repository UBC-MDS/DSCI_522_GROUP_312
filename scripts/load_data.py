# author: Group 312
# date: 2020-01-18

# This file loads The California Housing Price 
# csv dataset into a dataframe

import pandas as pd

def main():
  # read in data
  df = pd.read_csv('../data/src_data/housing.csv')

if __name__ == "__main__":
    main()