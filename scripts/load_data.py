# author: Group 312, George, Eithar, Sam
# date: 2020-1-18

"""
Downloads data csv data from the web to a local filepath as either a csv or feather file format.

Usage: load_data.py --url=<url> --file_path=<file_path> 

Options:
--file_path=<file_path>  Path (including filename) to the csv file.
--url=<file_path>  url from where to download the data
"""


  
from docopt import docopt
import requests
import os
import pandas as pd

opt = docopt(__doc__)

def main(url, file_path):
  
  data = pd.read_csv(url, header = None)
  data.to_csv(file_path, index = False)
  return

if __name__ == "__main__":
   main(url = opt["--url"], file_path = opt["--file_path"])
