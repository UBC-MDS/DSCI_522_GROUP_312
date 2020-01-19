# author: Group 312
# date: 2020-1-18

"""Downloads csv data from a URL and writes to a local filepath as a csv file 
Usage: /scripts/load_data.py --url=<url> --out_file=<out_file>
Options:
--url=<url>              URL from where to download the data (must be in standard csv format)
--out_file=<out_file>    Path (including filename) of where to locally write the file
"""
  
from docopt import docopt
import requests
import os
import pandas as pd
import feather

opt = docopt(__doc__)

def main(out_type, url, out_file):
  try: 
    request = requests.get(url)
    request.status_code == 200
  except Exception as req:
    print("No website exists for provided url, please check URL and try again")
    print(req)
    
  data = pd.read_csv(url, header=None)

  try:
    data.to_csv(out_file, index = False)
  except:
    os.makedirs(os.path.dirname(out_file))
    data.to_csv(out_file, index = False)

if __name__ == "__main__":
  main(opt["--url"], opt["--out_file"])