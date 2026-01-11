import opendatasets as od
import pandas as pd
import os

data_dir = 'home-credit-default-risk'

if not os.path.exists(data_dir):
    print("Trying to download dataset from Kaggle")
    dataset_url = 'https://www.kaggle.com/c/home-credit-default-risk'
    od.download(dataset_url)
    print("Downloading completed.")
else:
    print(f"Dataset already exists in '{data_dir}'.")

