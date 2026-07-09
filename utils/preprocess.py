import pandas as pd
import numpy as np

def load_raw_data(path):
    df = pd.read_csv(path, index_col=0)
    return df

def preprocess_data(df):
    return df

