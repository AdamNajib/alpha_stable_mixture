import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

def extract_serial_intervals(df):
    """
    Given a DataFrame with columns 'x.lb', 'x.ub', 'y' (all dates),
    compute the serial interval (SI) as:
    SI = y - mean(x.lb, x.ub)
    Returns the array of SI values and updates the DataFrame in-place.
    """
    df['x.lb'] = pd.to_datetime(df['x.lb'], format="%d/%m/%Y")
    df['x.ub'] = pd.to_datetime(df['x.ub'], format="%d/%m/%Y")
    df['y'] = pd.to_datetime(df['y'], format="%d/%m/%Y")

    midpoint = (df['x.lb'].view('int64') + df['x.ub'].view('int64')) // 2
    midpoint = pd.to_datetime(midpoint)

    df['SI'] = (df['y'] - midpoint).dt.days.astype(float)
    return df['SI'].dropna().values
