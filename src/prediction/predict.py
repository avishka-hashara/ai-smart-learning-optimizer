import numpy as np
import pandas as pd

def predict_scores(features_df):
    """Stage 1 Stub: Returns random predictions between 30 and 80."""
    n = len(features_df)
    return np.random.uniform(30, 80, size=n)