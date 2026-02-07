import pandas as pd


def one_hot_encoder(df, nan=True):
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    return pd.get_dummies(df, columns=cat_cols, dummy_na=nan, dtype=int)
