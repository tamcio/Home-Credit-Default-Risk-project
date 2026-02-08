import pandas as pd
import numpy as np
from src.features.bureau import get_bureau_features


def merge_data(data_path='', handle_outliers=True):
    app_train = pd.read_csv(f'{data_path}/application_train.csv')
    bureau_feats = get_bureau_features(data_path)
    
    df = app_train.merge(bureau_feats, on='SK_ID_CURR', how='left')
    
    if handle_outliers:
        df = remove_outliers(df)
    
    print(f"df shape: {df.shape}")
    return df


def remove_outliers(df, verbose=True):

    df = df.copy()
    
    error_thresholds = { 
        'CNT_CHILDREN': 15,
        'OWN_CAR_AGE': 70,
        'OBS_30_CNT_SOCIAL_CIRCLE': 40,
        'DEF_30_CNT_SOCIAL_CIRCLE': 30,
        'OBS_60_CNT_SOCIAL_CIRCLE': 40,
        'DEF_60_CNT_SOCIAL_CIRCLE': 20,
        'AMT_REQ_CREDIT_BUREAU_QRT': 15,
        'AMT_REQ_CREDIT_BUREAU_MON': 20,
        'AMT_REQ_CREDIT_BUREAU_YEAR': 22
    }
    
    if verbose:
        print("Handling outliers")

    if 'DAYS_EMPLOYED' in df.columns:
        n = (df['DAYS_EMPLOYED'] == 365243).sum()
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        if verbose and n > 0:
            print(f"DAYS_EMPLOYED: Replaced {n} anomalous values (365243) with nan")
    
    if 'AMT_INCOME_TOTAL' in df.columns:
        cap = df['AMT_INCOME_TOTAL'].quantile(0.9999)
        n = (df['AMT_INCOME_TOTAL'] > cap).sum()
        df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].clip(upper=cap)
        if verbose and n > 0:
            print(f"AMT_INCOME_TOTAL: Capped {n} values to {cap:.0f}")
    
    for col, threshold in error_thresholds.items():
        if col in df.columns:
            mask = df[col] > threshold
            n = mask.sum()
            if n > 0:
                df.loc[mask, col] = np.nan
                if verbose:
                    print(f"{col}: Set {n} outliers (>{threshold}) to nan")

    if verbose:
        print("Outlier handling completed")
    
    return df

