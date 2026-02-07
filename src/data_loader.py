import pandas as pd
from src.features.bureau import get_bureau_features


def merge_data(data_path=''):
    app_train = pd.read_csv(f'{data_path}/application_train.csv')
    bureau_feats = get_bureau_features(data_path)
    
    df = app_train.merge(bureau_feats, on='SK_ID_CURR', how='left')
    
    print(f"df shape: {df.shape}")
    return df