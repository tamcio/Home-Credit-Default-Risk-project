import pandas as pd
import numpy as np
from src.utils import one_hot_encoder


def get_bureau_features(data_path=''):
    bureau = pd.read_csv(f'{data_path}/bureau.csv')
    bb = pd.read_csv(f'{data_path}/bureau_balance.csv')
    
    valid_ids = bureau['SK_ID_BUREAU'].unique()
    bb = bb[bb['SK_ID_BUREAU'].isin(valid_ids)]
    
    bb_agg = _aggregate_bb(bb)
    
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    
    bureau = _bureau_feature_engineering(bureau)
    
    bureau = one_hot_encoder(bureau)
    
    result = _aggregate_to_client(bureau)
    
    return result


def _aggregate_bb(bb):
    bb_dummies = pd.get_dummies(bb, columns=['STATUS'], prefix='STATUS', dtype=int)
    
    agg_rules = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_dummies.columns:
        if 'STATUS_' in col:
            agg_rules[col] = ['mean', 'sum']
            
    bb_agg = bb_dummies.groupby('SK_ID_BUREAU').agg(agg_rules)
    bb_agg.columns = [f'BB_{c[0]}_{c[1].upper()}' for c in bb_agg.columns]
    bb_agg.reset_index(inplace=True)
    
    return bb_agg


def _bureau_feature_engineering(bureau):
    bureau = bureau.copy()
    
    bureau['DEBT_CREDIT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / (bureau['AMT_CREDIT_SUM'] + 1.0)
    bureau['CREDIT_DURATION'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT']
    
    return bureau


def _aggregate_to_client(bureau):
    num_cols = bureau.select_dtypes(include=[np.number]).columns
    
    agg_rules = {}
    for col in num_cols:
        if col not in ['SK_ID_CURR', 'SK_ID_BUREAU']:
            agg_rules[col] = ['mean', 'max', 'min', 'sum']
            
    result = bureau.groupby('SK_ID_CURR').agg(agg_rules)
    result.columns = [f'BUREAU_{c[0]}_{c[1].upper()}' for c in result.columns]
    result.reset_index(inplace=True)
    
    return result

