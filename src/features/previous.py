import pandas as pd
import numpy as np
import re
from src.utils import one_hot_encoder


def get_previous_features(data_path=''):
    prev = pd.read_csv(f'{data_path}/previous_application.csv')
    pay = pd.read_csv(f'{data_path}/installments_payments.csv')
    cc = pd.read_csv(f'{data_path}/credit_card_balance.csv')
    pos = pd.read_csv(f'{data_path}/POS_CASH_balance.csv')

    pay_agg = _aggregate_installments(pay)
    cc_agg = _aggregate_credit_card(cc)
    pos_agg = _aggregate_pos_cash(pos)

    prev['APPLICATION_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['APPLICATION_CREDIT_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT'].replace(0, 1)
    prev['REJECTED_FLAG'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)

    prev = one_hot_encoder(prev, nan=True)
    prev = prev.merge(pay_agg, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')
    prev = prev.merge(cc_agg, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')
    prev = prev.merge(pos_agg, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')

    result = _aggregate_to_client(prev)
    return _clean_column_names(result)


def _aggregate_installments(pay):
    pay['DPD'] = (pay['DAYS_ENTRY_PAYMENT'] - pay['DAYS_INSTALMENT']).clip(lower=0)
    pay['AMT_GAP'] = (pay['AMT_INSTALMENT'] - pay['AMT_PAYMENT']).clip(lower=0)

    return pay.groupby(['SK_ID_CURR', 'SK_ID_PREV']).agg({
        'DPD': ['max', 'mean'],
        'AMT_GAP': ['sum'],
        'DAYS_ENTRY_PAYMENT': ['std']
    }).pipe(_flatten_cols, 'PAY')


def _aggregate_credit_card(cc):
    cc['UTILIZATION'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, 1)

    return cc.groupby(['SK_ID_CURR', 'SK_ID_PREV']).agg({
        'UTILIZATION': ['max', 'mean'],
        'SK_DPD': ['max'],
        'AMT_DRAWINGS_ATM_CURRENT': ['sum']
    }).pipe(_flatten_cols, 'CC')


def _aggregate_pos_cash(pos):
    return pos.groupby(['SK_ID_CURR', 'SK_ID_PREV']).agg({
        'SK_DPD': ['max'],
        'CNT_INSTALMENT_FUTURE': ['last']
    }).pipe(_flatten_cols, 'POS')


def _aggregate_to_client(df):
    agg_dict = {}

    for col in [c for c in df.columns if 'NAME_CONTRACT_STATUS' in c or 'REJECTED_FLAG' in c]:
        agg_dict[col] = ['mean', 'sum']

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col not in agg_dict and col not in ['SK_ID_CURR', 'SK_ID_PREV']:
            agg_dict[col] = ['mean', 'max']

    result = df.groupby('SK_ID_CURR').agg(agg_dict)
    result.columns = [f'PREV_{c[0]}_{c[1].upper()}' for c in result.columns]
    return result.reset_index()


def _flatten_cols(df, prefix):
    df.columns = [f'{prefix}_{c[0]}_{c[1].upper()}' for c in df.columns]
    return df


def _clean_column_names(df):
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)).strip('_') for col in df.columns]
    return df