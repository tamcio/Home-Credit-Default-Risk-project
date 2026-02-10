import pandas as pd
import numpy as np
from src.utils import one_hot_encoder


def get_previous_features(data_path=''):
    """Main function to aggregate all historical loan information."""
    # 1. Load Datasets
    prev = pd.read_csv(f'{data_path}/previous_application.csv')
    pay = pd.read_csv(f'{data_path}/installments_payments.csv')
    cc = pd.read_csv(f'{data_path}/credit_card_balance.csv')
    pos = pd.read_csv(f'{data_path}/POS_CASH_balance.csv')

    # 2. Sub-aggregations (Extracting the "Golden Features")
    pay_agg = _aggregate_installments(pay)
    cc_agg = _aggregate_credit_card(cc)
    pos_agg = _aggregate_pos_cash(pos)

    # 3. Process Previous Application Table
    prev = _previous_feature_engineering(prev)
    prev = one_hot_encoder(prev)

    # 4. Merging all together
    # Merging historical behavioral data into previous applications
    prev = prev.merge(pay_agg, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')
    prev = prev.merge(cc_agg, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')
    prev = prev.merge(pos_agg, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')

    # 5. Final Aggregation to Client Level (SK_ID_CURR)
    result = _aggregate_to_client(prev)

    return result


def _aggregate_installments(pay):
    """Features from installments_payments.csv"""
    # Golden Feature: Payment Volatility
    pay['DAYS_DIFF_RAW'] = pay['DAYS_ENTRY_PAYMENT'] - pay['DAYS_INSTALMENT']

    # Standard aggregations + Volatility
    pay_agg = pay.groupby(['SK_ID_CURR', 'SK_ID_PREV']).agg({
        'DAYS_DIFF_RAW': ['std', 'mean', 'max'],
        'AMT_INSTALMENT': ['sum', 'mean'],
        'AMT_PAYMENT': ['sum', 'mean'],
        'DAYS_ENTRY_PAYMENT': ['max']
    })

    # Flatten columns and rename volatility
    pay_agg.columns = [f'PAY_{c[0]}_{c[1].upper()}' for c in pay_agg.columns]
    pay_agg.rename(columns={'PAY_DAYS_DIFF_RAW_STD': 'PAYMENT_VOLATILITY'}, inplace=True)
    pay_agg.reset_index(inplace=True)

    return pay_agg


def _aggregate_credit_card(cc):
    """Features from credit_card_balance.csv"""
    # Golden Feature: CC Danger Zone Frequency
    cc['UTILIZATION'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, 1)
    cc['HIGH_UTIL_FLAG'] = (cc['UTILIZATION'] > 0.9).astype(int)

    cc_agg = cc.groupby(['SK_ID_CURR', 'SK_ID_PREV']).agg({
        'HIGH_UTIL_FLAG': ['mean'],  # This is CC_DANGER_ZONE_FREQ
        'AMT_BALANCE': ['max', 'mean'],
        'AMT_DRAWINGS_ATM_CURRENT': ['sum', 'max'],
        'SK_DPD': ['max']
    })

    cc_agg.columns = [f'CC_{c[0]}_{c[1].upper()}' for c in cc_agg.columns]
    cc_agg.rename(columns={'CC_HIGH_UTIL_FLAG_MEAN': 'CC_DANGER_ZONE_FREQ'}, inplace=True)
    cc_agg.reset_index(inplace=True)

    return cc_agg


def _aggregate_pos_cash(pos):
    """Features from POS_CASH_balance.csv"""
    pos_agg = pos.groupby(['SK_ID_CURR', 'SK_ID_PREV']).agg({
        'SK_DPD': ['max', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['min', 'last']
    })
    pos_agg.columns = [f'POS_{c[0]}_{c[1].upper()}' for c in pos_agg.columns]
    pos_agg.reset_index(inplace=True)

    return pos_agg


def _previous_feature_engineering(prev):
    """Core engineering for previous_application.csv"""
    # Golden Feature: Previous Rejected Count
    # We create a flag to sum it later in the final aggregation
    prev['IS_REJECTED'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)

    # Other useful ratios
    prev['APPLICATION_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['APPLICATION_CREDIT_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT'].replace(0, 1)

    return prev


def _aggregate_to_client(prev):
    """Aggregates all features from SK_ID_PREV level to SK_ID_CURR level."""
    num_cols = prev.select_dtypes(include=[np.number]).columns

    agg_rules = {}
    for col in num_cols:
        if col not in ['SK_ID_CURR', 'SK_ID_PREV']:
            # For our special flags we want the sum/max
            if col in ['IS_REJECTED', 'CC_DANGER_ZONE_FREQ', 'PAYMENT_VOLATILITY']:
                agg_rules[col] = ['sum', 'max', 'mean']
            else:
                agg_rules[col] = ['mean', 'max', 'sum']

    result = prev.groupby('SK_ID_CURR').agg(agg_rules)
    result.columns = [f'PREV_{c[0]}_{c[1].upper()}' for c in result.columns]

    # Rename our key feature for clarity
    if 'PREV_IS_REJECTED_SUM' in result.columns:
        result.rename(columns={'PREV_IS_REJECTED_SUM': 'PREV_REJECTED_COUNT'}, inplace=True)

    result.reset_index(inplace=True)

    return result