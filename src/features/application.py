def application_feature_engineering(df):

    df = df.copy()
    
    df['INCOME_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / (df['AMT_CREDIT'] + 1)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] / 12 + 1)
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365
    df['IS_EMPLOYED'] = (df['DAYS_EMPLOYED'] != 365243).astype(int)
    
    return df
    