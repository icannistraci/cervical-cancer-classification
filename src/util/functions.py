import numpy as np
import pandas as pd
from collections import Counter


def check_missing_data(df):

    df.replace('?', np.NaN, inplace=True)
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


def check_equal_value(df, percentage):
    same_val = []
    for col in df.columns:
        df2 = df.groupby(col)['Id'].nunique().to_frame()
        for i, row in df2.iterrows():
            if row['Id'] / 1460 >= percentage:
                same_val.append(col)

    return list(dict.fromkeys(same_val))


def check_high_correlated_features(df, percentage):
    rows, cols = df.corr().shape
    flds = list(df.corr().columns)
    corr = df.corr().corr().values
    for i in range(cols):
        for j in range(i + 1, cols):
            if corr[i, j] > percentage:
                print(flds[i], flds[j], corr[i, j])
