import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, QuantileTransformer, \
    Normalizer


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
                print(f'{flds[i]} and {flds[j]}: {round(corr[i, j],2)}')


def scaling_data_multiple(X_train, X_test):
    scaler_a = StandardScaler()
    scaler_b = MinMaxScaler()
    scaler_c = MinMaxScaler(feature_range=(-1, 1))
    scaler_d = MaxAbsScaler()
    scaler_e = RobustScaler(quantile_range=(25, 75))
    scaler_f = QuantileTransformer(output_distribution='normal')
    scaler_g = QuantileTransformer(output_distribution='uniform')
    normalizer = Normalizer()

    # NB standard scaling has no good performance on data with outliers
    Scaled_Data = [
        ('data after standard scaling',
         scaler_a.fit_transform(X_train), scaler_a.transform(X_test)),
        # The outliers have an influence when computing the empirical mean and standard deviation which
        # shrink the range of the feature values. Because the outliers on each feature have different magnitudes,
        # the spread of the transformed data on each feature is very different.
        ('data after [0,1] min-max scaling',
         scaler_b.fit_transform(X_train), scaler_b.transform(X_test)),
        # As StandardScaler, MinMaxScaler is very sensitive to the presence of outliers.
        ('data after [-1,+1] min-max scaling',
         scaler_c.fit_transform(X_train), scaler_c.transform(X_test)),
        # As StandardScaler, MinMaxScaler is very sensitive to the presence of outliers.
        ('data after max-abs scaling',
         scaler_d.fit_transform(X_train), scaler_d.transform(X_test)),
        # This estimator scales and translates each feature individually such that the maximal absolute
        # value of each feature in the training set will be 1.0. It does not shift/center the data,
        # and thus does not destroy any sparsity.
        # On positive only data, this scaler behaves similarly to MinMaxScaler and therefore also suffers
        # from the presence of large outliers.
        ('data after robust scaling',
         scaler_e.fit_transform(X_train), scaler_e.transform(X_test)),
        #       This Scaler removes the median and scales the data according to the quantile range
        #       Centering and scaling happen independently on each feature by computing the relevant statistics
        #       on the samples in the training set. Median and interquartile range are then stored to be used
        #       on later data using the transform method. Standardization of a dataset is a common requirement for
        #       many machine learning estimators. This is done by removing the mean and scaling to unit variance.
        #       However, outliers can often influence the sample mean / variance in a negative way.
        #       In such cases, the median and the interquartile range often give better results.
        ('data after quantile transformation (gaussian pdf)',
         scaler_f.fit_transform(X_train), scaler_f.transform(X_test)),
        ('data after quantile transformation (uniform pdf)',
         scaler_g.fit_transform(X_train), scaler_f.transform(X_test)),
        ('data after sample-wise L2 normalizing',
         normalizer.fit_transform(X_train), normalizer.transform(X_test))
    ]

    return Scaled_Data

