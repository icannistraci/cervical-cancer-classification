import time
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
from collections import Counter
import matplotlib.pyplot as plt
from src.util import functions as util


# def preprocess_data():
start_time = time.time()
print(f'\n --- START Feature Engineering --- \n')

df = pd.read_csv('../../Data/dataset.csv', delimiter=',')

# check missing values
missing_data = util.check_missing_data(df)
# print(f'Missing data are: \n {missing_data} \n')

# nullity correlation:  how strongly the presence or absence of one variable affects the presence of another
# msno.heatmap(df)
# plt.show()

# dendogram: allows you to more fully correlate variable completion
# msno.dendrogram(df)
# plt.show()

# drop features with 92% of missing values
df.drop('STDs: Time since last diagnosis', inplace=True, axis=1)
df.drop('STDs: Time since first diagnosis', inplace=True, axis=1)

# drop other 3 target features
# df.drop('Hinselmann', inplace=True, axis=1)
# df.drop('Schiller', inplace=True, axis=1)
# df.drop('Citology', inplace=True, axis=1)

# tresh means at least N non-null to survive
# 105 instances deleted
df.dropna(axis=0, thresh=25, inplace=True)

print(f'Df shapes: {df.shape} \n')

# Analyze remaining features
for feature in df:
    print(f'{feature}, {Counter(df[feature])}')
print()

# STDs:cervical condylomatosis, Counter({'0.0': 753})
df.drop('STDs:cervical condylomatosis', inplace=True, axis=1)
# STDs:AIDS, Counter({'0.0': 753})
df.drop('STDs:AIDS', inplace=True, axis=1)
# Biopsy, Counter({0: 700, 1: 53})

# convert correctly categortical features to int
feat_to_convert_int = ['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes',
                       'Hormonal Contraceptives', 'IUD', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
                       'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
                       'STDs:pelvic inflammatory disease', 'STDs:genital herpes', 'STDs:molluscum contagiosum',
                       'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV']

# convert correctly categortical features to float
feat_to_convert_float = ['Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives (years)', 'IUD (years)']

for feat in feat_to_convert_int:
    df[feat] = df[feat].astype(str).astype(float)
    df[feat] = df[feat].fillna(0.0).astype(int)
    df[feat] = df[feat].apply(np.int64)

for feat in feat_to_convert_float:
    df[feat] = df[feat].astype(str).astype(float)

# estimate skewness and kurtosis
# print(f'Skewness is \b {df.skew()}')
# sns.distplot(df.skew(), color='blue', axlabel='Skewness')
# plt.show()

# print(f'Kurtosis is \n {df.kurtosis()}')
sns.distplot(df.kurtosis(), color='blue', axlabel='Kurtosis')
plt.show()

# correlation matrix
correlations = df.corr()
fig, ax = plt.subplots(figsize=(30, 30))
sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', square=True,
            linewidths=.5, annot=True, cbar_kws={"shrink": .70})
# plt.show()

# add id and rearrange columns
df['Id'] = df.index
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

print(f'\n --- END Feature Engineering in {(time.time() - start_time)} seconds ---')
