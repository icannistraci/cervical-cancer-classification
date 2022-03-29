import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from src import functions as util

warnings.filterwarnings('ignore')
start_time = time.time()

print(f'\n --- START Feature Engineering --- \n')


def preprocess_data():
    df = pd.read_csv('../data/dataset.csv', delimiter=',')

    # check missing values and replace ? with NaN
    missing_data = util.check_missing_data(df)
    print(f'\n Missing data are: \n {missing_data} \n')

    # check values Age and First sexual intercourse
    temp_val = pd.read_csv('../data/dataset.csv', delimiter=',')
    temp_val.replace('?', np.NaN, inplace=True)
    check_validity = temp_val[['Age', 'First sexual intercourse']].fillna(0)
    check_validity['First sexual intercourse'] = pd.to_numeric(check_validity['First sexual intercourse'], errors='coerce')
    check_validity['To check'] = np.where(check_validity['First sexual intercourse'] > check_validity['Age'], 'yes', 'no')

    # check the two wrong instances
    inst_312 = df.loc[312]
    df.loc[312, 'Age'] = 27
    df.loc[312, 'First sexual intercourse'] = 23

    inst_812 = df.loc[812]
    df.loc[812, 'Age'] = 16
    df.loc[812, 'First sexual intercourse'] = 14

    # drop features with 92% of missing values
    df.drop('STDs: Time since last diagnosis', inplace=True, axis=1)
    df.drop('STDs: Time since first diagnosis', inplace=True, axis=1)

    # TODO: decide what to do
    # tresh means at least N non-null to survive
    # 101 instances deleted
    # df.dropna(axis=0, thresh=20, inplace=True)

    # (SIMPLE) IMPUTATION OF MISSING VALUES #
    # Smokes
    # print(df['Smokes'].mode())
    # mode is 0!
    df['Smokes'].fillna('0.0', inplace=True)
    df['Smokes (years)'].fillna('0.0', inplace=True)
    df['Smokes (packs/year)'].fillna('0.0', inplace=True)
    # adjust 2 values (Smokes (years): 1.2669729090000001, Smokes (packs/year): 0.5132021277000001)
    # da un anno e mezzo
    df['Smokes (years)'].replace({'1.266972909': '1.50'}, inplace=True)
    # mezzo pacchetto
    df['Smokes (packs/year)'].replace({'0.5132021277': '0.50'}, inplace=True)

    # Hormonal Contraceptives
    # print(df['Hormonal Contraceptives'].mode())
    # mode is 1!
    df['Hormonal Contraceptives'].fillna('1.0', inplace=True)
    mean_hcy = round((pd.to_numeric(df['Hormonal Contraceptives (years)'], errors='coerce')).mean(), 2)
    df['Hormonal Contraceptives (years)'].fillna(str(mean_hcy), inplace=True)

    # IUD
    # print(df['IUD'].mode())
    # mode is 0!
    df['IUD'].fillna('0.0', inplace=True)
    df['IUD (years)'].fillna('0.0', inplace=True)

    # Num of pregnancies
    # replace with mean (2.0)
    temp = df[['Age', 'Num of pregnancies', 'First sexual intercourse']]
    missing_pregn = temp[(temp['Num of pregnancies'].isnull())]
    mean_nop = round((pd.to_numeric(df['Num of pregnancies'], errors='coerce')).mean(), 0)
    df['Num of pregnancies'].fillna(str(mean_nop), inplace=True)

    # Number of sexual partners
    # replace with mean (3.0)
    temp = df[['Age', 'Number of sexual partners', 'First sexual intercourse']]
    missing_sex = temp[(temp['Number of sexual partners'].isnull())]
    mean_nosp = round((pd.to_numeric(df['Number of sexual partners'], errors='coerce')).mean(), 0)
    df['Number of sexual partners'].fillna(str(mean_nosp), inplace=True)

    # First sexual intercourse
    # replace with mean (17.0)
    temp = df[['Age', 'Number of sexual partners', 'First sexual intercourse']]
    missing_first = temp[(temp['First sexual intercourse'].isnull())]
    mean_fsi = round((pd.to_numeric(df['First sexual intercourse'], errors='coerce')).mean(), 0)
    df['First sexual intercourse'].fillna(str(mean_fsi), inplace=True)

    # STDs - Analysis
    # replace with mode (0)
    stds = ['STDs', 'STDs (number)', 'STDs:condylomatosis', 'STDs:vaginal condylomatosis', 'STDs:cervical condylomatosis',
            'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease',
            'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV']
    df = df.fillna(0)

    for std in stds:
        print(f'{std}, {Counter(df[std])}')

    # convert object features to numeric (int and float)
    # convert correctly categorical features to int
    feat_to_convert_int = ['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes',
                           'Hormonal Contraceptives', 'IUD', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
                           'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
                           'STDs:pelvic inflammatory disease', 'STDs:genital herpes', 'STDs:molluscum contagiosum',
                           'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs:AIDS', 'STDs:cervical condylomatosis']

    # convert correctly categorical features to float
    feat_to_convert_float = ['Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives (years)', 'IUD (years)']

    for feat in feat_to_convert_int:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
        df[feat] = df[feat].astype('int64')

    for feat in feat_to_convert_float:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')

    # check validity of imputation
    impute = df[['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies']]

    impute['Valid age'] = np.where(impute['First sexual intercourse'] > impute['Age'], 'no', 'yes')
    impute['Valid preg'] = np.where(np.logical_and(impute['Num of pregnancies'] >= 0,
                                                   impute['First sexual intercourse'] > 0), 'yes', 'no')
    impute['Valid sex'] = np.where(np.logical_and(impute['Number of sexual partners'] > 0,
                                                  impute['First sexual intercourse'] > 0), 'yes', 'no')

    # empty df
    to_correct = impute.loc[(impute['Valid age'] == 'no') | (impute['Valid preg'] == 'no')
                            | (impute['Valid sex'] == 'no')]

    # Analyze remaining features
    for feature in df:
        print(f'{feature}, {Counter(df[feature])}')
    print()

    # STDs:cervical condylomatosis, Counter({'0.0': 858})
    df.drop('STDs:cervical condylomatosis', inplace=True, axis=1)
    # STDs:AIDS, Counter({'0.0': 858})
    df.drop('STDs:AIDS', inplace=True, axis=1)

    # Hinselmann, Counter({0: 823, 1: 35})
    # Schiller, Counter({0: 784, 1: 74})
    # Citology, Counter({0: 814, 1: 44})
    # Biopsy, Counter({0: 803, 1: 55})

    print(f'\n Df shapes: {df.shape} \n')

    # check the class distribution of the target variables
    colors = ["#fb9d8e", "#c53b50"]

    sns.countplot('Biopsy', data=df, palette=colors)
    plt.show()

    sns.countplot('Schiller', data=df, palette=colors)
    plt.show()

    sns.countplot('Hinselmann', data=df, palette=colors)
    plt.show()

    sns.countplot('Citology', data=df, palette=colors)
    plt.show()

    # correlation matrix
    correlation = df.corr()
    plt.subplots(figsize=(30, 30))
    sns.heatmap(correlation, vmax=1.0, center=0, fmt='.2f', square=True, annot=True, cbar_kws={'shrink': .70})
    plt.show()

    # drop multicorr
    df.drop('Smokes', inplace=True, axis=1)
    df.drop('IUD', inplace=True, axis=1)

    df.drop('STDs:vulvo-perineal condylomatosis', inplace=True, axis=1)
    df.drop('STDs: Number of diagnosis', inplace=True, axis=1)
    # df.drop('STDs', inplace=True, axis=1)

    return df


print(f'\n --- END Feature Engineering in {(time.time() - start_time)} seconds ---')
