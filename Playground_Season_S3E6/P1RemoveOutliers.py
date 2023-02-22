import warnings
warnings.simplefilter('ignore')

import pandas as pd
from imblearn.over_sampling import SMOTE

def split_squareMeters(x):
    if x < 25000:
        return 0
    elif x < 100000:
        return 1
    else:
        return 2

def remove_Outlier(df):
    # Remove outliers of others
    df = df.loc[(df['floors'] <= 1000) &
                (df['cityCode'] <= 100000) &
                (df['made'] <= 3000) &
                (df['basement'] <= 20000) &
                (df['attic'] <= 20000) &
                (df['garage'] <= 1900) &
                (df['garage'] > 90) &
                (df['squareMeters'] < 200000)]
    # Categorize price
    return df

def add_split(df):
    df['slope'] = df['price'] / df['squareMeters']
    mean = df['slope'].mean()
    std = df['slope'].std()
    df['split'] = df['slope'].apply(lambda x: split_squareMeters(x, mean, std))
    df.drop(['slope'], axis=1, inplace=True)
    return df

train = pd.read_csv('./input/PG_S03E06/train.csv')
train = remove_Outlier(train)
train_df = add_split(train)
train_df.to_csv('./input/PG_S03E06/train_df.csv', index=False)