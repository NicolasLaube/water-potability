import numpy as np
import pandas as pd
    
def impute_outlier(df, column):
    quantile_1 = np.quantile(df.loc[df["Potability"] == 1, column], 0.05)
    quantile_3 = np.quantile(df.loc[df["Potability"] == 1, column], 0.95)
    df.loc[df["Potability"] == 1, column] = df.loc[df["Potability"] == 1, column].apply(lambda x: quantile_1 if x < quantile_1 else x)
    df.loc[df["Potability"] == 1, column] = df.loc[df["Potability"] == 1, column].apply(lambda x: quantile_3 if x > quantile_3 else x)
    return df


def impute_all_outliers(df: pd.DataFrame):
    columns = df.columns
    for column in columns:
        quantile_1 = np.quantile(df.loc[df["Potability"] == 1, column], 0.05)
        quantile_3 = np.quantile(df.loc[df["Potability"] == 1, column], 0.95)
        df.loc[df["Potability"] == 1, column] = df.loc[df["Potability"] == 1, column].apply(lambda x: quantile_1 if x < quantile_1 else x)
        df.loc[df["Potability"] == 1, column] = df.loc[df["Potability"] == 1, column].apply(lambda x: quantile_3 if x > quantile_3 else x)
    return df

def del_outlier(df, column):
    quantile_1 = np.quantile(df.loc[df["Potability"] == 1, column], 0.05)
    quantile_3 = np.quantile(df.loc[df["Potability"] == 1, column], 0.95)
    df = df.drop(df[(df["Potability"] == 1) & ((df[column] < quantile_1) | (df[column] > quantile_3))].index, axis=0)

def del_all_outliers(df: pd.DataFrame):
    for column in df.columns:
        del_outlier(df, column)
    return df