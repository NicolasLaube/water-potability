import pandas as pd
import numpy as np
from configparser import ConfigParser
from sklearn.model_selection import train_test_split
from src.outliers_ import del_all_outliers, impute_all_outliers, del_all_outliers

config = ConfigParser()
config.read("config.ini")
DATA_FOLDER = config["DATA"]["DATA_FOLDER"]
TRAIN_TEST_SEED = int(config["SEEDS"]["TEST_SEED"])
TEST_SIZE = float(config["DATA"]["TEST_SIZE"])


def treat_outliers(df,columns,target,label):
    for c in columns:
        q1 = np.quantile(df.loc[df[target] == label,c],0.05)
        q3 = np.quantile(df.loc[df[target] == label,c],0.95)
        df.loc[df[target] == label,c] = df.loc[df[target] == label,c].apply(lambda x: q1 if x<q1 else x)
        df.loc[df[target] == label,c] = df.loc[df[target] == label,c].apply(lambda x: q3 if x>q3 else x)

def treat_del(df,columns,target,label):
    for c in columns:
        quantile_1 = np.quantile(df.loc[df[target] == label, c], 0.25)
        quantile_3 = np.quantile(df.loc[df[target] == label, c], 0.75)
        df = df.drop(df[(df[target] == label) & ((df[c] < quantile_1) | (df[c] > quantile_3))].index, axis=0)

# treat_del(df, df.columns, "Potability", 1)

class Loader:
    def __init__(self, shuffle: bool = True) -> None:
        df = pd.read_csv(DATA_FOLDER) # import the dataset
        df.sample(frac=1).reset_index(drop=shuffle) # shuffle entire DataFrame and reset index
        df_train, df_test = train_test_split(df, test_size=0.2) # split in train and test dataframe
        self.train = df_train
        self.test = df_test

    def process_train_outliers(self, method="impute"):
        if method == "impute":
            self.train = impute_all_outliers(self.train)
        else:
            self.train = del_all_outliers(self.train)

    def process_test_outliers(self, method="impute"):
        if method == "impute":
            self.test = impute_all_outliers(self.test)
        else:
            self.test = del_all_outliers(self.test)
        

    def get_train(self, keep_nan: bool = True):
        if keep_nan:
            X_train, y_train = self.train.loc[:, ~self.train.columns.isin(["Potability"])], self.train.Potability
        else:
            df_without_nan = self.train.dropna()
            X_train, y_train = df_without_nan.loc[:, ~df_without_nan.columns.isin(["Potability"])], df_without_nan.Potability
        return X_train, y_train

    def get_test(self, keep_nan: bool = True):
        if keep_nan:
            X_test, y_test = self.test.loc[:, ~self.test.columns.isin(["Potability"])], self.test.Potability
        else:
            df_without_nan = self.test.dropna()
            X_test, y_test = df_without_nan.loc[:, ~df_without_nan.columns.isin(["Potability"])], df_without_nan.Potability
        return X_test, y_test
