import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler


def preprocess_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a pandas dataframe and makes some transformations.
    Lowers the columns, removes OHE of the target variable, encodes "outside_global_index"
    variable, drops "typeofsteel_a400", removes row 391(outlier).

    :param dataframe: the initial dataframe
    :return: the processed dataframe
    """
    if not isinstance(dataframe, pd.DataFrame):
        print('The dataframe must be a pandas dataframe')

    dataframe.columns = map(lambda x: x.lower(), dataframe.columns)

    target_cols: list = ['pastry', 'z_scratch', 'k_scatch', 'stains', 'dirtiness', 'bumps', 'other_faults']
    enc_dict: dict = {'pastry': 0,
                'z_scratch': 1,
                'k_scatch': 2,
                'stains': 3,
                'dirtiness': 4,
                'bumps': 5,
                'other_faults': 6}
    dataframe['anomaly'] = pd.from_dummies(dataframe[target_cols]).replace(enc_dict)
    dataframe['outside_global_index'].replace({
        0: 0,
        0.5: 1,
        1: 2
    }, inplace = True)
    dataframe.drop(target_cols, axis=1, inplace=True)
    dataframe.drop('typeofsteel_a400', axis=1, inplace=True)

    return dataframe


def scale_df(dataframe: pd.DataFrame, bin_cols: List[str]) -> pd.DataFrame:
    """
    Normalize the dataframe using StandardScaler.

    :param dataframe: the dataframe to normalize
    :param bin_cols: the list of the categorical columns in the dataframe
    :return: the normalized dataframe
    """
    df_non_bin = dataframe.drop(bin_cols, axis=1)  # drop discrete columns

    sc = StandardScaler()
    non_bin_norm = sc.fit_transform(df_non_bin)
    df_non_bin_norm = pd.DataFrame(non_bin_norm, columns=df_non_bin.columns)
    dataframe = pd.concat([df_non_bin_norm, dataframe[bin_cols]], axis=1)
    return dataframe


def rounded_mean(array):
    mean = np.mean(array)
    return round(mean, 2)