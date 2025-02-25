import os
import pandas as pd
from imblearn.over_sampling import SMOTENC
from typing import List
from table_evaluator import TableEvaluator


from fegatini import preprocess_df, scale_df


def smotenc_oversampling(dataframe: pd.DataFrame, categorical_features: List[str], target_col: str) -> pd.DataFrame:
    """
    Over-samples the dataframe using SMOTENC.


    :param dataframe: the dataframe to over-sample
    :param categorical_features: list of categorical features in the dataframe
    :param target_col: target column name that needs class balancing
    :return: the dataframe with over-sampled columns
    """
    X = dataframe.drop(target_col, axis=1)
    y = dataframe[target_col]

    sm = SMOTENC(categorical_features = categorical_features)
    X_res, y_res = sm.fit_resample(X, y)

    df_sm = pd.merge(pd.DataFrame(X_res), pd.DataFrame(y_res), right_index=True, left_index=True)

    return df_sm


def evaluate_data(old_df: pd.DataFrame, new_df: pd.DataFrame, target_col: str, discrete_cols: List[str]) -> None:
    """
    Evaluate synthetic data created with SMOTENC. Creates visual evaluation of synthetic data and saves the images in a "Plots" directory.


    :param old_df: the dataframe before class balancing
    :param new_df: the dataframe after class balancing
    :param target_col: target column name that was over-sampled
    :param discrete_cols: list of discrete feature names of the dataframe
    :return: None
    """
    target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    len_old = len(old_df)
    synthetic_df = new_df[len_old:]

    for target in old_df[target_col].unique():
        if target not in synthetic_df[target_col].unique():
            print(f'Target {target} not in synthetic data')
            continue

        sub_df = old_df[old_df['anomaly'] == target]

        folder_path = f'Plots/{target}_{target_cols[target]}'

        table_evaluator = TableEvaluator(sub_df, synthetic_df[synthetic_df['anomaly'] == target], cat_cols=discrete_cols)
        table_evaluator.visual_evaluation(save_dir=folder_path)

        for filename in os.listdir(folder_path):
            name, ext = os.path.splitext(filename)

            if 'ctgan' in name or 'smotenc' in name:
                continue

            new_name = f"{name}_smotenc{ext}"
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_name)
            os.rename(old_file, new_file)


if __name__ == '__main__':
    # Import the data
    df = pd.read_csv('csv/faults.csv')

    # Make data more usable
    df = preprocess_df(df)
    df.drop(391, inplace=True)

    # Oversample using SMOTENC
    df_sm = smotenc_oversampling(df, ['outside_global_index', 'typeofsteel_a300'], 'anomaly')

    evaluate_data(df, df_sm, 'anomaly', discrete_cols=['outside_global_index', 'typeofsteel_a300', 'anomaly'])

    # Normalize data
    df_norm = scale_df(df_sm, ['typeofsteel_a300', 'outside_global_index', 'anomaly']).sample(frac=1)
    df_norm.reset_index(drop=True, inplace=True)

    anomaly_count_res = pd.DataFrame(df_norm['anomaly'].value_counts()).sort_index()
    anomaly_count_res.index = ['pastry', 'z_scratch', 'k_scatch', 'stains', 'dirtiness', 'bumps', 'other_faults']
    print(anomaly_count_res)