# import packages
import pandas as pd
import imblearn.over_sampling
from sklearn.preprocessing import StandardScaler
from ctgan import CTGAN
#from table_evaluator import TableEvaluator
import warnings

pd.options.display.max_columns = None
warnings.filterwarnings('ignore')


#### FUNCTIONS
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a pandas dataframe and makes some transformations.
    Lowers the columns, removes OHE of the target variable, encodes "outside_global_index"
    variable, drops "typeofsteel_a400", removes row 391(outlier).

    :param df: the initial dataframe
    :return: the processed dataframe
    """
    if not isinstance(df, pd.DataFrame):
        print('The dataframe must be a pandas dataframe')

    df.columns = map(str.lower, df.columns)  # lower columns

    target_cols: list = ['pastry', 'z_scratch', 'k_scatch', 'stains', 'dirtiness', 'bumps', 'other_faults']
    enc_dict: dict = {'pastry': 0,
                'z_scratch': 1,
                'k_scatch': 2,
                'stains': 3,
                'dirtiness': 4,
                'bumps': 5,
                'other_faults': 6}
    df['anomaly'] = pd.from_dummies(df[target_cols]).replace(enc_dict)
    df['outside_global_index'].replace({
        0: 0,
        0.5: 1,
        1: 2
    }, inplace = True)
    df.drop(target_cols, axis=1, inplace=True)
    df.drop('typeofsteel_a400', axis=1, inplace=True)

    return df


def smote_oversampling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Over-samples the dataframe class target until each sample count is = to the sample
    count of the fourth numerous target variable.


    :param df: the dataframe to over-sample
    :return: the oversampled dataframe
    """
    df.dropna(inplace=True)
    df_no256 = df[(df['anomaly'] != 2) & (df['anomaly'] != 5) & (df['anomaly'] != 6)]

    X = df_no256.drop('anomaly', axis=1)
    y = df_no256['anomaly']
    sm = imblearn.over_sampling.SMOTENC(categorical_features = [11, 19])
    X_res, y_res = sm.fit_resample(X, y)
    df_no256_over = pd.merge(pd.DataFrame(X_res), pd.DataFrame(y_res), right_index=True, left_index=True)

    df_smote = pd.concat([df_no256_over, df[(df['anomaly'] == 2) | (df['anomaly'] == 5) | (df['anomaly'] == 6)]],
                         axis=0)
    df_smote.reset_index(drop=True, inplace=True)
    return df_smote


def sample_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts the number of each "anomaly" in the dataframe.


    :param df: the dataframe with the "anomaly" column
    :return: the dataframe with the "anomaly" column
    """
    anomaly_count = pd.DataFrame(df['anomaly'].value_counts()).sort_index()
    return anomaly_count


def ctgan_oversampling(df: pd.DataFrame, discrete_cols: list, n_samples: int = 0) -> pd.DataFrame:
    """
    Over-samples the dataframe class target until each samples count is = to the sample count
    of the biggest target variable plus the n_samples integer.


    :param df: dataframe to over-sample
    :param discrete_cols: list of the discrete column names in df
    :param n_samples: number to samples to over-sample
    :return:
    """
    anomaly_count = sample_count(df)
    for idx in anomaly_count.index:
        print(f"Processing anomaly category: {idx}")

        sub_df = df[df['anomaly'] == idx]  # df for single anomaly

        if sub_df.empty:
            print(f"Skipping idx {idx}: No samples found.")
            continue

        num_samples = int(
            anomaly_count.max() - anomaly_count.loc[idx] + n_samples)  # check how many samples need to be generated

        if num_samples > 0:
            ctgan = CTGAN(batch_size=100)
            ctgan.fit(sub_df, discrete_columns = discrete_cols)  # fit ctgan

            synthetic_data = ctgan.sample(num_samples)  # generate synthetic data

            # uncomment the following lines to evaluate the synthetic data for sample n. 6
            '''
            if idx == 6:
                table_evaluator = TableEvaluator(sub_df, synthetic_data[:673], cat_cols=discrete_cols)
                table_evaluator.visual_evaluation()
                '''
            df = pd.concat([df, synthetic_data], axis=0)  # merge real and synthetic data
        else:
            print(f"Skipping idx {idx}: No synthetic data needed.")

    df.reset_index(drop=True, inplace=True)
    return df


def scale_df(df: pd.DataFrame, bin_cols: list) -> pd.DataFrame:
    """
    Normalize the dataframe using StandardScaler.

    :param df: the dataframe to normalize
    :param bin_cols: the list of the categorical columns in the dataframe
    :return: the normalized dataframe
    """
    df_non_bin = df.drop(bin_cols, axis=1)  # drop discrete columns

    sc = StandardScaler()
    non_bin_norm = sc.fit_transform(df_non_bin)
    df_non_bin_norm = pd.DataFrame(non_bin_norm, columns=df_non_bin.columns)
    df = pd.concat([df_non_bin_norm, df[bin_cols]], axis=1)
    return df


### END FUNCTIONS

if __name__ == '__main__':
    # Import the data
    df = pd.read_csv('csv/faults.csv')

    # Make data more usable
    df = preprocess_df(df)
    df.drop(391, inplace=True)

    # Over-sample with SMOTE until all class have at least the same number of samples as the class with the fourth biggest n. of samples
    df_smote = smote_oversampling(df)
    print('Anomaly count after oversampling with SMOTE:\n', sample_count(df_smote), '\n-----------------------------')

    # Over-sample with CTGAN, bringing all the classes to 1500 samples
    df = ctgan_oversampling(df_smote, ['typeofsteel_a300', 'outside_global_index', 'anomaly'], 327)
    print('Anomaly count after oversampling with CTGAN:\n', sample_count(df), '\n-----------------------------')

    # Normalize old and new synthetic data
    df_norm = scale_df(df, ['typeofsteel_a300', 'outside_global_index', 'anomaly']).sample(frac=1)
    df_norm.reset_index(drop=True, inplace=True)

    # Save the balanced dataset
    df_norm.to_csv('csv/balanced_normalized_steel_plates.csv')
else:
    pass
