import os
import pandas as pd
import imblearn.over_sampling
from ctgan import CTGAN
from table_evaluator import TableEvaluator
from utils.general import preprocess_df, scale_df
import warnings

pd.options.display.max_columns = None
warnings.filterwarnings('ignore')


#### FUNCTIONS
def smote_oversampling(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Over-samples the dataframe class target until each sample count is = to the sample
    count of the fourth numerous target variable.


    :param dataframe: the dataframe to over-sample
    :return: the oversampled dataframe
    """
    dataframe.dropna(inplace=True)
    df_no256 = dataframe[(dataframe['anomaly'] != 2) & (dataframe['anomaly'] != 5) & (dataframe['anomaly'] != 6)]

    X = df_no256.drop('anomaly', axis=1)
    y = df_no256['anomaly']
    sm = imblearn.over_sampling.SMOTENC(categorical_features = [11, 19])
    X_res, y_res = sm.fit_resample(X, y)
    df_no256_over = pd.merge(pd.DataFrame(X_res), pd.DataFrame(y_res), right_index=True, left_index=True)

    df_sm = pd.concat([df_no256_over, dataframe[(dataframe['anomaly'] == 2) | (dataframe['anomaly'] == 5) | (dataframe['anomaly'] == 6)]],
                      axis=0)
    df_sm.reset_index(drop=True, inplace=True)
    return df_sm


def sample_count(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Counts the number of each "anomaly" in the dataframe.


    :param dataframe: the dataframe with the "anomaly" column
    :return: the dataframe with the "anomaly" column
    """
    anomaly_count = pd.DataFrame(dataframe['anomaly'].value_counts()).sort_index()
    return anomaly_count


def ctgan_oversampling(dataframe: pd.DataFrame, discrete_cols: list, n_samples: int = 0, evaluate_data = False) -> pd.DataFrame:
    """
    Over-samples the dataframe class target until each samples count is = to the sample count
    of the biggest target variable plus the n_samples integer.


    :param dataframe: dataframe to over-sample
    :param discrete_cols: list of the discrete column names in df
    :param n_samples: number of samples to over-sample
    :param evaluate_data: whether to evaluate the over-sampling or not through visual evaluation
    :return:
    """
    anomaly_count = sample_count(dataframe)
    for idx in anomaly_count.index:
        print(f"Processing anomaly category: {idx}")

        sub_df = dataframe[dataframe['anomaly'] == idx]  # df for single anomaly

        if sub_df.empty:
            print(f"Skipping idx {idx}: No samples found.")
            continue

        num_samples = int(
            anomaly_count.max() - anomaly_count.loc[idx] + n_samples)  # check how many samples need to be generated

        if num_samples > 0:
            ctgan = CTGAN(batch_size=100)
            ctgan.fit(sub_df, discrete_columns=discrete_cols, epochs=1000)  # fit ctgan

            synthetic_data = ctgan.sample(num_samples).sample(frac=1)  # generate synthetic data

            if evaluate_data:
                target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
                folder_path = f'Plots/{idx}_{target_cols[idx]}'

                table_evaluator = TableEvaluator(sub_df, synthetic_data[:673], cat_cols=discrete_cols)
                table_evaluator.visual_evaluation(save_dir=folder_path)

                for filename in os.listdir(folder_path):
                    name, ext = os.path.splitext(filename)

                    if 'smotenc' in name or 'ctgan' in name:
                        continue

                    new_name = f"{name}_ctgan_ctgan{ext}"
                    old_file = os.path.join(folder_path, filename)
                    new_file = os.path.join(folder_path, new_name)
                    os.rename(old_file, new_file)

            dataframe = pd.concat([dataframe, synthetic_data], axis=0)  # merge real and synthetic data
        else:
            print(f"Skipping idx {idx}: No synthetic data needed.")

    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


### END FUNCTIONS

if __name__ == '__main__':
    # Import the data
    df = pd.read_csv('../csv/faults.csv')

    # Make data more usable
    df = preprocess_df(df)
    df.drop(391, inplace=True)

    # Over-sample with SMOTE until all class have at least the same number of samples as the class with the fourth biggest n. of samples
    #df_smote = smote_oversampling(df)
    #print('Anomaly count after oversampling with SMOTE:\n', sample_count(df_smote), '\n-----------------------------')

    # Over-sample with CTGAN, bringing all the classes to 1500 samples
    df = ctgan_oversampling(df, ['typeofsteel_a300', 'outside_global_index', 'anomaly'], 0, evaluate_data=True)
    print('Anomaly count after oversampling with CTGAN:\n', sample_count(df), '\n-----------------------------')

    # Normalize old and new synthetic data
    df_norm = scale_df(df, ['typeofsteel_a300', 'outside_global_index', 'anomaly']).sample(frac=1)

    # Save the balanced dataset
    df.to_csv('csv/smotenc_ctgan_normalized_steel_plates.csv')
else:
    pass
