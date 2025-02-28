import numpy as np
import pandas as pd
from utils.general import preprocess_df, scale_df
from scipy.stats import wasserstein_distance



def load_synthetic_dfs(file_paths: list, method_names: list) -> dict:
    method_dic = {}
    for idx, file_path in enumerate(file_paths):
        method_dic[method_names[idx]] = pd.read_csv(file_path, header=0, index_col=0)
    return method_dic


def load_real_df(file_path: str) -> pd.DataFrame:
    real_df = pd.read_csv(file_path)
    real_df = preprocess_df(real_df)
    real_df.drop(391, inplace=True)
    real_df = scale_df(real_df, ['typeofsteel_a300', 'outside_global_index', 'anomaly'])
    real_df.dropna(inplace=True)

    return real_df


def compute_wasserstein(real_data: pd.DataFrame, synthetic: pd.DataFrame, target_col: str):
    if real_data.shape[1] != synthetic.shape[1]:
        return print(f'Column length mismatch: real data shape {real_data.shape[1]}, synthetic data shape {synthetic.shape[1]}')

    temp_real_data = real_data.drop(target_col, axis=1)
    temp_synthetic = synthetic.drop(target_col, axis=1)

    distances = [wasserstein_distance(temp_real_data[i], temp_synthetic[i]) for i in temp_real_data.columns]

    return np.mean(distances)


def best_synthetic(real_data: pd.DataFrame, synthetic_dic: dict, target_col: str):
    methods_distances = {}
    for idx, synthetic in enumerate(synthetic_dic.values()):
        method = list(synthetic_dic.keys())[idx]
        print(f'Processing {method}')

        distance = compute_wasserstein(real_data, synthetic, target_col)
        if not isinstance(distance, float):
            break

        methods_distances[method] = distance
        print(f'Wasserstein distance for {method} is: {distance}\n----------------------')

    best_distance = np.min(list(methods_distances.values()))
    best_method = [key for key, value in methods_distances.items() if value == best_distance]
    return str(best_method[0])

def evaluate_synthetic_data(file_paths: list, method_names: list, real_file_path: str):
    real_data = load_real_df(real_file_path)
    synthetic_dict = load_synthetic_dfs(file_paths, method_names)
    target_col = real_data.columns[-1]
    best_synthetic(real_data, synthetic_dict, target_col)



if __name__ == '__main__':
    paths = [
        '../csv/smotenc_normalized_steel_plates.csv',
        '../csv/ctgan_normalized_steel_plates.csv',
        '../csv/smotenc_ctgan_normalized_steel_plates.csv'
    ]

    method_names_list = ['SMOTENC', 'CTGAN', 'SMOTENC-CTGAN']

    evaluate_synthetic_data(paths, method_names_list, '../csv/faults.csv')