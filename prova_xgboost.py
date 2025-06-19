import time
import pandas as pd
import numpy as np
import warnings
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from utils.general import (
    rounded_mean,
    metrics_of_prediction,
    plot_cm,
    scale_df,
    preprocess_df,
)

warnings.filterwarnings("ignore")


df = pd.read_csv("csv/faults.csv")
df.head()

df.drop(
    ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"],
    axis=1,
).describe().T
df.columns = map(str.lower, df.columns)
# list of the anomalies features present in the dataset
target_cols = [
    "pastry",
    "z_scratch",
    "k_scatch",
    "stains",
    "dirtiness",
    "bumps",
    "other_faults",
]

# list of the features of the steel plates
feat_col = [x for x in df.columns if x not in target_cols]

# mapper for the encoding
enc_dict = {
    "pastry": 0,
    "z_scratch": 1,
    "k_scatch": 2,
    "stains": 3,
    "dirtiness": 4,
    "bumps": 5,
    "other_faults": 6,
}

df["anomaly"] = pd.from_dummies(df[target_cols]).replace(enc_dict)
df.drop(target_cols, axis=1, inplace=True)
# remove troublesome outlier
df.drop(391, inplace=True)

feat_col.remove("typeofsteel_a400")
corr = df[feat_col].corr().abs()

threshold = 0.9

mask = np.triu(np.ones(corr.shape), k=1)

# Find feature pairs with high correlation
high_corr_pairs = [
    (corr.index[i], corr.columns[j], corr.iloc[i, j])
    for i in range(corr.shape[0])
    for j in range(corr.shape[1])
    if mask[i, j] and corr.iloc[i, j] > threshold
]

high_corr_pairs = pd.DataFrame(high_corr_pairs).T
high_corr_pairs

df_norm = pd.read_csv(
    "csv/smotenc_ctgan_normalized_steel_plates.csv", index_col=0, header=0
).sample(frac=1)
df_norm.reset_index(drop=True, inplace=True)

df_norm["aggregate1"] = df_norm[["y_minimum", "y_maximum"]].mean(axis=1)

df_norm["aggregate2"] = df_norm[["x_minimum", "x_maximum"]].mean(axis=1)

df_norm["aggregate4"] = df_norm[["x_perimeter", "pixels_areas"]].mean(axis=1)

df_norm["aggregate5"] = df_norm[["sum_of_luminosity", "aggregate4"]].mean(axis=1)
df_norm.drop([], axis=1, inplace=True)

df_norm["aggregate3"] = df_norm[["aggregate4", "aggregate5"]].mean(axis=1)
df_norm.drop(
    [
        "sum_of_luminosity",
        "x_perimeter",
        "y_minimum",
        "y_maximum",
        "x_minimum",
        "x_maximum",
        "pixels_areas",
        "aggregate4",
        "aggregate5",
    ],
    axis=1,
    inplace=True,
)

df_norm.dropna(inplace=True)

anomaly_col = df_norm.pop("anomaly")
df_norm["anomaly"] = anomaly_col

X = df_norm.drop("anomaly", axis=1)
y = df_norm["anomaly"]

metrics_df = pd.DataFrame([], columns=["Accuracy", "Precision", "Recall", "F1 Score"])
time_df = pd.DataFrame([], columns=["Avg. Time"])

# Define the K-fold Cross Validator
num_splits = 5
k_fold = KFold(n_splits=num_splits, shuffle=True, random_state=2)

acc_per_fold, precision_per_fold, recall_per_fold, f1_per_fold, time_per_fold = (
    [],
    [],
    [],
    [],
    [],
)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in k_fold.split(X, y):
    y_t = np.array(y.iloc[test])
    start_time = time.time()

    gnb = XGBClassifier()

    # Generate a print
    print("------------------------------------------------------------------------")
    print(f"Training for fold {fold_no} ...")
    gnb.fit(X.iloc[train], y.iloc[train])
    y_pred = gnb.predict(X.iloc[test])

    end_time = time.time()
    duration = end_time - start_time
    time_per_fold.append(duration)

    accuracy, precision, recall, f1 = metrics_of_prediction(y_t, y_pred, fold_no)

    acc_per_fold.append(accuracy)
    precision_per_fold.append(precision)
    recall_per_fold.append(recall)
    f1_per_fold.append(f1)

    if fold_no == num_splits:
        print("--------------------------------------------------------\n")
        print(f"Confusion matrix and classification report for fold {fold_no} ...")
        plot_cm(y_t, y_pred)

        metrics_df.loc["Gnb"] = [
            f"{rounded_mean(acc_per_fold)}%",
            f"{rounded_mean(precision_per_fold)}%",
            f"{rounded_mean(recall_per_fold)}%",
            f"{rounded_mean(f1_per_fold)}%",
        ]

    fold_no += 1

average_time_per_fold = sum(time_per_fold) / len(time_per_fold)
time_df.loc["Gnb"] = round(average_time_per_fold, 2)
print(f"Average time per fold: {average_time_per_fold:.2f} seconds")

print(metrics_df)
print(time_df.T)
