import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from rtdl_revisiting_models import FTTransformer


def create_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return device

def import_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col=0, header=0)

    return df

def create_tensors(df: pd.DataFrame, target: str, cat_features, device: torch.device):
    """
    Extracts arrays from the input dataframe and converts them into tensors to train a transformer.
    """
    col_to_drop = cat_features + [target]
    X_num = df.drop(col_to_drop, axis=1)
    X_cat = df[cat_features].astype(np.int64)
    y = df[target].astype(np.int64)

    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_num, X_cat, y, test_size=0.2, stratify=y, random_state=2
    )

    X_num_train = X_num_train.to_numpy(dtype=np.float32)
    X_num_test = X_num_test.to_numpy(dtype=np.float32)
    X_cat_train = X_cat_train.to_numpy(dtype=np.int64)
    X_cat_test = X_cat_test.to_numpy(dtype=np.int64)
    y_train = y_train.to_numpy(dtype=np.int64)
    y_test = y_test.to_numpy(dtype=np.int64)

    X_num_train = torch.tensor(X_num_train, dtype=torch.float32)
    X_num_test = torch.tensor(X_num_test, dtype=torch.float32)
    X_cat_train = torch.tensor(X_cat_train, dtype=torch.int64)
    X_cat_test = torch.tensor(X_cat_test, dtype=torch.int64)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    X_num_train = X_num_train.to(device)
    X_num_test = X_num_test.to(device)
    X_cat_train = X_cat_train.to(device)
    X_cat_test = X_cat_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    return X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test


def create_dataloader(batch_size: int, X_num_train: torch.Tensor, X_cat_train: torch.Tensor, y_train: torch.Tensor) -> DataLoader:
    """
    Create an iterable over the given dataset with a sample. Should be used with the train dataset

    :param batch_size: the sample size
    :param X_num_train: the numerical tensor
    :param X_cat_train: the categorical tensor
    :param y_train: the target tensor
    :return: iterator over the dataset
    """
    train_dataset = TensorDataset(X_num_train, X_cat_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader


def get_cardinalities(X_cat_train: torch.Tensor) -> list:
    """
    Get the cardinality from the categorical tensor columns.

    :param X_cat_train:
    :return: list of how many unique values in the tensor columns
    """
    card_0 = len(X_cat_train[:, 0].unique())
    card_1 = len(X_cat_train[:, 1].unique())

    return [card_0, card_1]


def compile_transformer(n_cont_features: int, cat_cardinalities, d_out: int, device: torch.device):
    """
    Create a FTTransformer object and compile it.

    :param n_cont_features: how many continuous features to use
    :param cat_cardinalities: cardinality of categorical features
    :param d_out: the output dimension
    :param device: the device where the model will be moved
    :return: the FTTransformer object
    :return: the adam optimizer
    :return: the cross_entropy loss
    """
    model = FTTransformer(
        n_cont_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        d_out=d_out,
        **FTTransformer.get_default_kwargs(),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    return model, optimizer, loss_fn


def train_transformer(model: FTTransformer, optimizer, loss_fn: torch.nn.CrossEntropyLoss, epochs: int, train_loader: DataLoader,
                      device: torch.device) -> FTTransformer:
    """
    Train the transformer on the given dataset.

    :param model: the FTTransformer object to train
    :param optimizer: the optimizer of the model
    :param loss_fn: the loss of the model
    :param epochs: for how many epochs the FTTransformer will be trained
    :param train_loader: the iterable of the train dataset with the batch size
    :param device: the device where the model is
    :return:
    """
    for _ in tqdm(range(epochs)):
        model.train()
        for X_num_batch, X_cat_batch, y_batch in train_loader:
            X_num_batch, X_cat_batch, y_batch = X_num_batch.to(device), X_cat_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_num_batch, X_cat_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

    return model


def get_prediction(model: FTTransformer, X_num_test: torch.Tensor, X_cat_test: torch.Tensor) -> torch.Tensor:
    """
    Use the FTTransformer to predict the anomaly score of the given dataset.


    :param model: the FTTransformer object
    :param X_num_test: The numerical test tensor
    :param X_cat_test: the categorical test tensor
    :return: the predicted tensor of the target variable
    """
    model.eval()
    with torch.no_grad():
        pred = model(X_num_test, X_cat_test).argmax(dim=1)  # Get predicted classes

    return pred


def accuracy_ftt(y_pred: torch.Tensor, y_test: torch.Tensor) -> int:
    accuracy = (y_pred == y_test).float().mean().item()

    return round(accuracy * 100, 2)


if __name__ == '__main__':
    dev = create_device()

    Df = import_csv('../csv/smotenc_normalized_steel_plates.csv')
    X_n_train, X_n_test, X_c_train, X_c_test, y_tr, y_te = create_tensors(Df, 'anomaly', ['typeofsteel_a300', 'outside_global_index'], dev)

    tr_loader = create_dataloader(512, X_n_train, X_c_train, y_tr)

    card = get_cardinalities(X_c_train)

    transformer, adam, loss_funct = compile_transformer(X_n_train.shape[1], card, 7, dev)

    transformer = train_transformer(transformer, adam, loss_funct, epochs=5, train_loader=tr_loader, device=dev)

    y_pr = get_prediction(transformer, X_n_test, X_c_test)
    print(type(y_te))
    acc = accuracy_ftt(y_te, y_pr)

    print(f"Test Accuracy: {acc}%")

else:
    pass