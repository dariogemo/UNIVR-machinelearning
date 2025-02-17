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

    col_to_drop = cat_features + [target]
    X_num = df.drop(col_to_drop, axis=1)
    X_cat = df[cat_features].astype(np.int64)
    y = df['anomaly'].astype(np.int64)

    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(X_num, X_cat, y, test_size=0.2)

    X_num_train, X_num_test = X_num_train.to_numpy(dtype=np.float32), X_num_test.to_numpy(dtype=np.float32)
    X_cat_train, X_cat_test = X_cat_train.to_numpy(dtype=np.int64), X_cat_test.to_numpy(dtype=np.int64)
    y_train, y_test = y_train.to_numpy(dtype=np.int64), y_test.to_numpy(dtype=np.int64)

    X_num_train, X_num_test = map(torch.tensor, (X_num_train, X_num_test))
    X_cat_train, X_cat_test = map(torch.tensor, (X_cat_train, X_cat_test))
    y_train, y_test = map(torch.tensor, (y_train, y_test))

    X_num_train, X_cat_train, y_train = X_num_train.to(device), X_cat_train.to(device), y_train.to(device)
    X_num_test, X_cat_test, y_test = X_num_test.to(device), X_cat_test.to(device), y_test.to(device)

    return X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test


def create_dataloader(batch_size: int, X_num_train: torch.Tensor, X_cat_train: torch.Tensor, y_train: torch.Tensor) -> DataLoader:
    train_dataset = TensorDataset(X_num_train, X_cat_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def get_cardinalities(X_cat_train: torch.Tensor) -> list:
    card_0 = len(X_cat_train[:, 0].unique())
    card_1 = len(X_cat_train[:, 1].unique())

    return [card_0, card_1]


def compile_transformer(n_cont_features: int, cat_cardinalities, d_out: int, device: torch.device):
    model = FTTransformer(
        n_cont_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        d_out=d_out,
        **FTTransformer.get_default_kwargs(),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    return model, optimizer, loss_fn


def train_transformer(model: FTTransformer, optimizer: object, loss_fn: object, epochs: int, train_loader: DataLoader,
                      device: torch.device) -> FTTransformer:
    #X_num_train, X_cat_train, y_train = tensors[0].to(device), tensors[2].to(device), tensors[4].to(device)
    #X_num_test, X_cat_test, y_test = tensors[1].to(device), tensors[3].to(device), tensors[5].to(device)

    for epoch in tqdm(range(epochs)):
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
    model.eval()
    with torch.no_grad():
        pred = model(X_num_test, X_cat_test).argmax(dim=1)  # Get predicted classes
        #pred = pred.numpy() if pred.device == torch.device('cpu') else pred.cpu().numpy()

    return pred


def accuracy_ftt(y_pred: torch.Tensor, y_test: torch.Tensor) -> int:
    accuracy = (y_pred == y_test).float().mean().item()

    return round(accuracy * 100, 2)


if __name__ == '__main__':
    device = create_device()

    df = import_csv('csv/balanced_normalized_steel_plates.csv')
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = create_tensors(df, 'anomaly', ['typeofsteel_a300', 'outside_global_index'], device)

    train_loader = create_dataloader(512, X_num_train, X_cat_train, y_train)

    cardinalities = get_cardinalities(X_cat_train)

    model, optimizer, loss_fn = compile_transformer(X_num_train.shape[1], cardinalities, 7, device)

    model = train_transformer(model, optimizer, loss_fn, epochs=5, train_loader=train_loader, device=device)

    y_pred = get_prediction(model, X_num_test, X_cat_test)
    print(type(y_test))
    accuracy = accuracy_ftt(y_test, y_pred)

    print(f"Test Accuracy: {accuracy}%")

else:
    pass