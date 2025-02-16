import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from rtdl_revisiting_models import FTTransformer

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
#  Load Data
# ===========================
df = pd.read_csv('csv/balanced_normalized_steel_plates.csv', index_col=0, header=0)

# Round categorical feature to ensure it's integer
df['outside_global_index'] = df['outside_global_index'].round().astype(np.int64)

# Split features
X_num = df.drop(['typeofsteel_a300', 'outside_global_index', 'anomaly'], axis=1)  # Numerical features
X_cat = df[['typeofsteel_a300', 'outside_global_index']].astype(np.int64)  # Categorical features
y = df['anomaly'].astype(np.int64)  # Ensure labels are integers

# Train/Test Split
X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)

# Convert to NumPy arrays (ensure correct dtype)
X_num_train, X_num_test = X_num_train.to_numpy(dtype=np.float32), X_num_test.to_numpy(dtype=np.float32)
X_cat_train, X_cat_test = X_cat_train.to_numpy(dtype=np.int64), X_cat_test.to_numpy(dtype=np.int64)
y_train, y_test = y_train.to_numpy(dtype=np.int64), y_test.to_numpy(dtype=np.int64)

# Convert to PyTorch tensors
X_num_train, X_num_test = map(torch.tensor, (X_num_train, X_num_test))
X_cat_train, X_cat_test = map(torch.tensor, (X_cat_train, X_cat_test))
y_train, y_test = map(torch.tensor, (y_train, y_test))

batch_size = 256  # Reduce batch size

# Create DataLoader
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(X_num_train, X_cat_train, y_train)
test_dataset = TensorDataset(X_num_test, X_cat_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Get categorical feature cardinalities
cat_cardinalities = [2, 3]  # Ensure these match the unique values in categorical features

# ===========================
#  Define FT-Transformer Model
# ===========================
model = FTTransformer(
    n_cont_features=X_num_train.shape[1],  # Number of numerical features
    cat_cardinalities=cat_cardinalities,  # Cardinalities of categorical features
    d_out=7,  # Number of classes (ensure it's correct)
    **FTTransformer.get_default_kwargs(),
).to(device)

# Optimizer & Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# Move data to GPU
X_num_train, X_cat_train, y_train = X_num_train.to(device), X_cat_train.to(device), y_train.to(device)
X_num_test, X_cat_test, y_test = X_num_test.to(device), X_cat_test.to(device), y_test.to(device)

# ===========================
#  Training Loop
# ===========================
for epoch in tqdm(range(50)):
    model.train()
    for X_num_batch, X_cat_batch, y_batch in train_loader:
        X_num_batch, X_cat_batch, y_batch = X_num_batch.to(device), X_cat_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_num_batch, X_cat_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()

# ===========================
#  Evaluation
# ===========================
model.eval()
with torch.no_grad():
    preds = model(X_num_test, X_cat_test).argmax(dim=1)  # Get predicted classes
    accuracy = (preds == y_test).float().mean().item()

print(f"Test Accuracy: {accuracy:.4f}")

# TODO: understand what's going on
# TODO: create functions to make it clearer