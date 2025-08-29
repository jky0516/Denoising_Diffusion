import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Make train and validation dataloaders
def _make_loaders(X, y, batch_size=128, test_split=0.2, preprocess_fn=None, shuffle=True):
    if test_split > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_split, random_state=42
        )
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1, 35, 31)
    if preprocess_fn is not None:
        y_train_t = preprocess_fn(y_train_t)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

    val_loader = None
    X_val_t = None
    y_val_t = None
    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).view(-1, 1, 35, 31)
        if preprocess_fn is not None:
            y_val_t = preprocess_fn(y_val_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train_t, y_train_t, X_val_t, y_val_t

# Load low Ka data
def load_lowKa(low_files, batch_size=128, test_split=0.2, preprocess_fn=None, shuffle=True):
    X = np.load(low_files[0])
    y = np.load(low_files[1])
    return _make_loaders(X, y, batch_size, test_split, preprocess_fn, shuffle)

# Load high Ka data
def load_highKa_eval(high_files, batch_size=128, preprocess_fn=None):
    X = np.load(high_files[0])
    y = np.load(high_files[1])
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).view(-1, 1, 35, 31)
    if preprocess_fn is not None:
        y_t = preprocess_fn(y_t)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, X_t, y_t
