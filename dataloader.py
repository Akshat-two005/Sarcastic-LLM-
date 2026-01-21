import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

BLOCK_SIZE = 128
BATCH_SIZE = 32


class TokenDataset(Dataset):
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data) - BLOCK_SIZE

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx + BLOCK_SIZE].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + BLOCK_SIZE + 1].astype(np.int64))
        return x, y

train_dataset = TokenDataset(f"{DATA_DIR}/train.bin")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)
print("Loading data from:", os.path.join(DATA_DIR, "train.bin"))
