import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class AgeDataset(Dataset):
    def __init__(self, csv, transform=None, binary=False):
        self.df = pd.read_csv(csv)
        self.transform = transform
        self.binary = binary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.normpath(self.df.iloc[idx]['file_path'])

        if self.binary:             # Binary label
            label = bool(self.df.iloc[idx]['binary_label']) 
            ret_label = torch.tensor(label, dtype=torch.bool)
        else:                       # Age label
            label = float(self.df.iloc[idx]['age'])
            label = (label - 5) / (70 - 5)
            ret_label = torch.tensor(label, dtype=torch.float32)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, ret_label
