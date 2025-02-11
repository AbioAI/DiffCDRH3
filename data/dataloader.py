import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
class PairedSequenceDataset(Dataset):
    def __init__(self, cdrh3_data, epitope_data):
        self.cdrh3_data = cdrh3_data
        self.epitope_data = epitope_data
        assert len(self.cdrh3_data) == len(self.epitope_data), "CDRH3 和 Epitope 数据不匹配"

    def __len__(self):
        return len(self.cdrh3_data)

    def __getitem__(self, index):
        cdrh3_onehot = torch.tensor(self.cdrh3_data[index], dtype=torch.float32)
        epitope_onehot = torch.tensor(self.epitope_data[index], dtype=torch.float32)
        return cdrh3_onehot, epitope_onehot


def get_dataloaders(cdrh3_npy_file, epitope_npy_file, batch_size=32, test_size=0.2, shuffle=True, num_workers=0):
    """
    创建训练和验证的 DataLoader。

    """
    cdrh3_data = np.load(cdrh3_npy_file, allow_pickle=True)
    epitope_data = np.load(epitope_npy_file, allow_pickle=True)

    # 划分训练集和验证集
    cdrh3_train, cdrh3_valid, epitope_train, epitope_valid = train_test_split(
        cdrh3_data, epitope_data, test_size=test_size, random_state=42, shuffle=shuffle
    )

    # 创建 Dataset
    train_dataset = PairedSequenceDataset(cdrh3_train, epitope_train)
    valid_dataset = PairedSequenceDataset(cdrh3_valid, epitope_valid)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader
