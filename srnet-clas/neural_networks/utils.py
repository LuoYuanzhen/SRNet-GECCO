import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        assert self.data.shape[0] == self.targets.shape[0]
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.data[index, :]), self.targets[index]
        return self.data[index, :], self.targets[index]