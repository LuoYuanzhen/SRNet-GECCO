import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float64)
        self.labels = labels
        assert self.data.shape[0] == self.labels.shape[0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :], self.labels[index]