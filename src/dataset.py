import torch
import numpy as np
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence


# Prepare padded hypergraphs for batched input
def transform_and_pad_input(x):
    tempX = []
    for ele in x:
        tempX.append(torch.tensor(ele).to(torch.float32))
    x_padded = pad_sequence(tempX, batch_first=True, padding_value=0)
    return x_padded


class MIMICiiiDataset(data.Dataset):
    def __init__(self, patient, label, pids, visit_len):
        self.patient = patient
        self.label = label
        self.pids = pids
        self.visit_len = visit_len

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        return self.patient[idx], self.label[idx], self.pids[idx], self.visit_len[idx]


class MIMICivDataset(data.Dataset):
    def __init__(self, x_path, v_lens, labels, pids, train_test_flag):
        self.X_path, self.v_lens, self.labels, self.pids = x_path, v_lens, labels, pids
        if train_test_flag == 'Train':
            self.anchor = np.load('../data/MIMIC_IV/anchor_train.npy')
        else:
            self.anchor = np.load('../data/MIMIC_IV/anchor_test.npy')

    def __len__(self):
        return len(self.pids)

    def transform_and_pad_input(self, source):
        temp_x = [torch.tensor(source).to(torch.float32), torch.tensor(self.anchor).to(torch.float32)]
        x_padded = pad_sequence(temp_x, batch_first=True, padding_value=0)
        return torch.transpose(x_padded[0], 0, 1)

    def __getitem__(self, idx):
        source = np.load(f'../data/MIMIC_IV/{self.X_path}_{idx}.npy')
        padded = self.transform_and_pad_input(source)
        return padded, self.labels[idx], self.pids[idx], self.v_lens[idx]

