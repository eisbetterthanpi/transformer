# @title char dataloader
# https://github.com/Sam-Armstrong/tinyGPT/blob/main/Training.py
# https://colab.research.google.com/github/karpathy/minGPT/blob/master/play_char.ipynb
# https://github.com/karpathy/nanoGPT
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CharDataset(Dataset): # https://github.com/karpathy/minGPT
    def __init__(self, raw_data, seq_len):
        data = ''.join(raw_data)
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars) # 283
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
        self.data = self.data_process(data) # list of int
        self.seq_len = seq_len

    def data_process(self, data): # str 10780437
        return torch.tensor([self.stoi.get(c) for c in data]) # list of int 4570571 # stoi.get(c,UNK_IDX)

    def __len__(self):
        # return len(self.data) - self.seq_len
        return len(self.data)//(self.seq_len+1)

    def __getitem__(self, idx):
        # dix = self.data[idx:idx + self.seq_len + 1]
        dix = self.data[idx*(self.seq_len+1) : (idx+1)*(self.seq_len+1)]
        x, y = dix[:-1], dix[1:]
        return x, y


seq_len = 100 # 128
train_data = CharDataset(text, seq_len) # one line of poem is roughly 50 characters
test_data = CharDataset(test_text, seq_len) # one line of poem is roughly 50 characters
from torch.utils.data.dataloader import DataLoader
batch_size = 64 #512
train_loader = DataLoader(train_data, shuffle = True, pin_memory = True, batch_size = batch_size, num_workers = 2) # num_workers = 4
test_loader = DataLoader(test_data, shuffle = True, pin_memory = True, batch_size = batch_size, num_workers = 0)


def encode(context): return torch.tensor([train_data.stoi.get(c) for c in context], device=device).unsqueeze(0)
def decode(x): return ''.join([train_data.itos[int(i)] for i in x])
# for x,y in train_loader:
#     break
