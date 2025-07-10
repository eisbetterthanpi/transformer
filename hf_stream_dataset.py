# @title hf stream dataset me
# !pip install -qU datasets # restart?
!pip install datasets==2.13.1 # restart
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import tiktoken # https://github.com/openai/tiktoken/tree/main

class StreamDataset(Dataset):
    def __init__(self, dataset, seq_len, buffer_size):
        self.enc = tiktoken.get_encoding("gpt2") # https://github.com/openai/tiktoken/blob/main/tiktoken/core.py
        self.vocab_size = self.enc.n_vocab # gpt2:50257
        self.dataset = dataset
        self.data = iter(dataset)
        self.seq_len = seq_len
        self.buffer_size = buffer_size  # must be ≥ seq_len
        self.buffer = []  # token buffer
        self.fill_buffer()

    def fill_buffer(self):
        while len(self.buffer) < self.buffer_size:
            x = next(self.data)
            tokens = self.enc.encode(x["text"]) # tiktoken
            self.buffer.extend(tokens)

    def __len__(self):
        # /4.5/(4/3)
        return 128000000
        # return self.length

    def __getitem__(self, idx):
        # print('get', idx)
        if idx == 0: self.data = iter(self.dataset)
        if len(self.buffer) < self.seq_len: self.fill_buffer()
        if len(self.buffer) < self.seq_len:
            raise StopIteration
        x = self.buffer[:self.seq_len]
        self.buffer = self.buffer[self.seq_len:]
        # return torch.tensor(x)
        return torch.tensor(x, dtype=torch.int32)

def collate_fn(batch):
    # print(batch)
    return torch.stack(batch)

name = 'Skylion007/openwebtext' if torch.cuda.is_available() else 'stas/openwebtext-10k'

dataset = load_dataset(name, trust_remote_code=True, split="train", streaming=True, cache_dir="/content/hf") # 8.7,3.8
# dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True, split="train", streaming=True, cache_dir="/content/hf") # 8.7,3.8
# dataset = load_dataset("deepmind/pg19", trust_remote_code=True, split="train", streaming=True, cache_dir="/content/hf") # 8.7,3.8

seq_len = 128*1+1 # 128
buffer_size = seq_len*1
train_data = StreamDataset(dataset, seq_len, buffer_size) # train_data = StreamDataset(dataset["train"], seq_len, buffer_size)
# del dataset

from torch.utils.data.dataloader import DataLoader
batch_size = 64 if torch.cuda.is_available() else 16 #64 512
train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True, num_workers=2)
del train_data
def encode(context):
    if type(context) == str: return torch.tensor([train_loader.dataset.enc.encode(context)], device=device)
    elif type(context) == list: return train_loader.dataset.enc.encode_batch(context)
    else: raise Exception
def decode(x): return train_loader.dataset.enc.decode(list(x))
# for x,y in train_loader:
#     break
# print(train_data.vocab_size)
