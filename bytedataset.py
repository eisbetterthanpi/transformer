# @title byte dataloader
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def txt_iter(filepath, chunk_size=8192):
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk: break
            yield chunk

class ByteDataset(Dataset):
    def __init__(self, path='train/', seq_len=129, buffer_size=1024):
        self.vocab_size = 256 # utf-8 # self.enc.n_vocab # gpt2:50257
        self.seq_len, self.buffer_size = seq_len, buffer_size  # must be ≥ seq_len
        file_list = [path+f for f in os.listdir(path)]
        random.shuffle(file_list)
        self.fileiter = iter(file_list)
        # self.process()
        self.textiter = txt_iter(next(self.fileiter))
        self.buffer = []  # token buffer
        self.fill_buffer()

    def fill_buffer(self):
        while len(self.buffer) < self.buffer_size:
            try: x = next(self.textiter)
            except StopIteration:
                self.textiter = txt_iter(next(self.fileiter))
                x = next(self.textiter)
            self.buffer.extend(x)

    def __len__(self): return 128000 # too large will cause long load first batch

# import re
# def strip_gutenberg_boilerplate(text):
#     start_re = re.compile(r"\*\*\* START OF (.*?) \*\*\*", re.IGNORECASE)
#     end_re = re.compile(r"\*\*\* END OF (.*?) \*\*\*", re.IGNORECASE)
#     start_match = start_re.search(text)
#     end_match = end_re.search(text)
#     start = start_match.end() if start_match else 0
#     end = end_match.start() if end_match else len(text)
#     return text[start:end].strip()

# def normalise_whitespace(text):
#     return re.sub(r'\s+', ' ', text).strip()

    def __getitem__(self, idx):
        if len(self.buffer) < self.seq_len: self.fill_buffer()
        if len(self.buffer) < self.seq_len: raise StopIteration
        x, self.buffer = self.buffer[:self.seq_len], self.buffer[self.seq_len:]
        return torch.tensor(x, dtype=torch.uint8)

seq_len = 129 # 128
train_data = ByteDataset('validation/', seq_len) # one line of poem is roughly 50 characters
# test_data = ByteDataset('test/', seq_len) # one line of poem is roughly 50 characters
batch_size = 64 #512
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
# test_loader = DataLoader(test_data, shuffle = True, pin_memory = True, batch_size = batch_size, num_workers = 0)

def encode(c): return torch.tensor(list(c.encode("utf-8")), dtype=torch.uint8)#, device=device)#.unsqueeze(0)
def decode(x): return bytes(x.tolist()).decode("utf-8")
# for x in train_loader:
#     break
# print(x)
