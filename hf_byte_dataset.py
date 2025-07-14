# @title hf byte dataset me
!pip install -qU datasets # restart?
import torch
from torch.utils.data import IterableDataset

class StreamDataset(IterableDataset):
    def __init__(self, dataset, seq_len=129, buffer_size=1024):
        self.vocab_size = 256 # utf-8 # self.enc.n_vocab # gpt2:50257
        self.data = iter(dataset)
        self.seq_len, self.buffer_size = seq_len, buffer_size  # must be ≥ seq_len
        self.buffer = []  # token buffer
        self.fill_buffer()

    def fill_buffer(self):
        while len(self.buffer) < self.buffer_size:
            x = next(self.data)
            # print(x)
            self.buffer.extend(x['text'].encode("utf-8"))

    def __iter__(self):
        while True:
            if len(self.buffer) < self.seq_len: self.fill_buffer()
            if len(self.buffer) < self.seq_len: return
            x, self.buffer = self.buffer[:self.seq_len], self.buffer[self.seq_len:]
            yield torch.tensor(x, dtype=torch.int32) # uint8 int32


from datasets import load_dataset
name = 'Skylion007/openwebtext' if torch.cuda.is_available() else 'stas/openwebtext-10k'
dataset = load_dataset(name, split="train", streaming=True, revision='refs/convert/parquet', cache_dir="/content/hf")

seq_len = 128*1+1 # 128
buffer_size = seq_len*1
train_data = StreamDataset(dataset, seq_len, buffer_size) # train_data = StreamDataset(dataset["train"], seq_len, buffer_size)
# del dataset

from torch.utils.data.dataloader import DataLoader
batch_size = 64 if torch.cuda.is_available() else 16 #64 512
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True, num_workers=2)
# del train_data
def encode(context):
    if type(context) == str: return torch.tensor([train_loader.dataset.enc.encode(context)], device=device)
    elif type(context) == list: return train_loader.dataset.enc.encode_batch(context)
    else: raise Exception
def decode(x): return train_loader.dataset.enc.decode(list(x))
for x in train_loader:
    print(x.shape, x)
    break
