# @title yelp data
!pip install -qU datasets
from datasets import load_dataset
# # dataset = load_dataset("Yelp/yelp_review_full", split="train", streaming=True, revision='refs/convert/parquet', cache_dir="/content/hf")
dataset = load_dataset("Yelp/yelp_review_full", split="train", revision='refs/convert/parquet', cache_dir="/content/hf")
# dataset = load_dataset("yelp_polarity") # yelp_polarity yelp_review_full

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def encode(c): return torch.tensor(list(c.encode("utf-8")), dtype=torch.int32, device=device)#.unsqueeze(0)
def decode(x): return bytes(x.tolist()).decode("utf-8", errors='replace') # replace ignore

from torch.utils.data import Dataset
# from torch.utils.data import IterableDataset
class ByteDataset(Dataset):
# class ByteDataset(IterableDataset):
    def __init__(self, data):
        self.data = data
        # self.data = iter(data)
        # self.data = self.data_process(data) # list of int
        self.vocab_size = 256 # utf-8 # self.enc.n_vocab # gpt2:50257
        self.num_classes = 5
    # def data_process(self, data): # str 10780437
    #     # return torch.tensor([self.stoi.get(c) for c in data]) # list of int 4570571 # stoi.get(c,UNK_IDX)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        xy = self.data[idx]
        return encode(xy['text']), xy['label']
    # def __iter__(self):
    #     while True:
    #         xy = next(self.data)
    #         yield encode(xy['text']), xy['label']

train_data = ByteDataset(dataset)

from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    x, y = zip(*batch)
    x = pad_sequence(x, batch_first=True, padding_value=0, padding_side='left')
    # print(x,y)
    return x, torch.tensor(y).unsqueeze(-1)

from torch.utils.data.dataloader import DataLoader
batch_size = 64 if torch.cuda.is_available() else 16 #64 512
train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=0)

# for i,(x,y) in enumerate(train_loader):
#     # print(x.shape, x)
#     print(i, x)
#     break
# print(decode(x[3]))
