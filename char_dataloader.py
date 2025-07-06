
# @title data
import requests
# url="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt" # https://www.tensorflow.org/text/tutorials/text_generation
url="https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt" # train test valid # https://pytorch.org/text/stable/datasets.html#penntreebank
out=requests.get(url)
with open("data.txt", "wb") as f:
    f.write(out.content)
text = open("data.txt", 'rb').read().decode(encoding='utf-8')

url="https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt" # train test valid # https://pytorch.org/text/stable/datasets.html#penntreebank
out=requests.get(url)
with open("data_.txt", "wb") as f: f.write(out.content)
test_text = open("data_.txt", 'rb').read().decode(encoding='utf-8')

# print(len(text))
# print(text[000:1000])
# data = ''.join(text)
# chars = sorted(list(set(data)))
# print(chars)


!pip install -qU datasets # restart?
# @title hf dataset
from datasets import load_dataset

# https://huggingface.co/datasets/Salesforce/wikitext
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1") # wikitext-103-raw-v1

text = dataset["train"]["text"]
test_text = dataset["test"]["text"]




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



# @title tiktoken dataloader
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import tiktoken # https://github.com/openai/tiktoken/tree/main

class CharDataset(Dataset): # https://github.com/karpathy/minGPT
    def __init__(self, raw_data, seq_len):
        data = ''.join(raw_data)
        self.enc = tiktoken.get_encoding("gpt2") # https://github.com/openai/tiktoken/blob/main/tiktoken/core.py
        self.vocab_size = self.enc.n_vocab # gpt2:50257
        self.data = self.data_process(data) # list of int
        self.seq_len = seq_len

    def data_process(self, data): # str 10780437
        return torch.tensor(self.enc.encode(data))

    def __len__(self):
        return len(self.data)//(self.seq_len+1)

    def __getitem__(self, idx):
        dix = self.data[idx*(self.seq_len+1) : (idx+1)*(self.seq_len+1)]
        x, y = dix[:-1], dix[1:]
        return x, y

seq_len = 100 # 128
train_data = CharDataset(text, seq_len) # one line of poem is roughly 50 characters
test_data = CharDataset(test_text, seq_len) # one line of poem is roughly 50 characters
from torch.utils.data.dataloader import DataLoader
batch_size = 64 #512
train_loader = DataLoader(train_data, shuffle=True, pin_memory=True, batch_size=batch_size, num_workers=2, drop_last=True) # num_workers = 4
test_loader = DataLoader(test_data, shuffle=True, pin_memory=True, batch_size=batch_size, num_workers=0, drop_last=True)

# https://github.com/openai/tiktoken/blob/main/tiktoken/core.py
def encode(context):
    if type(context) == str: return torch.tensor([train_loader.dataset.enc.encode(context)], device=device)
    elif type(context) == list: return train_loader.dataset.enc.encode_batch(context)
    else: raise Exception
def decode(x): return train_loader.dataset.enc.decode(list(x))
# for x,y in train_loader:
#     break


