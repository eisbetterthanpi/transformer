# @title Sliding Window Attention
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class Attention(nn.Module): # sliding window attention
    # def __init__(self, d_model, cond_dim=None, n_heads=None, d_head=8, drop=0.): # .1
    def __init__(self, query_dim, cond_dim=None, n_heads=8, d_head=8, drop=0, w=32):
        super().__init__()
        self.d_model = d_model = d_head * n_heads
        self.d_head, self.n_heads = d_head, n_heads
        self.cond_dim = cond_dim
        self.pos_enc = RoPE(d_model, base=10000) # 10000
        self.q = nn.Linear(query_dim, d_model, bias=False)
        self.kv = nn.Linear(cond_dim or query_dim, 2*d_model, bias=False)
        self.lin = zero_module(nn.Linear(d_model, d_model))
        self.drop = nn.Dropout(drop) # indp before q,k,v; after linout
        self.scale = self.d_head**-.5
        # torch.nn.init.normal_(self.qkv.weight, std=.02)
        # torch.nn.init.normal_(self.q.weight, std=1/(math.sqrt(query_dim)+math.sqrt(d_model)))
        # torch.nn.init.normal_(self.kv.weight, std=1/(math.sqrt(cond_dim or query_dim)+math.sqrt(d_model)))
        self.w = w

    def forward(self, x, cond=None, mask=None): # [b,t,d], [batch, num_tok, cond_dim], [b,t]
        b,t = x.shape[:2]
        if self.cond_dim==None: cond=x # is self attn
        q = self.pos_enc(self.q(x)).unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2) # [batch, T, d_model] -> [batch, n_heads, T, d_head]
        k, v = self.kv(cond).chunk(2, dim=-1)
        k = self.pos_enc(k).unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2) # [batch, n_heads, T/num_tok, d_head]
        v = v.unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2) # [batch, n_heads, T/num_tok, d_head]
        k, v = F.pad(k, (0,0,self.w-1,0)), F.pad(v, (0,0,self.w-1,0)) # [b, h, t+w-1, 2*d]
        k, v = k.unfold(-2, self.w, 1).transpose(-2,-1), v.unfold(-2, self.w, 1).transpose(-2,-1) # [b,h,t,w,d]

        attn = torch.einsum("bhtd,bhtwd->bhtw", q, k) * self.scale
        # print('attn fwd q mk attn', q.dtype, mk.dtype, attn.dtype)
        if mask != None: attn = attn.masked_fill(~mask[:,None,:,None], -torch.finfo(attn.dtype).max) # [b,t]->[b,1,t,1] # causal is built into swa, so mask is only for [b,t]
        attn = torch.softmax(attn, dim=-1) # [b,h,t,1,w]
        # attn = F.sigmoid(attn-math.log(attn.shape[-1])) # https://arxiv.org/pdf/2409.04431
        # out = (self.drop(attn) @ mv).squeeze(-2) # [b,h,t,1,w]@[b,h,t,w,d]=[b,h,t,1,d]
        out = torch.einsum("bhtw,bhtwd->bhtd", self.drop(attn), v)
        out = out.transpose(1, 2).flatten(2)
        return self.drop(self.lin(out)) # [batch, T, d_model]

# typically: gen td*td=tt
# up t1d*twd=tw, patch pd*tpd=p
# down td*(w+1)d=t*(w+1)

# t(2d+t)
# t((w+1)d + w)
# d+t >= wd+w
# d~64-512; w~32-256; t~128*4^3=8192=2^13
# 2^9; 2^8

# b,t,d = 64,100,512

# mask = torch.rand(b,t, device=device)>.5
# model = Attention(d, w=3).to(device) # 257 ms
# x = torch.rand(b,t,d, device=device)
# out = model(x, mask=mask)
