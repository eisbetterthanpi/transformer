import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RoPE(nn.Module): # Rotary Positional Embeddings
    def __init__(self, dim, seq_len=512, base=10000):
        super().__init__()
        self.dim, self.base = dim, base
        theta = 1.0 / (base ** (torch.arange(0, dim, step=2) / dim))
        pos = torch.arange(seq_len).unsqueeze(-1)
        angles = (pos * theta)[None,...,None] # [seq_len, 1] * [dim // 2] -> [1, seq_len, dim // 2, 1]
        self.rot_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2).to(device) # [seq_len, dim // 2, 2] -> [1, seq_len, dim]

    def forward(self, x):
        seq_len = x.size(1)
        if self.rot_emb.shape[0] < seq_len: self.__init__(self.dim, seq_len, self.base)
        return x * self.rot_emb[:seq_len]

# class LearnedRoPE(nn.Module): # learnt RoPE ; each tok is 1 pos
#     def __init__(self, dim):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(1, dim//2))

#     def forward(self, x): #
#         batch, seq_len, dim = x.shape
#         # if rot_emb.shape[0] < seq_len: self.__init__(dim, seq_len)
#         pos = torch.arange(seq_len).unsqueeze(1)
#         angles = (self.weights * pos * 2*torch.pi).unsqueeze(-1) # [seq_len, 1] * [dim // 2] -> [seq_len, dim // 2, 1]
#         rot_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # [seq_len, dim // 2, 2]
#         return x * rot_emb.flatten(-2).unsqueeze(0)

class LearnedRoPE(nn.Module): # learnt RoPE ; each tok is 1 pos
    def __init__(self, dim, seq_len=512):
        super().__init__()
        self.dim = dim
        self.weights = nn.Parameter(torch.randn(1, dim//2))
        pos = torch.arange(seq_len).unsqueeze(1)
        angles = (self.weights * pos * 2*torch.pi)[None,...,None] # [seq_len, 1] * [dim // 2] -> [1, seq_len, dim // 2, 1]
        self.rot_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2).to(device) # [seq_len, dim // 2, 2] -> [1, seq_len, dim]

    def forward(self, x): # [batch, seq_len, dim]
        seq_len = x.size(1)
        if self.rot_emb.shape[0] < seq_len: self.__init__(self.dim, seq_len)
        return x * self.rot_emb[:seq_len]




class RotEmb(nn.Module): # Rotary Positional Embeddings
    def __init__(self, dim, top=torch.pi, base=10000):
        super().__init__()
        self.theta = top / (base ** (torch.arange(0, dim, step=2, device=device) / dim))
        # self.theta = top / (base ** torch.linspace(0, 1, dim//2, device=device))

    def forward(self, pos): # [batch] in [0,1]
        angles = (pos.unsqueeze(-1) * self.theta).unsqueeze(-1) # [seq_len, 1] * [dim // 2] -> [seq_len, dim // 2, 1]
        rot_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # [seq_len, dim // 2, 2]
        return rot_emb.flatten(-2) # [seq_len, dim]

class LearnedRotEmb(nn.Module): # pos in R
    def __init__(self, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn((1, dim//2), device=device))

    def forward(self, pos): # [batch] in [0,1]
        angles = (self.weights * pos.unsqueeze(-1) * 2*torch.pi).unsqueeze(-1) # [batch, 1] * [1, dim//2] -> [batch, dim//2, 1]
        rot_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # [batch, dim // 2, 2]
        return rot_emb.flatten(-2) # [batch, dim]


class LearnedRotEmb2D(nn.Module): # pos in R
    def __init__(self, dim):
        super().__init__()
        self.weight_xy = nn.Parameter(torch.randn(2, dim//2))

    def forward(self, pos): # [batch,2] in [0,1]
        angles = (pos @ self.weight_xy * 2*math.pi).unsqueeze(-1) # [batch, 2] @ [2, dim//2] -> [batch, dim//2, 1]
        rot_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # [batch, dim//2, 2]
        return rot_emb.flatten(-2) # [batch, dim]





# class LearnedRoPE2D(nn.Module): # learnt RoPE ; each tok is 1 pos
#     def __init__(self, dim):
#         super().__init__()
#         self.weight_x, self.weight_y = nn.Parameter(torch.randn(1, dim//2)), nn.Parameter(torch.randn(1, dim//2))

#     def forward(self, img): #
#         batch, dim, h, w = img.shape
#         y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij") # [h,w], y:row_num, x:col_num
#         angles = self.weight_x * x.reshape(-1,1) + self.weight_y * y.reshape(-1,1) # [1, dim//2] * [h*w, 1] = [h*w, dim//2]
#         angles = (angles * 2*torch.pi)[None,...,None] # [1,h*w, dim//2,1]
#         rot_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # [1,h*w, dim//2,2]

#         return img * rot_emb.flatten(-2).transpose(-2,-1).reshape(1,dim,h,w).to(device)




# fixed, learnt, mixed learnt
# disc tok pos vs pos in R
# caartesian x y
# polar r theta


class RoPE2D(nn.Module): # Rotary Positional Embeddings
    def __init__(self, dim, h=224, w=224, base=10000):
        super().__init__()
        # theta = 1. / (base ** (torch.arange(0, dim, step=4) / dim))
        theta = 1. / (base**torch.linspace(0,1,dim//4)).unsqueeze(0)
        # pos = torch.arange(seq_len).unsqueeze(1)
        # angles = (pos * theta).unsqueeze(-1) # [seq_len, 1] * [dim // 2] -> [seq_len, dim // 2, 1]
        # self.rot_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # [seq_len, dim // 2, 2]
        self.dim, self.h, self.w = dim, h, w
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij") # print(y,x) # [h,w], y:row_num, x:col_num
        # y, x = y.reshape(-1,1), x.reshape(-1,1) # [h*w,1]
        # angles = (pos * theta).unsqueeze(-1) # [seq_len, 1] * [dim // 2] -> [seq_len, dim // 2, 1]
        y, x = (y.reshape(-1,1) * theta).unsqueeze(-1), (x.reshape(-1,1) * theta).unsqueeze(-1) # [h*w,1]*[1,dim//4] = [h*w, dim//4, 1]
        # self.rot_emb = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1).flatten(-2) # [h*w, dim//4 ,4] -> [h*w, dim]
        self.rot_emb = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1).reshape(dim, h, w).to(device) # [h*w, dim//4 ,4] -> [h, w, dim]

    def forward(self, img): #
        batch, dim, h, w = img.shape
        if self.h < h or self.w < w: self.__init__(self.dim, h, w)
        rot_emb = self.rot_emb[:, :h, :w].unsqueeze(0) # [1, h, w, dim]
        return img * rot_emb


def RoPE2D(dim=16, h=8, w=8, base=10000):
    # theta = 1. / (base ** (torch.arange(0, dim, step=4) / dim))
    theta = 1. / (base**torch.linspace(0,1,dim//4)).unsqueeze(0)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij") # print(y,x) # [h,w], y:row_num, x:col_num
    y, x = (y.reshape(-1,1) * theta).unsqueeze(-1), (x.reshape(-1,1) * theta).unsqueeze(-1) # [h*w,1]*[1,dim//4] = [h*w, dim//4, 1]
    rot_emb = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1).flatten(-2) # [h*w, dim//4 ,4] -> [h*w, dim]
    # rot_emb = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1)#.reshape(dim, h, w).to(device) # [h*w, dim//4 ,4] -> [h, w, dim]
    return rot_emb



def posemb_sincos_2d(h, w, dim, temp = 10000):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij") # print(y,x) # [h,w], y:row_num, x:col_num
    omega = 1. / (temp**torch.linspace(0,1,dim//4))
    # print(omega)
    y, x = y.reshape(-1,1) * omega.unsqueeze(0), x.reshape(-1,1) * omega.unsqueeze(0) # [h*w,1]*[1,dim//4] = [h*w,dim//4]
    # print(y.shape,x.shape) # [h,w], y:row num, x:col num
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1) # [h*w,dim]
    return pe



dim=16
# seq_len=512
# rope = RoPE(dim, seq_len, base=10000)
# rope = LearnedRoPE2D(dim)
rope = RoPE2D(dim)
# x = torch.rand(4,64,dim)
h,w = 3,5
x = torch.rand((4,dim,h,w), device=device)
out = rope(x)

# [batch] -> []

print(out.shape)

# rot_emb = rope.rot_emb
# print(rot_emb.shape)
# print(rot_emb[:7])
# # rot_emb = rot_emb.reshape(seq_len, dim // 2, 2)
# # print(rot_emb)



# class LearnedSinusoidalPosEmb(nn.Module):
#     """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """
#     def __init__(self, dim):
#         super().__init__()
#         half_dim = dim // 2
#         self.weights = nn.Parameter(torch.randn(1, half_dim))

#     def forward(self, x): # [b]
#         x = x.unsqueeze(-1)
#         freqs = x * self.weights * 2 * math.pi # [b, 1] * [1, half_dim] = [b, half_dim]
#         fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1) # [b, dim]
#         fouriered = torch.cat((x, fouriered), dim = -1) # [b, 1+dim]
#         return fouriered




