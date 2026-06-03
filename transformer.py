import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import matplotlib.pyplot as plt
import BPEtokenizer as bpe



with open('gutenberg.txt', 'r') as ff:
    data = ff.read()

bpe = bpe.BPE_tk(data)

bpe.train(200)

vocab_size = len(bpe.vocab)

token_list = bpe.encode(bpe.symbols)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")



train_data = torch.tensor(token_list[:int(.8*len(token_list))], dtype=torch.long, device = device)
val_data = torch.tensor(token_list[int(.8*len(token_list)):int(.9*len(token_list))], dtype=torch.long, device = device)
test_data = torch.tensor(token_list[int(.9*len(token_list)):], dtype=torch.long, device = device)


with torch.no_grad():
    def get_batch(split):
        if split=='train':
            data = train_data
        elif split=='val':
            data = val_data
        else:       
            data = test_data
        idx = torch.randint(len(data)-T-1, (B,), device =device)
        X = data[idx[:,None] + torch.arange(T,device = device)[None,:]]
        Y = data[idx[:,None] + torch.arange(T,device = device)[None,:] + 1]
        return X, Y



B = 64 #batch_size
T = 16  #block_size
d = 64 # token embedding dimension
dK = 16
num_heads = 4
num_layers = 4
steps = 2000
lr = 5e-4
epsilon = 1e-8


class LLM(nn.Module):
    def __init__(self, Rope=False):
        super().__init__()
        self.C = nn.Embedding(vocab_size, d)
        if not Rope:
            self.pos = nn.Embedding(T, d)
        self.pos = nn.Embedding(T, d)
        self.proj = nn.Linear(d,vocab_size)
        self.blocks = nn.Sequential(*[Block().to(device) for _ in range(num_layers)])


    def forward(self, X, target=None):
        B, T = X.shape
        x = self.C(X) + self.pos(torch.arange(T, device=device))
        y = self.blocks(x)
        logit = self.proj(y)
        if target is None:
            return logit
        else:
            loss = F.cross_entropy(logit.reshape(-1, vocab_size), target.reshape(-1))
            return logit, loss
    
    def generate(self, x):
        logits = self.forward(x)
        prob = F.softmax(logits[:,-1,:], dim=-1)
        idx_next = torch.multinomial(prob, 1)
        return idx_next
    
    def generate_n(self, x, n):
        for _ in range(n):
            x = torch.cat((x, self.generate(x)), dim=1)
        return x
    

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = Headss()
        self.MLP = MLP()
        self.LN1 = nn.LayerNorm(d, epsilon)
        self.LN2 = nn.LayerNorm(d, epsilon)

    def forward(self, x):
        B, T, dd = x.shape
        x = x + self.heads(self.LN1(x))
        x = x + self.MLP(self.LN2(x))
        return x
    

class Headss(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkvs = nn.Linear(d,3*num_heads*dK)
        self.WO = nn.Linear(num_heads*dK, d)
        
    def forward(self, x):

        B = x.shape[0]
        T = x.shape[1]
        QKVs = self.qkvs(x)
        Qs, Ks, Vs = QKVs.view(B, T, num_heads, -1).split(dK, dim=-1)
        M = torch.zeros(T,T, device=device).masked_fill(torch.tril(torch.ones(T, T, device=device))==0, -torch.inf)
        scores = F.softmax(torch.transpose(Qs, 1, 2) @ torch.transpose(torch.transpose(Ks, 1, 2), -2 , -1) * dK**-0.5 + M, dim = -1)
        out = scores @ torch.transpose(Vs, 1, 2)
        return self.WO(torch.transpose(out,1,2).reshape(B, T, -1))
        
class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head().to(device) for _ in range(num_heads)])
        self.O = nn.Linear(num_heads*dK, d)

    def forward(self, x):
        return self.O(torch.cat([self.heads[i](x) for i in range(num_heads)], dim=-1))
        


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(d,3*dK)
        
    def forward(self, x):
        B = x.shape[0]
        T = x.shape[1]
        M = torch.zeros(T,T, device=device).masked_fill(torch.tril(torch.ones(T, T, device=device))==0, -torch.inf)
        QKV = self.qkv(x)
        scores = F.softmax(QKV[:,:,:dK] @ torch.transpose(QKV[:,:,dK:2*dK], -2, -1) * dK**-0.5 + M, dim = -1)
        out = scores @ QKV[:, :, 2*dK:]
        return out



class MLP(nn.Module): 
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d,4*d, bias=True), nn.ReLU(), nn.Linear(4*d,d, bias=True))
        
    def forward(self ,x):
        return self.mlp(x)