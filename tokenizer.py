import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import matplotlib.pyplot as plt



with open('gutenberg.txt', 'r') as ff:
    dat = ff.read()


special_tokens = ["<bos>", "<eos>"]
def tokenize(txt):
    pieces = re.split("(<bos>|<eos>)", txt)
    tokens = []
    for el in pieces:
        if el == '' or el is None:
            continue
        elif el in special_tokens:
            tokens.append(el)
        else:
            tokens.extend(list(el))
    return tokens


tokens = tokenize(dat)
vocab = sorted(set(tokens))
vocab_size = len(vocab)
itos = {i:s for i,s in enumerate(vocab)}
stoi = {s:i for i, s in enumerate(vocab)}


def decode(tok_ids):
    text = []
    for id in tok_ids:
        text.append(itos[id])
    print("".join(text))


def encode(txt):
    tokens = tokenize(txt)
    return [stoi[tok] for tok in tokens]






token_list = encode(dat)








