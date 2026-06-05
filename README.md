# LLM-basics
Basic exercises in LLM architecture (see below) trained on a local GPU. Codes and description here were written by me. A wise guy said, you can outsource your thinking, but not your understanding.


######### What's in this?


The repo contains a basic multi-head and multi layer transformer (class LLM, subclass of nn.Module) with hyperparameters:

B: batch size
T: sequence length
d: size of embedding vector space
d_K: size of attention vector space (must be even if RoPE = True, i.e., rotary positional embedding is used)
num_heads: number of attention heads
num_layers: number of whole blocks. Each block is attention+MLP (both with residual layer norm)
steps: number of training steps
lr: learning rate
epsilon: used in layer norm to prevent infinity

Overview: take in a batch of T-long tokens, x ~ B * T. Embedd them (with basic position embedding or RoPE), y = Embedd(x) ~ B * T * C. Rearrange it with attention, i.e., y -> y + Att (LayerNorm (y)), where LayerNorm (y) just normalizes each y[b,t,:] across the C dimension, adding a learnable scale and bias, and Att() runs a single query attention mechanism. Then y -> y + MLP(LayerNorm(y)) where MLP is just a ReLU() feedforward layer. The loss function is relative entropy (aka, KL divergence, aka log probability).

I train (locally) on HF's "wikitext-103-raw-v1", with my BPEtokenizer trained with vocab size = 1000.

With hyperparameters:

B = 64
T = 128
d = 128
dK = 32
num_heads = 4
num_layers = 6
steps = 10000
lr: 4e-4
epsilon = 1e-8

The train and val loss obtained is 1.32 and 1.36 respectively.


######### Exercises

I check what basic tweaks does to performance:

1. Add a choice of positional vs. rotary (RoPE). I add RoPE embedding as an option to my LLM class. In the same setting as above, RoPE improves train and val loss to 1.32 and 1.36 respectively.

2. Implement KV cache for efficiency

3. Supervised Fine Tuning (SFT) on .

 The result before and after


4. Add a mixture of expert manually.



6. Basic mech-interp experiments:

7.
