# LLM-basics
Basic exercises in LLM architecture (see below) trained on a local GPU. Codes and description here were written by me. A wise guy said, you can outsource your thinking, but not your understanding.

The repo contains a basic multi-head and multi layer transformer (class LLM, subclass of nn.Module) with hyperparameters:

B: batch size
T: sequence length
d: size of embedding vector space
d_K: size of attention vector space
num_heads: number of attention heads
num_layers: number of whole blocks. Each block is attention+MLP (both with residual layer norm)
steps: number of training steps
lr = 5e-4
epsilon = 1e-8


It uses my BPEtokenizer class written and trained on a local text corpus.








I check what basic tweaks does to performance:



1. Add a choice of positional vs. rotary (RoPE).

2. Supervised Fine Tuning (SFT)  on 

3. Basic mech-interp:
