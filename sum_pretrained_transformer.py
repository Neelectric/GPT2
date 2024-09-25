import math
import json
from tqdm import tqdm
from dataclasses import dataclass
import torch
import torch.backends
import torch.nn as nn
from torch.nn import functional as F
from spt_tokenizer import SPTTokenizer
import random

# -------------------------------------------- #

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query and value projections for all heads, but batched!
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #technically a mask not a bias but follows HF/OAI naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate q, k and v for all heads in batch and move head forward to be the batch
        # nh is num_heads, hs is head_size, C (number of channels) is nh * ns
        # so for GPT-2 (124M), n_head=12, hs=64, nh*hs=C=768 channels in transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') #gpt2 used tanh approximation, dan hendrycks suggested in github comment, nowadays irrelevant
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        

@dataclass
class SPTConfig:
    block_size: int = 16
    vocab_size: int = 17
    print(f"VOCAB SIZE IS AT {vocab_size}")
    n_layer: int = 1
    n_head: int = 4
    n_embd: int = 8 #512 seems to work well. 256 can also generalize w bsz=< 1024. 128 or 64 works with 256 bsz and lr 8e-4. for 32 we need lr 8e-3. for 8 or 16 we need lr 1e-2.

class SPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # weight token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # weight positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # layers
            ln_f = nn.LayerNorm(config.n_embd), # final layernorm, introduced by GPT2 paper
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # final classification head

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        # sometimes idx is just a single sequence, so we unsqueeze to make it a batch of size 1:
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        # print(logits[0][0])
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # function does not like multi-dim tensors, so we flatten them to be BxT for all inputs and all targets
        return logits, loss
        # return logits

    def generate(self, input_ids, max_length=10):
        while True:
            if (len(input_ids) >= max_length) or (input_ids[-1] == 16):
                input_ids = input_ids.tolist()
                return input_ids
            logits = self(input_ids)
            logits = logits[0]
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, min(50, self.config.vocab_size), dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix).squeeze(-1)
            input_ids = torch.cat((input_ids, xcol), dim=0)
    
    def answer(self, prompt, max_length=10):
        tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        input_ids = tokens.to(self.device)
        output_ids = self.generate(input_ids, max_length=max_length)
        decoded = self.tokenizer.decode(output_ids)
        # print(decoded)
        return decoded

class DataLoaderLite:
    def __init__(self, B, T, data_location):
        self.B = B
        self.T = T
        vocab_path = 'tokenizer/vocab.json'
        tokenizer = SPTTokenizer(vocab_path)
        with open(data_location, 'r') as f:
            text = json.load(f)

        random.shuffle(text)
        num_eval = int(0.1 * len(text))
        eval_raw, train_raw = text[0:num_eval], text[num_eval+1:]
        self.trainset_size = len(train_raw)
        train = " ".join(train_raw)
        eval = " ".join(eval_raw)
        self.tokens_train = tokenizer(train, return_tensors="pt")["input_ids"][0]
        self.eval_raw = eval_raw
        self.tokens_eval = tokenizer(eval, return_tensors="pt")["input_ids"][0]
        print(f"loaded {len(self.tokens_train)} tokens")
        print(f"1 epoch = {len(self.tokens_train) // (B * T)} batches")
        self.current_position_train = 0

    def next_batch_train(self):
        B, T = self.B, self.T
        buf = self.tokens_train[self.current_position_train : self.current_position_train + B*T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the current position in tensor
        self.current_position_train += B * T
        # if loading next batch would be out of bounds, reset
        if self.current_position_train + (B * T + 1) > len(self.tokens_train):
            self.current_position_train = 0
        return x,y
    
    def next_batch_eval(self):
        B, T = self.B, self.T
        buf = self.tokens_eval[self.current_position_eval : self.current_position_eval + B*T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the current position in tensor
        self.current_position_eval += B * T
        # if loading next batch would be out of bounds, reset
        if self.current_position_eval + (B * T + 1) > len(self.tokens_eval):
            self.current_position_eval = 0
        return x,y