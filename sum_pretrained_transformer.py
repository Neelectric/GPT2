import math
from dataclasses import dataclass
import torch
import torch.backends
import torch.nn as nn
from torch.nn import functional as F
from spt_tokenizer import SPTTokenizer

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
    block_size: int = 1024
    vocab_size: int = 107
    print(f"VOCAB SIZE IS AT {vocab_size}")
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 768

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
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # function does not like multi-dim tensors, so we flatten them to be BxT for all inputs and all targets
        return logits, loss
        # return logits
    
# ---------------------------------------------------------------------------------------------------------
# attempt to auto recognize the device!
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device {device}")



class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        text = [
        "<bos> 18 + 19 = 37 <eos>",
        "<bos> 2 + 43 = 45 <eos>",
        "<bos> 3 + 4 = 7 <eos>",
        "<bos> 1 + 2 = 3 <eos>",
        ]
        text = " ".join(text)
        vocab_path = 'tokenizer/vocab.json'
        tokenizer = SPTTokenizer(vocab_path)
        self.tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the current position in tensor
        self.current_position += B * T
        # if loading next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x,y

train_loader = DataLoaderLite(2, 7)


# get logits
model = SPT(SPTConfig())
model.to(device)
# for loss: vocab size is like 50k. at initialisation we hope every token gets uniform logits. so they should all be 1/50k. 
# cross-entropy loss is just negative log likelihood

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # easy gains: decrease weights for different language tokens!
for i in range(500):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad() # always need to start with 0 gradient
    logits, loss = model(x, y)
    loss.backward() # this adds to gradients! which is why we need to zero_grad
    optimizer.step() # this actually updates the params
    print(f"step {i}, loss: {loss.item()}") #we use .item() because this is a tensor with a single element that lives on .device. .item() sends it to cpu

import sys; sys.exit(0)

num_return_sequences = 5
max_length = 30
    
model = SPT.from_pretrained('gpt2')
print('didnt crash!')
model.eval()
model.to(device)

# prefix tokens
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)
# print(x)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    logits = model(x)
    # print(logits)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    ix = torch.multinomial(topk_probs, 1)
    xcol = torch.gather(topk_indices, -1, ix)
    x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)