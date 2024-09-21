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
    block_size: int = 1024
    vocab_size: int = 107
    print(f"VOCAB SIZE IS AT {vocab_size}")
    n_layer: int = 8
    n_head: int = 16
    n_embd: int = 512

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
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix).squeeze(-1)
            input_ids = torch.cat((input_ids, xcol), dim=0)
    
    def answer(self, prompt, max_length=10):
        tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        input_ids = tokens.to(device)
        # print(input_ids)
        output_ids = self.generate(input_ids, max_length=max_length)
        # print(output_ids)
        decoded = self.tokenizer.decode(output_ids)
        print(decoded)
        return decoded
    
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
        with open('datasets/sum_dataset.json', 'r') as f:
            text = json.load(f)
        random.shuffle(text)
        # text = text[:20]
        # print(f"OVERRIDING TEXT TO BE {len(text)} SAMPLES")
        num_eval = int(0.1 * len(text))
        eval, train = text[0:num_eval], text[num_eval+1:]
        self.trainset_size = len(train)
        train = " ".join(train)
        vocab_path = 'tokenizer/vocab.json'
        tokenizer = SPTTokenizer(vocab_path)
        self.tokens = tokenizer(train, return_tensors="pt")["input_ids"][0]
        print(self.tokens[0:25])
        self.eval = eval
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
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

train_loader = DataLoaderLite(1, 10)


# MODEL SETUP
model = SPT(SPTConfig())
model.to(device)
vocab_path = 'tokenizer/vocab.json'
tokenizer = SPTTokenizer(vocab_path)
model.tokenizer = tokenizer
# for loss: vocab size is like 50k. at initialisation we hope every token gets uniform logits. so they should all be 1/50k. 
# cross-entropy loss is just negative log likelihood


# HYPERPARAMETERS FOR TRAINING
learning_rate = 1e-3
trainset_size = train_loader.trainset_size
epochs = 1
max_steps = epochs * (trainset_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # easy gains: decrease weights for different language tokens!
for i in tqdm(range(max_steps), dynamic_ncols=True):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad() # always need to start with 0 gradient
    logits, loss = model(x, y)
    loss.backward() # this adds to gradients! which is why we need to zero_grad
    optimizer.step() # this actually updates the params
    if i % 100 == 0:
        tqdm.write(f"step {i}, loss: {loss.item():.4f}") #we use .item() because this is a tensor with a single element that lives on .device. .item() sends it to cpu

# import sys; sys.exit(0)
num_return_sequences = 5
# max_length = 
model.eval()

torch.manual_seed(42)
torch.cuda.manual_seed(42)

eval_prompts = []
eval_ground_truths = []
for elt in train_loader.eval:
    eval_prompts.append(elt.split("=")[0] + "=")
    eval_ground_truths.append(elt)

num_correct = 0
for prompt, ground_truth in tqdm(zip(eval_prompts, eval_ground_truths), dynamic_ncols=True):
    prediction = model.answer(prompt)
    if prediction == ground_truth:
        num_correct += 1
        tqdm.write(f"CORRECT")

print(f"Accuracy (EM): {num_correct/len(eval_prompts):.3f}")