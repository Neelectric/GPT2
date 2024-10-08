# Let's train a Sum Pretrained Transformer with Neel Nanda's librar
# System imports

# External imports
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# Local imports
from sum_pretrained_transformer import SPT, SPTConfig, DataLoaderLite
from spt_tokenizer import SPTTokenizer

# Environment prep
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.mps.manual_seed(42)


# ------------------------------------------TRAINING-----------------------------------------------------------
# attempt to auto recognize the device!
device = "cpu"
if torch.cuda.is_available(): device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
print(f"using device {device}")


# Model parameters
n_layers = 1
d_vocab = 12
n_heads = 3 #@param
d_model = ( 512 // n_heads ) * n_heads    # About 512, and divisible by n_heads
d_head = d_model // n_heads
d_mlp = 4 * d_model
# Other seeds used 383171 # 387871 (good coloring) # 129000
seed = 42 #@param

#@markdown Data
n_digits = 5 #@param
n_ctx = 3 * n_digits + 3
act_fn = 'relu'
batch_size = 64 #@param

#@markdown Optimizer
lr = 0.00008 #@param
weight_decay = 0.1 #@param
n_epochs = 2000

# Special tokens
PLUS_INDEX = 10
EQUALS_INDEX = 11