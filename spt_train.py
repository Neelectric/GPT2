# Let's train a Sum Pretrained Transformer
# System imports

# External imports
import torch
from tqdm import tqdm

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


# MODEL SETUP
model = SPT(SPTConfig())
model.to(device)
model.device = device
vocab_path = 'tokenizer/vocab.json'
tokenizer = SPTTokenizer(vocab_path)
model.tokenizer = tokenizer


# HYPERPARAMETERS AND UTILITIES FOR TRAINING
batch_size = 64
train_loader = DataLoaderLite(batch_size, 10)
learning_rate = 8e-5
trainset_size = train_loader.trainset_size
epochs = 500
max_steps = epochs * (trainset_size) // batch_size
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # easy gains: decrease weights for different language tokens!


# TRAINING BEGINS
for i in tqdm(range(max_steps), dynamic_ncols=True):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad() # always need to start with 0 gradient
    logits, loss = model(x, y)
    loss.backward() # this adds to gradients! which is why we need to zero_grad
    optimizer.step() # this actually updates the params
    if i % 500 == 0:
        tqdm.write(f"step {i}, loss: {loss.item():.4f}") #we use .item() because this is a tensor with a single element that lives on .device. .item() sends it to cpu

# EVALUATE
model.eval()
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