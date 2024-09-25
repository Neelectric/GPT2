# Let's train a Sum Pretrained Transformer
# System imports

# External imports
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

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

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in model: {pytorch_total_params:,}")



# HYPERPARAMETERS AND UTILITIES FOR TRAINING, EVAL DATASET PREP
batch_size = 1024
num_tokens_per_sample = 10
data_location = 'datasets/sum_dataset.json'
train_loader = DataLoaderLite(B=batch_size, T=num_tokens_per_sample, data_location='datasets/sum_dataset.json')
learning_rate = 1e-2
trainset_size = train_loader.trainset_size
epochs = 2500
max_steps = epochs * (trainset_size) // batch_size
eval_intervals = max_steps // 8
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # easy gains: decrease weights for different language tokens!
print(f"max_steps: {max_steps}, eval_intervals: {eval_intervals}, learning_rate: {learning_rate}")

eval_prompts = []
eval_ground_truths = []
for elt in train_loader.eval_raw:
    eval_prompts.append(elt.split("=")[0] + "=")
    eval_ground_truths.append(elt)

def eval_naive(print_incorrect=False):
    model.eval()
    num_correct = 0
    for prompt, ground_truth in tqdm(zip(eval_prompts, eval_ground_truths), dynamic_ncols=True, disable=True):
        prediction = model.answer(prompt)
        if prediction == ground_truth:
            num_correct += 1
        elif print_incorrect:
            print(prompt, ground_truth, prediction)
    EM_score = num_correct/len(eval_prompts)
    if print_incorrect:
        print(f"Out of {len(eval_prompts)} questions, SPT got {num_correct} correct.")
    return EM_score


# TRAINING BEGINS
losses = []
accuracies = []
accuracy_steps = []

for i in tqdm(range(max_steps), dynamic_ncols=True):
    model.train()
    x, y = train_loader.next_batch_train()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad() # always need to start with 0 gradient
    logits, loss = model(x, y)
    loss.backward() # this adds to gradients! which is why we need to zero_grad
    optimizer.step() # this actually updates the params
    if i % eval_intervals == 0:
        em_score_reading = eval_naive() * 100
        tqdm.write(f"step {i}, train loss: {loss.item():.4f}, eval accuracy (EM): {em_score_reading:.2f}%") #we use .item() because this is a tensor with a single element that lives on .device. .item() sends it to cpu
        accuracies.append(em_score_reading)
        accuracy_steps.append(i)
        losses.append(loss.item())
    
final_em_score_reading = eval_naive(print_incorrect=True)
print(f"step {i}, train loss: {loss.item():.4f}, eval accuracy (EM): {final_em_score_reading:.2f}%") 

# PLOT
fig, ax1 = plt.subplots(figsize=(10, 5))
    
# Plot losses on the left y-axis
ax1.plot(accuracy_steps, losses, label='Losses', color='blue')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Losses', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis for accuracies
ax2 = ax1.twinx()
ax2.plot(accuracy_steps, accuracies, label='Accuracies', color='green', linestyle='-', marker='o')
ax2.set_ylabel('Accuracies (%)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Set y-ticks frequency for the right y-axis
ax2.yaxis.set_major_locator(MultipleLocator(5))

# Add grid
# ax1.grid(True)
ax2.grid(True)

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')

# Add title
plt.title('Training Losses and Accuracies')

# Display the plot
plt.show()