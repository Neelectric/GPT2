# Let's train a Sum Pretrained Transformer
# System imports

# External imports
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

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



# HYPERPARAMETERS AND UTILITIES FOR TRAINING, EVAL DATASET PREP
batch_size = 64
num_tokens_per_sample = 10
data_location = 'datasets/sum_dataset.json'
train_loader = DataLoaderLite(B=batch_size, T=num_tokens_per_sample, data_location='datasets/sum_dataset.json')
learning_rate = 8e-5
trainset_size = train_loader.trainset_size
epochs = 200
max_steps = epochs * (trainset_size) // batch_size
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # easy gains: decrease weights for different language tokens!
eval_prompts = []
eval_ground_truths = []
for elt in train_loader.eval_raw:
    eval_prompts.append(elt.split("=")[0] + "=")
    eval_ground_truths.append(elt)


def eval_loss(eval_prompts):
    return

def eval_naive():
    model.eval()
    num_correct = 0
    for prompt, ground_truth in tqdm(zip(eval_prompts, eval_ground_truths), dynamic_ncols=True, disable=True):
        prediction = model.answer(prompt)
        if prediction == ground_truth:
            num_correct += 1
            # tqdm.write(f"CORRECT")
    EM_score = num_correct/len(eval_prompts)
    return EM_score


# TRAINING BEGINS
losses = []
accuracies = []
for i in tqdm(range(max_steps), dynamic_ncols=True):
    model.train()
    x, y = train_loader.next_batch_train()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad() # always need to start with 0 gradient
    logits, loss = model(x, y)
    loss.backward() # this adds to gradients! which is why we need to zero_grad
    optimizer.step() # this actually updates the params
    if i % 250 == 0:
        em_score_reading = eval_naive() * 100
        tqdm.write(f"step {i}, loss: {loss.item():.4f}, accuracy (EM): {em_score_reading:.2f}%") #we use .item() because this is a tensor with a single element that lives on .device. .item() sends it to cpu
    accuracies.append(em_score_reading)
    losses.append(loss.item())
    

# PLOT
print(accuracies)
fig, ax1 = plt.subplots(figsize=(10, 5))
    
# Plot losses on the left y-axis
ax1.plot(losses, label='Losses', color='blue')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Losses', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis for accuracies
ax2 = ax1.twinx()
ax2.plot(accuracies, label='Accuracies', color='green', linestyle='-')
ax2.set_ylabel('Accuracies (%)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Add title
plt.title('Training Losses and Accuracies')

# Display the plot
plt.show()