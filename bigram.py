import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters:
seed = 1337
batch_size = 32
block_size = 8
max_iter = 1000
eval_interval = 300
learning_rate = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200
n_embd = 32

# Set the seed
torch.manual_seed(seed)

# Read the data:
input_file = pathlib.Path("input.txt")
if not input_file.exists():
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        str(input_file))
with open(str(input_file), "r", encoding="utf-8") as file:
    text = file.read()

# Find the unique chars:
chars = sorted(set(text))
vocab_size = len(chars)

# Create the encoding:
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda x: "".join([itos[i] for i in x])

# encode the data:
data = torch.tensor(encode(text), dtype=torch.long)

# Split the data:
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Define the batch getter:
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size, ))
    x = torch.stack([data[i: i+block_size] for i in ix], dim=0)
    y = torch.stack([data[i+1: i+block_size+1] for i in ix], dim=0)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets):
        B, T = idx.shape
        # idx and targets are (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # shape (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device).unsqueeze(0))  # (T, C)
        x = tok_emb + pos_emb
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) long tensor of current tokens
        for i in range(max_new_tokens):
            # get the predictions:
            logits, _ = self(idx, None)
            # look at last time step:
            logits = logits[:, -1, :]  # shape (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append the sample:
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# Create the model
model = BigramLanguageModel()
m = model.to(device)

# Create the optimizer:
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

for iter in range(max_iter):
    if iter % eval_interval == 0:
        losses = estimate_loss(m)
        print(f"Iter: {iter}, Loss: {losses}")

    # Sample data
    xb, yb = get_batch("train")

    # Evaluate the loss
    _, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate from the model:
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx, 500)[0].tolist()))