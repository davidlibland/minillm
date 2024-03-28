# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + jupyter={"outputs_hidden": true}
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# + jupyter={"outputs_hidden": false}
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# + jupyter={"outputs_hidden": false}
print(f"Number of chars in text: {len(text)}")

# + jupyter={"outputs_hidden": false}
print(text[:1000])

# + jupyter={"outputs_hidden": false}
chars = sorted(set(text))
vocab_size = len(chars)
print("".join(chars))
print(f"Vocabulary size: {vocab_size}")

# + jupyter={"outputs_hidden": false}
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda x: "".join([itos[i] for i in x])

print(encode("Hello"))
print(decode(encode("Hello")))

# + jupyter={"outputs_hidden": false}
import torch

# + jupyter={"outputs_hidden": false}
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

# + jupyter={"outputs_hidden": false}
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# + jupyter={"outputs_hidden": false}
block_size = 8
train_data[:block_size+1]

# + jupyter={"outputs_hidden": false}
torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size, ))
    x = torch.stack([data[i: i+block_size] for i in ix], dim=0)
    y = torch.stack([data[i+1: i+block_size+1] for i in ix], dim=0)
    return x, y

xb, yb = get_batch("train")
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"{context.tolist()} to {target}")
        

# + jupyter={"outputs_hidden": false}
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets):
        # idx and targets are (B, T) tensors of integers
        logits = self.token_embedding_table(idx) # shape (B, T, C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) long tensor of current tokens
        for i in range(max_new_tokens):
            # get the predictions:
            logits, _ = self(idx, None)
            # look at last time step:
            logits = logits[:, -1, :] # shape (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append the sample:
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
    
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape, loss)

idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx, 100)[0].tolist()))

# + jupyter={"outputs_hidden": false}
optimizer = torch.optim.Adam(m.parameters(), lr=3e-3)
torch.manual_seed(1337)
batch_size = 32
block_size = 8
n_steps = 10000
for steps in range(n_steps):
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(loss.item())

# + jupyter={"outputs_hidden": false}
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx, 100)[0].tolist()))

# + jupyter={"outputs_hidden": false}
B, T, C = 4, 8, 2
x = torch.rand(B, T, C)
x.shape

# + jupyter={"outputs_hidden": false}
xbow = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        xpast = x[b, :t+1, :]
        xbow[b, t] = xpast.mean(dim=0)
xbow

# + jupyter={"outputs_hidden": false}
# Version 2
weights = torch.tril(torch.ones(T, T))
weights = weights / weights.sum(dim=1, keepdim=True)
xbow2 = weights.unsqueeze(dim=0) @ x

# + jupyter={"outputs_hidden": false}
# version 3:
tril = torch.tril(torch.ones(T, T))
weights = torch.zeros(T, T)
weights = weights.masked_fill(tril==0, -float("inf"))
weights = F.softmax(weights, dim=1)
xbow3 = weights.unsqueeze(dim=0) @ x

# + jupyter={"outputs_hidden": false}
xbow3 - xbow

# + jupyter={"outputs_hidden": false}

