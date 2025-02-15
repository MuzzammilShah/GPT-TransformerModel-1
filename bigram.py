import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

with open('cleaned_dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))



# ======================
# This was my output lol:
# ======================

# step 0: train loss 4.8967, val loss 4.8902
# step 300: train loss 2.8335, val loss 2.8190
# step 600: train loss 2.5418, val loss 2.5389
# step 900: train loss 2.4926, val loss 2.4801
# step 600: train loss 2.5418, val loss 2.5389
# step 900: train loss 2.4926, val loss 2.4801
# step 1200: train loss 2.4656, val loss 2.4609
# step 1500: train loss 2.4646, val loss 2.4421
# step 1200: train loss 2.4656, val loss 2.4609
# step 1500: train loss 2.4646, val loss 2.4421
# step 1800: train loss 2.4539, val loss 2.4479
# step 2100: train loss 2.4395, val loss 2.4500
# step 1500: train loss 2.4646, val loss 2.4421
# step 1800: train loss 2.4539, val loss 2.4479
# step 2100: train loss 2.4395, val loss 2.4500
# step 1800: train loss 2.4539, val loss 2.4479
# step 2100: train loss 2.4395, val loss 2.4500
# step 2400: train loss 2.4451, val loss 2.4301
# step 2100: train loss 2.4395, val loss 2.4500
# step 2400: train loss 2.4451, val loss 2.4301
# step 2700: train loss 2.4464, val loss 2.4290
# step 2400: train loss 2.4451, val loss 2.4301
# step 2700: train loss 2.4464, val loss 2.4290
# step 2700: train loss 2.4464, val loss 2.4290



# Shelug llee, evonk Bavey abld t sin he ht le tod Bl, atorend, coanen ubixince^Pry d bope thingengrmee, weo urmu'ss thohelplerroptathe Squnt me, I le. cextumit, y welare Pre haip756Prrnd ban s, ft  millexpoten ane YoYe sks. ckn dly, y y d gle cor ptat sme ghotharyom woutooofupederernoung et s woound A J). w sw I Rong, ts tho atapedowe theld caw at athent Shelug llee, evonk Bavey abld t sin he ht le tod Bl, atorend, coanen ubixince^Pry d bope thingengrmee, weo urmu'ss thohelplerroptathe Squnt me, I le. cextumit, y welare Pre haip756Prrnd ban s, ft  millexpoten ane YoYe sks. ckn dly, y y d gle cor ptat sme ghotharyom woutooofupederernoung et s woound A J). w sw I Rong, ts tho atapedowe theld caw at athent thent n frlds cot t an.
# 56Prrnd ban s, ft  millexpoten ane YoYe sks. ckn dly, y y d gle cor ptat sme ghotharyom woutooofupederernoung et s woound A J). w sw I Rong, ts tho atapedowe theld caw at athent thent n frlds cot t an.

# thent n frlds cot t an.

# Du me -Sis wh idr? imbeve?

# Cand olery tis.

# Thack! amey, imar unad shetid houlid Vind caioghadmp dis parrong y w! to