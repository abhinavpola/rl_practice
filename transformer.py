# %%
from tinygrad import Tensor, nn
import numpy as np

# %%
# Hyperparams

training_steps = 1000
train_split = 0.9 # 90% of the data goes to training, 10% for validation
block_size = 8 # length of sequence
batch_size = 4 # number of sequences
vocab_size = 28
head_size = 16 # size of each attention head
learning_rate = 1e-4

# %%
def tokenize(input: str) -> np.ndarray:
    """
    Tokenizes an input string.
    """
    vocab = set(input)
    mapping = {v:i for i, v in enumerate(list(vocab))}
    return np.array([mapping[char] for char in input])

tokenize("The quick brown fox jumps over the lazy dog")

# %%
with open("datasets/tiny_shakespeare.txt") as f:
    input_text = f.read()
    data = tokenize(input_text)
    n = int(train_split * len(data))
    train_data = data[:n]
    test_data = data[n:]

# %%
def get_batch(split: str) -> tuple[Tensor, Tensor]:
    """
    Given input data of shape (N,), return a randomly sampled batch of shape (batch_size, block_size)
    """
    split_data = train_data if split == "train" else test_data
    sample = np.random.randint(0, split_data.shape[0]-block_size, size=(batch_size))
    x = np.stack([split_data[i:i+block_size] for i in sample])
    y = np.stack([split_data[i+1:i+block_size+1] for i in sample])
    return Tensor(x), Tensor(y)

get_batch("train")

# %%
class SelfAttentionHead:
    def __init__(self):
        self.queries = nn.Linear(vocab_size, head_size, bias=False)
        self.keys = nn.Linear(vocab_size, head_size, bias=False)
        self.values = nn.Linear(vocab_size, head_size, bias=False)
        self.projection = nn.Linear(head_size, vocab_size, bias=False)  # Add projection layer
    
    def __call__(self, input):
        # input: (B, T, C)
        q = self.queries(input)  # (B, T, 16)
        k = self.keys(input)      # (B, T, 16)
        v = self.values(input)    # (B, T, 16)
        x = q @ k.transpose(-2, -1) * head_size**-0.5  # (B, T, T)
        tril = Tensor.tril(Tensor.ones(block_size, block_size))
        x = x.masked_fill(tril == 0, float('-inf'))
        x = Tensor.softmax(x, axis=-1) @ v  # (B, T, 16)

        x = self.projection(x)  # Project output to vocab_size -> (B, T, vocab_size)
        return x


# %%
class FeedForwardNetwork:
    def __init__(self):
        self.l1 = nn.Linear(vocab_size, head_size)
        self.l2 = nn.Linear(head_size, vocab_size)
    def __call__(self, x):
        x = self.l1(x)
        x = Tensor.relu(x)
        x = self.l2(x)
        return x

# %%
class Model:
    def __init__(self):
        self.token_embeddings = nn.Embedding(vocab_size, vocab_size)
        self.positional_embeddings = nn.Embedding(block_size, vocab_size)
        self.attention = SelfAttentionHead()
        self.ff = FeedForwardNetwork()
        self.ln = nn.LayerNorm(vocab_size)
        self.lm_head = nn.Linear(vocab_size, vocab_size)

    def __call__(self, idx, targets=None):
        token_embedding = self.token_embeddings(idx)  # (B, T, C)
        pos_embedding = self.positional_embeddings(Tensor.arange(block_size))  # (T, C)
        x = token_embedding + pos_embedding  # (B, T, C)
        x = self.ln(x)  # Add normalization before attention
        x = x + self.attention(x)
        x = self.ln(x)  # Add normalization after attention
        x = x + self.ff(x)
        logits = self.lm_head(x)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = Tensor.sparse_categorical_crossentropy(logits, targets)
        return logits, loss


# %%
m = Model()
optim = nn.optim.AdamW(nn.state.get_parameters(m), lr=learning_rate)
with Tensor.train():
    for step in range(training_steps):
        xb, yb = get_batch("train") # (B, T)
        _, loss = m(xb, yb)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 100 == 0:
            print(f"Step {step+1} | Loss: {loss.numpy()}")



