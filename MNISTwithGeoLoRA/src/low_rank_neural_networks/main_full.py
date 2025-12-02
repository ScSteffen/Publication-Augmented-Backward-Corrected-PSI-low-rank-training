import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BPTTIterator, WikiText2
from torchtext.vocab import build_vocab_from_iterator
import math

# Define the model architecture
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        x = self.embedding(src)
        output = self.transformer(x)
        return self.fc(output)

# Hyperparameters
vocab_size = len(TEXT.vocab)
d_model = 512
nhead = 8
num_encoder_layers = 6
dim_feedforward = 2048
max_seq_len = 35  # This depends on your specific dataset
dropout = 0.1
batch_size = 64
epochs = 10
lr = 0.001

# Load the WikiText-2 dataset
TEXT = Field(tokenize='spacy', batch_first=True)
train, valid, test = WikiText2.splits(TEXT)

# Build the vocabulary
TEXT.build_vocab(train, max_size=10000, min_freq=2)

# Create data iterators
train_iter, valid_iter, test_iter = BPTTIterator.splits(
    (train, valid, test),
    batch_size=batch_size,
    bptt_len=max_seq_len,
    repeat=False
)

# Initialize the model
model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_len, dropout)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_iter:
        optimizer.zero_grad()
        src = batch.text
        trg = batch.target
        output = model(src)
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg.view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_iter)

    print(f"Epoch {epoch+1}, Train Loss: {average_loss:.4f}")

# Validation loop
model.eval()
total_loss = 0
with torch.no_grad():
    for batch in valid_iter:
        src = batch.text
        trg = batch.target
        output = model(src)
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg.view(-1)
        loss = criterion(output, trg)
        total_loss += loss.item()

average_loss = total_loss / len(valid_iter)
print(f"Validation Loss: {average_loss:.4f}")
