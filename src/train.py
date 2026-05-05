# ==================================================
# TRAINING SCRIPT
# ==================================================

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data import (
    data,
    eng_vocab,
    fr_vocab,
    tokenize
)

from model import (
    Encoder,
    Decoder,
    Seq2Seq
)

# ==================================================
# DEVICE CONFIGURATION
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# ==================================================
# CONVERT DATA TO TENSORS
# ==================================================

pairs = []

for eng, fr in data:

    src_tokens = tokenize(eng, eng_vocab)
    trg_tokens = tokenize(fr, fr_vocab)

    src_tensor = torch.tensor(src_tokens)
    trg_tensor = torch.tensor(trg_tokens)

    pairs.append((src_tensor, trg_tensor))

# ==================================================
# HYPERPARAMETERS
# ==================================================

input_dim = len(eng_vocab)
output_dim = len(fr_vocab)

embedding_dim = 64
hidden_dim = 128

num_layers = 1
dropout = 0.2

# ==================================================
# MODEL INITIALIZATION
# ==================================================

encoder = Encoder(
    input_dim,
    embedding_dim,
    hidden_dim,
    num_layers,
    dropout
)

decoder = Decoder(
    output_dim,
    embedding_dim,
    hidden_dim,
    num_layers,
    dropout
)

model = Seq2Seq(
    encoder,
    decoder,
    device
).to(device)

print(model)

# ==================================================
# LOSS FUNCTION & OPTIMIZER
# ==================================================

pad_idx = fr_vocab["<pad>"]

criterion = nn.CrossEntropyLoss(
    ignore_index=pad_idx
)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# ==================================================
# TRAINING LOOP
# ==================================================

num_epochs = 300

losses = []

model.train()

for epoch in range(num_epochs):

    epoch_loss = 0

    for src, trg in pairs:

        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(src, trg)

        # Reshape outputs
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)

        trg = trg[1:].view(-1)

        # Calculate loss
        loss = criterion(output, trg)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(pairs)

    losses.append(avg_loss)

    if (epoch + 1) % 20 == 0:

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Loss: {avg_loss:.4f}")
        print()

# ==================================================
# SAVE MODEL
# ==================================================

torch.save(
    model.state_dict(),
    "seq2seq_model.pth"
)

print("Model saved successfully!")

# ==================================================
# PLOT LOSS
# ==================================================

plt.figure(figsize=(8,5))

plt.plot(losses)

plt.title("Training Loss Over Epochs")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.show()