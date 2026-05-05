import torch
import torch.nn as nn
import random


# ==================================================
# ENCODER
# ==================================================

class Encoder(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            input_dim,
            embedding_dim
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        x = x.unsqueeze(1)

        embedded = self.dropout(
            self.embedding(x)
        )

        outputs, (hidden, cell) = self.lstm(embedded)

        return hidden, cell


# ==================================================
# DECODER
# ==================================================

class Decoder(nn.Module):

    def __init__(
        self,
        output_dim,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout
    ):

        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(
            output_dim,
            embedding_dim
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_dim
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, hidden, cell):

        x = x.unsqueeze(0).unsqueeze(1)

        embedded = self.dropout(
            self.embedding(x)
        )

        embedded = embedded.squeeze(2)

        outputs, (hidden, cell) = self.lstm(
            embedded,
            (hidden, cell)
        )

        prediction = self.fc(outputs.squeeze(0))

        return prediction, hidden, cell


# ==================================================
# SEQ2SEQ MODEL
# ==================================================

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device


    def forward(
        self,
        source,
        target,
        teacher_forcing_ratio=0.5
    ):

        target_len = len(target)

        target_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(
            target_len,
            1,
            target_vocab_size
        ).to(self.device)

        hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_len):

            output, hidden, cell = self.decoder(
                x,
                hidden,
                cell
            )

            outputs[t] = output

            best_guess = output.argmax(1)

            teacher_force = random.random() < teacher_forcing_ratio

            x = target[t] if teacher_force else best_guess

        return outputs