# ==================================================
# INFERENCE SCRIPT
# ==================================================

import torch

from data import (
    eng_vocab,
    fr_vocab,
    fr_idx_to_word,
    tokenize
)

from model import (
    Encoder,
    Decoder,
    Seq2Seq
)

# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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
# LOAD MODEL
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

model.load_state_dict(
    torch.load("seq2seq_model.pth")
)

model.eval()

print("Model loaded successfully!")

# ==================================================
# TRANSLATION FUNCTION
# ==================================================

def translate_sentence(sentence):

    tokens = tokenize(sentence, eng_vocab)

    src_tensor = torch.tensor(tokens).to(device)

    with torch.no_grad():

        hidden, cell = model.encoder(src_tensor)

    outputs = []

    x = torch.tensor(
        [fr_vocab["<sos>"]]
    ).to(device)

    for _ in range(20):

        with torch.no_grad():

            output, hidden, cell = model.decoder(
                x,
                hidden,
                cell
            )

        best_guess = output.argmax(1).item()

        if best_guess == fr_vocab["<eos>"]:
            break

        outputs.append(
            fr_idx_to_word[best_guess]
        )

        x = torch.tensor(
            [best_guess]
        ).to(device)

    return " ".join(outputs)

# ==================================================
# TEST SENTENCES
# ==================================================

test_sentences = [

    "i am happy",
    "he is tall",
    "good morning",
    "i love you"

]

for sentence in test_sentences:

    translation = translate_sentence(sentence)

    print(f"Input      : {sentence}")
    print(f"Translation: {translation}")
    print()