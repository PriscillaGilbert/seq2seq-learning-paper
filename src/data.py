import torch

# ==================================================
# SIMPLE ENGLISH → FRENCH DATASET
# ==================================================

data = [

    ("i am happy", "je suis heureux"),
    ("i am sad", "je suis triste"),
    ("he is tall", "il est grand"),
    ("she is kind", "elle est gentille"),
    ("i love you", "je t aime"),
    ("how are you", "comment allez vous"),
    ("i am hungry", "j ai faim"),
    ("he is strong", "il est fort"),
    ("she is smart", "elle est intelligente"),
    ("good morning", "bonjour")

]

# ==================================================
# SPECIAL TOKENS
# ==================================================

special_tokens = ["<pad>", "<sos>", "<eos>"]

eng_vocab = {}
fr_vocab = {}

# Add special tokens
for idx, token in enumerate(special_tokens):

    eng_vocab[token] = idx
    fr_vocab[token] = idx


# ==================================================
# BUILD VOCABULARY
# ==================================================

def build_vocab(sentences, vocab):

    index = len(vocab)

    for sentence in sentences:

        for word in sentence.split():

            if word not in vocab:

                vocab[word] = index
                index += 1


# Build vocabularies
build_vocab([pair[0] for pair in data], eng_vocab)
build_vocab([pair[1] for pair in data], fr_vocab)


# ==================================================
# TOKENIZATION FUNCTION
# ==================================================

def tokenize(sentence, vocab):

    tokens = [vocab["<sos>"]]

    for word in sentence.split():

        tokens.append(vocab[word])

    tokens.append(vocab["<eos>"])

    return tokens


# ==================================================
# REVERSE VOCAB
# ==================================================

eng_idx_to_word = {

    idx: word for word, idx in eng_vocab.items()

}

fr_idx_to_word = {

    idx: word for word, idx in fr_vocab.items()

}