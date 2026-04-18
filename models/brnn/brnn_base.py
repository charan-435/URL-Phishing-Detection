from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding


class BrnnBase:
    def __init__(self, embed_dim, sequence_length):
        self.embed_dim = embed_dim
        self.seq_len = sequence_length

    def build(self, char_index):
        vocab_size = len(char_index)
        print(f"[brnn_base] vocab size: {vocab_size}")

        model = Sequential(name="brnn_base")
        # embed characters
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len, name="embedding"))
        # bidirectional lstm - reads the url from both directions
        model.add(Bidirectional(LSTM(128), name="bilstm"))
        # binary classification output
        model.add(Dense(1, activation="sigmoid", name="output"))

        return model