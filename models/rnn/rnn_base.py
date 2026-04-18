from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding


class RnnBase:
    def __init__(self, embed_dim, sequence_length):
        self.embed_dim = embed_dim
        self.seq_len = sequence_length

    def build(self, char_index):
        vocab_size = len(char_index)
        print(f"[rnn_base] vocab size: {vocab_size}")

        model = Sequential(name="rnn_base")
        # embed characters into vectors
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len, name="embedding"))
        # single lstm layer to capture sequential patterns
        model.add(LSTM(128, name="lstm"))
        # binary output
        model.add(Dense(1, activation="sigmoid", name="output"))

        return model