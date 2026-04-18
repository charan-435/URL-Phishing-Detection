from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding

# stacked bidirectional rnn model
class BrnnComplex:
    def __init__(self, embed_dim, seq_len):
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def build(self, char_index):
        # build deep bidirectional model
        vocab_size = len(char_index)
        model = Sequential(name="brnn_complex")
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len))

        # stack of layers
        for _ in range(6):
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(Dropout(0.2))

        # final pooling layer
        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(0.2))

        model.add(Dense(1, activation="sigmoid"))
        return model