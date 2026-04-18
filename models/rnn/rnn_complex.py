from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding

# stacked bidirectional lstm model
class RnnComplex:
    def __init__(self, embed_dim, seq_len):
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def build(self, char_index):
        # build deep rnn
        vocab_size = len(char_index)
        model = Sequential(name="rnn_complex")
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len))

        # initial lstm
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))

        # stacking bidirectional layers
        for size in [64, 64, 64, 128, 128]:
            model.add(Bidirectional(LSTM(size, return_sequences=True)))
            model.add(Dropout(0.2))

        # final pooling
        model.add(LSTM(128))
        model.add(Dropout(0.2))

        model.add(Dense(1, activation="sigmoid"))
        return model