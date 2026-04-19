from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Embedding

# complex cnn model based on Table 5
class CnnComplex:
    def __init__(self, embed_dim, seq_len):
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def build(self, char_index):
        # 17-layer architecture from paper
        vocab_size = len(char_index)
        model = Sequential(name="cnn_complex")

        # layer 1: embedding
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len))

        # layers 2-4: conv block 1
        model.add(Conv1D(128, 3, activation="relu", padding="same"))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.2))

        # layers 5-6: conv block 2
        model.add(Conv1D(128, 3, activation="relu", padding="same"))
        model.add(Dropout(0.2))

        # layers 7-9: conv block 3
        model.add(Conv1D(128, 3, activation="relu", padding="same"))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.2))

        # layers 10-11: conv block 4
        model.add(Conv1D(128, 3, activation="relu", padding="same"))
        model.add(Dropout(0.2))

        # layers 12-14: conv block 5
        model.add(Conv1D(128, 3, activation="relu", padding="same"))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.2))

        # layers 15-16: conv block 6
        model.add(Conv1D(128, 3, activation="relu", padding="same"))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.2))

        # layer 17: flatten
        model.add(Flatten())

        # final dense
        model.add(Dense(1, activation="sigmoid"))
        return model