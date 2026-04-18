from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization, Embedding

# deeper cnn model 
class CnnComplex:
    def __init__(self, embed_dim, seq_len):
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def build(self, char_index):
        # multi-layer cnn 
        vocab_size = len(char_index)
        model = Sequential(name="cnn_complex")

        # start with embedding
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len))

        # conv block 1
        model.add(Conv1D(64, 3, activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.2))

        # conv block 2
        model.add(Conv1D(128, 5, activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.2))

        # conv block 3
        model.add(Conv1D(256, 3, activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # pooling and dense
        model.add(GlobalMaxPooling1D())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation="sigmoid"))
        return model