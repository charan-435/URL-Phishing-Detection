from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout,
    Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    BatchNormalization, Embedding
)


class CnnComplex:
    def __init__(self, embed_dim, sequence_length):
        self.embed_dim = embed_dim
        self.seq_len = sequence_length

    def build(self, char_index):
        vocab_size = len(char_index)
        print(f"[cnn_complex] vocab size: {vocab_size}")

        model = Sequential(name="cnn_complex")

        # character embedding layer
        # tip: keep embed_dim small (16-32) when running on cpu
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len, name="embedding"))

        # block 1 - detect short local patterns (kernel size 3)
        model.add(Conv1D(64, 3, activation="relu", padding="same", name="conv1"))
        model.add(BatchNormalization(name="bn1"))
        model.add(MaxPooling1D(2, name="pool1"))
        model.add(Dropout(0.2, name="drop1"))

        # block 2 - detect slightly longer patterns (kernel size 5)
        model.add(Conv1D(128, 5, activation="relu", padding="same", name="conv2"))
        model.add(BatchNormalization(name="bn2"))
        model.add(MaxPooling1D(2, name="pool2"))
        model.add(Dropout(0.2, name="drop2"))

        # block 3 - higher level features
        model.add(Conv1D(256, 3, activation="relu", padding="same", name="conv3"))
        model.add(BatchNormalization(name="bn3"))
        model.add(Dropout(0.3, name="drop3"))

        # collapse the whole sequence into a single vector
        model.add(GlobalMaxPooling1D(name="global_pool"))

        # classifier head
        model.add(Dense(128, activation="relu", name="dense1"))
        model.add(Dropout(0.3, name="drop4"))
        model.add(Dense(1, activation="sigmoid", name="output"))

        return model