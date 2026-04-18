from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, GlobalMaxPooling1D, Embedding


class CnnBase:
    def __init__(self, embed_dim, sequence_length):
        self.embed_dim = embed_dim
        self.seq_len = sequence_length

    def build(self, char_index):
        vocab_size = len(char_index)
        print(f"[cnn_base] vocab size: {vocab_size}")

        model = Sequential(name="cnn_base")
        # embed each character into a dense vector
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len, name="embedding"))
        # single conv layer to detect local patterns
        model.add(Conv1D(128, 3, activation="tanh", name="conv1"))
        model.add(Flatten(name="flatten"))
        # binary output: phishing or not
        model.add(Dense(1, activation="sigmoid", name="output"))

        return model
