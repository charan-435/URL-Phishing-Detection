from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Embedding

# standard cnn model
class CnnBase:
    def __init__(self, embed_dim, seq_len):
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def build(self, char_index):
        # build cnn with embedding
        vocab_size = len(char_index)
        model = Sequential(name="cnn_base")
        
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len))
        model.add(Conv1D(128, 3, activation="tanh"))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))
        return model
