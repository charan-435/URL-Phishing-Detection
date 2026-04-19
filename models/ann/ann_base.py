from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding

# base ann model using character sequences
class AnnBase:
    def __init__(self, embed_dim, seq_len):
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def build(self, char_index):
        # build ann with embedding
        vocab_size = len(char_index)
        model = Sequential(name="ann_base")
        
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len))
        model.add(Dense(128, activation="relu"))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))
        return model
