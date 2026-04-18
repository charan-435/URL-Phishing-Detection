from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding

# bidirectional rnn model
class BrnnBase:
    def __init__(self, embed_dim, seq_len):
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def build(self, char_index):
        # build model
        vocab_size = len(char_index)
        model = Sequential(name="brnn_base")
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len))
        
        # bidirectional lstm
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(1, activation="sigmoid"))
        return model