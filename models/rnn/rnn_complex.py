from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

# complex rnn model with 7 stacked lstm layers
class RnnComplex:
    def __init__(self, embed_dim, seq_len):
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def build(self, char_index):
        # 7-layer architecture from paper
        vocab_size = len(char_index)
        model = Sequential(name="rnn_complex")
        
        # embed chars
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len))
        
        # initial stacking (layers 1-6 return sequences)
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        
        # final layer
        model.add(LSTM(128))
        
        model.add(Dense(1, activation="sigmoid"))
        return model