from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding


class BrnnComplex:
    def __init__(self, embed_dim, sequence_length):
        self.embed_dim = embed_dim
        self.seq_len = sequence_length

    def build(self, char_index):
        vocab_size = len(char_index)
        print(f"[brnn_complex] vocab size: {vocab_size}")

        model = Sequential(name="brnn_complex")
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len, name="embedding"))

        # block 1 - three layers of bidirectional lstms
        model.add(Bidirectional(LSTM(64, return_sequences=True), name="bilstm1"))
        model.add(Dropout(0.2, name="drop1"))

        model.add(Bidirectional(LSTM(64, return_sequences=True), name="bilstm2"))
        model.add(Dropout(0.2, name="drop2"))

        model.add(Bidirectional(LSTM(64, return_sequences=True), name="bilstm3"))
        model.add(Dropout(0.2, name="drop3"))

        # block 2 - three more bidirectional layers
        model.add(Bidirectional(LSTM(64, return_sequences=True), name="bilstm4"))
        model.add(Dropout(0.2, name="drop4"))

        model.add(Bidirectional(LSTM(64, return_sequences=True), name="bilstm5"))
        model.add(Dropout(0.2, name="drop5"))

        model.add(Bidirectional(LSTM(64, return_sequences=True), name="bilstm6"))
        model.add(Dropout(0.2, name="drop6"))

        # final layer - no return_sequences so this collapses down to a single vector
        model.add(Bidirectional(LSTM(128), name="bilstm_final"))
        model.add(Dropout(0.2, name="drop_final"))

        model.add(Dense(1, activation="sigmoid", name="output"))

        return model