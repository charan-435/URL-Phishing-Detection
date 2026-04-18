from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding


class RnnComplex:
    def __init__(self, embed_dim, sequence_length):
        self.embed_dim = embed_dim
        self.seq_len = sequence_length

    def build(self, char_index):
        vocab_size = len(char_index)
        print(f"[rnn_complex] vocab size: {vocab_size}")

        model = Sequential(name="rnn_complex")
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len, name="embedding"))

        # block 1 - forward lstm
        model.add(LSTM(128, return_sequences=True, name="lstm1"))
        model.add(Dropout(0.2, name="drop1"))

        # block 2 - stack of bidirectional lstms (reads url left-to-right AND right-to-left)
        model.add(Bidirectional(LSTM(64, return_sequences=True), name="bilstm1"))
        model.add(Dropout(0.2, name="drop2"))

        model.add(Bidirectional(LSTM(64, return_sequences=True), name="bilstm2"))
        model.add(Dropout(0.2, name="drop3"))

        model.add(Bidirectional(LSTM(64, return_sequences=True), name="bilstm3"))
        model.add(Dropout(0.2, name="drop4"))

        # block 3 - deeper bidirectional layers
        model.add(Bidirectional(LSTM(128, return_sequences=True), name="bilstm4"))
        model.add(Dropout(0.2, name="drop5"))

        model.add(Bidirectional(LSTM(128, return_sequences=True), name="bilstm5"))
        model.add(Dropout(0.2, name="drop6"))

        # final lstm collapses the sequence dimension
        model.add(LSTM(128, name="lstm_final"))
        model.add(Dropout(0.2, name="drop_final"))

        # output layer
        model.add(Dense(1, activation="sigmoid", name="output"))

        return model