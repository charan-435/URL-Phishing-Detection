from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, LSTM, Bidirectional, Embedding, MultiHeadAttention,
    Flatten, Dropout, Input, LayerNormalization
)


class AttComplex:
    def __init__(self, embed_dim, sequence_length):
        self.embed_dim = embed_dim
        self.seq_len = sequence_length

    def build(self, char_index):
        vocab_size = len(char_index)
        print(f"[att_complex] vocab size: {vocab_size}")

        inputs = Input(shape=(self.seq_len,), name="input")
        x = Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len, name="embedding")(inputs)

        # block 1: lstm -> bidirectional lstm -> self-attention
        x = LSTM(128, return_sequences=True, name="lstm1")(x)
        x = Dropout(0.2, name="drop1")(x)

        x = Bidirectional(LSTM(128, return_sequences=True), name="bilstm1")(x)
        x = Dropout(0.2, name="drop2")(x)

        # self-attention + residual connection + layer norm
        # (the residual helps with training stability)
        attn1 = MultiHeadAttention(num_heads=4, key_dim=self.embed_dim, name="attention1")(x, x)
        x = LayerNormalization(name="norm1")(x + attn1)
        x = Dropout(0.2, name="drop3")(x)

        # block 2: another bilstm -> lstm -> attention stack
        x = Bidirectional(LSTM(128, return_sequences=True), name="bilstm2")(x)
        x = Dropout(0.2, name="drop4")(x)

        x = LSTM(128, return_sequences=True, name="lstm2")(x)
        x = Dropout(0.2, name="drop5")(x)

        attn2 = MultiHeadAttention(num_heads=4, key_dim=self.embed_dim, name="attention2")(x, x)
        x = LayerNormalization(name="norm2")(x + attn2)
        x = Dropout(0.2, name="drop6")(x)

        # block 3: final lstm
        x = LSTM(128, return_sequences=True, name="lstm3")(x)
        x = Dropout(0.2, name="drop7")(x)

        # flatten and output
        x = Flatten(name="flatten")(x)
        outputs = Dense(1, activation="sigmoid", name="output")(x)

        model = Model(inputs, outputs, name="att_complex")
        return model