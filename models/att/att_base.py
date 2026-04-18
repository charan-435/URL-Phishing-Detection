from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, MultiHeadAttention,
    Flatten, Dropout, Input, GlobalAveragePooling1D
)


class AttBase:
    def __init__(self, embed_dim, sequence_length):
        self.embed_dim = embed_dim
        self.seq_len = sequence_length

    def build(self, char_index):
        vocab_size = len(char_index)
        print(f"[att_base] vocab size: {vocab_size}")

        # using functional api here because MultiHeadAttention needs query + value args
        inputs = Input(shape=(self.seq_len,), name="input")

        x = Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len, name="embedding")(inputs)

        # self-attention: the url attends to itself
        # (query and value are the same tensor)
        x = MultiHeadAttention(num_heads=4, key_dim=self.embed_dim, name="self_attention")(x, x)

        # average pooling to collapse sequence dimension
        x = GlobalAveragePooling1D(name="gap")(x)

        outputs = Dense(1, activation="sigmoid", name="output")(x)

        model = Model(inputs, outputs, name="att_base")
        return model