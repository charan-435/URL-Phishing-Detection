from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention, Input, GlobalAveragePooling1D

# basic attention model
class AttBase:
    def __init__(self, embed_dim, seq_len):
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def build(self, char_index):
        # uses functional api for attention
        vocab_size = len(char_index)
        inputs = Input(shape=(self.seq_len,))

        # embed chars
        x = Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len)(inputs)

        # multi-head attention
        x = MultiHeadAttention(num_heads=4, key_dim=self.embed_dim)(x, x)

        # mean pooling 
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation="sigmoid")(x)

        return Model(inputs, outputs, name="att_base")