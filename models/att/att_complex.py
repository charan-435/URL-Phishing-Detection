from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, MultiHeadAttention, Flatten, Dropout, Input, LayerNormalization

# deep attention and rnn model
class AttComplex:
    def __init__(self, embed_dim, seq_len):
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def build(self, char_index):
        # build complex model
        vocab_size = len(char_index)
        inputs = Input(shape=(self.seq_len,))
        x = Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len)(inputs)

        # block 1: lstm + attention
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        
        # residual attention block
        attn1 = MultiHeadAttention(num_heads=4, key_dim=self.embed_dim)(x, x)
        x = LayerNormalization()(x + attn1)
        x = Dropout(0.2)(x)

        # block 2: more rnn + attention
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = LSTM(128, return_sequences=True)(x)
        
        attn2 = MultiHeadAttention(num_heads=4, key_dim=self.embed_dim)(x, x)
        x = LayerNormalization()(x + attn2)
        x = Dropout(0.2)(x)

        # final part
        x = LSTM(128, return_sequences=True)(x)
        x = Flatten()(x)
        outputs = Dense(1, activation="sigmoid")(x)

        return Model(inputs, outputs, name="att_complex")