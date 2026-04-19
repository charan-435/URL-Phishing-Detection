from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from keras_self_attention import SeqSelfAttention

# attention model based on Table 5
class AttBase:
    def __init__(self, embed_dim, seq_len):
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def build(self, char_index):
        # build architecture
        vocab_size = len(char_index)
        model = Sequential(name="att_base")
        
        # embed chars
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len))
        
        # self attention layer
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        
        # flatten and final dense
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))
        
        return model