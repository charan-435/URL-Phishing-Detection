from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Dropout


class AnnComplex:
    """
    ANN that works directly on character sequences (like the CNN/RNN models).
    It treats the URL as a sequence of character IDs, embeds them, 
    and then uses dense layers to classify.
    """

    def __init__(self, embed_dim, sequence_length):
        self.embed_dim = embed_dim
        self.seq_len = sequence_length

    def build(self, char_index):
        vocab_size = len(char_index)
        print(f"[ann_complex] vocab size: {vocab_size}")

        model = Sequential(name="ann_complex")
        
        # embed each character into a dense vector
        # (vocab_size + 1 because of the padding token)
        model.add(Embedding(vocab_size + 1, self.embed_dim, input_length=self.seq_len, name="embedding"))
        
        # for a "complex" ANN, we flatten the sequence and use deep dense layers
        model.add(Flatten(name="flatten"))
        
        model.add(Dense(128, activation="relu", name="dense1"))
        model.add(Dropout(0.3, name="dropout1"))
        
        model.add(Dense(64, activation="relu", name="dense2"))
        
        # binary output: phishing or legitimate
        model.add(Dense(1, activation="sigmoid", name="output"))

        return model
