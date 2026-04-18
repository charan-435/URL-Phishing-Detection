from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input


class AnnBase:
    """
    Artificial Neural Network (ANN) that uses hand-crafted numeric features.
    These features (length, count of special chars, entropy, etc.) are 
    extracted from the URL string before being fed into this network.
    """

    def __init__(self, feature_count):
        # feature_count will be the number of metrics we extract per URL (approx 28)
        self.feature_count = feature_count

    def build(self):
        print(f"[ann_base] input feature count: {self.feature_count}")

        model = Sequential(name="ann_base")
        model.add(Input(shape=(self.feature_count,)))

        # a couple of standard dense layers with relu
        model.add(Dense(64, activation="relu", name="dense1"))
        model.add(Dropout(0.2, name="dropout1"))  # prevent overfitting
        
        model.add(Dense(32, activation="relu", name="dense2"))
        
        # binary output: phishing (0) or legitimate (1)
        model.add(Dense(1, activation="sigmoid", name="output"))

        return model
