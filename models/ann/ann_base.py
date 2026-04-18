from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# ann using manual features
class AnnBase:
    def __init__(self, feature_count):
        self.feature_count = feature_count

    def build(self):
        # build simple dense network
        model = Sequential(name="ann_base")
        model.add(Input(shape=(self.feature_count,)))
        
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2)) 
        
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        return model
