from data_loader import load_data_embeddings
from embedding_features import load_embeddings
from models import get_model
from sklearn.metrics import accuracy_score, classification_report


TRAIN_PATH = "dataset/train/train.txt"
VAL_PATH   = "dataset/val/val.txt"
TEST_PATH  = "dataset/test/test.txt"
# Load embeddings (from DL model output)
char_embeddings = load_embeddings("char_embeddings.json")

# Load data
X_train, y_train = load_data_embeddings(TRAIN_PATH, char_embeddings)
X_val, y_val = load_data_embeddings(VAL_PATH , char_embeddings)
X_test, y_test = load_data_embeddings(TEST_PATH, char_embeddings)

# Model
model = get_model("RF")

# Train
model.fit(X_train, y_train)

# Evaluate
print("Validation:", accuracy_score(y_val, model.predict(X_val)))
print("Test:", accuracy_score(y_test, model.predict(X_test)))

print(classification_report(y_test, model.predict(X_test)))