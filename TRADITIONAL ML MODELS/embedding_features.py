import json
import numpy as np

# load trained embeddings (from DL model)
def load_embeddings(path):
    return json.load(open(path))


def url_to_embedding(url, char_embeddings):
    vectors = []

    for ch in url:
        if ch in char_embeddings:
            vectors.append(char_embeddings[ch])

    if len(vectors) == 0:
        return np.zeros(len(next(iter(char_embeddings.values()))))

    return np.mean(np.array(vectors), axis=0)