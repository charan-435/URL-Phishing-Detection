import json
from tensorflow.keras.preprocessing.text import Tokenizer

# read training urls
def get_urls(path):
    url_list = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                # clean url
                u = parts[1].lower().replace("http://", "").replace("https://", "").replace("www.", "")
                url_list.append(u)
    return url_list

def run():
    # load train data
    urls = get_urls("dataset/train/train.txt")

    # setup character tokenizer
    tokener = Tokenizer(lower=True, char_level=True, oov_token="-n-")
    tokener.fit_on_texts(urls)

    # save mapping
    with open("dataset/char_index", "w") as f:
        f.write(json.dumps(tokener.word_index))

    print(f"done: {len(tokener.word_index)} chars")

if __name__ == '__main__':
    run()