import json
from tensorflow.keras.preprocessing.text import Tokenizer


def load_data(file_path):
    # reads the training file and returns a list of cleaned urls
    urls = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            url = parts[1]
            # lowercase and strip off the scheme + www prefix
            url = url.lower().replace("http://", "").replace("https://", "").replace("www.", "")
            urls.append(url)
    return urls


def main():
    urls = load_data("dataset/train/train.txt")

    # fit a character-level tokenizer on all the training urls
    tokener = Tokenizer(lower=True, char_level=True, oov_token="-n-")
    tokener.fit_on_texts(urls)

    # save the character index to a file so we can reload it later
    char_index = tokener.word_index
    with open("dataset/char_index", "w") as f:
        f.write(json.dumps(char_index))

    print(f"char index built - {len(char_index)} unique characters found")


if __name__ == '__main__':
    main()