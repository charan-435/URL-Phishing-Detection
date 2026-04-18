import re
import json
import numpy as np
from urllib.parse import urlparse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence as keras_sequence


# these are all the features we compute per URL
# grouped into categories so its easier to understand whats going on
feature_names = [
    # length stuff
    "url_length",        # total chars in the url
    "domain_length",     # how long is just the domain
    "path_length",       # how long is the path part
    "subdomain_count",   # how many subdomains (dots in hostname - 1)
    "path_depth",        # number of slashes in path

    # counting special characters
    "count_dots",        # dots -> phishing urls usually have loads (a.b.c.evil.com)
    "count_hyphens",     # hyphens -> spoofed domains love hyphens
    "count_underscores", # underscores
    "count_slashes",     # slashes
    "count_at",          # @ sign -> http://legit.com@evil.com trick
    "count_question",    # question marks -> query string
    "count_equals",      # equals sign
    "count_ampersand",   # ampersand
    "count_percent",     # percent -> url encoding tricks
    "count_digits",      # total number of digit characters

    # ratios
    "digit_ratio",       # digits / total length
    "letter_ratio",      # letters / total length
    "special_char_ratio",# non-alphanumeric / total length

    # entropy (how random/chaotic the string looks)
    "url_entropy",       # shannon entropy of full url
    "domain_entropy",    # shannon entropy of just the domain

    # yes/no flags
    "has_ip_address",    # 1 if it looks like an IP e.g. 192.168.1.1
    "has_port",          # 1 if theres a port number in the url
    "has_https",         # 1 if it starts with https
    "has_http",          # 1 if it starts with http://
    "has_at_symbol",     # 1 if @ appears anywhere
    "has_double_slash",  # 1 if // appears in path -> redirect trick
    "has_dash_in_domain",# 1 if domain has a dash in it
    "is_shortened",      # 1 if its a known url shortener (bit.ly etc)
]

# known shortener domains - we flag these as suspicious
shortener_list = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co",
    "buff.ly", "rebrand.ly", "cutt.ly", "is.gd", "bl.ink",
    "short.io", "tiny.cc", "lc.chat", "soo.gd", "s2r.co",
}


class FeatureExtractor:

    def __init__(self, char_index_path="../dataset/char_index"):
        # load the character tokenizer from a saved json file
        # this was made during training with build_char_index.py
        self.tokener = Tokenizer(lower=True, char_level=True, oov_token="-n-")
        self.tokener.word_index = json.loads(open(char_index_path).read())

        # storage for loaded data
        self.urls = []
        self.labels = []

    # ---- loading data ----

    def load_from_file(self, filepath):
        # reads a tab-separated file: label\turl
        # label is either "phishing" or "legitimate"
        self.urls = []
        self.labels = []

        with open(filepath, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        skipped = 0
        for line in lines:
            parts = [p for p in line.split("\t") if p]
            if len(parts) < 2:
                skipped += 1
                continue

            label_str = parts[0]
            url = parts[1]

            if label_str == "phishing":
                label = 0
            elif label_str == "legitimate":
                label = 1
            else:
                skipped += 1
                continue

            # strip off http/https so urls are consistent
            clean = url.lower().replace("http://", "").replace("https://", "")
            self.urls.append(clean)
            self.labels.append(label)

        print("loaded {} urls ({} skipped) from {}".format(len(self.urls), skipped, filepath))

    # ---- feature generation ----

    def get_sequences(self, sequence_length=512):
        # converts each url to a sequence of character-level token ids
        # then pads them all to the same length
        self._check_loaded()
        seqs = self.tokener.texts_to_sequences(self.urls)
        padded = keras_sequence.pad_sequences(seqs, maxlen=sequence_length)
        print("sequences shape:", padded.shape)
        return padded

    def get_handcrafted(self):
        # computes all hand-engineered numeric features for each url
        self._check_loaded()
        matrix = np.array([self._extract_one(u) for u in self.urls], dtype=np.float32)
        print("handcrafted features shape:", matrix.shape, "| num features:", len(feature_names))
        return matrix

    def get_labels(self):
        # returns labels as a numpy array (0=phishing, 1=legit)
        self._check_loaded()
        return np.array(self.labels, dtype=np.int32)

    def get_feature_names(self):
        return feature_names.copy()

    # ---- single url feature extraction ----

    def _extract_one(self, url):
        # need the scheme prefix so urlparse handles it properly
        parse_url = "http://" + url if not url.startswith("http") else url
        parsed = urlparse(parse_url)
        domain = parsed.hostname or ""
        path = parsed.path or ""
        full = url  # use the scheme-stripped version for char counts

        n = len(full)
        dom_len = len(domain)
        path_len = len(path)

        # length features
        num_subdomains = max(domain.count(".") - 1, 0)
        path_depth = path.count("/")

        # character count features
        dots = full.count(".")
        hyphens = full.count("-")
        underscores = full.count("_")
        slashes = full.count("/")
        at = full.count("@")
        qmarks = full.count("?")
        equals = full.count("=")
        amps = full.count("&")
        pct = full.count("%")
        digits = sum(c.isdigit() for c in full)

        # ratios
        digit_ratio = digits / n if n else 0
        letter_ratio = sum(c.isalpha() for c in full) / n if n else 0
        special_ratio = sum(not c.isalnum() for c in full) / n if n else 0

        # entropy
        url_ent = self._shannon_entropy(full)
        dom_ent = self._shannon_entropy(domain)

        # binary flags
        is_ip = int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain)))
        has_port = int(parsed.port is not None)
        is_https = int(parse_url.startswith("https"))
        is_http = int(parse_url.startswith("http://"))
        has_at = int("@" in full)
        double_slash = int("//" in path)
        dash_in_domain = int("-" in domain)
        shortened = int(domain in shortener_list)

        return [
            n, dom_len, path_len, num_subdomains, path_depth,
            dots, hyphens, underscores, slashes,
            at, qmarks, equals, amps,
            pct, digits,
            digit_ratio, letter_ratio, special_ratio,
            url_ent, dom_ent,
            is_ip, has_port, is_https, is_http, has_at,
            double_slash, dash_in_domain, shortened,
        ]

    # ---- helpers ----

    @staticmethod
    def _shannon_entropy(s):
        # measures how random/chaotic a string is
        # higher = more suspicious looking (random subdomains etc)
        if not s:
            return 0.0
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        n = len(s)
        return -sum((v / n) * np.log2(v / n) for v in freq.values())

    def _check_loaded(self):
        if not self.urls:
            raise RuntimeError("no data loaded - call load_from_file() first")
