import re
import json
import numpy as np
from urllib.parse import urlparse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence as keras_sequence
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

# list of features we look at
feature_names = [
    "url_length", "domain_length", "path_length", "subdomain_count", "path_depth",
    "count_dots", "count_hyphens", "count_underscores", "count_slashes",
    "count_at", "count_question", "count_equals", "count_ampersand",
    "count_percent", "count_digits",
    "digit_ratio", "letter_ratio", "special_char_ratio",
    "url_entropy", "domain_entropy",
    "has_ip_address", "has_port", "has_https", "has_http", "has_at_symbol",
    "has_double_slash", "has_dash_in_domain", "is_shortened",
]

# common redirect links
shortener_list = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co",
    "buff.ly", "rebrand.ly", "cutt.ly", "is.gd", "bl.ink",
    "short.io", "tiny.cc", "lc.chat", "soo.gd", "s2r.co",
}

class FeatureExtractor:
    def __init__(self, char_index_path="../dataset/char_index"):
        # load tokenizer
        self.tokener = Tokenizer(lower=True, char_level=True, oov_token="-n-")
        self.tokener.word_index = json.loads(open(char_index_path).read())
        self.urls = []
        self.labels = []

    def load_from_file(self, filepath):
        # load data from txt file
        self.urls = []
        self.labels = []
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        print(f"Loading from {filepath}...")
        skipped = 0
        for line in tqdm(lines, desc="loading"):
            parts = [p for p in line.split("\t") if p]
            if len(parts) < 2:
                skipped += 1
                continue

            label_str, url = parts[0], parts[1]
            if label_str == "phishing":
                label = 0
            elif label_str == "legitimate":
                label = 1
            else:
                skipped += 1
                continue

            # normalize url
            clean = url.lower().replace("http://", "").replace("https://", "")
            self.urls.append(clean)
            self.labels.append(label)
        print(f"done: {len(self.urls)} urls")

    def get_sequences(self, sequence_length=512):
        # convert urls to numbers
        self._check_loaded()
        seqs = self.tokener.texts_to_sequences(self.urls)
        padded = keras_sequence.pad_sequences(seqs, maxlen=sequence_length)
        return padded

    def get_handcrafted(self, n_jobs=-1):
        # compute manual features in parallel
        self._check_loaded()
        cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        print(f"extracting on {cores} cores...")
        results = Parallel(n_jobs=cores)(
            delayed(self._extract_one)(u) for u in tqdm(self.urls, desc="features")
        )
        return np.array(results, dtype=np.float32)

    def get_labels(self):
        # return labels array
        self._check_loaded()
        return np.array(self.labels, dtype=np.int32)

    def _extract_one(self, url):
        # get features for 1 url
        parse_url = "http://" + url if not url.startswith("http") else url
        parsed = urlparse(parse_url)
        domain = parsed.hostname or ""
        path = parsed.path or ""
        full = url

        n = len(full)
        dom_len = len(domain)
        path_len = len(path)

        # counts
        num_subdomains = max(domain.count(".") - 1, 0)
        path_depth = path.count("/")
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

        # chaos score
        url_ent = self._shannon_entropy(full)
        dom_ent = self._shannon_entropy(domain)

        # flags
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

    @staticmethod
    def _shannon_entropy(s):
        # entropy score
        if not s: return 0.0
        freq = {}
        for c in s: freq[c] = freq.get(c, 0) + 1
        n = len(s)
        return -sum((v / n) * np.log2(v / n) for v in freq.values())

    def _check_loaded(self):
        if not self.urls:
            raise RuntimeError("load data first!")
