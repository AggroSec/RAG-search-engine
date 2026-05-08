import json
import os
import string
from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
DATA_STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stop_words.txt")
SCORE_PRECISION = 4
RRF_K = 60


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stop_words() -> list[str]:
    with open(DATA_STOP_WORDS_PATH, "r") as f:
        stop_words = f.read().splitlines()
        return stop_words

def preprocess_text(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    text = text.lower()
    return text

def tokenize(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    return valid_tokens

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for token in query_tokens:
        for title_token in title_tokens:
            if token in title_token:
                return True
    return False

def remove_stop_words(tokens: list[str]) -> list[str]:
    stop_words = load_stop_words()
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)
    return filtered_tokens

def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens
