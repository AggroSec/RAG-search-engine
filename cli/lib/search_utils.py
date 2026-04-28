import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
DATA_STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stop_words.txt")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stop_words() -> list[str]:
    with open(DATA_STOP_WORDS_PATH, "r") as f:
        stop_words = f.read().splitlines()
        return stop_words