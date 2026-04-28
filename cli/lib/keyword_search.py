import string
from lib.search_utils import load_movies, DEFAULT_SEARCH_LIMIT, load_stop_words

def search_movies(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movie_data = load_movies()

    #set up results list and loop through movies to find matches
    results = []

    for movie in movie_data:
        query_tokens = remove_stop_words(tokenize(query))
        title_tokens = remove_stop_words(tokenize(movie["title"]))
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results

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