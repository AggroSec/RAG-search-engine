import string
from lib.search_utils import load_movies, DEFAULT_SEARCH_LIMIT, load_stop_words, tokenize, remove_stop_words, stem_tokens
from lib.InvertedIndex import InvertedIndex

def search_movies(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movie_index = InvertedIndex()
    movie_index.load()
    query_tokens = tokenize(query)
    query_tokens = remove_stop_words(query_tokens)
    query_tokens = stem_tokens(query_tokens)
    matching_movies = []
    for token in query_tokens:
        doc_ids = movie_index.get_documents(token)
        for doc_id in doc_ids:
            matching_movies.append(movie_index.docmap[doc_id])
            if len(matching_movies) >= limit:
                break
        if len(matching_movies) >= limit:
            break
    return matching_movies


