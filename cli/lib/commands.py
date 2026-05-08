from lib.constants import BM25_B, BM25_K1
from lib.keyword_search import search_movies
from lib.InvertedIndex import InvertedIndex
from lib.search_utils import tokenize, remove_stop_words, stem_tokens
import math

def search_cmd(query: str) -> None:
    print(f"Searching for: {query}")
    movie_list = search_movies(query)
    for movie in movie_list:
        print(f"ID:{movie['id']} TITLE: {movie['title']}")
    pass

def build_cmd() -> None:
    print("Building inverted index...")
    index = InvertedIndex()
    index.build()
    index.save()
    print("Inverted index built and saved.")

def tf_cmd(doc_id: int, term: str) -> float:
    print(f"Getting term frequency for doc_id: {doc_id}, term: {term}")
    index = InvertedIndex()
    index.load()
    tf = index.get_tf(doc_id, term)
    return tf

def idf_cmd(term: str) -> float:
    print(f"Getting inverse document frequency for term: {term}")
    index = InvertedIndex()
    index.load()
    total_docs = len(index.docmap)
    token = tokenize(term)
    token = remove_stop_words(token)
    token = stem_tokens(token)
    if not token:
        print(f"Term '{term}' is a stop word or has no valid tokens after preprocessing.")
        return
    doc_freq = len(index.get_documents(token[0]))
    idf = math.log((total_docs + 1) / (doc_freq + 1))
    return idf

def tfidf_cmd(doc_id: int, term: str) -> float:
    tf = tf_cmd(doc_id, term)
    idf = idf_cmd(term)
    ftidf = tf * idf
    return ftidf

def bm25_idf_cmd(term: str) -> float:
    print(f"Getting BM25 IDF for term: {term}")
    index = InvertedIndex()
    index.load()
    total_docs = len(index.docmap)
    token = tokenize(term)
    token = remove_stop_words(token)
    token = stem_tokens(token)
    if not token or len(token) > 1:
        print(f"Term '{term}' is a stop word or has no valid tokens after preprocessing.")
        return
    return index.get_bm25_idf(token[0])

def bm25_tf_cmd(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    index = InvertedIndex()
    index.load()
    token = tokenize(term)
    token = remove_stop_words(token)
    token = stem_tokens(token)
    if not token or len(token) > 1:
        print(f"Term '{term}' is a stop word or has no valid tokens after preprocessing.")
        return
    return index.get_bm25_tf(doc_id, token[0], k1, b)

def bm25search_cmd(query: str, limit: int = 5) -> None:
    index = InvertedIndex()
    index.load()
    results, _ = index.bm25_search(query, limit)
    for i, (doc_id, score) in enumerate(results):
        print(f"{i+1}. ({doc_id}) {index.docmap.get(doc_id, {}).get('title', 'Unknown')} - Score: {score:.2f}")