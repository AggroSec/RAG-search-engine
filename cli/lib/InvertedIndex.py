import math

from lib.search_utils import load_movies, tokenize, remove_stop_words, stem_tokens
from pickle import dump, load
from pathlib import Path
from collections import Counter
from lib.constants import BM25_K1, BM25_B

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        tokens = remove_stop_words(tokens)
        tokens = stem_tokens(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        self.term_frequencies[doc_id].update(tokens)

        self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self, term):
        doc_ids = self.index.get(term.lower(), set())
        id_list = list(doc_ids)
        return sorted(id_list)
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")

    def save(self):
        index_path = Path("cache/index.pkl")
        docmap_path = Path("cache/docmap.pkl")
        term_frequencies_path = Path("cache/term_frequencies.pkl")
        doc_lengths_path = Path("cache/doc_lengths.pkl")
        if not index_path.parent.exists():
            index_path.parent.mkdir(parents=True)
        with index_path.open("wb") as f:
            dump(self.index, f)
        with docmap_path.open("wb") as f:
            dump(self.docmap, f)
        with term_frequencies_path.open("wb") as f:
            dump(self.term_frequencies, f)
        with doc_lengths_path.open("wb") as f:
            dump(self.doc_lengths, f)

    def load(self):
        index_path = Path("cache/index.pkl")
        docmap_path = Path("cache/docmap.pkl")
        term_frequencies_path = Path("cache/term_frequencies.pkl")
        doc_lengths_path = Path("cache/doc_lengths.pkl")
        if index_path.exists() and docmap_path.exists() and term_frequencies_path.exists() and doc_lengths_path.exists():
            with index_path.open("rb") as f:
                self.index = load(f)
            with docmap_path.open("rb") as f:
                self.docmap = load(f)
            with term_frequencies_path.open("rb") as f:
                self.term_frequencies = load(f)
            with doc_lengths_path.open("rb") as f:
                self.doc_lengths = load(f)
        else:
            raise FileNotFoundError("Index files not found. Please build the index first.")

    def get_tf(self, doc_id, term) -> float:
        token = term.lower()
        token = tokenize(token)
        token = remove_stop_words(token)
        token = stem_tokens(token)
        if len(token) < 1 or len(token) > 1:
            raise ValueError("Term must be a single token after processing.")
        return self.term_frequencies.get(doc_id, {}).get(term, 0)
    
    def get_bm25_idf(self, term: str) -> float:
        total_docs = len(self.docmap)
        doc_freq = len(self.get_documents(term))
        idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        return idf
    
    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths.get(doc_id, 0) / avg_doc_length)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf

    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths) if self.doc_lengths else 0.0
    
    def bm25(self, doc_id, term) -> float:
        bm25_idf = self.get_bm25_idf(term)
        bm25_tf = self.get_bm25_tf(doc_id, term)
        return bm25_idf * bm25_tf
    
    def bm25_search(self, query, limit):
        tokens = tokenize(query)
        tokens = remove_stop_words(tokens)
        tokens = stem_tokens(tokens)

        scores = {}
        for token in tokens:
            for doc_id in self.get_documents(token):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += self.bm25(doc_id, token)

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs[:limit]