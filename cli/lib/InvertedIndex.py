from lib.search_utils import load_movies, tokenize, remove_stop_words, stem_tokens
from pickle import dump, load
from pathlib import Path
from collections import Counter

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

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
        if not index_path.parent.exists():
            index_path.parent.mkdir(parents=True)
        with index_path.open("wb") as f:
            dump(self.index, f)
        with docmap_path.open("wb") as f:
            dump(self.docmap, f)
        with term_frequencies_path.open("wb") as f:
            dump(self.term_frequencies, f)

    def load(self):
        index_path = Path("cache/index.pkl")
        docmap_path = Path("cache/docmap.pkl")
        term_frequencies_path = Path("cache/term_frequencies.pkl")
        if index_path.exists() and docmap_path.exists() and term_frequencies_path.exists():
            with index_path.open("rb") as f:
                self.index = load(f)
            with docmap_path.open("rb") as f:
                self.docmap = load(f)
            with term_frequencies_path.open("rb") as f:
                self.term_frequencies = load(f)
        else:
            raise FileNotFoundError("Index files not found. Please build the index first.")

    def get_tf(self, doc_id, term):
        token = term.lower()
        token = tokenize(token)
        token = remove_stop_words(token)
        token = stem_tokens(token)
        if len(token) < 1 or len(token) > 1:
            raise ValueError("Term must be a single token after processing.")
        return self.term_frequencies.get(doc_id, {}).get(term, 0)