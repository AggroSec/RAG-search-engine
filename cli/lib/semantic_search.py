from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from lib.search_utils import SCORE_PRECISION, load_movies
import re, json

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("text cannot be empty or just whitespace.")
        embedding = self.model.encode([text])
        return embedding[0]
    
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        movie_texts = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            movie_texts.append(f"{doc['title']} {doc['description']}")

        self.embeddings = self.model.encode(movie_texts, show_progress_bar=True)
        embedding_path = Path("cache/movie_embeddings.npy")
        if not embedding_path.parent.exists():
            embedding_path.parent.mkdir(parents=True)
        np.save(embedding_path, self.embeddings)
        print(f"Embeddings saved to {embedding_path}")
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]):
        embedding_path = Path("cache/movie_embeddings.npy")
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        if embedding_path.exists():
            print(f"Loading embeddings from {embedding_path}")
            self.embeddings = np.load(embedding_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings  
        else:
            print("Embeddings not found, building new embeddings.")
            return self.build_embeddings(documents)
        
    def search(self, query: str, limit: int):
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        similarity_list = []
        for i, embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embedding, embedding)
            similarity_tup = (similarity_score, self.documents[i])
            similarity_list.append(similarity_tup)
        similarity_list.sort(key=lambda x: x[0], reverse=True)
        return similarity_list[:limit]



def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    movies = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1:
        return [sentences[0].strip()]
    chunks = []
    current_chunk = []
    for sentence in sentences:
        stripped_sentence = sentence.strip()
        if not stripped_sentence:
            continue
        current_chunk.append(stripped_sentence)
        if len(current_chunk) >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            if overlap > 0:
                current_chunk = current_chunk[overlap:]
            else:
                current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        all_chunks = []
        metadata = []
        for doc in documents:
            if doc['description'] == "":
                continue
            chunks = semantic_chunk_text(doc['description'], max_chunk_size=4, overlap=1)
            all_chunks.extend(chunks)
            for i, chunk in enumerate(chunks):
                metadata.append({
                    "movie_idx": doc['id'],
                    "chunk_idx": i,
                    "total_chunks": len(chunks)
                })
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata
        embedding_path = Path("cache/chunk_embeddings.npy")
        if not embedding_path.parent.exists():
            embedding_path.parent.mkdir(parents=True)
        np.save(embedding_path, self.chunk_embeddings)
        print(f"Chunk embeddings saved to {embedding_path}")
        metadata_path = Path("cache/chunk_metadata.json")
        if not metadata_path.parent.exists():
            metadata_path.parent.mkdir(parents=True)
        with open(metadata_path, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        print(f"Chunk metadata saved to {metadata_path}")
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        embedding_path = Path("cache/chunk_embeddings.npy")
        metadata_path = Path("cache/chunk_metadata.json")
        if embedding_path.exists() and metadata_path.exists():
            print(f"Loading chunk embeddings from {embedding_path}")
            self.chunk_embeddings = np.load(embedding_path)
            with open(metadata_path, "r") as f:
                metadata_json = json.load(f)
                self.chunk_metadata = metadata_json["chunks"]
            if len(self.chunk_embeddings) == metadata_json["total_chunks"]:
                return self.chunk_embeddings
        else:
            print("Chunk embeddings not found, building new chunk embeddings.")
            return self.build_chunk_embeddings(documents)
        
    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        preprocessed_query = query.strip()
        if not preprocessed_query:
            return []
        query_chunks = semantic_chunk_text(preprocessed_query, max_chunk_size=4, overlap=1)
        query_embedding = self.generate_embedding(preprocessed_query)
        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_score = {
                "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": score,
            }
            chunk_scores.append(chunk_score)
        score_map = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if movie_idx not in score_map or chunk_score["score"] > score_map[movie_idx]["score"]:
                score_map[movie_idx] = chunk_score
        sorted_scores = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)
        top_results = sorted_scores[:limit]
        final_results = []
        for result in top_results:
            movie = self.document_map[result["movie_idx"]]
            final_results.append({
                "id": movie["id"],
                "title": movie["title"],
                "description": movie["description"][:100],
                "score": round(result["score"], SCORE_PRECISION),
                "metadata": self.chunk_metadata[result["chunk_idx"]] or {}
            })
        return final_results
        
def embed_chunks():
    semantic_chunk_search = ChunkedSemanticSearch()
    movies = load_movies()
    chunk_embeddings = semantic_chunk_search.load_or_create_chunk_embeddings(movies)
    print(f"Generated 72909 chunked embeddings")
