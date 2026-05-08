from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from pathlib import Path
from lib.search_utils import *
from itertools import islice

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not Path("cache/index.pkl").exists():
            self.idx.build()
            self.idx.save()
        self.idx.load()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        _, bm25_results = self.idx.bm25_search(query, limit*500)  # Get more results for better weighting
        semantic_results = self.semantic_search.search_chunks(query, limit*500)
        norm_bm25_list = []
        
        for result in bm25_results:
            norm_bm25_list.append(result['score'])
        norm_bm25_list = self.normalize(norm_bm25_list)
        norm_semantic_list = []
        for result in semantic_results:
            norm_semantic_list.append(result['score'])
        norm_semantic_list = self.normalize(norm_semantic_list)
        bm25_by_id = {}
        for i, result in enumerate(bm25_results):
            doc_id = result['id']
            bm25_by_id[doc_id] = norm_bm25_list[i]
        semantic_by_id = {}
        for i, result in enumerate(semantic_results):
            doc_id = result['id']
            semantic_by_id[doc_id] = norm_semantic_list[i]
        combined_scores = {}
        all_doc_ids = set(bm25_by_id.keys()) | set(semantic_by_id.keys())
        for doc_id in all_doc_ids:
            bm25_score = bm25_by_id.get(doc_id, 0)
            semantic_score = semantic_by_id.get(doc_id, 0)
            combined_scores[doc_id] = {'doc': self.semantic_search.document_map[doc_id], 'bm25': bm25_score, 'semantic': semantic_score, 'hybrid': self.hybrid_score(bm25_score, semantic_score, alpha)}
        sorted_scores = sorted(combined_scores.values(), key=lambda x: x['hybrid'], reverse=True)
        return sorted_scores

    def rrf_search(self, query, k, limit=10):
        _, bm25_results = self.idx.bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)
        combined_scores = {}
        current_rank = 1
        for result in bm25_results:
            doc_id = result['id']
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {'doc': self.idx.docmap[doc_id], 'bm25_rank': 0, 'semantic_rank': 0, 'rrf_score': 0}
            combined_scores[doc_id]['bm25_rank'] = current_rank
            current_rank += 1
        current_rank = 1
        for result in semantic_results:
            doc_id = result['id']
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {'doc': self.semantic_search.document_map[doc_id], 'bm25_rank': 0, 'semantic_rank': 0, 'rrf_score': 0}
            combined_scores[doc_id]['semantic_rank'] = current_rank
            current_rank += 1
        for doc_id, scores in combined_scores.items():
            current_rrf_score = 0
            if scores['bm25_rank'] != 0:
                current_rrf_score += self.get_rrf_score(scores['bm25_rank'], k)
            if scores['semantic_rank'] != 0:
                current_rrf_score += self.get_rrf_score(scores['semantic_rank'], k)
            combined_scores[doc_id]['rrf_score'] = current_rrf_score
        sorted_scores = sorted(combined_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
        return sorted_scores
    
    def normalize(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        return normalized_scores
    
    def hybrid_score(self, bm25_score, semantic_score, alpha=0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def get_rrf_score(self, rank: int, k: int = RRF_K) -> float:
        return 1 / (k + rank)