#!/usr/bin/env python3

import argparse
from lib.semantic_search import *


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the semantic search functionality")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embedding for")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embedding generation for all documents")

    embed_query_text_parser = subparsers.add_parser("embed_query", help="Generate embedding for a query text")
    embed_query_text_parser.add_argument("query", type=str, help="Query text to generate embedding for")

    search_parser = subparsers.add_parser("search", help="Search for similar documents based on a query")
    search_parser.add_argument("query", type=str, help="Query text to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a given text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size to chunk the text into")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantically chunk a given text into smaller pieces")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to semantically chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum size to chunk the text into")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Generate embeddings for chunks of a given text")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for similar documents based on a query using chunked embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Query text to search for")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search = SemanticSearch()
            movies = load_movies()
            semantic_search.load_or_create_embeddings(movies)
            results = semantic_search.search(args.query, args.limit)
            for i, (score, doc) in enumerate(results):
                print(f"{i+1}. {doc['title']} (score: {score:.4f})")
                if len(doc["description"]) > 200:
                    print(f"{doc['description'][:200].rsplit(' ', 1)[0]}...")
                else:
                    print(doc["description"])
        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")
        case "semantic_chunk":
            print(f"Semantically chunking {len(args.text)} characters")
            chunks = semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            semantic_chunk_search = ChunkedSemanticSearch()
            movies = load_movies()
            semantic_chunk_search.load_or_create_chunk_embeddings(movies)
            results = semantic_chunk_search.search_chunks(args.query, args.limit)
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']} (score: {result['score']:.4f})")
                print(f"{result['description']}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()