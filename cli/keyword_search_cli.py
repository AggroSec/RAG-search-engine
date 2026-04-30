import argparse, math
from lib.commands import *
from lib.InvertedIndex import InvertedIndex
from lib.search_utils import tokenize, stem_tokens, remove_stop_words

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to check frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to check IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF for a document and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to check TF-IDF for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a document and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="BM25 k1 parameter (default: 1.5)")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="BM25 b parameter (default: 0.75)")

    bm25_search_parser = subparsers.add_parser("bm25search", help="Search movies using BM25")
    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument("limit", type=int, nargs="?", default=10, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "search":
            search_cmd(args.query)
        case "build":
            build_cmd()
        case "tf":
            tf = tf_cmd(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document {args.doc_id}: {tf}")
        case "idf":
            idf = idf_cmd(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf =tfidf_cmd(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")
        case "bm25idf":
            bm25_idf = bm25_idf_cmd(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            bm25_tf = bm25_tf_cmd(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document {args.doc_id}: {bm25_tf:.2f}")
        case "bm25search":
            bm25search_cmd(args.query, args.limit)
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()