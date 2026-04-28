import argparse, math
from lib.keyword_search import search_movies
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

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movie_list = search_movies(args.query)
            for movie in movie_list:
                print(f"ID:{movie['id']} TITLE: {movie['title']}")
            pass
        case "build":
            print("Building inverted index...")
            index = InvertedIndex()
            index.build()
            index.save()
            print("Inverted index built and saved.")
        case "tf":
            print(f"Getting term frequency for doc_id: {args.doc_id}, term: {args.term}")
            index = InvertedIndex()
            index.load()
            tf = index.get_tf(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document {args.doc_id}: {tf}")
        case "idf":
            print(f"Getting inverse document frequency for term: {args.term}")
            index = InvertedIndex()
            index.load()
            total_docs = len(index.docmap)
            token = tokenize(args.term)
            token = remove_stop_words(token)
            token = stem_tokens(token)
            if not token:
                print(f"Term '{args.term}' is a stop word or has no valid tokens after preprocessing.")
                return
            doc_freq = len(index.get_documents(token[0]))
            idf = math.log((total_docs + 1) / (doc_freq + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()