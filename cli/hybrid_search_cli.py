import argparse, os

from lib.hybrid_search import *
from lib.search_utils import RRF_K
from google import genai
from dotenv import load_dotenv
from lib.prompts import get_prompt

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="List of scores to normalize")

    weighted_search_parser = subparser.add_parser("weighted-search", help="Perform a weighted hybrid search")
    weighted_search_parser.add_argument("query", help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 scores")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    rrf_search_parser = subparser.add_parser("rrf-search", help="Perform an RRF search, giving weight by ranks")
    rrf_search_parser.add_argument("query", help="Search query")
    rrf_search_parser.add_argument("--k", type=int, default=RRF_K, help="constant used to give more or less weight to rank, defaults to 60")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Limit how many results are returned, defaults to 5")
    rrf_search_parser.add_argument(
    "--enhance",
    type=str,
    choices=["spell", "rewrite", "expand"],
    help="Query enhancement method",
)

    args = parser.parse_args()

    match args.command:
        case "normalize":
            hybrid_search = HybridSearch(load_movies())
            normalized_scores = hybrid_search.normalize(args.scores)
            print("Normalized scores:", normalized_scores)
        case "weighted-search":
            hybrid_search = HybridSearch(load_movies())
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            i = 0
            while i < args.limit:
                print(f"{i+1}. {results[i]['doc']['title']}")
                print(f"Hybrid Score: {results[i]['hybrid']}")
                print(f"BM25: {results[i]['bm25']}, Semantic: {results[i]['semantic']}")
                if len(results[i]['doc']['description']) > 100:
                    print(f"{results[i]['doc']['description'][:100]}...")
                else:
                    print(results[i]['doc']['description'])
                i += 1
        case "rrf-search":
            query = args.query
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            model = os.environ.get("GEMINI_MODEL")
            if args.enhance:
                match args.enhance:
                    case "spell":
                        response = client.models.generate_content(model=model, contents=get_prompt("spell", query))
                    case "rewrite":
                        response = client.models.generate_content(model=model, contents=get_prompt("rewrite", query))
                    case "expand":
                        response = client.models.generate_content(model=model, contents=get_prompt("expand", query))
                    case _:
                        raise ValueError("incorrect enhancement used")
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{response.text}'")
                query = response.text
            hybrid_search = HybridSearch(load_movies())
            results = hybrid_search.rrf_search(query=query, k=args.k, limit=args.limit)
            i = 0
            while i < args.limit:
                print(f"{i+1}. {results[i]['doc']['title']}")
                print(f"RRF Score: {results[i]['rrf_score']}")
                print(f"BM25 rank: {results[i]['bm25_rank']}, Semantic Rank: {results[i]['semantic_rank']}")
                if len(results[i]['doc']['description']) > 100:
                    print(f"{results[i]['doc']['description'][:100]}...")
                else:
                    print(results[i]['doc']['description'])
                i += 1
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()