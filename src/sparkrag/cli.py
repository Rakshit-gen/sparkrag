import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="sparkrag")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="ingest a directory of documents")
    ingest_p.add_argument("--input", required=True, help="path to a folder of docs")

    query_p = sub.add_parser("query", help="ask a question against ingested docs")
    query_p.add_argument("question", help="your question")
    query_p.add_argument("--k", type=int, default=4, help="how many chunks to retrieve")

    args = parser.parse_args()

    if args.command == "ingest":
        from sparkrag.ingest import run_ingest

        count = run_ingest(args.input)
        print(f"ingested {count} chunks from {args.input}")
    elif args.command == "query":
        from sparkrag.chain import ask

        try:
            print(ask(args.question, k=args.k))
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
