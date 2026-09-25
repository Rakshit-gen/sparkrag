import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sparkrag")


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

        try:
            count = run_ingest(args.input)
        except ValueError as e:
            # common user mistake: bad or empty --input path, no need for a
            # spark stack trace on top of it
            logger.error(str(e))
            sys.exit(1)

        logger.info("ingested %d chunks from %s", count, args.input)
    elif args.command == "query":
        from sparkrag.chain import ask

        try:
            answer = ask(args.question, k=args.k)
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)

        # the answer itself stays plain stdout, not a log line, so it can
        # still be piped or redirected cleanly
        print(answer)


if __name__ == "__main__":
    main()
