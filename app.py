"""Command-line entry point for the RAG pipeline.

Three commands, matching the two pipelines the system actually has:

    python app.py ingest              # index time: build/refresh the index
    python app.py ingest --rebuild    # force a full rebuild
    python app.py ask "your question" # query time: one question, then exit
    python app.py chat                # query time: interactive session

Index time and query time are separate commands because they are separate
things -- one is slow and occasional, the other is fast and repeated. Ingestion
is idempotent, so running it before a session costs nothing when the corpus is
unchanged.

This module is wiring and I/O only; all pipeline logic lives in ``src``.
"""

import argparse
import sys
from typing import List, Tuple

from src.search import DEFAULT_MIN_SCORE, RAGSearch


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI surface.

    Kept in its own function so the argument definitions are testable and the
    help text stays close to the defaults it documents.
    """
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Ask questions about the documents in data/ using RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["ingest", "ask", "chat"],
        help="ingest: build/refresh the index. ask: one question. chat: interactive.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="The question to ask (required for 'ask').",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a full re-index even if the corpus is unchanged.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Cosine-similarity floor; chunks below this are ignored.",
    )
    parser.add_argument("--data-dir", default="data", help="Corpus directory.")
    parser.add_argument(
        "--persist-dir", default="vector_store", help="Index directory."
    )
    parser.add_argument(
        "--embedding-model", default="all-MiniLM-L6-v2", help="Sentence-transformer id."
    )
    parser.add_argument(
        "--llm-model", default="llama-3.1-8b-instant", help="Groq model id."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling randomness; 0 keeps answers close to the sources.",
    )
    return parser


def print_answer(result) -> None:
    """Render an ``Answer`` for the terminal, citations included.

    Sources are printed only when the answer is grounded -- a refusal has
    nothing to cite, and listing chunks under it would imply otherwise.
    """
    print(f"\n{result.text}\n")
    for source in result.sources:
        print(f"  {source.label()}")
    if result.sources:
        print()


def run_chat(rag: RAGSearch, top_k: int) -> int:
    """Interactive question loop with conversational memory.

    History is kept as ``(role, text)`` pairs and passed to ``RAGSearch.answer``,
    which condenses follow-ups into standalone questions before retrieval. Only
    the history is carried forward -- retrieved context is fetched fresh each
    turn rather than accumulated, which is what stops the prompt from growing
    without bound over a long session.

    Returns:
        Process exit code.
    """
    history: List[Tuple[str, str]] = []

    print("\nAsk a question. Commands: /clear to reset history, /exit to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D / Ctrl-C is a normal way to end a session, not an error.
            print()
            break

        if not question:
            continue
        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if question.lower() == "/clear":
            history.clear()
            print("(history cleared)\n")
            continue

        try:
            result = rag.answer(question, top_k=top_k, history=history)
        except RuntimeError as exc:
            # Keep the session alive: a transient API failure should not end
            # the conversation and lose the accumulated history.
            print(f"\n[ERROR] {exc}\n")
            continue

        if result.standalone_question:
            print(f"  (searched for: {result.standalone_question})")
        print_answer(result)

        history.append(("user", question))
        history.append(("assistant", result.text))

    print("Bye.")
    return 0


def main(argv=None) -> int:
    """Parse arguments, run the requested command, and map failures to exit codes.

    Returns:
        0 on success, 1 on a handled error (bad usage, missing index, missing
        API key, failed request). Unexpected exceptions are left to propagate so
        their tracebacks are not hidden.
    """
    args = build_parser().parse_args(argv)

    if args.command == "ask" and not args.question:
        print('Usage: python app.py ask "your question here"', file=sys.stderr)
        return 1

    try:
        rag = RAGSearch(
            persist_dir=args.persist_dir,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            data_dir=args.data_dir,
            temperature=args.temperature,
            min_score=args.min_score,
            # Ingestion is entirely local, so it must not demand an API key.
            require_api_key=args.command != "ingest",
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.command == "ingest":
        try:
            rag.ingest(force=args.rebuild)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        return 0

    try:
        rag.load()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.command == "ask":
        try:
            print_answer(rag.answer(args.question, top_k=args.top_k))
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        return 0

    return run_chat(rag, args.top_k)


if __name__ == "__main__":
    sys.exit(main())
