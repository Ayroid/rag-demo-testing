"""Query-time orchestration: retrieval, prompt construction, and generation.

The top layer of the system and the only module that talks to an external API.
It wires a ``VectorStore`` (local, free, fast) to a Groq-hosted LLM (remote,
metered, network-bound) and exposes the calls that turn a question into a
grounded, cited answer.

Everything below this module is deterministic and offline; everything this
module adds is not. That boundary is why the LLM integration is isolated here.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from src.data_loader import LOADERS, load_all_documents
from src.manifest import fingerprint_corpus
from src.vectorstore import VectorStore
from langchain_groq import ChatGroq

# Read .env at import time so GROQ_API_KEY is present before any class below is
# constructed. Import-time side effects are usually worth avoiding, but the
# alternative -- every caller remembering to call this first -- is worse.
# Real environment variables take precedence over .env by default.
load_dotenv()

# Cosine-similarity floor for a chunk to be considered relevant at all.
#
# Calibrated for all-MiniLM-L6-v2, where genuinely on-topic passages typically
# land above ~0.35 and unrelated text clusters near zero. The value is
# deliberately conservative: too high silently drops usable context, too low
# readmits the failure this guards against -- answering confidently over
# irrelevant chunks because FAISS always returns *something*.
DEFAULT_MIN_SCORE = 0.30

# The refusal string the model is instructed to emit verbatim. Kept as a
# constant so callers can recognise a refusal without parsing prose.
NO_ANSWER = "I don't know based on the provided documents."

# Grounded-QA system prompt. Four jobs, in order: assign the role, fence the
# knowledge to the supplied context, license refusal, and mandate citations.
# The refusal clause does real work -- an LLM handed context will otherwise
# strain to produce something from it rather than admit the answer is absent.
SYSTEM_PROMPT = f"""You answer questions using ONLY the provided context.

Rules:
- If the context does not contain the answer, reply with exactly: "{NO_ANSWER}"
- Never use outside knowledge. Never guess.
- Cite the source after each claim using the bracketed numbers of the context
  blocks you used, like [1] or [2].
- Be concise and direct."""

# Rewrites a context-dependent follow-up into a standalone question.
#
# This exists because retrieval happens *before* the LLM ever sees the
# conversation: "Who wrote it?" gets embedded literally, and "it" carries no
# semantic content, so the search returns noise. Condensing first is what makes
# multi-turn chat work at all -- passing history to the generation step alone
# does not fix retrieval.
CONDENSE_PROMPT = """Given the conversation and a follow-up question, rewrite the
follow-up as a standalone question that makes sense without the conversation.
Preserve the original wording where possible. Output ONLY the rewritten
question, with no preamble.

Conversation:
{history}

Follow-up: {question}
Standalone question:"""

# How many prior turns to feed the condenser. Enough to resolve pronouns and
# recent topic references without spending tokens on the whole conversation.
CONDENSE_HISTORY_TURNS = 6


@dataclass
class Source:
    """One cited chunk, resolved into something printable.

    Exists so callers can render citations without knowing the metadata schema
    or which loader produced a record.
    """

    number: int
    file: str
    page: Optional[int]
    score: float

    def label(self) -> str:
        """Render as ``[1] Nature-of-code.pdf p.24 (score 0.71)``.

        Page numbers are stored zero-based by ``PyPDFLoader`` and shown
        one-based, matching what a reader sees on the page.
        """
        page = f" p.{self.page + 1}" if self.page is not None else ""
        return f"[{self.number}] {self.file}{page} (score {self.score:.2f})"


@dataclass
class Answer:
    """The result of one question: the text, and what it was based on.

    Replaces the bare ``str`` the previous implementation returned. Callers need
    the sources to display citations and ``grounded`` to distinguish a real
    answer from a refusal, neither of which survives being flattened to text.
    """

    text: str
    sources: List[Source] = field(default_factory=list)
    grounded: bool = True
    standalone_question: Optional[str] = None


class RAGSearch:
    """Answers questions over the indexed corpus using a Groq-hosted LLM.

    Owns the query-time pipeline: condense, embed, retrieve, filter, prompt,
    generate. It is the façade the application layer uses -- ``app.py``
    constructs one of these and knows nothing about FAISS, chunking, or prompts.

    Ingestion is a separate, explicit call (``ingest``). The constructor no
    longer builds an index as a side effect of the first query, so index time
    and query time are now distinct operations on the same object.
    """

    def __init__(
        self,
        persist_dir: str = "vector_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
        data_dir: str = "data",
        temperature: float = 0.0,
        min_score: float = DEFAULT_MIN_SCORE,
        require_api_key: bool = True,
    ):
        """Prepare the retriever and the LLM client.

        Args:
            persist_dir: Directory holding the index files.
            embedding_model: Must match the model the index was built with; the
                manifest enforces this on load.
            llm_model: A Groq model id.
            data_dir: Corpus root, used by ``ingest`` and by staleness checks.
            temperature: Sampling randomness. Defaults to 0 for grounded QA --
                ChatGroq's own default is 0.7, which invites embellishment
                beyond the retrieved text.
            min_score: Cosine-similarity floor for retrieved chunks.
            require_api_key: When True, a missing GROQ_API_KEY raises here
                rather than surfacing as an opaque auth error on the first
                request. Set False for ingestion, which needs no LLM.

        Raises:
            RuntimeError: when the API key is required but absent.

        Side effects:
            Loads the embedding model and constructs an API client. Expensive
            enough that instances are meant to be created once and reused. Does
            **not** read the index -- call ``load`` or ``ingest``.
        """
        self.data_dir = data_dir
        self.min_score = min_score
        self.vectorstore = VectorStore(persist_dir, embedding_model)

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key and require_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file or export it:\n"
                "    export GROQ_API_KEY=your-key-here"
            )

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
            temperature=temperature,
        )
        print(f"[INFO] Groq LLM initialized: {llm_model} (temperature={temperature})")

    # -- Index lifecycle ----------------------------------------------------

    def fingerprint(self) -> Dict[str, str]:
        """Hash the corpus as ingestion would see it.

        Uses the same directory and glob patterns as ``load_all_documents``, so
        the fingerprint covers exactly the files that would be indexed.
        """
        return fingerprint_corpus(self.data_dir, list(LOADERS.keys()))

    def ingest(self, force: bool = False) -> bool:
        """Build the index, skipping the work when nothing has changed.

        Idempotent by design: safe to run before every session. This is what
        replaces the old "comment the build block back in" workflow, and what
        makes adding a document to ``data/`` actually take effect.

        Args:
            force: Rebuild even when the fingerprint matches.

        Returns:
            True if an index was built, False if the existing one was current.

        Side effects:
            On a rebuild, overwrites everything in the persistence directory.
        """
        current_files = self.fingerprint()

        if not force and self.vectorstore.exists():
            try:
                self.vectorstore.load()
            except (RuntimeError, FileNotFoundError) as exc:
                print(f"[INFO] Existing index unusable, rebuilding. Reason: {exc}")
            else:
                stale = self.vectorstore.check_stale(current_files)
                if stale is None:
                    print("[INFO] Index is up to date; nothing to ingest.")
                    return False
                print(f"[INFO] Corpus changed ({stale}); rebuilding index.")

        documents = load_all_documents(self.data_dir)
        if not documents:
            raise ValueError(
                f"No documents were loaded from {self.data_dir!r}. "
                f"Supported extensions: {', '.join(p.split('.')[-1] for p in LOADERS)}."
            )
        self.vectorstore.build_from_documents(documents, files=current_files)
        return True

    def load(self, warn_if_stale: bool = True) -> None:
        """Load the index for querying, failing loudly if it is unusable.

        Args:
            warn_if_stale: Print a warning when the corpus has changed since the
                index was built. A warning rather than an error, because a
                slightly stale index still answers most questions -- the user
                just needs to know why a new document is not being found.

        Raises:
            FileNotFoundError: no index present.
            RuntimeError: index incompatible with the current configuration.
        """
        self.vectorstore.load()
        if warn_if_stale:
            stale = self.vectorstore.check_stale(self.fingerprint())
            if stale:
                print(
                    f"[WARN] Index is out of date ({stale}). "
                    "Run: python app.py ingest"
                )

    # -- Query pipeline -----------------------------------------------------

    def condense(self, question: str, history: Sequence[Tuple[str, str]]) -> str:
        """Rewrite a follow-up into a question that stands on its own.

        Args:
            question: The user's latest input, possibly referring to earlier
                turns ("who wrote it?", "what about the second one?").
            history: ``(role, text)`` pairs, oldest first.

        Returns:
            A standalone question, or ``question`` unchanged when there is no
            history to resolve against or the rewrite fails.

        Side effects:
            One extra LLM call per turn when history is non-empty. On
            ``llama-3.1-8b-instant`` the added latency is small relative to the
            answer call, and it is what makes retrieval work for follow-ups.

        Failure handling: if the rewrite errors or comes back empty, the
        original question is used. A degraded retrieval beats a crashed turn.
        """
        if not history:
            return question

        recent = "\n".join(
            f"{role}: {text}" for role, text in history[-CONDENSE_HISTORY_TURNS:]
        )
        try:
            response = self.llm.invoke(
                CONDENSE_PROMPT.format(history=recent, question=question)
            )
            rewritten = (response.content or "").strip()
        except Exception as exc:
            print(f"[WARN] Question condensation failed ({exc}); using it verbatim.")
            return question

        return rewritten or question

    def build_context(
        self, hits: List[Dict[str, Any]]
    ) -> Tuple[str, List[Source]]:
        """Turn retrieved chunks into a numbered context block plus source list.

        Numbering is the mechanism that makes citation possible: the model is
        told to cite ``[n]``, and the same ``n`` indexes the returned sources,
        so a claim can be traced to a file and page.

        Args:
            hits: Result dicts from ``VectorStore.query``.

        Returns:
            ``(context, sources)`` where context is the prompt-ready text and
            sources are the corresponding ``Source`` records.
        """
        blocks: List[str] = []
        sources: List[Source] = []

        for number, hit in enumerate(hits, 1):
            meta = hit["metadata"] or {}
            raw_source = meta.get("source", "unknown")
            page = meta.get("page")
            source = Source(
                number=number,
                file=Path(str(raw_source)).name,
                page=page if isinstance(page, int) else None,
                score=hit["score"],
            )
            sources.append(source)

            header = f"[{number}] ({source.file}"
            header += f" p.{source.page + 1})" if source.page is not None else ")"
            blocks.append(f"{header}\n{meta.get('content', '')}")

        return "\n\n".join(blocks), sources

    def answer(
        self,
        query: str,
        top_k: int = 5,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Answer:
        """Answer a question from the indexed corpus.

        The complete query-time path: condense, retrieve, filter by score,
        build a numbered context, generate under grounding rules.

        Args:
            query: The user's question.
            top_k: How many chunks to consider. Raises recall and token cost
                together.
            history: Prior ``(role, text)`` turns, enabling follow-ups.

        Returns:
            An ``Answer``. When nothing clears ``min_score``, returns the
            refusal text with ``grounded=False`` and no sources -- **without**
            calling the LLM, which saves a request and removes any opportunity
            to hallucinate over irrelevant context.

        Side effects:
            One LLM call, or two when history is present (condensation).

        Raises:
            RuntimeError: when the LLM call fails, wrapping the underlying
                client error with context about what was being attempted.
        """
        standalone = self.condense(query, history or [])
        hits = self.vectorstore.query(standalone, top_k=top_k, min_score=self.min_score)

        if not hits:
            return Answer(
                text=NO_ANSWER,
                sources=[],
                grounded=False,
                standalone_question=standalone if standalone != query else None,
            )

        context, sources = self.build_context(hits)
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {standalone}\n\n"
            "Answer:"
        )

        try:
            response = self.llm.invoke(
                [("system", SYSTEM_PROMPT), ("human", prompt)]
            )
        except Exception as exc:
            raise RuntimeError(f"Groq request failed: {exc}") from exc

        text = (response.content or "").strip()
        grounded = bool(text) and NO_ANSWER.rstrip(".").lower() not in text.lower()

        return Answer(
            text=text or NO_ANSWER,
            # A refusal cites nothing, even though chunks cleared the threshold.
            sources=sources if grounded else [],
            grounded=grounded,
            standalone_question=standalone if standalone != query else None,
        )

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Answer a question and return just the text.

        Backwards-compatible wrapper for the original single-string API, kept so
        existing callers and notebooks keep working. Prefer ``answer``, which
        also returns sources and whether the response was grounded.
        """
        return self.answer(query, top_k=top_k).text
