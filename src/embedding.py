"""Stages 2 and 3 of the ingestion pipeline: chunking and embedding.

Splits loaded documents into retrieval-sized pieces, then converts each piece
into a fixed-length numeric vector. These two steps live together because they
share one tuning concern: the chunk is the unit that gets embedded, and
therefore the unit that gets retrieved. Chunk boundaries decide what the
retriever is even capable of returning.

Serves both pipelines. ``chunk_documents`` and ``embed_chunks`` run at index
time; ``embed_query`` runs at query time. One instance, owned by
``VectorStore``, handles both -- which is what guarantees the two sides use
identical encoding settings.
"""

from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingPipeline:
    """Turns Documents into chunks, and chunks into vectors.

    Owns all text-to-vector conversion. It exists so that chunking and
    embedding configuration (model, chunk size, overlap) travels as one object
    and is applied consistently -- both across an entire index build and
    between index time and query time.

    Constructed once by ``VectorStore.__init__`` and reused for that store's
    lifetime, so the model is loaded exactly once per process. Beyond its
    configuration and the loaded model it holds no state.

    Vectors from two different models are not comparable. ``src.manifest``
    records which model built an index and refuses to load a mismatched one,
    which is what keeps that assumption enforced rather than merely documented.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Load the sentence-transformer model and record chunking settings.

        Args:
            model_name: A sentence-transformers model id. The default,
                all-MiniLM-L6-v2, produces 384-dimensional vectors. It is
                downloaded from the Hugging Face Hub on first use and cached
                under ~/.cache thereafter. Changing it changes the vector
                dimension and invalidates any existing index.
            chunk_size: Target maximum **characters** (not tokens) per chunk.
            chunk_overlap: Characters repeated between consecutive chunks, so a
                sentence spanning a boundary survives intact in one of them.

        Side effects:
            Loads the model into memory, downloading it on first run. This is
            the slow part of constructing the object, which is why it happens
            once here rather than per call.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[DEBUG] Initialized SentenceTransformer model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Split documents into chunks using RecursiveCharacterTextSplitter.

        "Recursive" describes the separator strategy: the splitter tries the
        first separator in the list, and only falls back to the next when a
        piece is still larger than ``chunk_size``. The order below --
        paragraph, line, word, character -- makes cuts land on the largest
        natural boundary available, keeping chunks semantically whole instead
        of slicing mid-sentence.

        Args:
            documents: ``Document`` objects from ``load_all_documents``.

        Returns:
            A flat list of ``Document`` objects, not strings. ``split_documents``
            copies each parent document's ``.metadata`` onto every chunk derived
            from it, so ``source`` (file path) and ``page`` (for PDFs) survive
            into the vector store and remain available for citations.

        Side effects:
            Prints a single summary line.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            # Ordered coarse -> fine; "" is the last-resort hard character cut.
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"[DEBUG] {len(documents)} documents split into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """Encode chunk texts into a matrix of embedding vectors.

        Runs locally on CPU (or GPU, if torch finds one) -- no network call and
        no per-token cost, unlike the LLM step. sentence-transformers batches
        internally, so passing the whole corpus in one call is both correct and
        far faster than encoding chunk by chunk.

        Args:
            chunks: ``Document`` chunks from ``chunk_documents``. Only
                ``.page_content`` is embedded; metadata is not seen by the model.

        Returns:
            A ``(len(chunks), 384)`` array for the default model, where row i
            corresponds to ``chunks[i]``. That positional correspondence is the
            contract ``VectorStore`` depends on to map a search hit back to its
            text -- the vectors and the metadata list must never be reordered
            independently.

            ``normalize_embeddings=True`` scales every row to unit length. This
            is what makes the inner-product index in ``VectorStore`` compute
            cosine similarity, so it must stay in sync with the matching flag in
            ``VectorStore.query``.

        Side effects:
            Prints a progress bar to stderr; this is the long pole of a build.
        """
        texts = [chunk.page_content for chunk in chunks]
        print(f"[DEBUG] Embedding {len(texts)} chunks.")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        print(f"[DEBUG] Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def embedding_dimension(self) -> int:
        """Return the vector length this model produces (384 for MiniLM-L6-v2).

        Wraps a sentence-transformers API that was renamed in 5.x
        (``get_sentence_embedding_dimension`` -> ``get_embedding_dimension``),
        so callers do not have to care which version is installed. Used by
        ``VectorStore.load`` to catch a dimension mismatch up front rather than
        as an opaque FAISS assertion on the first query.
        """
        # Both looked up dynamically so the deprecated name is never referenced
        # directly -- otherwise type checkers flag the compatibility branch.
        getter = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension"
        )
        return int(getter())

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single search query into one embedding vector.

        The query-time counterpart to ``embed_chunks``. It exists so that both
        sides of the system encode through the same object with the same
        settings -- in particular ``normalize_embeddings=True``, without which
        the inner-product index would not compute cosine similarity. Keeping the
        two methods adjacent makes any future divergence obvious.

        Args:
            query: Raw question text.

        Returns:
            A ``(1, dim)`` float32 array, shaped as a batch of one because that
            is what ``faiss.Index.search`` expects.

        Side effects:
            None. No progress bar -- a single short encode is not worth one.
        """
        return self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
