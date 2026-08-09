"""Stage 4 of ingestion, and the retrieval half of query time: the vector index.

Wraps a FAISS index plus a parallel list of per-chunk records, and handles
persisting both to disk. This is the one component used by *both* pipelines:
``build_from_documents`` is index time, ``query`` is query time.

Storage layout under ``persistence_directory``:
    faiss.index    -- the FAISS index (vectors + similarity structure)
    metadata.pkl   -- pickled list of dicts, positionally aligned to the index
    manifest.json  -- provenance: model, dimension, chunk settings, file hashes

The first two files are meaningless apart: FAISS returns integer row ids, and
only ``metadata.pkl`` can turn a row id back into text. The manifest is what
makes that pairing verifiable rather than assumed -- see ``src.manifest``.
"""

import os, faiss, pickle
import numpy as np
from typing import Any, Dict, List, Optional
from src.embedding import EmbeddingPipeline
from src import manifest as manifest_mod


class VectorStore:
    """A persistent FAISS vector index with attached chunk records.

    Responsible for building an index from documents, saving and loading it,
    and answering nearest-neighbour queries. It is the system's long-term
    memory -- the only component whose output survives between runs.

    The similarity metric is cosine, implemented as an inner-product index over
    unit-length vectors (see ``add_embeddings``). Scores are therefore bounded
    in [-1, 1] and **higher is better**, the opposite of a distance.

    Owns a single ``EmbeddingPipeline`` used for both indexing and querying, so
    the model is loaded once per process and the two sides cannot drift apart
    in their encoding settings.
    """

    def __init__(
        self,
        persistence_directory: str = "vector_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Prepare the storage directory and load the embedding pipeline.

        Constructing this object does **not** read any existing index; the
        instance starts empty and the caller must invoke ``load()`` or
        ``build_from_documents()`` before querying.

        Args:
            persistence_directory: Where the three index files live. Created if
                absent.
            embedding_model: Model id used for both chunk and query encoding.
                Recorded in the manifest and enforced on load.
            chunk_size: Characters per chunk, applied at build time.
            chunk_overlap: Characters shared between neighbouring chunks.

        Side effects:
            Creates the directory, and loads the embedding model into memory
            (downloading it on first run).
        """
        # Creating the persistence directory if it doesn't exist
        self.persistence_directory = persistence_directory
        os.makedirs(self.persistence_directory, exist_ok=True)

        # Populated by load() or add_embeddings(); querying before either is an
        # error that search() reports rather than raises.
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.manifest: Optional[Dict[str, Any]] = None

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # One pipeline for the whole lifetime of the store: index-time chunking
        # and embedding, and query-time encoding, all go through this object.
        self.pipeline = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        print(
            f"[DEBUG] Initialized VectorStore with persistence directory: {self.persistence_directory}"
        )

    @property
    def faiss_path(self) -> str:
        """Absolute-or-relative path to the FAISS index file."""
        return os.path.join(self.persistence_directory, "faiss.index")

    @property
    def metadata_path(self) -> str:
        """Path to the pickled metadata records."""
        return os.path.join(self.persistence_directory, "metadata.pkl")

    def exists(self) -> bool:
        """Whether a previously saved index is present on disk.

        Checks only for the two files the index cannot function without; the
        manifest is optional at this level so that a pre-manifest index is
        still reported as existing (and then rejected by ``load``, which can
        explain why).
        """
        return os.path.exists(self.faiss_path) and os.path.exists(self.metadata_path)

    def build_from_documents(self, documents: List[Any], files: Optional[Dict[str, str]] = None):
        """Run the full ingestion pipeline and persist the result.

        Chains chunk -> embed -> add -> save, and is the only method that writes
        a complete index.

        Args:
            documents: ``Document`` objects from ``load_all_documents``.
            files: Optional path -> SHA-256 map from
                ``manifest.fingerprint_corpus``, recorded so staleness can be
                detected later. Omitting it still builds a usable index, but
                one whose corpus cannot be diffed against disk.

        Returns:
            None. The result is the side effect: three files written to
            ``persistence_directory``.

        Raises:
            ValueError: when ``documents`` yields no chunks. Previously this
                surfaced as an opaque IndexError from reading ``shape[1]`` of an
                empty array; an empty corpus is a configuration mistake and
                deserves to say so.

        Side effects:
            Spends most of its wall-clock time embedding, then overwrites every
            file in the persistence directory.

        Resets in-memory state first, so this always *replaces* rather than
        appends. Calling it twice no longer doubles the index.
        """
        print(f"[DEBUG] Building vector store from {len(documents)} documents.")

        # Discard any previously loaded index so a rebuild is a true rebuild.
        self.index = None
        self.metadata = []

        chunks = self.pipeline.chunk_documents(documents)
        if not chunks:
            raise ValueError(
                f"No chunks were produced from {len(documents)} documents. "
                "Check that the data directory contains supported, non-empty files."
            )

        embeddings = self.pipeline.embed_chunks(chunks)
        # Flatten each chunk into one storable record: the text under "content",
        # plus the loader-supplied metadata (source, page, ...) merged in at the
        # top level via ** unpacking. Keeping text and provenance in a single
        # dict is what lets a search hit be rendered as a citation.
        metadatas = [
            {"content": chunk.page_content, **chunk.metadata} for chunk in chunks
        ]
        self.add_embeddings(np.array(embeddings).astype(np.float32), metadatas)

        self.manifest = manifest_mod.build_manifest(
            embedding_model=self.embedding_model,
            dimension=int(self.index.d),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunk_count=len(chunks),
            files=files or {},
        )
        self.save()
        print(
            f"[DEBUG] Vector store built and saved with {len(chunks)} chunks at {self.persistence_directory}"
        )

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]):
        """Append vectors and their records to the in-memory index.

        Args:
            embeddings: float32 array of shape (n, dim). Assumed unit-length --
                the inner-product index below only equals cosine similarity if
                that holds. Normalisation happens in
                ``EmbeddingPipeline.embed_chunks``, not here, and is not verified.
            metadatas: n records, positionally aligned with ``embeddings``.

        Side effects:
            Creates the FAISS index on first call, inferring dimension from the
            data, and mutates ``self.index`` / ``self.metadata``. Does not save.

        IndexFlatIP is exhaustive and exact: every query is compared against
        every stored vector. That is the right trade at this corpus size (it
        avoids the accuracy loss and tuning burden of an approximate index) but
        search cost grows linearly with the number of chunks.
        """
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            print(f"[DEBUG] Created new FAISS index with dimension: {dimension}")

        self.index.add(embeddings)
        self.metadata.extend(metadatas) if metadatas else []
        print(
            f"[DEBUG] Added {len(embeddings)} embeddings to the vector store. Total embeddings: {self.index.ntotal}"
        )

    def save(self):
        """Write the index, its records, and the manifest to disk.

        Side effects:
            Overwrites all three files without backup or confirmation.

        Write order matters: the manifest goes last, so its presence implies the
        index and metadata are already complete. A crash mid-save therefore
        leaves an index with no (or an older) manifest, which ``load`` rejects
        -- failing closed rather than serving a half-written index.

        Prints an error and does nothing when there is no index to write, since
        an empty corpus is a plausible misconfiguration rather than a crash.
        """
        if self.index is None:
            print("[ERROR] No index to save.")
            return

        faiss.write_index(self.index, self.faiss_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        if self.manifest is not None:
            manifest_mod.save_manifest(self.persistence_directory, self.manifest)
        print(f"[DEBUG] Vector store saved at {self.persistence_directory}")

    def load(self, strict: bool = True):
        """Restore a previously saved index, verifying it is safe to query.

        Args:
            strict: When True, an index that cannot be shown compatible with the
                current configuration raises instead of loading. When False the
                problem is printed as a warning and the index is used anyway --
                intended for inspection and debugging, not for answering.

        Raises:
            FileNotFoundError: when either required file is missing.
            RuntimeError: when the manifest is absent or names a different
                embedding model.

        Side effects:
            Replaces ``self.index``, ``self.metadata`` and ``self.manifest``.

        Why this validates at all: querying an index built by a *different*
        embedding model is the worst failure mode in the system. A different
        vector dimension at least raises inside FAISS, but a same-dimension
        model returns plausible-looking neighbours that are simply wrong. The
        manifest turns that silent corruption into an explicit error.

        Security note: ``pickle.load`` executes arbitrary code embedded in the
        file. Safe for a file this program wrote; never point this at a
        metadata.pkl from an untrusted source.
        """
        if not self.exists():
            raise FileNotFoundError(
                f"No saved vector store in {self.persistence_directory!r}. "
                "Run ingestion first (python app.py ingest)."
            )

        stored_manifest = manifest_mod.load_manifest(self.persistence_directory)
        problem = manifest_mod.check_compatibility(stored_manifest, self.embedding_model)
        if problem:
            message = (
                f"Refusing to load the index in {self.persistence_directory!r}: {problem}. "
                "Rebuild it with: python app.py ingest --rebuild"
            )
            if strict:
                raise RuntimeError(message)
            print(f"[WARN] {message}")

        self.index = faiss.read_index(self.faiss_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.manifest = stored_manifest

        # A dimension disagreement would otherwise surface as an opaque FAISS
        # assertion on the first query, far from its cause.
        expected_dim = self.pipeline.embedding_dimension()
        if self.index.d != expected_dim:
            raise RuntimeError(
                f"Index dimension {self.index.d} does not match model "
                f"{self.embedding_model!r} (dimension {expected_dim}). "
                "Rebuild it with: python app.py ingest --rebuild"
            )

        print(
            f"[DEBUG] Vector store loaded from {self.persistence_directory} "
            f"({self.index.ntotal} vectors)"
        )

    def check_stale(self, current_files: Dict[str, str]) -> Optional[str]:
        """Report whether the source corpus changed since the index was built.

        Args:
            current_files: Result of ``manifest.fingerprint_corpus``.

        Returns:
            None when up to date, otherwise a description of what changed.
            Delegates to ``manifest.is_stale``; exposed here so callers do not
            need to reach into the stored manifest themselves.
        """
        return manifest_mod.is_stale(self.manifest, current_files)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Find the ``top_k`` most similar stored chunks to a query vector.

        Args:
            query_embedding: float32 array of shape (1, dim). Only the first row
                is read, so this is effectively single-query.
            top_k: How many neighbours to consider.
            min_score: Optional cosine-similarity floor. Hits below it are
                dropped, so the result may contain fewer than ``top_k`` entries
                -- including none at all.

        Returns:
            A list of dicts, best match first, each with:
                index    -- row id in the FAISS index
                metadata -- the aligned record, or None if the lists disagree
                score    -- cosine similarity in [-1, 1]; higher is more similar

            Returns [] when no index is loaded.

        Why ``min_score`` matters: FAISS always returns its k nearest
        neighbours, however weak. Without a floor, a question unrelated to the
        corpus still yields top_k rows, the caller builds a non-empty context
        from them, and the LLM answers confidently over irrelevant text. The
        threshold is what lets the system distinguish "no good match" from
        "here are the best matches".
        """
        if self.index is None:
            print(
                "[ERROR] No index loaded. Please load or build the vector store first."
            )
            return []

        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            # FAISS pads with -1 when the index holds fewer than top_k vectors.
            if idx < 0:
                continue
            if min_score is not None and float(score) < min_score:
                # Sorted best-first, so everything after this is weaker too.
                break
            # Defensive: a metadata list shorter than the index means the two
            # files fell out of sync, so degrade to None instead of IndexError.
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": int(idx), "metadata": meta, "score": float(score)})
        return results

    def query(
        self, query: str, top_k: int = 5, min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Embed a natural-language question and retrieve matching chunks.

        The query-time counterpart to ``build_from_documents`` and the entry
        point ``RAGSearch`` calls. Encoding goes through the same
        ``EmbeddingPipeline`` used at index time, which is what keeps the two
        sides consistent.

        Args:
            query: The question text, embedded as-is. Any conversational
                rewriting happens upstream in ``RAGSearch``.
            top_k: Number of chunks to consider.
            min_score: Optional similarity floor; see ``search``.

        Returns:
            The result list described in ``search``.
        """
        return self.search(self.pipeline.embed_query(query), top_k, min_score)
