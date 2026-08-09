"""Index provenance: what was indexed, with which model, and when.

Exists to answer two questions that a bare FAISS index cannot:

1. **Is the index stale?** Source files change, but nothing in faiss.index
   records which files went into it. Hashing the corpus and comparing lets
   ingestion become idempotent -- safe to run every time, doing work only when
   something actually changed.
2. **Was the index built by a compatible configuration?** Query vectors from a
   different embedding model live in a different space than the stored ones.
   FAISS raises only when the dimension also differs; a same-dimension model
   swap produces silently wrong rankings. Recording the model name turns that
   into a loud failure.

The manifest is written alongside faiss.index and metadata.pkl as
``manifest.json``, and is plain JSON so it can be inspected by hand.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_FILENAME = "manifest.json"

# Read in blocks rather than whole-file so a large PDF does not have to be held
# in memory purely to be fingerprinted.
_HASH_BLOCK_SIZE = 1024 * 1024


def hash_file(path: str) -> str:
    """Return a SHA-256 hex digest of a file's contents.

    Content-based rather than mtime-based: copying a file or touching it must
    not read as a change, while an edit that preserves size must.

    Args:
        path: File to fingerprint.

    Returns:
        Hex digest string.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_corpus(data_dir: str, patterns: List[str]) -> Dict[str, str]:
    """Hash every source file that ingestion would pick up.

    Mirrors the discovery logic in ``load_all_documents`` -- the same directory
    and the same glob patterns -- so the fingerprint covers exactly the files
    that would be indexed, no more and no less. Passing the patterns in (rather
    than importing LOADERS) keeps this module free of the heavyweight document
    parsers.

    Args:
        data_dir: Corpus root, searched recursively.
        patterns: Glob patterns, normally ``LOADERS.keys()``.

    Returns:
        Mapping of path (relative to cwd where possible, for portability) to
        SHA-256 digest. Sorted by key so the JSON is diff-friendly.

        A file that cannot be read is skipped rather than raising -- it would
        also fail during loading, and reporting that is the loader's job.
    """
    root = Path(data_dir).resolve()
    fingerprints: Dict[str, str] = {}

    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            try:
                key = os.path.relpath(path, Path.cwd())
            except ValueError:
                # Different drive on Windows; absolute path is the best we can do.
                key = str(path)
            try:
                fingerprints[key] = hash_file(str(path))
            except OSError:
                continue

    return dict(sorted(fingerprints.items()))


def build_manifest(
    embedding_model: str,
    dimension: int,
    chunk_size: int,
    chunk_overlap: int,
    chunk_count: int,
    files: Dict[str, str],
) -> Dict[str, Any]:
    """Assemble the manifest record for an index that was just built.

    Every field is something that, if changed, invalidates the index: a
    different model or dimension makes vectors incomparable, different chunk
    settings make retrieval behave differently, and different file hashes mean
    the content is out of date.

    Returns:
        A JSON-serialisable dict.
    """
    return {
        "embedding_model": embedding_model,
        "dimension": dimension,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_count": chunk_count,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }


def save_manifest(persistence_directory: str, manifest: Dict[str, Any]) -> None:
    """Write the manifest as indented JSON next to the index files.

    Side effects:
        Overwrites ``manifest.json``. Written last by ``VectorStore.save`` so
        that its presence implies the index files are already complete.
    """
    path = os.path.join(persistence_directory, MANIFEST_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(persistence_directory: str) -> Optional[Dict[str, Any]]:
    """Read the manifest, or return None when it is absent or unreadable.

    Returns None rather than raising for the legitimate case of an index built
    before manifests existed, and for a corrupt file -- in both situations the
    caller's correct response is the same: treat the index as unverifiable.
    """
    path = os.path.join(persistence_directory, MANIFEST_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def check_compatibility(
    manifest: Optional[Dict[str, Any]], embedding_model: str
) -> Optional[str]:
    """Verify a stored index can be queried with the current configuration.

    This is the check that prevents the worst silent failure in the system:
    querying an index with a model it was not built with. A dimension mismatch
    would at least raise inside FAISS; a same-dimension mismatch would not, and
    would simply return meaningless neighbours.

    Args:
        manifest: Result of ``load_manifest``, possibly None.
        embedding_model: The model the caller intends to query with.

    Returns:
        None when compatible, otherwise a human-readable reason. A missing
        manifest is reported as a reason too -- an unverifiable index is treated
        as suspect, since it predates this check and may have been built with
        anything.
    """
    if manifest is None:
        return (
            f"no {MANIFEST_FILENAME} found, so the index cannot be verified "
            "(it predates manifest support, or was built by other code)"
        )

    stored_model = manifest.get("embedding_model")
    if stored_model != embedding_model:
        return (
            f"index was built with embedding model {stored_model!r} but "
            f"{embedding_model!r} is configured; their vectors are not comparable"
        )

    return None


def is_stale(
    manifest: Optional[Dict[str, Any]], current_files: Dict[str, str]
) -> Optional[str]:
    """Compare the indexed corpus against the corpus on disk right now.

    Args:
        manifest: Result of ``load_manifest``, possibly None.
        current_files: Result of ``fingerprint_corpus``.

    Returns:
        None when the index is up to date, otherwise a summary of what changed
        -- added, removed, and modified files, capped so the message stays
        readable on a large corpus.
    """
    if manifest is None:
        return "no manifest to compare against"

    indexed = manifest.get("files", {})
    added = sorted(set(current_files) - set(indexed))
    removed = sorted(set(indexed) - set(current_files))
    modified = sorted(
        path
        for path in set(indexed) & set(current_files)
        if indexed[path] != current_files[path]
    )

    if not (added or removed or modified):
        return None

    def summarise(label: str, paths: List[str]) -> str:
        shown = ", ".join(Path(p).name for p in paths[:3])
        extra = f" (+{len(paths) - 3} more)" if len(paths) > 3 else ""
        return f"{len(paths)} {label} [{shown}{extra}]"

    parts = [
        summarise(label, paths)
        for label, paths in (
            ("added", added),
            ("removed", removed),
            ("modified", modified),
        )
        if paths
    ]
    return "; ".join(parts)
