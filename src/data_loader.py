"""Stage 1 of the ingestion pipeline: turn files on disk into LangChain Documents.

This module is the only place that knows about file formats. Everything
downstream (chunking, embedding, storage) operates on a uniform
``langchain_core.documents.Document`` stream and never touches a raw file
again. Adding support for a new format means adding one entry to ``LOADERS``
and changing nothing else in the pipeline.
"""

from pathlib import Path
from typing import Callable, List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, JSONLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

# Glob pattern -> (label used in logs, loader factory taking the file path)
#
# The registry is the extension point of this module: the loop in
# load_all_documents() is format-agnostic and simply walks these entries. Each
# value is a *factory* (not an instance) because loaders bind to one file path
# at construction time and must be re-created per file.
#
# Note that different loaders produce different document granularity:
# PyPDFLoader emits one Document per page (with a "page" metadata key), while
# TextLoader emits a single Document for the whole file. Chunking downstream
# normalises this difference away.
LOADERS: dict[str, tuple[str, Callable[[str], object]]] = {
  "**/*.pdf": ("PDF", PyPDFLoader),
  "**/*.txt": ("TXT", TextLoader),
  "**/*.csv": ("CSV", CSVLoader),
  "**/*.docx": ("DOCX", Docx2txtLoader),
  # JSONLoader has no usable default: jq_schema="." selects the whole document
  # and text_content=False permits non-string values, which together make it
  # behave as a generic "dump the JSON as text" loader.
  "**/*.json": ("JSON", lambda path: JSONLoader(path, jq_schema=".", text_content=False)),
  "**/*.xlsx": ("Excel", UnstructuredExcelLoader),
}

def load_all_documents(data_dir: str) -> List[Document]:
  """Load all supported documents and convert them into LangChain document structure.
     Supported documents: pdf, txt, csv, docx, json, excel

  Walks ``data_dir`` recursively once per registered format and concatenates
  everything into a single flat list. This is the entry point of the ingestion
  pipeline; its output feeds ``EmbeddingPipeline.chunk_documents``.

  Args:
    data_dir: Directory to scan. Resolved to an absolute path, and searched
      recursively (the patterns in ``LOADERS`` are all prefixed with ``**/``).

  Returns:
    A flat list of ``Document`` objects. Each carries ``.page_content`` (the
    extracted text) and ``.metadata`` (loader-dependent, but normally includes
    at least ``source`` with the file path, plus ``page`` for PDFs).

    Ordering is by format, following ``LOADERS`` insertion order (all PDFs,
    then all TXTs, ...), not by filename or directory.

  Side effects:
    Prints progress and errors to stdout. Reads from disk but writes nothing.

  Failure behaviour:
    A file that fails to load is reported and skipped -- one corrupt PDF cannot
    abort a whole ingest. This is deliberate, but it is a *silent partial
    success*: the caller receives no signal that anything was missing, so the
    printed [ERROR] lines are the only record. A missing or empty ``data_dir``
    is likewise not an error; ``glob`` simply yields nothing and this function
    returns an empty list.
  """

  # Use project root data folder
  data_path = Path(data_dir).resolve()
  print(f"[DEBUG] Data Path: {data_path}")
  documents = []

  for pattern, (label, make_loader) in LOADERS.items():
    files = list(data_path.glob(pattern))
    print(f"[DEBUG] Found {len(files)} {label} files: {[str(f) for f in files]}")
    for file in files:
      print(f"[DEBUG] Loading {label}: {file}")
      try:
        loaded_docs = make_loader(str(file)).load()
        print(f"[DEBUG] Loaded {len(loaded_docs)} documents from {label}: {file}")
        documents.extend(loaded_docs)
      except Exception as e:
        # Broad catch is intentional: loaders raise a wide variety of
        # library-specific exceptions (parse errors, encoding errors, missing
        # optional dependencies) and none of them should be fatal to the batch.
        print(f"[ERROR] Failed to load {label} {file}: {e}")

  return documents
