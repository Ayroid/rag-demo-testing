"""RAG pipeline components.

Modules are ordered by their position in the data flow:

    data_loader  -- files on disk           -> Document objects
    embedding    -- Document objects        -> chunks -> vectors
    vectorstore  -- vectors + chunk records -> persisted, searchable FAISS index
    search       -- question                -> retrieved context -> LLM answer

Dependencies run one way, upward through that list: ``search`` imports
``vectorstore``, which imports ``embedding``, which imports ``data_loader``.

Deliberately left free of re-exports. Modules import heavyweight dependencies
(torch, faiss, document parsers) at import time, so pulling them all in from the
package root would make even a trivial import slow.
"""
