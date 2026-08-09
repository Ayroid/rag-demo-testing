# RAG Pipeline — Complete Documentation

**From "what is AI?" to "how does this specific codebase work?"**

This document assumes no prior knowledge of AI, language models, embeddings, vector databases, or RAG. It builds each concept up in order, then shows exactly how this project implements it, referencing real files, classes, and methods.

> **A note on accuracy.** Sections 1–7 describe **what the code does today**. Where the implementation has a gap, the gap is stated plainly rather than glossed over. Anything not yet built appears only under headings explicitly marked **Recommended** or **Not implemented**. Nothing in the "how it works" sections describes aspirational behaviour.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [What is RAG?](#2-what-is-rag)
3. [Why RAG is Needed](#3-why-rag-is-needed)
4. [Core RAG Concepts](#4-core-rag-concepts)
5. [Architecture of This Implementation](#5-architecture-of-this-implementation)
6. [The Codebase, File by File](#6-the-codebase-file-by-file)
7. [End-to-End Execution Flow](#7-end-to-end-execution-flow)
8. [Configuration Reference](#8-configuration-reference)
9. [Error Handling and Edge Cases](#9-error-handling-and-edge-cases)
10. [Security and Production Considerations](#10-security-and-production-considerations)
11. [Performance Considerations](#11-performance-considerations)
12. [Example Walkthrough](#12-example-walkthrough)
13. [Technical Glossary](#13-technical-glossary)
14. [Known Gaps and Recommended Improvements](#14-known-gaps-and-recommended-improvements)

---

## 1. Introduction

### What is AI, at a high level?

"Artificial Intelligence" here means one specific thing: **software that learned a task from examples rather than from rules a programmer wrote**.

A traditional program is explicit. To detect spam you might write `if subject contains "FREE MONEY" then spam`. Every rule is authored by hand, and the program does exactly what it was told.

A machine-learned program is different. You show it a million emails already labelled spam or not-spam, and a training process adjusts millions of internal numbers ("parameters") until the system reproduces those labels well. Nobody writes the rules. They are *discovered*, and they end up encoded as numbers nobody can read directly.

That distinction explains most of the strange behaviour later in this document — including why these systems make things up.

### What is an LLM?

A **Large Language Model** is that idea applied to text, at enormous scale.

An LLM is trained on a very large amount of text with one deceptively simple objective: **predict the next word.**

> "The capital of France is ___"

To predict `Paris` reliably, the model has to internalise facts. To continue a Python function correctly, it has to internalise syntax and logic. Trained on trillions of words, next-word prediction forces the model to absorb grammar, facts, reasoning patterns, and style — all as numeric parameters.

When you "chat" with an LLM, it is still only doing this. It reads your text and generates a plausible continuation, one piece at a time. This project uses **`llama-3.1-8b-instant`**, served by **Groq**.

Two consequences matter enormously for RAG:

1. **The model has no database.** It cannot look anything up. Facts are smeared across billions of parameters, not stored as retrievable records.
2. **It is trained to be plausible, not correct.** Nothing in "predict the next word" rewards accuracy over fluency. A confident wrong answer and a confident right answer look identical to the model's objective.

### The problems with using an LLM alone

**Problem 1 — Its knowledge is frozen.** Training ended on some date. Ask about anything after it and the model either declines or invents an answer.

**Problem 2 — It has never seen your private data.** It was trained on public text. It has never read your company's HR policy, your internal wiki, or the PDF sitting in this project's [`data/`](../data/) folder. It cannot know what it never saw.

**Problem 3 — It hallucinates.** When it lacks a fact, it does not reliably say "I don't know." It generates the most *plausible-sounding* continuation. Ask about a policy it has never seen and it will happily invent one, in the correct tone, with realistic specifics. This is the single most dangerous failure mode.

**Problem 4 — No sources.** Even when correct, it cannot tell you *where* it learned something. There is nothing to verify against.

### Why can't we just give the LLM all our data?

Two reasons.

**Retraining is out of reach.** Training a model from scratch costs millions of dollars and months of compute. Fine-tuning an existing one is cheaper but still requires expertise and a training run for *every* document change.

**The context window is finite.** You could paste documents directly into your question — the model's working memory is called the **context window**. But it has a hard size limit measured in **tokens** (roughly ¾ of a word each). This project's model tops out around 128,000 tokens ≈ 100,000 words. The corpus in this repo — one book-length PDF plus two short text files — produced **1,099 chunks of up to 1,000 characters each**, on the order of 250,000 tokens. It does not fit. And even when content fits, you pay for every token on every question, and accuracy tends to fall as irrelevant material crowds the window.

### What RAG solves

RAG resolves the tension directly: **don't send everything — find the few relevant pieces first, and send only those.**

The model no longer needs to *know* your data. It needs to *read* the excerpt you hand it and answer from that. Facts move out of the frozen parameters and into the prompt, where they can be updated by simply changing a file.

---

## 2. What is RAG?

**Retrieval-Augmented Generation.** Read it backwards:

- **Generation** — an LLM writes an answer.
- **Augmented** — its input has been enriched with extra material.
- **Retrieval** — that material was fetched from your documents by searching them.

> **The core idea in one sentence:** before asking the LLM anything, search your documents for the passages most relevant to the question, paste those passages into the prompt, and instruct the model to answer using them.

### The basic flow

```text
                        User Question
                              ↓
                 Retrieve Relevant Information
                              ↓
                Provide Retrieved Info to the LLM
                              ↓
                     LLM Generates Answer
                              ↓
                    Answer + Sources/Citations
```

### The open-book exam analogy

Picture a student sitting an exam.

**Without RAG — closed book.** The student answers purely from memory. Strong on well-studied topics; on anything half-remembered, a confident guess. They cannot cite a page. *This is a bare LLM.*

**With RAG — open book, with a research assistant.** The student gets the question, and an assistant sprints to the library, finds the three most relevant pages, and drops them on the desk. The student reads those pages and answers from them, quoting page numbers.

Mapping the analogy onto the stages:

| Stage | In the analogy | In this codebase |
|---|---|---|
| User Question | The exam question | The `question` argument to `python app.py ask` |
| Retrieve | Assistant finds the right pages | `VectorStore.query()` searches the FAISS index |
| Provide to LLM | Pages placed on the desk | `RAGSearch.build_context()` numbers and labels the chunks |
| Generate | Student writes the answer | `ChatGroq.invoke()` calls the Groq API |
| Answer + Sources | The answer, with page numbers cited | `Answer.text` plus `Answer.sources` — see §4.10 |

The assistant is the crucial part. It never *answers* anything; it only *finds*. Everything from §4.5 to §4.7 is about building a good assistant.

---

## 3. Why RAG is Needed

| Problem | Why it hurts | How RAG addresses it | Status in this project |
|---|---|---|---|
| **Knowledge cutoff** | Model knows nothing past training | Answers come from files read at query time | ✅ Works — edit a file, re-ingest, answers change |
| **Private data** | Model never saw your documents | Your documents *are* the knowledge source | ✅ Works — indexes [`data/`](../data/) |
| **Outdated information** | Retraining to refresh is infeasible | Replace a file and re-index | ✅ Works — `python app.py ingest` detects changes by file hash (§4.11) |
| **Hallucination** | Invents plausible falsehoods | Grounds the answer in real retrieved text | ✅ Works — score threshold + grounding rules + explicit refusal (§4.7, §4.8) |
| **No source attribution** | Cannot verify claims | Every chunk knows which file it came from | ✅ Works — numbered citations with file and page (§4.10) |
| **Large collections** | Too big for the context window | Send only the top-K relevant chunks | ✅ Works — 1,099 chunks indexed, 5 sent |

---

## 4. Core RAG Concepts

Concepts build in the order data flows through the system.

### 4.1 Documents

A **document** is any file holding knowledge you want answerable: a PDF, a Word file, a spreadsheet, a plain text file.

Files cannot be searched *meaningfully* in their raw form. A PDF is a layout format describing glyph positions, not a bag of sentences. An `.xlsx` is a zip archive of XML. Before anything else, every format must become plain text in a uniform structure.

In this project, that uniform structure is LangChain's `Document` object — two fields:

```python
Document(
    page_content="The actual text extracted from the file...",
    metadata={"source": "data/pdf_files/Nature-of-code.pdf", "page": 0}
)
```

The exact metadata keys depend on the loader. `PyPDFLoader` is generous — a real chunk from this corpus carries nine: `source`, `page`, `page_label`, `total_pages`, `title`, `creator`, `producer`, `creationdate`, `moddate`. `TextLoader` supplies only `source`.

`page_content` is what gets searched. `metadata` is what makes citation possible. **Keeping metadata attached from this point onward is what determines whether the finished system can cite its sources** — a theme that recurs in §4.4.

### 4.2 Document Loading

**Loading** is discovering files on disk and reading them in.

Implemented in [`src/data_loader.py`](../src/data_loader.py) by `load_all_documents(data_dir)`. It resolves the directory to an absolute path, then walks it **recursively** — every pattern is prefixed `**/`, so nested folders like `data/pdf_files/` are found automatically.

The current corpus:

```text
data/
├── pdf_files/
│   └── Daniel-Shiffman_Nature-of-code.pdf   (a book on simulating nature with code)
└── text_files/
    ├── python_intro.txt
    └── typescript_intro.txt
```

### 4.3 Document Parsing

**Parsing** is extracting readable text from a format-specific container. Each format needs its own logic — a PDF parser is useless on a spreadsheet.

The project uses a **registry** pattern. `LOADERS` in [`src/data_loader.py`](../src/data_loader.py) maps a glob pattern to a label and a loader class:

| Pattern | Label | Loader | Library | Granularity |
|---|---|---|---|---|
| `**/*.pdf` | PDF | `PyPDFLoader` | `pypdf` | **One Document per page** |
| `**/*.txt` | TXT | `TextLoader` | built-in | One Document per file |
| `**/*.csv` | CSV | `CSVLoader` | built-in | One Document per row |
| `**/*.docx` | DOCX | `Docx2txtLoader` | `docx2txt` | One Document per file |
| `**/*.json` | JSON | `JSONLoader` | `jq` | Depends on schema |
| `**/*.xlsx` | Excel | `UnstructuredExcelLoader` | `unstructured` | One Document per sheet |

Two things worth understanding here:

**Granularity varies by format.** PDFs arrive pre-split by page, text files arrive whole. Chunking (§4.4) normalises this away, so the rest of the pipeline never notices.

**The registry is the extension point.** The loop over `LOADERS` is entirely format-agnostic. Supporting Markdown means adding one line; no other file changes.

> **Dependency caveat.** `Docx2txtLoader` and `UnstructuredExcelLoader` need `docx2txt` and `unstructured`, which are **not** declared in [pyproject.toml](../pyproject.toml). Those two formats will fail at load time unless the packages are installed. The failure is non-fatal (§9), so it appears as an `[ERROR]` line and the file is skipped.

### 4.4 Chunking

**Chunking** splits long documents into smaller pieces. It is the least intuitive step and the one that most determines retrieval quality.

**Why not index whole documents?** Three reasons:

1. **Precision.** If a 300-page book is one searchable unit, a search can only tell you "the answer is somewhere in this book."
2. **Context window.** You cannot paste a whole book into a prompt (§1).
3. **Dilution.** Embeddings (§4.5) compress text into a single vector. Compress a whole book and you get a vague "this is about programming" average, where every specific idea has been washed out. Small chunks produce sharp, specific vectors.

**Why not tiny chunks then?** Because a chunk must be self-contained enough to *answer* something. `"It uses a Perlin noise function."` is precise but useless — what does "it" refer to? Chunking is a trade-off between precision and sufficient context.

#### Chunk size and overlap

**Chunk size** is the target maximum length of a piece. This project uses **1,000 characters** (~150–200 words, a healthy paragraph). Note this counts *characters*, not tokens.

**Chunk overlap** means consecutive chunks share their boundary text. This project uses **200 characters**. Overlap exists because a hard cut can sever a sentence that answers the question:

```text
Without overlap:
  Chunk 1: "...the algorithm uses three forces. The first is"
  Chunk 2: "gravity, applied downward each frame..."
           ↑ Neither chunk alone answers "what is the first force?"

With 200-char overlap:
  Chunk 1: "...the algorithm uses three forces. The first is"
  Chunk 2: "The first is gravity, applied downward each frame..."
           ↑ Chunk 2 is now self-contained.
```

The cost is redundancy: ~20% more chunks to store and search.

#### The strategy actually implemented

`EmbeddingPipeline.chunk_documents()` in [`src/embedding.py`](../src/embedding.py) uses LangChain's `RecursiveCharacterTextSplitter` with:

```python
separators=["\n\n", "\n", " ", ""]
```

**"Recursive" refers to that separator list.** The splitter tries to cut on the *first* separator; if a resulting piece still exceeds 1,000 characters, it retries with the next, and so on:

1. `"\n\n"` — split on paragraph breaks. Best case: chunks are whole paragraphs.
2. `"\n"` — paragraph still too long? Split on line breaks.
3. `" "` — still too long? Split on spaces, keeping words intact.
4. `""` — last resort: cut mid-word at exactly 1,000 characters.

The effect is that cuts land on the **largest natural boundary available**, so chunks stay semantically whole instead of being sliced arbitrarily.

**Measured result:** the current corpus — 523 loaded Documents (521 PDF pages + 2 text files) — produces **1,099 chunks**.

The method calls `splitter.split_documents(documents)`, which returns `Document` objects and **copies each parent's metadata onto every derived chunk**. This is what carries `source` and `page` forward into the vector store, and is precisely what makes citations possible later.

### 4.5 Embeddings

This is the conceptual heart of RAG.

**The problem.** How does a computer find text that is *relevant* to a question? Keyword matching fails constantly:

- Question: *"How do I make objects fall realistically?"*
- Document: *"Gravitational acceleration is applied to each vector every frame."*

Zero shared keywords. Same topic. Keyword search finds nothing.

**The solution.** Convert text into a list of numbers — a **vector**, or **embedding** — positioned such that *similar meanings produce nearby numbers*.

A tiny illustration in 3 dimensions (the real model uses 384):

```text
"cat"     → [0.9, 0.8, 0.1]
"kitten"  → [0.9, 0.7, 0.2]   ← nearly identical to "cat"
"dog"     → [0.8, 0.9, 0.1]   ← close: also a pet animal
"database"→ [0.1, 0.2, 0.9]   ← far away: unrelated concept
```

Meaning has become **geometry**. "Find related text" becomes "find nearby points" — and that is arithmetic a computer does extremely fast.

An **embedding model** is a neural network trained to produce these vectors. It learned from millions of examples that "cat" and "kitten" appear in similar contexts and should land near each other.

#### Semantic similarity

Two vectors' closeness is measured by **cosine similarity** — the angle between them, ignoring length:

| Score | Meaning |
|---|---|
| `1.0` | Identical direction — same meaning |
| `0.7` | Strongly related |
| `0.4` | Loosely related |
| `0.0` | Unrelated (perpendicular) |
| `-1.0` | Opposed |

Back to the failing example — semantically, these land close together despite sharing no words:

```text
"How do I make objects fall realistically?"       → [0.71, 0.22, ...]
"Gravitational acceleration applied per frame"    → [0.68, 0.25, ...]
                                    cosine similarity ≈ 0.8  ✓ found
```

#### The model actually used

**`all-MiniLM-L6-v2`**, run locally via the `sentence-transformers` library.

| Property | Value |
|---|---|
| Output dimension | **384** *(verified against the built index)* |
| Runs | Locally, on CPU |
| Cost | **Free** — no API calls |
| Download | ~90 MB from Hugging Face on first run, then cached in `~/.cache` |
| Speed | Thousands of chunks per minute on CPU |

It is small and fast rather than maximally accurate — the right trade for a project of this size, and it means ingestion has no API cost at all.

#### Normalisation — why it matters here

Both call sites pass `normalize_embeddings=True`, scaling every vector to length 1. This is not cosmetic. For unit-length vectors, the **inner product is mathematically identical to cosine similarity**. Normalising lets a fast inner-product index compute true cosine similarity for free (§4.6).

The two call sites must agree — index time in `embed_chunks()`, query time in `VectorStore.query()`. Both currently do.

### 4.6 Vector Database

A **vector database** stores embeddings and answers the question *"which stored vectors are closest to this one?"* quickly.

#### Why not just use SQL?

A SQL database excels at exact matching:

```sql
SELECT * FROM docs WHERE content LIKE '%gravity%';
```

That finds the literal string `gravity`. It cannot find *"objects falling downward"*, because SQL has no notion of meaning — only characters.

You might try storing the 384 numbers in columns and computing similarity in SQL. It works, and it is unusably slow: comparing against every row with no index for the purpose. **Vector databases exist to make nearest-neighbour search over high-dimensional vectors fast**, which is a genuinely hard problem that ordinary B-tree indexes do not solve.

#### The vector store actually used

This project uses **FAISS** (Facebook AI Similarity Search) — a library, not a server. There is no daemon and no connection string; the index is an in-memory object saved to a file.

Implemented in `VectorStore` in [`src/vectorstore.py`](../src/vectorstore.py), which persists **two files** to `vector_store/`:

```text
vector_store/
├── faiss.index     # the 384-dim vectors + search structure
└── metadata.pkl    # Python list of dicts: the chunk text and its provenance
```

**These two files are meaningless apart.** FAISS stores only vectors and returns only integer row IDs. Row 47 means nothing until `metadata.pkl` is consulted:

```python
metadata[47] == {
    "content": "Gravitational acceleration is applied...",
    "source":  "data/pdf_files/Daniel-Shiffman_Nature-of-code.pdf",
    "page":    23,
    "page_label": "24", "total_pages": 521, "title": "The Nature of Code",
    "creator": "...", "producer": "...", "creationdate": "...", "moddate": "...",
}
```

Every loader-supplied key is merged in by the `**chunk.metadata` unpacking in `build_from_documents()`, so records carry whatever the loader produced — for PDFs, all nine keys above alongside `content`.

The alignment is **purely positional** — vector row *i* ↔ `metadata[i]`. Nothing enforces it. Mixing files from different builds produces wrong text for every hit, with no error raised.

#### The index type

`add_embeddings()` creates **`faiss.IndexFlatIP`**:

- **`Flat`** — exhaustive. Every query is compared against all 1,099 vectors, guaranteeing the *exact* best matches. Approximate index types trade accuracy for speed at large scale; at ~1,000 vectors that trade is unnecessary.
- **`IP`** — Inner Product. Combined with normalised vectors (§4.5), this computes cosine similarity.

**Higher scores are better** — this is a similarity, not a distance. (An `IndexFlatL2` would invert that: lower = better.)

The index dimension is inferred from the first batch of data rather than configured, so it always matches the embedding model that built it.

### 4.7 Retrieval

**Retrieval** is the search step: question in, most relevant chunks out.

`VectorStore.query(query, top_k)` in [`src/vectorstore.py`](../src/vectorstore.py) does it in two steps:

```python
# 1. Encode the question with the same model, normalised the same way
query_embedding = self.model.encode(
    [query], convert_to_numpy=True, normalize_embeddings=True
).astype(np.float32)

# 2. Exhaustive nearest-neighbour search
score, indices = self.index.search(query_embedding, top_k)
```

`search()` then assembles results, best match first:

```python
{"index": 47, "metadata": {...}, "score": 0.72}
```

Two implementation details in `search()` are worth knowing:

- **`if idx < 0: continue`** — FAISS pads results with `-1` when the index holds fewer vectors than `top_k`. Skipping them prevents a nonsense lookup at index `-1` (which in Python would silently return the *last* element).
- **`meta = self.metadata[idx] if idx < len(self.metadata) else None`** — defensive guard against the two files falling out of sync; degrades to `None` instead of raising `IndexError`.

#### The relevance threshold

FAISS *always* returns K results, however weak the match. Ask *"What is the capital of France?"* against a corpus about generative art and it will still hand back 5 chunks — the least-irrelevant ones, scoring perhaps 0.05. Feeding those to an LLM is how a RAG system produces confident nonsense.

The guard is a **minimum cosine similarity**, applied inside `search()`:

```python
if min_score is not None and float(score) < min_score:
    break          # results are sorted best-first, so the rest are weaker too
```

`DEFAULT_MIN_SCORE = 0.30`, defined in [`src/search.py`](../src/search.py). It is calibrated for `all-MiniLM-L6-v2`, where on-topic passages typically score above ~0.35 and unrelated text clusters near zero. Observed scores on this corpus: a well-matched question returns hits in the **0.46–0.71** range; an off-topic one returns nothing at all.

When every hit is filtered out, `RAGSearch.answer()` returns the refusal **without calling the LLM** — saving a request and removing any chance to hallucinate over irrelevant text.

#### The retrieval strategy, stated precisely

**Pure dense vector search, top-K, exhaustive, score-filtered.**

- *Dense* — semantic embeddings only. No keyword/BM25 search, no hybrid.
- *Top-K* — a fixed ceiling, default **5**, overridable with `--top-k`.
- *Exhaustive* — exact, not approximate.
- *Score-filtered* — hits below `--min-score` (default 0.30) are dropped, so a query may return fewer than K chunks, or none.
- *No re-ranking* — vector order is final. A cross-encoder re-ranker would improve ordering; not implemented (§14).

### 4.8 Prompt Construction

A **prompt** is the complete text sent to the LLM. Prompt construction assembles retrieved chunks and the user's question into one string.

The prompt is built in two pieces: a fixed **system prompt** carrying the rules, and a **user message** carrying the numbered context and the question.

##### The system prompt

`SYSTEM_PROMPT` in [`src/search.py`](../src/search.py):

```text
You answer questions using ONLY the provided context.

Rules:
- If the context does not contain the answer, reply with exactly: "I don't know based on the provided documents."
- Never use outside knowledge. Never guess.
- Cite the source after each claim using the bracketed numbers of the context
  blocks you used, like [1] or [2].
- Be concise and direct.
```

Four jobs, in order: **assign the role**, **fence the knowledge** to the supplied context, **license refusal**, and **mandate citations**.

That refusal clause does real work. An LLM handed context will otherwise strain to produce *something* from it rather than admit the answer is absent — giving it an exact sentence to emit makes refusal the path of least resistance. The same string is exported as the constant `NO_ANSWER`, so callers can recognise a refusal without parsing prose.

##### The context block

`RAGSearch.build_context()` numbers each chunk and labels it with its origin:

```text
Context:
[1] (Daniel-Shiffman_Nature-of-code.pdf p.254)
To apply a force to an object, add the force vector to...

[2] (Daniel-Shiffman_Nature-of-code.pdf p.115)
<chunk 2 text>

Question: How do I apply a force like gravity to a moving object?

Answer:
```

**The numbering is the citation mechanism.** The model is told to cite `[n]`; the same `n` indexes the `Source` records returned alongside the answer, so a claim traces back to a file and page (§4.10).

##### Conversational memory

`chat` mode maintains history, but history alone would not fix retrieval — see §4.9 for why, and for the condensation step that does.

### 4.9 Generation

**Generation** is the LLM writing the final answer from the assembled prompt.

```python
response = self.llm.invoke([("system", SYSTEM_PROMPT), ("human", prompt)])
```

`self.llm` is a **`ChatGroq`** client (from `langchain_groq`), the project's only external API integration:

| Aspect | Detail |
|---|---|
| Provider | **Groq** — an inference service known for very fast token generation |
| Model | **`llama-3.1-8b-instant`** — Meta's Llama 3.1, 8-billion parameters |
| Auth | `GROQ_API_KEY`, validated at startup (§9) |
| Call style | Synchronous and blocking; no streaming |
| Temperature | **`0.0`**, set explicitly — `ChatGroq`'s own default is 0.7, which invites embellishment beyond the retrieved text |
| Errors | Caught and re-raised as `RuntimeError` with context; `chat` mode survives them |

#### Query condensation — why generation needs two calls

A multi-turn conversation breaks retrieval in a way that is easy to miss:

```text
You: What is the Nature of Code about?
Bot: It's about simulating natural systems with code... [1]
You: Who wrote it?
```

*"Who wrote it?"* gets embedded and searched **literally**. The word "it" carries no semantic content, so the search returns noise.

> **The key insight: chat history does not fix retrieval.** Retrieval happens *before* the LLM ever sees the conversation. Passing history to the generation step is too late — the wrong chunks have already been fetched.

The fix is **query condensation**: a cheap extra LLM call that rewrites the follow-up into a standalone question *before* embedding it. `RAGSearch.condense()` uses `CONDENSE_PROMPT` with the last **6** turns of history (`CONDENSE_HISTORY_TURNS`).

Verified working on this corpus:

```text
You: Who wrote it?
  (searched for: Who wrote The Nature of Code?)
Daniel Shiffman [1, 3, 4, 5].
```

This costs two LLM calls per turn instead of one; on `llama-3.1-8b-instant` the added latency is small next to the answer call. If the rewrite fails or returns empty, the original question is used — a degraded retrieval beats a crashed turn.

Note also what is **not** accumulated: only the history list is carried forward. Retrieved context is fetched fresh each turn rather than appended, which is what stops the prompt growing without bound over a long session.

### 4.10 Citations / Source Attribution

A **citation** tells the user which document an answer came from, so claims can be verified. It is a large part of what makes a RAG system trustworthy rather than merely fluent.

**Current status: implemented end to end.**

The provenance chain is unbroken:

```text
load_all_documents()      → Document.metadata = {"source": "...pdf", "page": 254, ...}
        ↓                    ✅ produced by the loader
chunk_documents()         → split_documents() copies metadata onto every chunk
        ↓                    ✅ preserved
build_from_documents()    → {"content": ..., "source": ..., "page": 254, ...}
        ↓                    ✅ stored in metadata.pkl
search()                  → returns the full metadata dict with its score
        ↓                    ✅ available
build_context()           → numbers each block, emits Source records
        ↓                    ✅ [1] (file p.254) ... in the prompt
answer()                  → Answer(text=..., sources=[Source, ...])
                             ✅ returned to the caller
```

Two dataclasses carry it out of the pipeline:

```python
@dataclass
class Source:
    number: int      # matches the [n] the model cites
    file: str        # basename, e.g. "Daniel-Shiffman_Nature-of-code.pdf"
    page: int | None # zero-based in storage, displayed one-based
    score: float

@dataclass
class Answer:
    text: str
    sources: list[Source]
    grounded: bool               # False when the model refused
    standalone_question: str | None   # set when a follow-up was rewritten
```

Real output from `python app.py ask`:

```text
You can apply a force like gravity to a moving object by calling the
applyForce() function on the object's body... [1]

  [1] Daniel-Shiffman_Nature-of-code.pdf p.254 (score 0.54)
  [2] Daniel-Shiffman_Nature-of-code.pdf p.115 (score 0.52)
```

`grounded` is what lets callers distinguish a real answer from a refusal without string-matching prose. When the model refuses, `sources` is deliberately emptied — a refusal cites nothing, and listing chunks beneath it would imply otherwise.

### 4.11 Index Freshness (the manifest)

A RAG system is only as current as its index. Adding a file to `data/` does nothing until the index is rebuilt — and a system with no way to *know* it is out of date will quietly answer from stale content forever.

[`src/manifest.py`](../src/manifest.py) solves this by recording what was indexed, alongside the index itself, as `manifest.json`:

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "dimension": 384,
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "chunk_count": 1099,
  "built_at": "2026-08-09T13:55:41+00:00",
  "files": {
    "data/pdf_files/Daniel-Shiffman_Nature-of-code.pdf": "26f91f978c34…",
    "data/text_files/python_intro.txt": "…",
    "data/text_files/typescript_intro.txt": "…"
  }
}
```

Files are fingerprinted by **SHA-256 of their contents**, not modification time — copying a file must not read as a change, while an edit that happens to preserve the file size must.

This buys two things:

**1. Ingestion becomes idempotent.** `python app.py ingest` hashes the corpus, diffs against the manifest, and does nothing when they match:

```text
$ python app.py ingest
[INFO] Index is up to date; nothing to ingest.

$ python app.py ingest        # after adding a file
[INFO] Corpus changed (1 added [notes.txt]); rebuilding index.
```

Safe to run before every session. This is what replaces the old "uncomment the build block" workflow. Query commands also warn without blocking:

```text
[WARN] Index is out of date (1 added [notes.txt]). Run: python app.py ingest
```

**2. Wrong-model use becomes impossible.** Vectors from two different embedding models are not comparable. A *different*-dimension model raises inside FAISS; a *same*-dimension one does not — it silently returns meaningless neighbours, the worst kind of failure. `check_compatibility()` catches both:

```text
[ERROR] Refusing to load the index in 'vector_store': index was built with
embedding model 'all-MiniLM-L6-v2' but 'all-mpnet-base-v2' is configured;
their vectors are not comparable. Rebuild it with: python app.py ingest --rebuild
```

An index with **no** manifest is also rejected — it predates this check and could have been built with anything, so it is treated as unverifiable rather than assumed compatible.

---

## 5. Architecture of This Implementation

### 5.1 Ingestion / indexing pipeline

Runs occasionally and offline. Slow. Produces the index.

```text
   data/  (PDF, TXT, CSV, DOCX, JSON, XLSX)
     │
     ↓
┌────────────────────────────────────────────┐
│ load_all_documents()                       │  src/data_loader.py
│ · walks data/ recursively per LOADERS entry│
│ · per-file try/except: failures skipped    │
└────────────────────┬───────────────────────┘
                     ↓  List[Document]  (523: 521 PDF pages + 2 text files)
┌────────────────────────────────────────────┐
│ EmbeddingPipeline.chunk_documents()        │  src/embedding.py
│ · RecursiveCharacterTextSplitter           │
│ · size 1000 / overlap 200 / chars          │
│ · split_documents → metadata preserved     │
└────────────────────┬───────────────────────┘
                     ↓  List[Document]  (1,099 chunks)
┌────────────────────────────────────────────┐
│ EmbeddingPipeline.embed_chunks()           │  src/embedding.py
│ · all-MiniLM-L6-v2, local CPU              │
│ · normalize_embeddings=True                │
└────────────────────┬───────────────────────┘
                     ↓  ndarray (1099, 384) float32
┌────────────────────────────────────────────┐
│ VectorStore.add_embeddings()               │  src/vectorstore.py
│ · IndexFlatIP(384), created on first add   │
│ · metadata list kept positionally aligned  │
└────────────────────┬───────────────────────┘
                     ↓
┌────────────────────────────────────────────┐
│ VectorStore.save()                         │  src/vectorstore.py
│ · faiss.index   (binary)                   │
│ · metadata.pkl  (pickle)                   │
│ · manifest.json (written LAST — its        │  src/manifest.py
│   presence implies the pair is complete)   │
└────────────────────────────────────────────┘
```

Orchestrated end-to-end by `VectorStore.build_from_documents()`, which is reached through `RAGSearch.ingest()` — the layer that decides whether a build is needed at all:

```text
  python app.py ingest
        ↓
  RAGSearch.ingest(force=False)
        ↓
  fingerprint_corpus()  → {path: sha256}
        ↓
  manifest matches on disk?  ── yes ──→  "Index is up to date"  (stop)
        │ no
        ↓
  load_all_documents()  →  build_from_documents(documents, files=…)
```

### 5.2 Query pipeline

Runs per question. Fast. Consumes the index.

```text
                    ┌─────────────────────────┐
                    │      User Query         │  app.py ask / chat
                    └───────────┬─────────────┘
                                ↓
                    ┌─────────────────────────┐
                    │  Query Condensation     │  RAGSearch.condense()
                    │  follow-up → standalone │  LLM call #1
                    │  (skipped on turn 1)    │  history: last 6 turns
                    └───────────┬─────────────┘
                                ↓
                    ┌─────────────────────────┐
                    │   Query Embedding       │  EmbeddingPipeline
                    │   MiniLM, normalised    │    .embed_query()
                    └───────────┬─────────────┘  → (1, 384) float32
                                ↓
                    ┌─────────────────────────┐
                    │  Vector Search (FAISS)  │  VectorStore.search()
                    │  IndexFlatIP, exhaustive│  top_k = 5
                    └───────────┬─────────────┘
                                ↓
                    ┌─────────────────────────┐
                    │   Score Filter ≥ 0.30   │  DEFAULT_MIN_SCORE
                    │   may yield 0 results ──┼──→ refuse, no LLM call
                    └───────────┬─────────────┘    Answer(grounded=False)
                                ↓
                    ┌─────────────────────────┐
                    │     Prompt Builder      │  build_context()
                    │  numbered [n] blocks    │  + SYSTEM_PROMPT:
                    │  labelled (file p.N)    │  context-only, cite, refuse
                    └───────────┬─────────────┘
                                ↓
                    ┌─────────────────────────┐
                    │   LLM  (Groq, remote)   │  ChatGroq.invoke()
                    │   llama-3.1-8b, temp 0  │  LLM call #2
                    │   errors → RuntimeError │
                    └───────────┬─────────────┘
                                ↓
                    ┌─────────────────────────┐
                    │  Answer + Sources       │  Answer dataclass
                    │  text, sources[],       │  printed with citations
                    │  grounded, standalone_q │
                    └─────────────────────────┘
```

### 5.3 How the two pipelines meet

```text
    INDEX TIME  ──────────────┐
    (occasional, offline)     │
                              ↓
                    ┌───────────────────┐
                    │   vector_store/   │   the only shared artifact
                    │   faiss.index     │
                    │   metadata.pkl    │
                    └───────────────────┘
                              ↑
    QUERY TIME  ──────────────┘
    (per question, online)
```

The index is the sole interface between them, and the two are now **fully decoupled**: `RAGSearch.__init__` no longer builds anything. Ingestion is an explicit `ingest()` call reached by its own CLI command, and query commands `load()` an existing index and fail with an actionable message if there isn't one:

```text
[ERROR] No saved vector store in 'vector_store'. Run ingestion first (python app.py ingest).
```

The manifest is what makes that separation safe rather than merely tidy — without it, "load an existing index" would mean trusting whatever happens to be on disk (§4.11).

---

## 6. The Codebase, File by File

### 6.1 Project structure

```text
RAG/
├── app.py                   CLI — ingest / ask / chat
├── pyproject.toml           Dependencies (uv-managed)
├── requirements.txt         Older pip dependency list
├── .env                     GROQ_API_KEY (gitignored)
├── .gitignore               excludes .env and vector_store/
│
├── src/
│   ├── __init__.py          Package docstring; no re-exports
│   ├── data_loader.py       Files            → Documents
│   ├── embedding.py         Documents        → chunks → vectors; query vectors
│   ├── vectorstore.py       Vectors          → persisted FAISS index; search
│   ├── manifest.py          Index provenance → staleness + compatibility
│   └── search.py            Question         → context → LLM → Answer
│
├── data/                    Source corpus
│   ├── pdf_files/           Nature of Code (PDF)
│   └── text_files/          python_intro.txt, typescript_intro.txt
│
├── vector_store/            Generated index (gitignored)
│   ├── faiss.index
│   ├── metadata.pkl
│   └── manifest.json
│
├── notebook/                Exploratory scratch work, not imported
│   ├── document.ipynb
│   └── pdf_loader.ipynb
│
└── docs/
    └── RAG_DOCUMENTATION.md This document
```

Two pieces of cleanup are reflected above: `data/vector_store/` — a 15 MB orphaned **Chroma** database left by `notebook/pdf_loader.ipynb`, referenced by no application code — has been deleted, and `vector_store/` is now gitignored, since a binary index and a pickle are build artifacts rather than source.

### 6.2 Data flow through the real classes

Using actual names from the source:

```text
INDEX TIME
                 fingerprint_corpus()          [src/manifest.py]
                          ↓  {path: sha256}
                 load_all_documents()          [src/data_loader.py]
                          ↓  List[Document]
                 EmbeddingPipeline             [src/embedding.py]
                   .chunk_documents()
                          ↓  List[Document]
                   .embed_chunks()
                          ↓  ndarray (n, 384)
                 VectorStore                   [src/vectorstore.py]
                   .add_embeddings()
                   .save()  → + manifest.json
QUERY TIME
                 RAGSearch.answer()            [src/search.py]
                   .condense()                 → standalone question
                          ↓
                 VectorStore.query()           [src/vectorstore.py]
                   → EmbeddingPipeline.embed_query()
                   → .search(min_score=0.30)
                          ↓  List[dict]
                 RAGSearch.build_context()     [src/search.py]
                          ↓  numbered context + List[Source]
                 ChatGroq.invoke()             [langchain_groq → Groq API]
                          ↓  Answer
                     app.py  print_answer()
```

### 6.3 Module reference

#### `src/data_loader.py`

**Purpose:** the only module that knows about file formats.

| Member | Kind | Responsibility |
|---|---|---|
| `LOADERS` | `dict` | Registry: glob pattern → (log label, loader factory) |
| `load_all_documents(data_dir)` | function | Walk the directory, run each loader, return `List[Document]` |

**Interactions:** imported by `VectorStore` (indirectly, at build time) and lazily by `RAGSearch.__init__`. Depends on nothing else in `src/`.

**Design note:** values in `LOADERS` are *factories*, not instances — LangChain loaders bind to one path at construction, so a new one is built per file.

#### `src/embedding.py`

**Purpose:** chunking and vectorisation. Index-time only.

| Member | Kind | Responsibility |
|---|---|---|
| `EmbeddingPipeline` | class | Holds chunking config + the loaded model |
| `.__init__(model_name, chunk_size, chunk_overlap)` | method | Loads `SentenceTransformer` (slow; downloads on first run) |
| `.chunk_documents(documents)` | method | `RecursiveCharacterTextSplitter` → `List[Document]`, metadata preserved |
| `.embed_chunks(chunks)` | method | Encode `.page_content` → normalised `(n, 384)` array |
| `.embed_query(query)` | method | Encode one question → normalised `(1, 384)` float32 |
| `.embedding_dimension()` | method | Vector length (384); wraps a sentence-transformers 5.x rename |

**Interactions:** constructed **once** by `VectorStore.__init__` and reused for that store's lifetime, serving both index time and query time. Keeping `embed_chunks` and `embed_query` on one object is what guarantees both sides encode identically — same model, same `normalize_embeddings=True`.

#### `src/vectorstore.py`

**Purpose:** the persistent index. The only component used by *both* pipelines.

| Member | Kind | Responsibility |
|---|---|---|
| `VectorStore` | class | FAISS index + aligned metadata list + persistence |
| `.__init__(persistence_directory, embedding_model, chunk_size, chunk_overlap)` | method | Creates the dir, builds the shared `EmbeddingPipeline`. Does **not** load the index. |
| `.faiss_path` / `.metadata_path` | property | Paths to the two index files |
| `.exists()` | method | Whether a saved index is present |
| `.build_from_documents(documents, files)` | method | Full ingest: reset → chunk → embed → add → manifest → save. Raises `ValueError` on an empty corpus. |
| `.add_embeddings(embeddings, metadatas)` | method | Creates `IndexFlatIP` on first call; appends |
| `.save()` | method | Writes `faiss.index`, `metadata.pkl`, then `manifest.json` **last** |
| `.load(strict=True)` | method | Reads all three, validating model and dimension; raises rather than serving a mismatched index |
| `.check_stale(current_files)` | method | Diffs the manifest's file hashes against disk |
| `.search(query_embedding, top_k, min_score)` | method | Vector-in → `[{index, metadata, score}]`, score-filtered |
| `.query(query, top_k, min_score)` | method | Text-in: encodes via the shared pipeline, then `.search()` |

**Interactions:** constructed by `RAGSearch`; owns one `EmbeddingPipeline`; delegates provenance to `src.manifest`.

> **One model copy.** The pipeline is created in `__init__` and reused, so the embedding model is loaded exactly once per process — for both indexing and querying.

> **Rebuilds replace rather than append.** `build_from_documents` clears `self.index` and `self.metadata` first, so calling it twice no longer doubles the index.

#### `src/manifest.py`

**Purpose:** records what an index was built from, so staleness and incompatibility become detectable (§4.11). Deliberately free of heavyweight imports — no torch, no document parsers — so it stays cheap to import.

| Member | Kind | Responsibility |
|---|---|---|
| `MANIFEST_FILENAME` | constant | `"manifest.json"` |
| `hash_file(path)` | function | SHA-256 of file contents, read in 1 MB blocks |
| `fingerprint_corpus(data_dir, patterns)` | function | `{path: sha256}` for every file ingestion would pick up |
| `build_manifest(...)` | function | Assemble the record: model, dimension, chunk settings, count, timestamp, files |
| `save_manifest` / `load_manifest` | function | JSON write/read; `load` returns `None` when absent or corrupt |
| `check_compatibility(manifest, model)` | function | `None` if safe to query, else the reason |
| `is_stale(manifest, current_files)` | function | `None` if current, else a summary of added/removed/modified |

**Interactions:** used by `VectorStore` (save/load/validate) and `RAGSearch` (fingerprinting, staleness warnings).

#### `src/search.py`

**Purpose:** query-time orchestration and the sole external API integration.

| Member | Kind | Responsibility |
|---|---|---|
| `load_dotenv()` | module-level call | Loads `.env` at import time |
| `DEFAULT_MIN_SCORE` | constant | `0.30` — cosine floor for relevance |
| `NO_ANSWER` | constant | The exact refusal string the model is told to emit |
| `SYSTEM_PROMPT` | constant | Grounding rules: context-only, refuse, cite |
| `CONDENSE_PROMPT` | constant | Rewrites a follow-up into a standalone question |
| `Source` | dataclass | One citation: number, file, page, score; `.label()` renders it |
| `Answer` | dataclass | `text`, `sources`, `grounded`, `standalone_question` |
| `RAGSearch` | class | Façade: index lifecycle + retrieval + prompt + generation |
| `.__init__(...)` | method | Builds `VectorStore`, validates `GROQ_API_KEY`, constructs `ChatGroq` at `temperature=0`. Does **not** touch the index. |
| `.fingerprint()` | method | Hash the corpus as ingestion would see it |
| `.ingest(force=False)` | method | Idempotent build; returns whether work was done |
| `.load(warn_if_stale=True)` | method | Load for querying; warn if the corpus moved on |
| `.condense(question, history)` | method | Follow-up → standalone question (LLM call #1) |
| `.build_context(hits)` | method | Numbered context block + `List[Source]` |
| `.answer(query, top_k, history)` | method | The full query path → `Answer` |
| `.search_and_summarize(query, top_k)` | method | Back-compat wrapper returning just `.text` |

**Interactions:** the top of the stack. Uses `VectorStore`, `load_all_documents`, and `src.manifest`.

#### `app.py`

**Purpose:** CLI — argument parsing and terminal I/O only. No pipeline logic.

| Member | Responsibility |
|---|---|
| `build_parser()` | Defines `ingest` / `ask` / `chat` and the shared options |
| `print_answer(result)` | Renders text plus citations; prints sources only when grounded |
| `run_chat(rag, top_k)` | The REPL: history, `/clear`, `/exit`, Ctrl-D; survives API errors |
| `main(argv=None)` | Wires it together; maps handled failures to exit code 1 |

```bash
python app.py ingest                          # build/refresh (idempotent)
python app.py ingest --rebuild                # force a full rebuild
python app.py ask "How do forces work?"       # one question
python app.py chat                            # interactive session
```

Shared options: `--top-k`, `--min-score`, `--data-dir`, `--persist-dir`, `--embedding-model`, `--llm-model`, `--temperature`.

#### `notebook/`

Exploratory Jupyter work: loader experiments, a `PyPDFLoader`→`PyMuPDFLoader` fallback prototype, and Chroma trials. **Not imported by the application.** Useful as history; the `src/` modules are the real implementation.

---

## 7. End-to-End Execution Flow

### 7.1 What happens when a document is added

Adding a file to `data/` does not index it on its own — nothing watches the directory. Running `python app.py ingest` does, and detects the change by content hash:

```text
  New file dropped into data/pdf_files/
             ↓
  python app.py ingest
             ↓
  RAGSearch.ingest(force=False)                     [search.py]
   · fingerprint_corpus("data", LOADERS.keys())     [manifest.py]
       → SHA-256 of every matching file
   · vectorstore.load()  → reads existing manifest
   · check_stale(current_files)
       → "1 added [new-file.pdf]"  → proceed
     (had it returned None: "Index is up to date", stop here)
             ↓
  load_all_documents("data")                        [data_loader.py]
   · Path("data").resolve()
   · for each LOADERS pattern → data_path.glob(pattern)
   · PyPDFLoader(path).load()  → one Document per page
   · failures caught per-file, printed, skipped
             ↓  List[Document]
  VectorStore.build_from_documents(documents,       [vectorstore.py]
                                   files=fingerprints)
   · resets index + metadata (rebuild, not append)
             ↓
  self.pipeline                                     [embedding.py]
   · already loaded in VectorStore.__init__; no second copy
             ↓
  .chunk_documents(documents)
   · RecursiveCharacterTextSplitter(1000, 200, ["\n\n","\n"," ",""])
   · split_documents() → metadata copied to each chunk
             ↓  List[Document]  ← 1,099 for the current corpus
  .embed_chunks(chunks)
   · texts = [c.page_content for c in chunks]
   · model.encode(..., normalize_embeddings=True)   ← the slow step
             ↓  ndarray (1099, 384)
  metadatas = [{"content": c.page_content, **c.metadata} for c in chunks]
             ↓
  .add_embeddings(embeddings.astype(float32), metadatas)
   · index is None → faiss.IndexFlatIP(384)
   · index.add(...)          ← vectors stored
   · metadata.extend(...)    ← records stored, positionally aligned
             ↓
  build_manifest(model, dim, chunk_size, overlap,   [manifest.py]
                 chunk_count, files)
             ↓
  .save()
   · faiss.write_index(index, "vector_store/faiss.index")
   · pickle.dump(metadata,   "vector_store/metadata.pkl")
   · save_manifest(...)     → "vector_store/manifest.json"   ← written LAST
             ↓
        Vector Stored ✓
```

Write order is deliberate: the manifest goes last, so its presence implies the index and metadata are already complete. A crash mid-save therefore leaves an index with a missing or older manifest, which `load()` rejects — failing closed rather than serving a half-written index.

### 7.2 What happens when a user asks a question

```text
  python app.py chat
             ↓
  RAGSearch.__init__(...)                            [search.py]
   · VectorStore(...) → EmbeddingPipeline loads MiniLM (once)
   · os.getenv("GROQ_API_KEY")  → missing? raise RuntimeError here
   · ChatGroq(api_key, "llama-3.1-8b-instant", temperature=0.0)
             ↓
  rag.load()                                         [search.py]
   · vectorstore.load()                              [vectorstore.py]
       · exists()? no  → FileNotFoundError with a fix-it message
       · check_compatibility(manifest, model)        [manifest.py]
             mismatch → RuntimeError, refuses to load
       · faiss.read_index()  → IndexFlatIP in memory
       · pickle.load()       → 1,099 metadata dicts
       · index.d == pipeline.embedding_dimension()?  else RuntimeError
   · check_stale(fingerprint())  → [WARN] if the corpus moved on
             ↓
  User Question  ("Who wrote it?")
             ↓
  RAGSearch.answer(query, top_k=5, history=[...])
             ↓
  .condense(question, history)                       LLM call #1
   · history empty? → return question unchanged
   · else rewrite:  "Who wrote it?"
                 →  "Who wrote The Nature of Code?"
   · failure → fall back to the original question
             ↓  Standalone Question
  VectorStore.query(standalone, top_k=5, min_score=0.30)
   · pipeline.embed_query()   → (1, 384) float32     Query Embedding
   · index.search(...)        → exhaustive over 1,099 vectors
   · skip idx == -1 padding
   · break on score < 0.30    ← the relevance threshold
             ↓  Relevant Chunks Retrieved — 0 to 5 dicts, best first
   · no hits? → Answer(NO_ANSWER, grounded=False)    ← no LLM call at all
             ↓
  .build_context(hits)
   · "[1] (Nature-of-code.pdf p.2)\n<text>\n\n[2] ..."
   · sources = [Source(1, file, page, score), ...]
             ↓
  ChatGroq.invoke([("system", SYSTEM_PROMPT),         LLM call #2
                   ("human", context + question)])
   · HTTPS to Groq; blocking; failures → RuntimeError
             ↓
  LLM Response  → Answer(text, sources, grounded, standalone_question)
             ↓
  print_answer(result)                               in app.py
   · text, then "  [1] Nature-of-code.pdf p.2 (score 0.71)"
             ↓
  Sources Returned ✓
```

---

## 8. Configuration Reference

Defaults live in [`app.py`](../app.py)'s argument parser and in the `RAGSearch` signature; every one can be overridden on the command line. The only environment variable is the API key.

### 8.1 LLM configuration

| Setting | Value | Set in | What it does → Why → Effect of change |
|---|---|---|---|
| `--llm-model` | `llama-3.1-8b-instant` | CLI / `RAGSearch` | Which model writes the answer. Chosen for speed and low cost. A larger model (e.g. `llama-3.3-70b-versatile`) follows instructions better and hallucinates less, at higher latency and cost. |
| `--temperature` | **`0.0`**, set explicitly | CLI / `RAGSearch` | Randomness. Zero keeps the answer close to the retrieved text, which is what grounded QA wants. `ChatGroq`'s own default is 0.7 — high enough to invite embellishment — so this is deliberately overridden. Raise it only if you want more varied phrasing. |
| `max_tokens` | `None` — no cap | — | Caps answer length. `None` defers to the provider; a long answer could be truncated with no signal. |
| `request_timeout` | `None` | — | **No client-side deadline.** A hung request blocks indefinitely (§14). |
| `max_retries` | `2` — `ChatGroq`'s default | — | Two automatic retries on transient failures. Inherited, not chosen. |

### 8.2 Embedding configuration

| Setting | Value | Set in | What it does → Why → Effect of change |
|---|---|---|---|
| `--embedding-model` | `all-MiniLM-L6-v2` | CLI / `VectorStore` / `EmbeddingPipeline` | Turns text into 384-dim vectors. Small, fast, free, local. A stronger model (e.g. `all-mpnet-base-v2`, 768-dim) improves retrieval but is slower and **changes the dimension**. |
| Vector dimension | **384** | Inferred from data; recorded in the manifest | Set automatically from the first batch — never configured by hand. |
| `normalize_embeddings` | `True` | `embed_chunks()`, `embed_query()` | Scales vectors to unit length so inner product = cosine similarity. Both call sites live on one class, so they cannot drift apart. |

> **Changing the embedding model requires a rebuild** — `python app.py ingest --rebuild`. You will not forget by accident: the manifest records which model built the index and `load()` refuses a mismatch with an explicit error (§4.11), instead of silently returning garbage rankings.

### 8.3 Chunking configuration

| Setting | Value | Set in | What it does → Why → Effect of change |
|---|---|---|---|
| `chunk_size` | `1000` chars | `VectorStore` / `EmbeddingPipeline` defaults; recorded in the manifest | Max characters per chunk. **Smaller** → sharper, more precise embeddings but risks cutting context. **Larger** → more context per hit, but diluted vectors and more prompt tokens. Not exposed on the CLI; changing it requires `--rebuild`. |
| `chunk_overlap` | `200` chars | same | Text shared between neighbours so boundary-spanning sentences survive. **Larger** → safer, ~more chunks. **Zero** → risks orphaned facts. 10–20% of chunk size is typical; 200/1000 = 20%. |
| `separators` | `["\n\n","\n"," ",""]` | `chunk_documents()` | Cut preference, coarse → fine, so cuts land on natural boundaries. |
| Unit | **characters** | `length_function=len` | Not tokens. ~1,000 chars ≈ 250 tokens. |

### 8.4 Vector store configuration

| Setting | Value | Set in | Notes |
|---|---|---|---|
| `--persist-dir` | `"vector_store"` | CLI | Where the three index files live. |
| `--data-dir` | `"data"` | CLI | Corpus root for ingestion and fingerprinting. |
| Index type | `IndexFlatIP` | `add_embeddings()` | Exhaustive + exact. Fine to ~100k vectors; beyond that consider `IndexIVFFlat` or `IndexHNSWFlat`. |
| Metadata format | `pickle` | `save()` / `load()` | Fast, but Python-only and **unsafe to load from an untrusted source** (§10). |
| Manifest format | JSON | `src/manifest.py` | Human-readable and inspectable, unlike the pickle. |
| Rebuild trigger | Content hash mismatch, or `--rebuild` | `RAGSearch.ingest()` | Detects added, removed, and modified files (§4.11). |

### 8.5 Retrieval configuration

| Setting | Value | Set in | What it does → Why → Effect of change |
|---|---|---|---|
| `--top-k` | **5** | CLI | Maximum chunks retrieved and pasted into the prompt. **Higher** → better recall, more tokens, more cost, more noise. **Lower** → cheaper, risks missing the answer. |
| `--min-score` | **0.30** (`DEFAULT_MIN_SCORE`) | CLI / `src/search.py` | Cosine-similarity floor. **Higher** → stricter; more "I don't know", fewer ungrounded answers. **Lower** → more recall, and eventually the hallucination this exists to prevent. Tuned for MiniLM, where on-topic hits score ~0.46–0.71 here. Model-specific — re-tune if you change the embedding model. |
| Re-ranking | **none** | — | Not implemented. Vector order is final (§14). |

### 8.6 Prompt configuration

| Setting | Value | Notes |
|---|---|---|
| `SYSTEM_PROMPT` | Context-only, refuse-if-absent, cite `[n]`, be concise | Constant in [`src/search.py`](../src/search.py) |
| `NO_ANSWER` | `"I don't know based on the provided documents."` | The exact refusal string; also used to detect refusals |
| User message | `"Context:\n{numbered blocks}\n\nQuestion: {q}\n\nAnswer:"` | Built by `build_context()` + `answer()` |
| Chunk format | `[n] (file p.N)` header, then text, blank line between | Numbering is what makes citation possible |
| `CONDENSE_PROMPT` | Rewrites a follow-up into a standalone question | Only used when history is non-empty |
| `CONDENSE_HISTORY_TURNS` | `6` | Prior turns fed to the condenser; caps token spend |
| Chat history | `(role, text)` pairs, in-memory, per session | Cleared with `/clear`; not persisted between runs |

### 8.7 Environment variables

| Variable | Required | Read in | Notes |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes**, except for `ingest` | `RAGSearch.__init__` via `os.getenv` | Loaded from `.env` by `load_dotenv()` at import. **Validated at startup**: missing → `RuntimeError` with instructions, before any work happens. `ingest` passes `require_api_key=False`, since indexing is entirely local. |
| `FREELLMAPI_BASE_URL` | No | — | Present but **commented out** in `.env`; unused by the code. |
| `FREELLMAPI_API_KEY` | No | — | Same. |
| `FREELLMAPI_MODEL` | No | — | Same. |

`.env` is correctly listed in [.gitignore](../.gitignore).

### 8.8 Dependency notes

| Package | Declared in `pyproject.toml`? | Note |
|---|---|---|
| `python-dotenv` | ✅ Yes | Imported by `search.py`. Previously resolved only transitively via `chromadb`; now declared directly. |
| `langchain-text-splitters` | ✅ Yes | Imported by `embedding.py`. Previously transitive via `langchain`; now declared directly. |
| `docx2txt` | ❌ **No** | Needed for `.docx`; that loader will fail without it. |
| `unstructured` | ❌ **No** | Needed for `.xlsx`; that loader will fail without it. |
| `openai` | ✅ Yes | **Unused** by any code in `src/` or `app.py`. |
| `chromadb` | ✅ Yes | Used only by the notebooks, not the application. |

---

## 9. Error Handling and Edge Cases

Honest assessment of each case. ✅ handled, ⚠️ partly, ❌ unhandled.

| Case | Behaviour today | Status |
|---|---|---|
| **Invalid / corrupt document** | `try/except Exception` per file in `load_all_documents`. Prints `[ERROR]`, skips, continues. One bad PDF cannot abort the batch. | ✅ |
| **Unsupported file type** | No matching glob → silently ignored. No warning that files were skipped. | ⚠️ |
| **Missing `data/` directory** | `Path.glob` on a non-existent dir yields nothing. Returns `[]` — indistinguishable from an empty corpus. | ⚠️ |
| **Empty corpus** | `build_from_documents` raises `ValueError` naming the directory and the supported extensions; `ingest` raises earlier still if no documents loaded at all. Previously an opaque `IndexError`. | ✅ |
| **Failed embedding** | No handling. A `SentenceTransformer` error aborts the whole build; partial work is lost. | ❌ |
| **Missing index files** | `load()` raises `FileNotFoundError`: *"No saved vector store in 'vector_store'. Run ingestion first (python app.py ingest)."* `main()` catches it and exits 1. | ✅ |
| **Index / metadata out of sync** | `search()` guards with `idx < len(self.metadata)` → `None` instead of `IndexError`. Those hits are then skipped when building context. | ✅ |
| **Interrupted save** | Writes are still not atomic, but the manifest is written **last**, so a partial save leaves a missing/older manifest that `load()` rejects. Fails closed instead of serving a mismatched pair. | ⚠️ |
| **Stale index** | Detected by SHA-256 file hashes. `ingest` rebuilds; `ask`/`chat` warn and continue (§4.11). | ✅ |
| **Wrong embedding model** | `check_compatibility()` refuses to load and names both models. A dimension mismatch is caught separately against the live model. | ✅ |
| **Index with no manifest** | Treated as unverifiable and rejected, with a `--rebuild` instruction. | ✅ |
| **No relevant documents found** | Chunks below `--min-score` are dropped. If nothing survives, `answer()` returns `NO_ANSWER` with `grounded=False` — **without calling the LLM**. Verified: *"What is the capital of France?"* returns the refusal. | ✅ |
| **Empty user query** | The REPL skips blank input. A whitespace-only query passed to `ask` would still embed, but its hits will not clear the threshold. | ⚠️ |
| **Missing `GROQ_API_KEY`** | `RAGSearch.__init__` raises immediately with the export command to run. `ingest` exempts itself, since it needs no LLM. | ✅ |
| **LLM / API failure** | Wrapped as `RuntimeError("Groq request failed: …")`. `ask` exits 1 with the message; `chat` prints it and **keeps the session alive**, so a transient failure does not lose accumulated history. | ✅ |
| **Condensation failure** | Caught separately: warns and falls back to the verbatim question. A degraded retrieval beats a crashed turn. | ✅ |
| **LLM returns empty** | Falls back to `NO_ANSWER` rather than printing nothing. | ✅ |
| **Duplicate documents** | No deduplication by content hash or path. The same file in two folders is indexed twice, and both copies can occupy top-K slots. | ❌ |
| **Re-running a build** | `build_from_documents` resets index and metadata first, so a rebuild replaces rather than doubles. | ✅ |

### Exit codes

`main()` maps handled failures to exit code **1** and success to **0**, so the CLI composes with shell scripts and CI. Unexpected exceptions are deliberately *not* caught — their tracebacks are more useful than a swallowed error.

| Situation | Exit code |
|---|---|
| `ingest` / `ask` / `chat` succeeded | 0 |
| `ingest` with nothing to do | 0 |
| `ask` with no question argument | 1 |
| Missing or incompatible index | 1 |
| Missing `GROQ_API_KEY` | 1 |
| Groq request failed (`ask`) | 1 |

### Remaining weaknesses

Stated plainly, since §9 is where a reader looks for them:

- **No request timeout.** A hung Groq call blocks forever; `request_timeout` is unset.
- **No embedding-failure recovery.** A model error mid-build loses all progress; there is no checkpointing.
- **No deduplication.** Identical content in two files is indexed twice and can occupy multiple top-K slots.
- **Non-atomic save.** Mitigated by manifest-last ordering, but a temp-file-and-rename would be strictly better.
- **No input length cap.** A pathologically long query is embedded as-is.

---

## 10. Security and Production Considerations

### Already implemented

| Control | Detail |
|---|---|
| **API key not hardcoded** | Read via `os.getenv("GROQ_API_KEY")`; never a literal in source. |
| **API key validated at startup** | Missing key raises immediately with instructions, rather than surfacing as an opaque auth error mid-request. |
| **`.env` is gitignored** | [.gitignore](../.gitignore) excludes `.env`, keeping the key out of version control. |
| **Index artifacts gitignored** | `vector_store/` is excluded, so document text embedded in `metadata.pkl` is not committed to the repository. |
| **`.env` loading** | `python-dotenv` for local development; real environment variables take precedence. |
| **Ingestion runs locally** | Embedding happens on-device, and `ingest` does not require an API key. Document text is **not** sent to any third party during indexing. |
| **Loader fault isolation** | A malformed file cannot crash the ingest. |
| **Index provenance** | The manifest records what was indexed and with which model, and `load()` refuses mismatches — closing off a class of silent-corruption bugs. |

### Recommended improvements — not implemented

| Risk | Current state | Recommendation |
|---|---|---|
| **Prompt injection** | **Partially mitigated, not solved.** Context now sits in a user message under a system prompt, and chunks are delimited by numbered `[n]` headers — both raise the bar. But nothing asserts that context is *data rather than instructions*, so a document containing *"Ignore previous instructions and…"* may still steer the model. Anyone who can write to `data/` has this leverage. | Add an explicit "treat context as untrusted data" clause to the system prompt; validate documents at ingest; restrict write access to the corpus |
| **Retrieved-content trust** | Chunks are treated as authoritative regardless of origin | Track per-source trust; restrict who may write to `data/` |
| **Data leakage to the API** | Every retrieved chunk is sent to Groq, a third party. Confidential document text leaves the machine on every query. | Review the provider's data-retention terms; for sensitive corpora, self-host the LLM |
| **Access control** | **None.** No users, no roles, no per-document permissions. Anyone who can run the script can query every indexed document. | Add authentication and per-document filtering at retrieval time |
| **Logging sensitive data** | `[DEBUG]` lines print full file paths; a failing loader can print exception text containing document content. All to stdout, unconditional. | Use the `logging` module with levels; suppress content in production logs |
| **Input validation** | The REPL rejects blank input, but there is no length cap on a query. | Cap query length before embedding |
| **Pickle deserialisation** | `metadata.pkl` is loaded with `pickle.load`, which **executes arbitrary code** in the file. Safe for self-generated files; a supply-chain risk if an index is ever shared or downloaded. | Use JSON or Parquet for metadata, or verify integrity before loading |
| **Rate limiting / cost control** | None. Nothing caps queries or spend. | Add client-side rate limiting and budget alerts |
| **PII handling** | No detection or redaction | Scan and redact at ingest if the corpus warrants it |

---

## 11. Performance Considerations

### Where the time goes

**Index time** (one-off, ~1,099 chunks):

| Step | Relative cost | Notes |
|---|---|---|
| Loading + parsing | Low–medium | PDF parsing dominates; scales with page count |
| Chunking | Negligible | Pure string manipulation |
| **Embedding** | **Dominant** | 1,099 forward passes through MiniLM on CPU |
| FAISS add + save | Low | Sequential write of a 1.6 MB index |

Model **loading** costs seconds per `SentenceTransformer` construction. The pipeline is now built once in `VectorStore.__init__` and reused for both indexing and querying, so a build loads the model **once**, not twice.

Ingestion also short-circuits entirely when the corpus is unchanged: `ingest` hashes the files and stops before loading any documents, making a no-op run essentially free apart from model loading.

**Query time** (per question):

| Step | Typical cost | Notes |
|---|---|---|
| Query condensation | **Hundreds of ms** | LLM call #1 — **only on follow-up turns** |
| Query embedding | ~10 ms | One short forward pass |
| FAISS search | **< 1 ms** | 1,099 × 384 exhaustive comparison is trivial |
| Prompt assembly | Negligible | String join |
| **LLM answer call** | **Hundreds of ms to seconds** | **Dominant — network + generation** |

> **The headline:** at this corpus size, retrieval is effectively free and **the LLM calls are essentially the entire query latency**. Optimising FAISS would be pointless; reducing tokens or streaming output would not be.

Two structural notes:

- **Multi-turn chat costs roughly double.** Condensation adds a second round trip on every turn after the first. It buys working follow-ups, which is usually worth it — but a first turn, or `ask`, pays only one call.
- **A refused question costs nothing.** When no chunk clears `--min-score`, the answer call is skipped entirely.

Startup is significant and easy to overlook: every `python app.py` run reloads the embedding model and re-reads the index before answering anything.

### Embedding cost

**Zero monetary cost** — MiniLM runs locally. The cost is CPU time and a one-off ~90 MB download. This is a genuine architectural advantage: re-indexing is free, so it can be done as often as needed.

By contrast, every LLM call is billed per token, and `top_k` is the main lever on that bill.

### Vector search performance

`IndexFlatIP` is exhaustive: cost grows **linearly** with chunk count.

| Chunks | Approximate search time | Verdict |
|---|---|---|
| 1,099 (today) | < 1 ms | Ideal |
| 100,000 | ~10 ms | Still fine |
| 1,000,000+ | ~100 ms+ | Switch to `IndexIVFFlat` / `IndexHNSWFlat` |

Memory is `n × 384 × 4` bytes: today ~1.7 MB, and ~1.5 GB at a million chunks. Both index and metadata are held **fully in RAM**.

### Trade-offs

**`top_k`** — the main cost/quality dial. Each extra chunk adds ~250 tokens to the prompt, raising both price and latency, while improving the odds the answer is present. At the default `top_k=5` the prompt carries roughly 1,250 tokens of context — though the score filter often trims it below that.

**`chunk_size`** — larger chunks mean fewer, more diluted vectors and more tokens per hit; smaller chunks mean sharper retrieval but more risk of severing context.

### Optimisations

**Implemented:**

| Optimisation | Benefit |
|---|---|
| **One shared embedding model** | Model loaded once per process, not twice per build (§6.3) |
| **Batched ingestion** | `embed_chunks` encodes the whole corpus in one batched call |
| **Hash-based ingest short-circuit** | An unchanged corpus skips loading, chunking, and embedding entirely |
| **Threshold before generation** | An irrelevant question costs zero API calls |
| **`chat` reuses one process** | Model load and index read are paid once for the whole session, not per question |

**Not implemented:**

| Opportunity | Benefit |
|---|---|
| **Query embedding cache** | Repeated questions skip encoding entirely |
| **Answer cache** | Identical question + same top-K → skip the API call, the expensive step |
| **Streaming responses** | Same total latency, dramatically better perceived speed |
| **Incremental indexing** | Avoids re-embedding the whole corpus when one file changes; currently any change triggers a full rebuild |
| **Skip condensation heuristically** | A question with no pronouns rarely needs rewriting; detecting that would halve many turns' cost |

---

## 12. Example Walkthrough

All three examples below are **real captured output** from this implementation, not illustrations.

### Example A — a question the corpus can answer

```bash
$ python app.py ask "How do I apply a force like gravity to a moving object?"
```

**Step 1 — Condensation.** `ask` passes no history, so `condense()` returns the question unchanged. No LLM call.

**Step 2 — Query embedding.** `EmbeddingPipeline.embed_query()` encodes it with MiniLM, normalised:

```text
"How do I apply a force like gravity to a moving object?"
        ↓  model.encode(..., normalize_embeddings=True)
[0.031, -0.118, 0.204, ..., 0.077]      # 384 floats, unit length
```

**Step 3 — Semantic search.** `index.search()` compares against all 1,099 stored vectors — exhaustively, exactly — and returns the 5 best. The winning chunks need not share a single word with the question; they win on meaning.

**Step 4 — Score filter.** All five clear the 0.30 floor (0.54 down to 0.49), so all five survive.

**Step 5 — Prompt assembly.** `build_context()` numbers and labels each chunk:

```text
Context:
[1] (Daniel-Shiffman_Nature-of-code.pdf p.254)
...box2d.setGravity(0, -20); ... body.applyForce(force, pos); ...

[2] (Daniel-Shiffman_Nature-of-code.pdf p.115)
<chunk 2 text>

Question: How do I apply a force like gravity to a moving object?

Answer:
```

...sent under the grounding `SYSTEM_PROMPT` (§4.8).

**Step 6 — Final answer, with citations:**

```text
You can apply a force like gravity to a moving object by calling the
applyForce() function on the object's body, passing in the force vector
and the position of the object.

box2d.setGravity(0, -20); [1]
body.applyForce(force, pos); [1]

  [1] Daniel-Shiffman_Nature-of-code.pdf p.254 (score 0.54)
  [2] Daniel-Shiffman_Nature-of-code.pdf p.115 (score 0.52)
  [3] Daniel-Shiffman_Nature-of-code.pdf p.286 (score 0.50)
  [4] Daniel-Shiffman_Nature-of-code.pdf p.91 (score 0.50)
  [5] Daniel-Shiffman_Nature-of-code.pdf p.90 (score 0.49)
```

The model cited `[1]` and the reader can open page 255 to verify it.

### Example B — a question the corpus cannot answer

The case that previously produced an ungrounded answer.

```bash
$ python app.py ask "What is the capital of France?"
```

**Steps 1–3.** Embedded and searched exactly as before. The corpus is about generative art, Python, and TypeScript, so nothing is relevant — but FAISS is asked for 5 neighbours and dutifully returns its 5 least-irrelevant chunks, scoring near zero.

**Step 4 — The threshold fires.** Every hit falls below 0.30 and is dropped. `hits` is empty.

**Step 5 — Refusal, with no LLM call:**

```text
I don't know based on the provided documents.
```

`Answer(text=NO_ANSWER, sources=[], grounded=False)`. No sources are listed, because none were used.

> **Why this is the important example:** the previous implementation answered *"Paris"* here — correct, but drawn from the model's training data rather than from the documents, which is exactly what RAG exists to prevent. The system now distinguishes *"I found nothing"* from *"here is an answer."*

### Example C — a follow-up question (multi-turn)

```bash
$ python app.py chat
```

```text
You: What is the Nature of Code about?

The Nature of Code is about capturing the unpredictable evolutionary and
emergent properties of nature in software... [2]

  [1] Daniel-Shiffman_Nature-of-code.pdf p.2 (score 0.71)
  [2] Daniel-Shiffman_Nature-of-code.pdf p.521 (score 0.56)

You: Who wrote it?
  (searched for: Who wrote The Nature of Code?)

Daniel Shiffman [1, 3, 4, 5].

  [1] Daniel-Shiffman_Nature-of-code.pdf p.2 (score 0.71)
  [2] Daniel-Shiffman_Nature-of-code.pdf p.503 (score 0.55)
```

The second turn is the interesting one. *"Who wrote it?"* contains no searchable content — embedding it literally would retrieve noise. `condense()` rewrote it to *"Who wrote The Nature of Code?"* using the previous turn, and the `(searched for: …)` line surfaces that so the behaviour is inspectable rather than magic.

---

## 13. Technical Glossary

| Term | Definition |
|---|---|
| **AI** | Software that learns a task from examples rather than from explicitly programmed rules. |
| **LLM** | *Large Language Model.* A neural network trained on vast amounts of text to predict the next word, which makes it able to answer, summarise, and write. This project uses `llama-3.1-8b-instant`. |
| **Prompt** | The complete text sent to an LLM as input — here, the instruction plus the retrieved context plus the question. |
| **Token** | The unit an LLM reads and bills by: roughly ¾ of a word, or ~4 characters. |
| **Context** | The retrieved document text placed into the prompt for the model to answer from. |
| **Context window** | The maximum number of tokens a model can process at once — its working memory. Exceeding it truncates input. |
| **RAG** | *Retrieval-Augmented Generation.* Search your documents for relevant passages, put them in the prompt, and have the LLM answer from them. |
| **Retrieval** | The search step: finding the document pieces most relevant to a question. |
| **Generation** | The writing step: the LLM producing an answer from the prompt. |
| **Embedding** | A list of numbers representing a piece of text's meaning, arranged so that similar meanings sit near each other. Here, 384 numbers per chunk. |
| **Vector** | An ordered list of numbers; a point in high-dimensional space. An embedding is a vector. |
| **Vector database** | Storage optimised for finding the vectors nearest to a query vector. This project uses **FAISS**, a local library rather than a server. |
| **Semantic search** | Search by meaning rather than by matching keywords. Finds *"gravity"* when you ask about *"falling objects."* |
| **Chunk** | A small piece of a document (here, ~1,000 characters) — the unit that gets embedded, stored, and retrieved. |
| **Chunk overlap** | Text repeated between consecutive chunks (here, 200 characters) so sentences spanning a boundary are not lost. |
| **Similarity search** | Finding stored vectors closest to a query vector; the operation a vector database is built for. |
| **Cosine similarity** | A closeness measure based on the angle between two vectors, from `-1` (opposed) to `1` (identical). Used here via an inner-product index over normalised vectors. |
| **Top-K** | The number of best-matching chunks retrieved. This project uses **K = 3**. |
| **Retriever** | The component that performs retrieval. Here, `VectorStore.query()`. |
| **Hallucination** | An LLM producing confident, fluent, false information — its default behaviour when it lacks a fact. |
| **Citation** | A pointer to the source document a claim came from, allowing verification. Data is captured for this but not yet surfaced (§4.10). |
| **Prompt injection** | An attack where text inside a document contains instructions ("ignore previous instructions…") that the model obeys as if they came from the user. No defence currently exists here (§10). |
| **Normalisation** | Scaling a vector to length 1, which makes inner product equal cosine similarity. |
| **Index (FAISS)** | The searchable structure holding all stored vectors. `IndexFlatIP` here — exhaustive and exact. |
| **Ingestion** | The offline pipeline that turns documents into a searchable index. |

---

## 14. Known Gaps and Recommended Improvements

Everything below is **not implemented**. Collected here so the rest of the document can describe only reality.

### Resolved since the first draft

For readers comparing against an earlier version of this document, these gaps are now closed and are documented as working behaviour in the sections cited:

| Former gap | Now | Where |
|---|---|---|
| No relevance threshold | `--min-score` 0.30; refuses without an LLM call | §4.7, §12B |
| Prompt asked for a summary, permitted outside knowledge | Grounded `SYSTEM_PROMPT` with an explicit refusal string | §4.8 |
| Citations not surfaced | Numbered `[n]` blocks → `Answer.sources` with file and page | §4.10 |
| No index validation on load | Manifest checks model and dimension; refuses mismatches | §4.11 |
| No re-indexing story | SHA-256 corpus fingerprints; idempotent `ingest` | §4.11 |
| Single hardcoded query, no loop, no memory | `ingest`/`ask`/`chat` + REPL + query condensation | §4.9, §6.3 |
| API errors uncaught, key unvalidated | `RuntimeError` wrapping, startup key check, `chat` survives failures | §9 |
| `temperature` inherited at 0.7 | Set explicitly to 0.0 | §8.1 |
| Two `SentenceTransformer` copies | One shared `EmbeddingPipeline` | §6.3, §11 |
| Orphaned Chroma DB, index not gitignored | Deleted; `vector_store/` ignored | §6.1 |
| `python-dotenv`, `langchain-text-splitters` undeclared | Declared directly | §8.8 |
| Unused import in `embedding.py` | Removed | — |

### Still outstanding

Ordered by impact.

#### 1. Prompt injection is only partially mitigated

Retrieved text is delimited and placed in a user message under a system prompt, but nothing tells the model that context is **data rather than instructions**. A document containing *"Ignore previous instructions and…"* may still steer it. Anyone who can write to `data/` has this leverage (§10).

**Fix:** add an explicit untrusted-data clause to `SYSTEM_PROMPT`; consider screening documents at ingest.

#### 2. No request timeout

`request_timeout` is unset, so a hung Groq call blocks indefinitely. `max_retries=2` is inherited rather than chosen (§8.1).

**Fix:** set an explicit timeout and a deliberate retry policy with backoff.

#### 3. No re-ranking

Vector similarity order is final. Dense retrieval alone sometimes ranks a loosely-related chunk above the precisely-relevant one.

**Fix:** a cross-encoder re-ranker over the top ~20 hits, or hybrid dense + BM25 retrieval with score fusion.

#### 4. Full rebuilds only

Any corpus change re-embeds everything. Fine at 1,099 chunks; painful at 100,000.

**Fix:** incremental indexing — embed only new and modified files. Requires deleting rows from FAISS, which `IndexFlatIP` does not support directly (`IndexIDMap` plus `remove_ids` is the usual route).

#### 5. No deduplication

The same content in two files is indexed twice and both copies can occupy top-K slots, wasting prompt space (§9).

**Fix:** hash chunk text at build time and drop exact duplicates.

#### 6. Smaller items

- **Non-atomic save.** Mitigated by writing the manifest last, but temp-file-and-rename would be strictly better (§9).
- **No embedding-failure recovery.** A model error mid-build loses all progress; no checkpointing.
- **No query length cap.** The REPL rejects blank input but not pathologically long input.
- **`print()`-based output throughout**; no `logging` levels, so debug noise cannot be suppressed in production (§10).
- **`docx2txt` and `unstructured` remain undeclared** (§8.8). `LOADERS` advertises DOCX and XLSX support, but those loaders fail without the packages. They are heavy dependencies and no file in the current corpus needs them, so they were deliberately left out — add them if you intend to use those formats.
- **`openai` is declared but unused**; `chromadb` is used only by the notebooks.
- **`[project.scripts] rag = "rag:main"`** in [pyproject.toml](../pyproject.toml) points at a module that does not exist, so an installed console script would fail. The project is run as `python app.py`, so this is latent rather than active.
- **No tests.** Everything in this document was verified by hand.

---

*This document describes the codebase as of the current commit. Sections 1–13 reflect implemented behaviour, verified against the running system; §14 collects what is not.*
