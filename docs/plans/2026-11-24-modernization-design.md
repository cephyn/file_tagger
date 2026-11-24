# File Tagger Local-First Modernization Design

**Status:** Approved
**Date:** 2026-11-24
**Implementation scope:** Planning only
**Platform baseline:** Windows 10 22H2 and Windows 11
**Python baseline:** 3.13+

## Goal

Modernize File Tagger into a local-first document tagging, hybrid search, and
retrieval-augmented generation (RAG) application. All AI inference will run
through a user-installed Ollama instance on the local machine. Cloud AI
providers, direct model-file loading, and downloadable embedding models outside
Ollama will be removed.

The primary end-state use case is:

1. A user selects an optional Markdown context document, such as an RPG
   adventure outline.
2. The user asks a broad question, such as "Find appropriate encounters,
   treasure, and random tables for this adventure."
3. The app translates the request and context into several targeted searches.
4. SQLite FTS5 and vector search retrieve relevant chunks from local files.
5. A local Ollama generation model ranks the evidence and produces a cited
   answer.

## Confirmed Product Decisions

- Ollama installation and model downloads are user-controlled. The app detects
  Ollama, checks versions and installed models, recommends model tags, and
  displays install/pull commands. It does not install Ollama or pull models on
  the user's behalf.
- Ollama-dependent features are blocked until readiness checks pass. File
  browsing and ordinary tag management may remain available, but indexing,
  semantic search, AI tagging, and RAG must show actionable setup guidance
  rather than fall back to a cloud provider or a different local runtime.
- Windows 10 22H2 and Windows 11 are the supported platforms.
- Python 3.13 is the minimum supported version.
- The approved architecture is the full local storage rebuild: one SQLite
  database with FTS5 and `sqlite-vec`, replacing the split SQLAlchemy/Chroma
  persistence model.
- PDF text extraction is preferred over whole-page vision embedding. Vision is
  an augmentation for pages whose text/layout quality is insufficient or whose
  diagrams contain important meaning not represented in extracted text.
- The application must cover large files completely. It must not silently
  truncate documents before embedding.

## Current-State Findings

The repository is currently a flat PySide6 desktop application with several
overlapping AI and document-processing paths:

- [`ai_service.py`](../../ai_service.py) selects among OpenAI, Anthropic, Gemini,
  llama.cpp, and CTransformers. Model names and provider behavior are embedded
  in service and UI code.
- [`config.py`](../../config.py) stores cloud API keys, one direct local model
  path, and a PDF extraction preference in encrypted configuration.
- [`vector_search/vector_search.py`](../../vector_search/vector_search.py)
  stores text and vectors in a separate Chroma database, uses a Sentence
  Transformers model, and generates document summaries through the provider
  service.
- [`vector_search/content_extractor.py`](../../vector_search/content_extractor.py)
  competes between Docling and MarkItDown, with pypdf used separately for AI
  tagging.
- [`vector_search/document_chunker.py`](../../vector_search/document_chunker.py)
  chunks by character count, not by the selected model's token context.
- [`search.py`](../../search.py) sends full extracted documents into chat and
  duplicates provider-specific generation code.
- [`pyproject.toml`](../../pyproject.toml) and
  [`requirements.txt`](../../requirements.txt) disagree about the Python floor
  and dependency set.
- There is no automated test suite or schema migration system.

This creates four coupled modernization risks: a provider-lifecycle change can
break extraction, search, tagging, or UI flows; Chroma and SQLite can become
inconsistent; changing embedding models mixes incompatible vector spaces; and
large files can be partially omitted because limits are byte- or
character-based.

## Architecture Decision

### Considered Approaches

1. **Focused adapter swap.** Replace providers and embeddings in place while
   retaining current UI, Chroma, and config structure. Fast, but it preserves
   the existing coupling and migration hazards.
2. **Modular local-first core with Chroma.** Add explicit service boundaries and
   keep Chroma temporarily. Lower migration risk, but persistence remains split.
3. **Single SQLite rebuild.** Replace Chroma with SQLite FTS5 plus `sqlite-vec`
   and move all index state into one transactional store. Highest migration
   scope, but it removes duplicated state and best supports versioned
   extraction, hybrid ranking, provenance, and recovery.

### Approved Approach

Use option 3.

The implementation should be a layered desktop application:

```text
PySide6 UI
    |
    v
Application services
  - readiness/setup
  - ingest/indexing
  - tag classification
  - hybrid search
  - contextual RAG
    |
    v
Local domain adapters
  - Ollama client
  - hardware profiler/recommender
  - Docling extractor
  - SQLite search repository
    |
    v
Single SQLite database
  - relational metadata
  - FTS5 text index
  - sqlite-vec vector index
```

UI code must not branch on model names or parse model output. Domain services
should receive typed records and depend on adapter interfaces.

## Target Components

### Ollama Gateway

The gateway is the only component allowed to call Ollama.

Responsibilities:

- Restrict the base URL to loopback (`127.0.0.1` or `localhost`) by default and
  reject remote or cloud endpoints for normal operation.
- Check the Ollama server version and require 0.35 or newer because decision
  models use the System One endpoint.
- List installed models and retrieve model details/context metadata.
- Check readiness for the selected embedding, decision, and generation models.
- Call `/api/embed` with truncation disabled.
- Use `ollama.systemone` for typed Nimble/Tev1 tagging decisions.
- Use Ollama chat with structured outputs for query planning, reranking, and
  final cited answers.
- Reject model names that represent cloud-hosted variants.
- Record model name, digest, capability, context length, latency, and any
  observed memory usage.

### Hardware Profiler and Model Recommender

The profiler produces an advisory capability report rather than selecting
models from RAM alone:

- CPU architecture, logical processors, and feature support.
- Total and currently available RAM.
- Disk space for model downloads and local index growth.
- Windows GPU name and VRAM when detectable.
- Ollama's observed runtime behavior and loaded-model VRAM when a smoke test is
  available.

Feasibility is a hard gate. Quality preference is a weighted score applied only
after feasibility:

1. Reserve enough memory and disk headroom for Windows, Docling, SQLite, and the
   app.
2. Exclude a model if its published download footprint plus a runtime reserve
   cannot fit.
3. Prefer a GPU-resident model when known VRAM is sufficient; otherwise permit
   a CPU/offload plan.
4. Add the strongest quality weight to Inkling 2 Mini for generation/search.
5. Run short, local smoke benchmarks and adjust the recommendation with
   measured embed latency, load time, and tokens per second.
6. Permit a user override, but explain the measured reason for the advisory
   warning.

Because embedding, decision, and generation models may not fit concurrently,
jobs should use short `keep_alive` values and unload or allow eviction between
roles. The app must not assume that all three models remain resident.

### Extraction Service

Use Docling as the single primary local document converter.

For PDFs:

- Prefer native text/layout extraction.
- Enable OCR only where a page has no useful text layer or fails text-quality
  checks.
- Preserve page numbers, reading order, headings, tables, code, and figures in
  Markdown/structured form.
- Produce quality diagnostics so the user can see when a PDF is scanned,
  encrypted, partially extracted, or image-heavy.
- Render only a selective set of pages for vision augmentation. Do not embed
  every PDF page as an image by default.

The current MarkItDown, `pypdf`, `PyPDF2`, direct Office-reader, and duplicate
DOCX/PPTX/XLSX paths should be removed unless Docling or a verified format
exception requires one narrowly scoped fallback.

### Full-Coverage Chunking and Embedding

The chunker must use document structure and runtime model constraints rather
than fixed character limits.

Ordering:

1. Split by page and document boundary.
2. Split by heading/section.
3. Split by paragraph or table row group.
4. Add bounded overlap only where a semantic boundary would otherwise lose
   context.
5. Validate the final chunk by calling Ollama with `truncate=false`.
6. If Ollama rejects a chunk as too long, recursively split it and retry.
7. Abort with a visible, resumable job error if a chunk cannot be embedded
   without truncation.

The model's runtime context must be read from Ollama, not assumed from an old
README, and the app must reserve room for task prefixes and metadata. A
character or byte limit may be used as an early conservative splitter, but it
is not proof of token coverage; the `truncate=false` call is the acceptance
check.

EmbeddingGemma 2 variants:

| Intended profile | Ollama tag | Use |
| --- | --- | --- |
| Text-first / constrained hardware | `embeddinggemma-2:270m` | Default PDF and document indexing |
| Text plus selective PDF images | `embeddinggemma-2:440m` | Visual augmentation for image-heavy documents |
| Full multimodal profile | `embeddinggemma-2:740m` | Only when image/audio/video behavior is intentionally useful and the hardware benchmark supports it |

The parameter tags include modular modality encoders; a larger tag must not be
described as automatically producing a better text vector. For storage, start
with 256-dimensional Matryoshka vectors, with 512 and 768 available as
higher-quality reindex profiles. A model or dimension change is an index
version change and triggers an explicit full reindex; embeddings from different
spaces must never be mixed.

### Tagging Pipeline

Use System One decision models for the final typed classification:

1. Extract a bounded document representation: title, headings, a small set of
   section summaries, keywords, and relevant chunks.
2. Retrieve likely existing tags from the local tag index.
3. Optionally use the selected generation model to propose a short list of new
   candidate tags.
4. Ask Nimble or Tev1 a batch of typed `choice`, `noul`, or `score` questions
   about those candidates.
5. Validate and normalize the typed response with Pydantic.
6. Present confidence/probability data to the user; do not apply tags silently
   unless the user explicitly enables a high-confidence policy.

Selection:

- Preferred: `nimble:9b`.
- First fallback: `tev1:4b`.
- Constrained fallback: `tev1:0.8b`.

The current free-form response format and regex parser should be deleted.
Decision-model state should stay short enough for the model's effective task
range; never send an entire long file to Tev1/Nimble.

### Unified SQLite Repository

Use SQLAlchemy 2.1 with Alembic for relational schema management and direct SQL
for FTS5 and `sqlite-vec` setup/querying. At a high level, the database should
contain:

- `files`: canonical path, hash, size, mtime, MIME type, and indexing state.
- `tags` and `file_tags`: user-managed classification data.
- `documents`: extraction version, file hash, parser, page count, extraction
  status, and parser diagnostics.
- `pages`: normalized Markdown/text and page provenance.
- `chunks`: stable chunk identifier, parent page/document, text, heading,
  ordinal, model/index version, and content hash.
- `chunks_fts`: an external-content FTS5 table over chunk text and relevant
  metadata.
- `chunk_embeddings`: a `sqlite-vec` `vec0` table with one supported dimension
  per index generation.
- `index_jobs`: resumable batch/one-file job status, counts, failures, and
  retry data.
- `model_profiles`: selected model tags, digests, dimensions, benchmark data,
  and timestamps.

All writes for a file revision should commit as one transaction: new document,
pages, chunks, FTS rows, and vectors become visible together. Failed jobs leave
the prior successful revision searchable. Deletions cascade through the text
and vector indexes.

`sqlite-vec` is pre-v1, so vector-table shape and extension loading must be
tested in startup and migration tests. The application should fail clearly if
the extension is unavailable rather than substitute an in-memory or cloud
index.

## Hybrid Search and RAG

### Retrieval

Each search request produces:

1. A normalized user query.
2. Optional FTS queries generated from exact terms/entities.
3. A query embedding from EmbeddingGemma 2.
4. FTS5/BM25 top-k results.
5. `sqlite-vec` nearest-neighbor top-k results.
6. Reciprocal-rank fusion of the two lists.
7. SQL filtering by tags, path, file type, and user scope.
8. File-level grouping, duplicate suppression, and source diversity.

The app should keep deterministic retrieval scoring visible and use the
generation model for reranking or synthesis, not as the sole recall mechanism.
Store the query plan and result source for debugging.

### Generation Model

For search planning, reranking, summaries, and cited answers:

- Prefer `inkling-2-mini` whenever the hardware profile fits its 18-20 GB model
  footprint and 32 GB+ system guidance.
- Otherwise choose the strongest feasible Gemma 4 local tag:
  - `gemma4:e2b` for constrained systems.
  - `gemma4:e4b` or `gemma4:12b` when additional memory/context is available.
  - `gemma4:26b` or `gemma4:31b` for workstation-class hardware when Inkling is
    not feasible or fails the local benchmark.
- Never select a `gemma4:cloud` or other remote/cloud variant.

Inkling receives the strongest quality weight because it has a 1M-token
published context and strong deep-research/document benchmark results. A large
published context does not justify dumping the library into a prompt; retrieval
must still bound and cite evidence.

### Question Plus Markdown Context

The contextual-search flow is:

1. Extract the optional Markdown context as structured sections.
2. Ask the selected generation model for a typed search plan containing
   intents, exact entities, exclusions, and several subqueries.
3. Execute all subqueries through hybrid retrieval.
4. Merge and diversify candidates by evidence and source file.
5. Rerank a bounded candidate set with Inkling/Gemma 4.
6. Build a prompt with the user question, relevant context excerpts, and
   page/chunk citations.
7. Stream a cited answer. If no retrieved evidence supports a conclusion, say
   so explicitly.
8. Reuse the evidence set for follow-up chat until the query/context changes.

Answers should identify file, page or section, and chunk. They must not present
uncited model knowledge as a result from the user's files.

## Readiness and User Experience

The application setup flow should show:

- Ollama server status and version.
- Whether each role has an installed, compatible local model.
- A recommended profile with the reasons based on detected hardware.
- Alternative Nimble/Tev1 and Inkling/Gemma 4 tiers.
- Required disk space and the commands the user may run themselves.
- A readiness report after the user installs the models.

The model selector must never pull automatically. The app may copy a suggested
command to the clipboard or link to official Ollama model pages.

When readiness fails:

- Do not silently use a smaller generation provider or cloud endpoint.
- Do not make an "AI unavailable" result look like a valid empty search result.
- Keep the failure reason, missing model tag, and next action visible.
- Allow normal file browsing/tag management, but block Ollama-dependent actions.

## Dependency Upgrade Plan

The two dependency manifests must be consolidated into `pyproject.toml`.
Versions below are the latest stable PyPI versions verified on 2026-11-24.
They are planning targets and must be locked by a reproducible resolver before
implementation ships.

### Runtime Dependencies

| Dependency | Planned version | Role |
| --- | ---: | --- |
| PySide6 | 6.12.0 | Desktop UI |
| SQLAlchemy | 2.1.9 | Relational persistence |
| Alembic | 1.20.1 | Database migrations |
| cryptography | 50.0.2 | Existing encrypted configuration/secrets migration |
| ollama | 0.6.3 | Local embedding, System One, and chat client |
| Docling | 2.152.0 | Local multi-format extraction, PDF layout, tables, OCR |
| sqlite-vec | 0.1.9 | Local vector storage/search |
| Pydantic | 2.14.4 | Typed model and internal contracts |
| psutil | 7.2.2 | CPU/RAM/disk profiling |
| platformdirs | 4.14.8 | Stable per-user application directories |
| pywin32 | 312 | Optional Windows hardware/configuration probing |
| nvidia-ml-py | 13.615.71 | Optional NVIDIA VRAM measurement |

### Development and Build Dependencies

| Dependency | Planned version | Role |
| --- | ---: | --- |
| Hatchling | 1.33.0 | Build backend |
| pytest | 9.2.0 | Unit/integration test runner |
| pytest-qt | 4.5.0 | PySide6 UI testing |
| pytest-cov | 7.1.0 | Coverage reporting |
| Ruff | 0.17.6 | Linting and formatting |
| Mypy | 2.4.1 | Static type checking |

### Dependencies to Remove

- Cloud AI: `openai`, `google-generativeai`, `anthropic`.
- In-process generation: `llama-cpp-python`, `ctransformers`.
- Direct embedding stack: `sentence-transformers`, `transformers`.
- Separate vector store and related ONNX stack: `chromadb`.
- Competing PDF/document paths: `pypdf`, `PyPDF2`, `markitdown`, `docx2txt`,
  `python-docx`, `python-pptx`, `openpyxl`, and `beautifulsoup4` unless a
  verified Docling format exception requires one.
- `python-dotenv` if no remaining non-test runtime uses it.
- `tqdm` as a direct dependency unless app code directly requires it.

Before implementation, generate a lock file and inspect the transitive Python
3.13 Windows resolution, especially Docling/OCR extras and PySide6 wheels.
Direct dependencies should be imported by app code or documented as intentional
runtime extras; transitive dependencies should not be duplicated.

## Migration Strategy

### Existing SQLite Database

Use Alembic to move `file_tags.db` from its current shape into the versioned
schema. Preserve file paths, tags, file-tag links, and valid suggestion cache
metadata. Store a pre-migration backup in the new per-user application data
directory and verify counts/checksums before activating the new schema.

### Existing Chroma Database

The old `.chroma` embeddings must not be copied into the new vector space.

1. Detect the legacy database and enumerate known file paths/metadata.
2. Preserve useful file/tag associations if they are absent from SQLite.
3. Mark all legacy indexed files for re-extraction and re-embedding.
4. Keep `.chroma` untouched until the new index passes migration verification
   and the user confirms cleanup.
5. Provide a resumable full reindex with per-file errors and progress.

### Encrypted Configuration

Keep the password/recovery mechanism only where it protects intentional
settings. Migrate provider/API-key fields out of active use. Do not retain
cloud keys in any new config schema, and provide a one-time migration that
removes or archives obsolete cloud settings according to the user's password
flow.

Move databases, indexes, logs, and configuration from the repository directory
to `platformdirs`-defined per-user paths. Development overrides may use a
fixture/test path.

## Error Handling and Observability

Replace placeholder success values and broad fallback content with explicit
domain results:

- Not indexed
- Extracted with warnings
- Ollama unavailable
- Model missing
- Model incompatible
- Context rejected
- OCR failed
- Password/encryption failure
- Embedding/index transaction failed

Log:

- Model tag/digest and index version.
- Per-file extraction/page/chunk counts.
- Truncation rejection/splitting events.
- Per-stage duration and memory estimates.
- Retrieval query plan, candidate ranks, and citation IDs.
- Job completion, failure, retry, and cancellation state.

Logs must not include full user document contents, model prompts, API keys, or
raw recovery secrets by default.

## Quality Gates

Implementation is not complete until all of the following are represented by
automated tests and, where real model behavior is required, opt-in integration
tests:

### Extraction and Coverage

- Born-digital PDF, scanned PDF, encrypted PDF, tables, multi-column content,
  Markdown, DOCX, PPTX, XLSX, and plain text.
- Every extracted section/page maps to at least one accepted embedding chunk.
- An embedding-context rejection recursively splits rather than truncates.
- A chunk-level failure leaves the previous valid index revision intact.

### Search and RAG

- Exact identifier/name recall through FTS5.
- Paraphrase recall through vectors.
- Measurable improvement or non-regression from reciprocal-rank fusion.
- RPG-context golden set: encounters, treasure, random tables, setting/tone,
  exclusions, and multi-file evidence.
- All generated claims map to returned citation chunk IDs.
- "Not found" behavior for unsupported answers.

### Tagging

- Existing-tag classification.
- New-tag candidate proposal plus final System One decision.
- Threshold handling and malformed response rejection.
- Nimble → Tev1 4B → Tev1 0.8B selection on synthetic hardware profiles.
- No long-document state exceeds the decision endpoint limit.

### Readiness and Security

- Ollama offline, too old, missing model, wrong model capability, remote host,
  or cloud-tag rejection.
- No network request except loopback Ollama during normal AI operation.
- Missing `sqlite-vec`/FTS5 diagnostics.
- Database backup and migration rollback.
- Password/config migration without exposing secrets in logs.

### Desktop and Operations

- PySide6 startup/readiness states.
- Cancellation and resume of large indexing jobs.
- Responsive UI during extraction and Ollama calls.
- Windows/Python 3.13 and 3.14 CI.
- Clean packaged build with per-user data paths and no repository-local
  runtime database.

## Delivery Phases

1. **Foundation:** packaging, Python 3.13, locked dependencies, test harness,
   per-user directories, typed errors, and current-behavior characterization.
2. **Storage migration:** Alembic schema, FTS5/sqlite-vec setup, SQLite tag
   repository, backup, and legacy detection.
3. **Extraction:** Docling adapter, page provenance, OCR quality diagnostics,
   structured chunking, and extraction job state.
4. **Ollama readiness:** local gateway, hardware profiler, recommendations,
   model installation guidance, and smoke benchmarks.
5. **Indexing/search:** EmbeddingGemma 2, full-coverage embedding, hybrid
   retrieval, citations, and Chroma removal.
6. **Tagging:** Nimble/Tev1 System One integration, candidate proposal, typed
   validation, and cache invalidation by model/profile.
7. **Contextual RAG:** Inkling/Gemma 4 query planning, optional Markdown
   context, reranking, cited answers, and follow-up chat.
8. **Packaging and release:** Windows packaged app, migration drills, fixture
   quality baseline, performance checks, and updated user documentation.

Each phase should end with a working acceptance boundary. A later feature must
not depend on partially migrated database state.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| `sqlite-vec` is pre-v1 and may change | Pin 0.1.9, isolate extension/table creation, migration-test every upgrade |
| Embedding model/dimension changes invalidate the index | Persist profile/digest/dimension and require a deliberate versioned reindex |
| Large PDFs are slow or memory-heavy | Page-level streaming, resumable jobs, bounded OCR concurrency, visible progress |
| One large model evicts another during multi-role workflows | Short keep-alive, sequential role execution, readiness/performance warnings |
| Hardware detection is incomplete on mixed AMD/Intel/NVIDIA systems | Treat VRAM data as advisory, use conservative available-RAM fit, support user override |
| Inkling may not fit or be fast enough on CPU | Hard feasibility gate, measured benchmark, Gemma 4 E2B/E4B/12B/26B/31B ladder |
| Decision models are poor at open-ended tag generation | Separate candidate generation from final typed classification |
| Vision pages add cost without improving search | Text-first by default, selective image embedding, quality and recall tests |
| Unified database makes packaging paths and backups more important | Use `platformdirs`, transactional writes, startup integrity checks, backups before migrations |

## Out of Scope for This Modernization

- Hosting or bundling Ollama.
- Automatic model downloads.
- Cloud AI models or remote Ollama endpoints.
- Cross-platform first-class support beyond Windows 10/11.
- Replacing PySide6 or rewriting the app as a web application.
- Fully automatic tag application without explicit user policy.
- General file content modification, OCR model training, or custom embedding
  fine-tuning.

## Research Sources

Research was checked on 2026-11-24.

- [Ollama EmbeddingGemma 2](https://ollama.com/library/embeddinggemma-2)
- [Ollama Inkling 2 Mini](https://ollama.com/library/inkling-2-mini)
- [Ollama Nimble](https://ollama.com/library/nimble)
- [Ollama Tev1](https://ollama.com/library/tev1)
- [Ollama Gemma 4](https://ollama.com/library/gemma4)
- [Ollama embed API](https://docs.ollama.com/api/embed)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama running model API](https://docs.ollama.com/api/ps)
- [Ollama Windows requirements](https://docs.ollama.com/windows)
- [Ollama Python 0.6.3](https://pypi.org/project/ollama/)
- [Docling 2.152.0](https://pypi.org/project/docling/)
- [sqlite-vec 0.1.9](https://pypi.org/project/sqlite-vec/)
- [PyPI project metadata](https://pypi.org/) for all dependency versions listed
  above
