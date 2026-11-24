# File Tagger Local-First Modernization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild File Tagger as a Python 3.13+, Windows-first, Ollama-only
application with transactional SQLite/FTS5/vector search, full-coverage local
document ingestion, typed tag decisions, and cited contextual RAG.

**Architecture:** Move behavior into a `src/file_tagger` package with UI,
application, domain, and adapter layers. Preserve the current user workflows
while replacing Chroma and cloud/direct model loading with one SQLite database,
`sqlite-vec`, Docling, and a loopback-only Ollama gateway.

**Tech Stack:** Python 3.13, PySide6 6.12.0, SQLAlchemy 2.1.9, Alembic 1.20.1,
SQLite FTS5, `sqlite-vec 0.1.9`, Docling 2.152.0, Ollama 0.6.3, Pydantic
2.14.4, pytest 9.2.0, Ruff 0.17.6, Mypy 2.4.1.

---

## Execution Rules

1. Implement with `@superpowers:test-driven-development`.
2. Use `@superpowers:systematic-debugging` for every unexpected failure.
3. Run focused tests after every task and
   `@superpowers:verification-before-completion` before merging a phase.
4. Keep Ollama models and application tests separate: default tests use a fake
   local Ollama adapter; real-model tests are explicitly opted in.
5. Never add a cloud fallback, remote Ollama host, or `-cloud` model tag.
6. Never copy an embedding produced by the old Sentence Transformers model
   into a new EmbeddingGemma index.
7. Commit after each task with the Co-authored-by trailer required by the
   repository.

## Reference Documents

- Approved architecture:
  [`2026-11-24-modernization-design.md`](./2026-11-24-modernization-design.md)
- Current runtime entrypoint: [`main.py`](../../main.py)
- Current provider service: [`ai_service.py`](../../ai_service.py)
- Current Chroma adapter:
  [`vector_search/vector_search.py`](../../vector_search/vector_search.py)
- Current extraction and chunking:
  [`vector_search/content_extractor.py`](../../vector_search/content_extractor.py)
  and
  [`vector_search/document_chunker.py`](../../vector_search/document_chunker.py)

## Phase 0: Project Foundation

### Task 1: Lock the Modern Python Toolchain

**Files:**
- Modify: `pyproject.toml`
- Delete after migration: `requirements.txt`
- Create: `uv.lock`
- Create: `tests/test_project_metadata.py`

**Step 1: Write the failing metadata test**

```python
from pathlib import Path
import tomllib


def test_project_targets_python_313_and_has_one_dependency_manifest() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert project["requires-python"].startswith(">=3.13")
    assert not Path("requirements.txt").exists()
```

**Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_project_metadata.py -v`
Expected: FAIL because the project still declares Python 3.8 and
`requirements.txt` exists.

**Step 3: Write the minimal project metadata**

Replace the dependency tables with exact compatible releases from the approved
design, including optional hardware dependencies. Configure Hatch, Ruff,
pytest, Mypy, and pytest coverage in `pyproject.toml`. Generate `uv.lock` with
the Python 3.13 Windows resolution.

**Step 4: Run the test and resolver checks**

Run:

```powershell
uv lock
uv sync --locked --all-extras
python -m pytest tests/test_project_metadata.py -v
uv run ruff check .
```

Expected: lock/sync succeed, metadata test passes, and Ruff reports only
pre-existing findings selected for the next package-migration task.

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/test_project_metadata.py
git rm requirements.txt
git commit -m "build: lock modern Python toolchain"
```

### Task 2: Create the Importable Application Package

**Files:**
- Create: `src/file_tagger/__init__.py`
- Create: `src/file_tagger/__main__.py`
- Move: `main.py` to `src/file_tagger/main.py`
- Move: all current root application modules into `src/file_tagger/`
- Move: `vector_search/` to `src/file_tagger/vector_search/`
- Modify: all moved import statements to package-relative imports
- Test: `tests/smoke/test_package_import.py`

**Step 1: Write the failing package test**

```python
def test_application_package_imports() -> None:
    from file_tagger.main import main

    assert callable(main)
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/smoke/test_package_import.py -v`
Expected: FAIL because `file_tagger` is not a package.

**Step 3: Move code without changing behavior**

Use `git mv` for the existing modules and nested `vector_search` package. Update
imports only; do not refactor behavior. Fix `[project.scripts]` to
`file-tagger = "file_tagger.main:main"`.

**Step 4: Run package and legacy smoke checks**

Run:

```powershell
pytest tests/smoke/test_package_import.py -v
python -m file_tagger --help
file-tagger
```

Expected: package import passes, `python -m file_tagger` reaches the GUI/login
path, and the installed script resolves the same entrypoint.

**Step 5: Commit**

```bash
git add src tests pyproject.toml
git commit -m "refactor: create file_tagger package"
```

### Task 3: Define Per-User Application Paths

**Files:**
- Create: `src/file_tagger/infrastructure/app_paths.py`
- Modify: `src/file_tagger/main.py`
- Modify: `src/file_tagger/config.py`
- Modify: `src/file_tagger/initialization.py`
- Create: `tests/infrastructure/test_app_paths.py`

**Step 1: Write the failing path-isolation test**

```python
from file_tagger.infrastructure.app_paths import AppPaths


def test_paths_are_built_under_platformdirs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    assert paths.database == tmp_path / "file_tagger.sqlite3"
    assert paths.legacy_database != paths.database
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/infrastructure/test_app_paths.py -v`
Expected: FAIL because `file_tagger.infrastructure.app_paths` does not exist.

**Step 3: Implement `AppPaths`**

Return typed absolute paths for data, config, logs, cache, temporary extraction,
legacy `file_tags.db`, and legacy `.chroma`. Support the
`FILE_TAGGER_DATA_DIR` environment override only for tests and portable
development.

**Step 4: Run path and startup tests**

Run: `pytest tests/infrastructure/test_app_paths.py tests/smoke -v`
Expected: all pass, and logs are no longer written beside `main.py`.

**Step 5: Commit**

```bash
git add src/file_tagger/infrastructure tests/infrastructure src/file_tagger/main.py
git commit -m "feat: isolate application data paths"
```

### Task 4: Add Typed Domain Records and Explicit Errors

**Files:**
- Create: `src/file_tagger/domain/errors.py`
- Create: `src/file_tagger/domain/contracts.py`
- Test: `tests/domain/test_contracts.py`

**Step 1: Write the failing contract test**

```python
import pytest
from pydantic import ValidationError
from file_tagger.domain.contracts import ModelProfile


def test_model_profile_rejects_cloud_model() -> None:
    with pytest.raises(ValidationError):
        ModelProfile(
            role="generation",
            model="gemma4:cloud",
            digest="sha256:test",
            context_length=256_000,
        )
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/domain/test_contracts.py -v`
Expected: FAIL with a missing domain package.

**Step 3: Implement contracts and errors**

Add Pydantic records for hardware reports, Ollama readiness, model profiles,
extract/page/chunk records, search candidates, citations, tag decisions, and
RAG plans. Add exception subclasses for every externally visible failure named
in the design.

**Step 4: Run domain tests and Mypy**

Run: `pytest tests/domain -v && mypy src/file_tagger/domain`
Expected: tests and type checking pass.

**Step 5: Commit**

```bash
git add src/file_tagger/domain tests/domain
git commit -m "feat: add typed local AI contracts"
```

## Phase 1: Unified SQLite Storage

### Task 5: Load FTS5 and sqlite-vec on Every Connection

**Files:**
- Create: `src/file_tagger/infrastructure/database.py`
- Test: `tests/infrastructure/test_database.py`

**Step 1: Write the failing extension test**

```python
def test_database_has_fts5_and_vector_extensions(tmp_path) -> None:
    from file_tagger.infrastructure.database import create_session_factory

    session_factory = create_session_factory(tmp_path / "app.sqlite3")
    with session_factory() as session:
        assert session.scalar(text("SELECT fts5(?)", ("ready",))) == 1
        assert session.scalar(text("SELECT vec_version() IS NOT NULL"))
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/infrastructure/test_database.py -v`
Expected: FAIL because the database module does not exist.

**Step 3: Implement the engine factory**

Enable foreign keys, WAL, `busy_timeout`, and predictable transactions.
Register a SQLAlchemy connection event that loads `sqlite_vec.load(...)`.
Check FTS5 and `vec_version()` at startup and raise a typed diagnostic error
instead of continuing without either extension.

**Step 4: Run database integration tests**

Run: `pytest tests/infrastructure/test_database.py -v`
Expected: FTS5 and sqlite-vec checks pass on Windows and the test reports an
explicit error for a monkeypatched missing extension.

**Step 5: Commit**

```bash
git add src/file_tagger/infrastructure/database.py tests/infrastructure/test_database.py
git commit -m "feat: bootstrap sqlite search extensions"
```

### Task 6: Create the Versioned Unified Schema

**Files:**
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/versions/0001_unified_search_schema.py`
- Create: `src/file_tagger/persistence/models.py`
- Modify: `src/file_tagger/infrastructure/database.py`
- Test: `tests/persistence/test_schema.py`

**Step 1: Write the failing schema test**

```python
def test_new_database_has_unified_search_tables(test_session) -> None:
    names = set(test_session.execute(text(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
    )).scalars())
    assert {
        "files",
        "tags",
        "file_tags",
        "documents",
        "pages",
        "chunks",
        "index_jobs",
        "model_profiles",
    } <= names
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/persistence/test_schema.py -v`
Expected: FAIL because the unified schema is absent.

**Step 3: Implement models and migration**

Map relational tables with SQLAlchemy. Create FTS5 and one `vec0` table per
configured embedding dimension in an isolated schema helper. Add foreign-key
cascades and uniqueness constraints around file revision, model digest, and
chunk ordinal.

**Step 4: Run migration tests**

Run:

```powershell
alembic upgrade head
pytest tests/persistence/test_schema.py -v
alembic downgrade base
alembic upgrade head
```

Expected: migration is repeatable and all required tables/indexes exist.

**Step 5: Commit**

```bash
git add alembic.ini alembic src/file_tagger/persistence tests/persistence
git commit -m "feat: add unified search database schema"
```

### Task 7: Migrate Existing Tags, Files, and Config

**Files:**
- Create: `src/file_tagger/migration/sqlite_migration.py`
- Create: `src/file_tagger/migration/config_migration.py`
- Test: `tests/migration/test_legacy_sqlite.py`
- Test: `tests/migration/test_config_migration.py`

**Step 1: Write the failing migration test**

```python
def test_legacy_file_tags_are_preserved(legacy_db, target_session) -> None:
    report = migrate_legacy_sqlite(legacy_db, target_session)
    assert report.files_migrated == 1
    assert target_session.scalar(select(func.count(Tag.id))) == 2
    assert report.errors == []
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/migration -v`
Expected: FAIL because the migration module does not exist.

**Step 3: Implement backup-first migration**

Create a timestamped backup before writes. Preserve paths, tags, file/tag
associations, and migration-safe suggestion data. Verify source and target
counts/checksums. Remove active use of cloud key and provider fields but never
log key contents.

**Step 4: Run migration and failure-path tests**

Run: `pytest tests/migration -v`
Expected: pass for current schema, an older minimal schema, duplicate records,
and a forced target-write failure that leaves the backup intact.

**Step 5: Commit**

```bash
git add src/file_tagger/migration tests/migration
git commit -m "feat: migrate legacy tags and settings"
```

### Task 8: Detect and Retire Legacy Chroma State

**Files:**
- Create: `src/file_tagger/migration/legacy_chroma.py`
- Create: `tests/migration/test_legacy_chroma.py`

**Step 1: Write the failing legacy-index test**

```python
def test_chroma_paths_are_reindexed_not_vector_copied(legacy_collection) -> None:
    plan = inspect_legacy_chroma(legacy_collection)
    assert plan.paths == ["C:/library/source.pdf"]
    assert plan.reuse_embeddings is False
    assert plan.requires_full_reindex is True
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/migration/test_legacy_chroma.py -v`
Expected: FAIL with the missing migration module.

**Step 3: Implement inventory-only migration**

Read only legacy paths and metadata. Mark each valid path for extraction and
embedding. Do not import old vectors. Do not delete `.chroma`; expose a separate
user-confirmed cleanup operation after successful migration verification.

**Step 4: Run tests with malformed and missing Chroma data**

Run: `pytest tests/migration/test_legacy_chroma.py -v`
Expected: valid metadata is inventoried, malformed rows become explicit
warnings, and missing `.chroma` is a no-op.

**Step 5: Commit**

```bash
git add src/file_tagger/migration/legacy_chroma.py tests/migration/test_legacy_chroma.py
git commit -m "feat: inventory legacy chroma content"
```

## Phase 2: Ollama and Hardware Readiness

### Task 9: Add the Loopback-Only Ollama Gateway

**Files:**
- Create: `src/file_tagger/adapters/ollama/gateway.py`
- Create: `tests/adapters/ollama/test_gateway.py`

**Step 1: Write the failing local-host test**

```python
@pytest.mark.parametrize("host", [
    "http://localhost:11434",
    "http://127.0.0.1:11434",
])
def test_loopback_host_is_allowed(host) -> None:
    assert OllamaGateway(host).host == host


def test_remote_host_is_rejected() -> None:
    with pytest.raises(RemoteOllamaHostError):
        OllamaGateway("https://example.invalid")
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/adapters/ollama/test_gateway.py -v`
Expected: FAIL because `OllamaGateway` is absent.

**Step 3: Implement gateway readiness**

Use `ollama.AsyncClient`. Implement version/model probes, installed model
lookup, model metadata, embeddings with `truncate=False`, System One, structured
chat, streaming, and typed connection/time-out errors. Parse and reject cloud
tags in all model entrypoints.

**Step 4: Run gateway contract tests**

Run: `pytest tests/adapters/ollama/test_gateway.py -v`
Expected: local hosts pass, remote/cloud cases fail explicitly, and fake-client
responses are converted into domain contracts.

**Step 5: Commit**

```bash
git add src/file_tagger/adapters/ollama tests/adapters/ollama
git commit -m "feat: add local Ollama gateway"
```

### Task 10: Recommend Models from Hardware

**Files:**
- Create: `src/file_tagger/ollama/hardware.py`
- Create: `src/file_tagger/ollama/model_recommender.py`
- Create: `tests/ollama/test_model_recommender.py`

**Step 1: Write the failing recommendation test**

```python
@pytest.mark.parametrize(
    ("available_ram_gb", "expected_generation"),
    [(8, "gemma4:e2b"), (24, "gemma4:12b"), (32, "inkling-2-mini")],
)
def test_generation_model_follows_feasible_size(
    available_ram_gb, expected_generation
) -> None:
    recommendation = recommend_models(HardwareReport(
        available_ram_gb=available_ram_gb,
        free_disk_gb=100,
        logical_cores=8,
        vram_gb=None,
    ))
    assert recommendation.generation.model == expected_generation
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/ollama/test_model_recommender.py -v`
Expected: FAIL because the recommender does not exist.

**Step 3: Implement hard gates and weighted scores**

Define the model catalog and published footprints in versioned code. Use
available RAM/disk/headroom and optional NVIDIA/Windows GPU data as hard gates.
Weight Inkling highest among feasible generation models. Select Nimble, then
Tev1 4B, then Tev1 0.8B. Select EmbeddingGemma 270m by default, 440m for visual
profiles, and 740m only when the requested multimodal profile is feasible.

**Step 4: Run the hardware matrix**

Run: `pytest tests/ollama/test_model_recommender.py -v`
Expected: tests cover low RAM, CPU-only, discrete VRAM, low disk, user
override, Inkling, every Gemma 4 tier, and all decision fallbacks.

**Step 5: Commit**

```bash
git add src/file_tagger/ollama tests/ollama
git commit -m "feat: recommend Ollama models from hardware"
```

### Task 11: Benchmark and Persist Model Profiles

**Files:**
- Create: `src/file_tagger/ollama/benchmarks.py`
- Create: `src/file_tagger/ollama/readiness.py`
- Test: `tests/ollama/test_readiness.py`

**Step 1: Write the failing readiness test**

```python
def test_missing_decision_model_blocks_tagging(fake_gateway) -> None:
    report = readiness.check(required=required_profile())
    assert report.ready is False
    assert report.blocked_roles == ["tagging"]
    assert "ollama pull" in report.instructions
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/ollama/test_readiness.py -v`
Expected: FAIL because readiness is absent.

**Step 3: Implement readiness and smoke benchmarks**

Check Ollama 0.35+, installed model tags, capabilities, digests, and context.
Benchmark a small embed, one decision, and a short generation without sending
user content. Store the selected model digest/profile and measured latency.
Generate manual pull commands; never pull automatically.

**Step 4: Run readiness tests**

Run: `pytest tests/ollama/test_readiness.py -v`
Expected: old server, missing model, cloud tag, wrong capability, and slow but
successful benchmarks all receive distinct reports.

**Step 5: Commit**

```bash
git add src/file_tagger/ollama tests/ollama
git commit -m "feat: add Ollama readiness benchmarks"
```

## Phase 3: Extraction, Chunking, and Indexing

### Task 12: Replace PDF/Office Extraction with Docling

**Files:**
- Create: `src/file_tagger/extraction/docling_adapter.py`
- Create: `src/file_tagger/extraction/quality.py`
- Delete behavior in: `src/file_tagger/vector_search/content_extractor.py`
- Test: `tests/extraction/test_docling_adapter.py`

**Step 1: Write the failing extraction test**

```python
def test_pdf_extraction_preserves_page_provenance(sample_pdf, fake_docling) -> None:
    document = extract_document(sample_pdf)
    assert document.pages[0].number == 1
    assert document.pages[0].markdown.startswith("# Title")
    assert document.parser == "docling"
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/extraction/test_docling_adapter.py -v`
Expected: FAIL because the adapter is absent.

**Step 3: Implement the Docling adapter**

Use one conversion pipeline for PDF, DOCX, PPTX, XLSX, images, HTML, Markdown,
and plain text. Export normalized Markdown plus structured page/table
provenance. Convert conversion warnings, encrypted documents, OCR failure, and
unsupported format conditions into typed results.

**Step 4: Run extraction fixtures**

Run: `pytest tests/extraction -v`
Expected: born-digital, scanned, table, multi-column, Office, and plain-text
fixtures pass; encrypted and unsupported files return visible typed failures.

**Step 5: Commit**

```bash
git add src/file_tagger/extraction tests/extraction
git commit -m "feat: extract documents with Docling"
```

### Task 13: Add Selective Vision Augmentation

**Files:**
- Create: `src/file_tagger/extraction/vision_candidates.py`
- Modify: `src/file_tagger/extraction/docling_adapter.py`
- Test: `tests/extraction/test_vision_candidates.py`

**Step 1: Write the failing selective-vision test**

```python
def test_born_digital_text_page_is_not_rendered() -> None:
    candidate = classify_page(text_chars=2400, image_area=0.05, tables=0)
    assert candidate.needs_vision_embedding is False


def test_diagram_only_page_is_rendered() -> None:
    candidate = classify_page(text_chars=20, image_area=0.75, tables=0)
    assert candidate.needs_vision_embedding is True
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/extraction/test_vision_candidates.py -v`
Expected: FAIL because page classification is absent.

**Step 3: Implement quality classification**

Flag pages with missing/low-quality OCR text, dominant diagram/figure coverage,
or tables whose extracted representation fails structural checks. Render only
flagged pages, and request the visual embedding only when the selected
EmbeddingGemma profile supports it.

**Step 4: Run extraction and vision selection tests**

Run: `pytest tests/extraction -v`
Expected: text-heavy pages avoid rendering, visual pages are selected, and
text-only profiles return an explicit "visual augmentation disabled" report.

**Step 5: Commit**

```bash
git add src/file_tagger/extraction tests/extraction
git commit -m "feat: add selective PDF vision fallback"
```

### Task 14: Make Chunking Structure- and Context-Aware

**Files:**
- Create: `src/file_tagger/indexing/chunker.py`
- Delete behavior in: `src/file_tagger/vector_search/document_chunker.py`
- Test: `tests/indexing/test_chunker.py`

**Step 1: Write the failing coverage test**

```python
def test_every_source_span_is_covered_by_a_chunk() -> None:
    pages = [page_with_sections_and_table()]
    chunks = chunk_document(pages, soft_limit_tokens=200)
    covered = merge_spans(chunk.source_span for chunk in chunks)
    assert covered == [(0, pages[0].source_length)]
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/indexing/test_chunker.py -v`
Expected: FAIL because the new chunker does not exist.

**Step 3: Implement structural chunking**

Split page → heading/section → paragraph/table. Give every chunk source
offsets, page, heading, and stable content hash. Keep overlap bounded and
semantic. Do not expose a maximum file byte/character count.

**Step 4: Run coverage and ordering tests**

Run: `pytest tests/indexing/test_chunker.py -v`
Expected: complete ordered coverage for long headings, tables, non-ASCII text,
Markdown, and pages with no headings.

**Step 5: Commit**

```bash
git add src/file_tagger/indexing tests/indexing
git commit -m "feat: add provenance-aware document chunking"
```

### Task 15: Enforce Full Coverage at the Embedding Boundary

**Files:**
- Create: `src/file_tagger/indexing/embedder.py`
- Modify: `src/file_tagger/adapters/ollama/gateway.py`
- Test: `tests/indexing/test_embedder.py`

**Step 1: Write the failing recursive-split test**

```python
def test_context_rejection_splits_until_every_span_is_embedded(
    oversized_chunk, rejecting_then_accepting_gateway
) -> None:
    result = embed_document_chunks([oversized_chunk])
    assert len(result.vectors) > 1
    assert result.complete_source_coverage is True
    assert all(call.truncate is False for call in result.calls)
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/indexing/test_embedder.py -v`
Expected: FAIL because full-coverage embedding does not exist.

**Step 3: Implement adaptive embedding**

Call `embed(..., truncate=False, dimensions=profile.dimensions)`. On a typed
context-too-long error, bisect the chunk at the safest structural boundary and
retry. Preserve source spans and order. Refuse mixed model digests/dimensions.

**Step 4: Run context, batching, and failure tests**

Run: `pytest tests/indexing/test_embedder.py -v`
Expected: accepted chunks are complete, rejected chunks split, unsplittable
failures are explicit, and no call requests truncation.

**Step 5: Commit**

```bash
git add src/file_tagger/indexing/embedder.py tests/indexing/test_embedder.py
git commit -m "feat: guarantee complete embedding coverage"
```

### Task 16: Add Transactional, Resumable Index Jobs

**Files:**
- Create: `src/file_tagger/indexing/service.py`
- Create: `src/file_tagger/indexing/repository.py`
- Test: `tests/indexing/test_service.py`

**Step 1: Write the failing atomicity test**

```python
def test_vector_insert_failure_keeps_previous_revision_searchable(
    test_session, repository, service, file_revision
) -> None:
    repository.fail_next_vector_insert()
    result = service.reindex(file_revision)
    assert result.status == "failed"
    assert repository.current_revision(file_revision.path) == "previous"
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/indexing/test_service.py -v`
Expected: FAIL because the indexing service is absent.

**Step 3: Implement revision swap and resumability**

Track queued/running/failed/completed job state. Extract and embed outside the
database write lock, then atomically replace document/pages/chunks/FTS/vectors
for one revision. Store partial progress, retry count, and a safe error summary.

**Step 4: Run transactional and cancellation tests**

Run: `pytest tests/indexing/test_service.py -v`
Expected: old revisions survive failures, cancellation checkpoints resume, and
all indexes are committed or rolled back together.

**Step 5: Commit**

```bash
git add src/file_tagger/indexing/service.py src/file_tagger/indexing/repository.py tests/indexing/test_service.py
git commit -m "feat: add resumable transactional indexing"
```

## Phase 4: Hybrid Retrieval and Citations

### Task 17: Implement FTS5 and Vector Search with RRF

**Files:**
- Create: `src/file_tagger/search/hybrid.py`
- Create: `src/file_tagger/search/rrf.py`
- Create: `tests/search/test_hybrid.py`

**Step 1: Write the failing fusion test**

```python
def test_rrf_preserves_complementary_lexical_and_vector_hits() -> None:
    fused = reciprocal_rank_fusion(
        {"lexical": ["exact", "shared"], "vector": ["shared", "semantic"]},
        k=60,
    )
    assert fused[0].chunk_id == "shared"
    assert {item.chunk_id for item in fused} == {"exact", "shared", "semantic"}
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/search/test_hybrid.py -v`
Expected: FAIL because hybrid search does not exist.

**Step 3: Implement safe hybrid queries**

Escape/normalize user input for FTS5 MATCH. Query exact metadata and BM25
separately from sqlite-vec. Fuse by rank, apply tag/path/type filters in SQL,
group by file, and return candidates with stable chunk IDs and page/heading
citation data.

**Step 4: Run integration recall tests**

Run: `pytest tests/search/test_hybrid.py -v`
Expected: exact-term, semantic, mixed, empty, and filtered searches pass with no
SQL injection or duplicate file regressions.

**Step 5: Commit**

```bash
git add src/file_tagger/search tests/search
git commit -m "feat: add SQLite hybrid search"
```

### Task 18: Add Citation-First Search Results

**Files:**
- Modify: `src/file_tagger/search/hybrid.py`
- Create: `src/file_tagger/search/results.py`
- Create: `tests/search/test_results.py`

**Step 1: Write the failing citation test**

```python
def test_search_result_cites_page_and_chunk() -> None:
    result = make_result(seed_chunk())
    citation = result.citations[0]
    assert citation.path.endswith("adventure.pdf")
    assert citation.page_number == 12
    assert citation.chunk_id
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/search/test_results.py -v`
Expected: FAIL because citation result records are absent.

**Step 3: Implement result hydration**

Hydrate paths, tags, summary, matched sections, page/heading, and stable chunk
IDs from one search transaction. Keep displayed snippets bounded and generated
from indexed text, not file re-extraction.

**Step 4: Run result consistency tests**

Run: `pytest tests/search/test_results.py -v`
Expected: citations resolve to stored chunks, stale files are marked, and
duplicates are merged deterministically.

**Step 5: Commit**

```bash
git add src/file_tagger/search tests/search
git commit -m "feat: add cited hybrid search results"
```

## Phase 5: Typed Tagging

### Task 19: Replace Free-Form Tag Parsing with Nimble/Tev1

**Files:**
- Create: `src/file_tagger/tagging/decision_service.py`
- Delete: `src/file_tagger/ai_service.py`
- Delete: old provider branches in `src/file_tagger/tag_suggestion.py`
- Test: `tests/tagging/test_decision_service.py`

**Step 1: Write the failing System One test**

```python
def test_tag_decision_uses_installed_model(fake_ollama) -> None:
    result = decide_tags(
        document_state="encounter table for a flooded ruin",
        candidates=["dungeon", "rpg", "invoice"],
        model="nimble:9b",
    )
    assert result.accepted[0].tag == "dungeon"
    assert 0.0 <= result.accepted[0].confidence <= 1.0
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/tagging/test_decision_service.py -v`
Expected: FAIL because the decision service is absent.

**Step 3: Implement typed tagging decisions**

Build one System One request with up to the supported number of questions.
Parse `choice`, `noul`, and score responses into validated Pydantic records.
Use Nimble by default and the same contract for Tev1 fallbacks.

**Step 4: Run model-contract and malformed-response tests**

Run: `pytest tests/tagging/test_decision_service.py -v`
Expected: valid confidence/probabilities pass; missing answers, unknown
candidates, out-of-range values, and oversized state fail visibly.

**Step 5: Commit**

```bash
git add src/file_tagger/tagging tests/tagging
git commit -m "feat: add typed Ollama tag decisions"
```

### Task 20: Generate and Cache Tag Candidates

**Files:**
- Create: `src/file_tagger/tagging/candidate_service.py`
- Create: `src/file_tagger/tagging/cache.py`
- Test: `tests/tagging/test_candidate_service.py`

**Step 1: Write the failing cache-invalidation test**

```python
def test_model_digest_change_invalidates_cached_tags(cache, stored) -> None:
    assert cache.lookup(stored.file_hash, model_digest="old") == stored
    assert cache.lookup(stored.file_hash, model_digest="new") is None
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/tagging/test_candidate_service.py -v`
Expected: FAIL because candidate/cache services are absent.

**Step 3: Implement candidate generation and caching**

Retrieve existing tags semantically, generate a bounded list of new candidates
through Inkling/Gemma 4, send all candidates to the decision model, and cache by
file hash, extraction version, candidate-model digest, and decision-model
digest.

**Step 4: Run cache and candidate tests**

Run: `pytest tests/tagging/test_candidate_service.py -v`
Expected: existing tags remain preferred, duplicate/case variants collapse,
changed models invalidate cache, and low-confidence tags require confirmation.

**Step 5: Commit**

```bash
git add src/file_tagger/tagging tests/tagging
git commit -m "feat: cache typed tag recommendations"
```

## Phase 6: Contextual RAG

### Task 21: Build a Typed Search Plan from Question Plus Markdown

**Files:**
- Create: `src/file_tagger/rag/planner.py`
- Create: `src/file_tagger/rag/context.py`
- Test: `tests/rag/test_planner.py`

**Step 1: Write the failing contextual-plan test**

```python
def test_rpg_outline_creates_targeted_search_intents() -> None:
    plan = build_search_plan(
        question="Find encounters, treasure, and random tables",
        markdown_context="Level 3 party exploring a flooded temple",
    )
    assert {"encounters", "treasure", "random tables"} <= plan.intent_names
    assert "level 3" in plan.normalized_constraints
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/rag/test_planner.py -v`
Expected: FAIL because the RAG planner is absent.

**Step 3: Implement structured planning**

Normalize Markdown sections and entities. Ask Inkling first or the selected
Gemma 4 fallback for a Pydantic-validated `SearchPlan` with exact terms,
paraphrase queries, filters, exclusions, and the evidence categories needed.

**Step 4: Run planner tests**

Run: `pytest tests/rag/test_planner.py -v`
Expected: valid plans cover intents and constraints; malformed JSON, cloud
models, and empty plans fail before retrieval.

**Step 5: Commit**

```bash
git add src/file_tagger/rag/planner.py src/file_tagger/rag/context.py tests/rag/test_planner.py
git commit -m "feat: plan searches from Markdown context"
```

### Task 22: Rerank Evidence and Generate Cited Answers

**Files:**
- Create: `src/file_tagger/rag/service.py`
- Create: `src/file_tagger/rag/citations.py`
- Test: `tests/rag/test_service.py`

**Step 1: Write the failing citation-validation test**

```python
def test_answer_without_valid_citation_is_rejected(fake_model, evidence) -> None:
    fake_model.returns(
        "The adventure should include a dragon. [source: invented]"
    )
    with pytest.raises(UncitedAnswerError):
        answer_question("Find suitable encounters", evidence)
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/rag/test_service.py -v`
Expected: FAIL because citation-aware RAG is absent.

**Step 3: Implement bounded RAG**

Execute search-plan subqueries, merge/diversify candidates, optionally rerank a
bounded top-k with Inkling/Gemma 4, allocate prompt space from runtime context,
and require a structured answer containing claims mapped to known chunk IDs.
Use "not found in indexed files" when evidence is insufficient.

**Step 4: Run RAG quality and failure tests**

Run: `pytest tests/rag/test_service.py -v`
Expected: citation IDs resolve, follow-up chat reuses evidence, unsupported
claims fail, and model context overflow triggers further evidence reduction.

**Step 5: Commit**

```bash
git add src/file_tagger/rag tests/rag
git commit -m "feat: generate citation-validated local answers"
```

## Phase 7: PySide6 Integration

### Task 23: Replace API Settings with Ollama Readiness Settings

**Files:**
- Create: `src/file_tagger/ui/ollama_settings.py`
- Delete: `src/file_tagger/api_settings.py`
- Modify: `src/file_tagger/file_tag_manager.py`
- Test: `tests/ui/test_ollama_settings.py`

**Step 1: Write the failing UI test**

```python
def test_missing_model_disables_ai_features(qtbot, readiness_report) -> None:
    dialog = OllamaSettingsDialog(readiness_report)
    qtbot.addWidget(dialog)
    assert dialog.ok_button.isEnabled() is False
    assert "ollama pull nimble:9b" in dialog.instructions.text()
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/ui/test_ollama_settings.py -v`
Expected: FAIL because the readiness settings UI is absent.

**Step 3: Implement the settings UI**

Show Ollama host/version, detected hardware, recommendations, installed
models, disk fit, benchmark results, manual pull commands, and a refresh
button. Permit only local loopback and model selections from the installed
catalog.

**Step 4: Run UI tests**

Run: `pytest tests/ui/test_ollama_settings.py -v`
Expected: missing, outdated, ready, and user-override states render without
network/model downloads.

**Step 5: Commit**

```bash
git add src/file_tagger/ui/ollama_settings.py tests/ui src/file_tagger/file_tag_manager.py
git rm src/file_tagger/api_settings.py
git commit -m "feat: replace provider settings with Ollama readiness"
```

### Task 24: Wire Indexing, Reindex, and Migration Progress

**Files:**
- Modify: `src/file_tagger/initialization.py`
- Modify: `src/file_tagger/file_tag_manager.py`
- Create: `src/file_tagger/ui/indexing_status.py`
- Test: `tests/ui/test_indexing_status.py`

**Step 1: Write the failing UI cancellation test**

```python
def test_index_cancel_does_not_delete_previous_revision(qtbot, service) -> None:
    status = IndexingStatus(service)
    status.cancel()
    assert service.cancel_requested is True
    assert service.previous_revision_available is True
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/ui/test_indexing_status.py -v`
Expected: FAIL because the indexing status UI is absent.

**Step 3: Implement progress wiring**

Replace direct `VectorSearch` calls with index-service commands. Show
per-file extraction/embed/commit status, resumable errors, reindex-all,
single-file reindex, migration inventory, and safe legacy cleanup.

**Step 4: Run UI and service integration tests**

Run: `pytest tests/ui/test_indexing_status.py tests/indexing -v`
Expected: progress updates do not freeze the UI and failed/cancelled jobs
preserve the previous revision.

**Step 5: Commit**

```bash
git add src/file_tagger/ui/indexing_status.py src/file_tagger/initialization.py src/file_tagger/file_tag_manager.py tests/ui/test_indexing_status.py
git commit -m "feat: expose resumable indexing in desktop UI"
```

### Task 25: Add Contextual Markdown Search and Chat

**Files:**
- Modify: `src/file_tagger/file_tag_manager.py`
- Modify: `src/file_tagger/search.py`
- Create: `src/file_tagger/ui/contextual_search.py`
- Test: `tests/ui/test_contextual_search.py`

**Step 1: Write the failing contextual UI test**

```python
def test_question_accepts_optional_markdown_context(qtbot, tmp_path) -> None:
    dialog = ContextualSearchDialog()
    dialog.question.setText("Find encounters and treasure")
    dialog.set_context_file(tmp_path / "outline.md")
    assert dialog.submission().context_file.name == "outline.md"
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/ui/test_contextual_search.py -v`
Expected: FAIL because the contextual-search UI is absent.

**Step 3: Implement the workflow**

Add an optional Markdown context picker, question input, readiness gate,
streaming answer, citations, evidence-file list, and follow-up prompt. Reuse the
RAG service rather than duplicating provider logic in the dialog.

**Step 4: Run UI and RAG integration tests**

Run: `pytest tests/ui/test_contextual_search.py tests/rag -v`
Expected: optional context and query-only workflows pass, citations activate the
source location, and missing models disable submission with setup guidance.

**Step 5: Commit**

```bash
git add src/file_tagger/ui/contextual_search.py src/file_tagger/search.py src/file_tagger/file_tag_manager.py tests/ui
git commit -m "feat: add Markdown-contextual local search"
```

## Phase 8: Cleanup, Packaging, and Acceptance

### Task 26: Remove Cloud and Competing Runtime Stacks

**Files:**
- Delete: `src/file_tagger/ai_service.py` if not already removed
- Delete: `src/file_tagger/vector_search/vector_search.py`
- Delete: `src/file_tagger/vector_search/content_extractor.py`
- Delete: `src/file_tagger/vector_search/document_chunker.py`
- Delete: `src/file_tagger/vector_search/search_utils.py`
- Modify: `src/file_tagger/file_tag_manager.py`
- Modify: `src/file_tagger/tag_suggestion.py`
- Modify: `src/file_tagger/search.py`
- Modify: `pyproject.toml`
- Test: `tests/test_local_only_dependencies.py`

**Step 1: Write the failing local-only test**

```python
FORBIDDEN = {
    "openai",
    "anthropic",
    "google-generativeai",
    "chromadb",
    "sentence-transformers",
    "transformers",
    "llama-cpp-python",
    "ctransformers",
}


def test_runtime_dependencies_are_local_ollama_only(project) -> None:
    names = {dependency.partition(">=")[0] for dependency in project.dependencies}
    assert names.isdisjoint(FORBIDDEN)
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/test_local_only_dependencies.py -v`
Expected: FAIL while old runtime dependencies remain.

**Step 3: Remove old branches and dependencies**

Delete unused imports, configuration getters, direct model-path UI, provider
parsers, debug scripts that depend on Chroma, and cloud/local duplicate
dependencies. Retain compatibility behavior only in explicit migration code.

**Step 4: Run focused suite and source audit**

Run:

```powershell
pytest tests/test_local_only_dependencies.py tests/ui tests/search tests/tagging tests/rag -v
rg "openai|anthropic|google-generativeai|chromadb|llama-cpp|ctransformers" src pyproject.toml
```

Expected: focused tests pass and the source/dependency audit returns no runtime
matches.

**Step 5: Commit**

```bash
git add src tests pyproject.toml uv.lock
git commit -m "refactor: remove cloud and legacy model stacks"
```

### Task 27: Add Golden Migration and Retrieval Acceptance Tests

**Files:**
- Create: `tests/acceptance/test_full_upgrade.py`
- Create: `tests/fixtures/legacy/file_tags.sqlite3`
- Create: `tests/fixtures/legacy/config.encrypted`
- Create: `tests/fixtures/library/*.pdf` and Markdown context fixtures
- Test: `tests/acceptance/test_full_upgrade.py`

**Step 1: Write the failing end-to-end acceptance test**

```python
def test_upgrade_indexes_and_answers_from_local_evidence(
    legacy_install, fake_ollama
) -> None:
    app = start_acceptance_app(legacy_install)
    app.upgrade()
    app.reindex_all()
    answer = app.ask(
        "Find encounters and treasure",
        context=legacy_install / "adventure-outline.md",
    )
    assert answer.citations
    assert answer.evidence_categories >= {"encounters", "treasure"}
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/acceptance/test_full_upgrade.py -v`
Expected: FAIL until all migration and RAG stages are connected.

**Step 3: Wire the accepted end-to-end flow**

Keep fixtures synthetic and license-safe. Exercise database/config backup,
legacy index inventory, per-file extraction, complete embedding, hybrid
search, RAG plan, and citation validation. Use the fake Ollama server by
default.

**Step 4: Run acceptance and coverage**

Run:

```powershell
pytest tests/acceptance/test_full_upgrade.py -v
pytest --cov=file_tagger --cov-report=term-missing
```

Expected: end-to-end migration passes and uncovered critical error branches
are either tested or explicitly documented for the opt-in real-model suite.

**Step 5: Commit**

```bash
git add tests/acceptance tests/fixtures
git commit -m "test: cover local upgrade and contextual search"
```

### Task 28: Package the Windows Application and Per-User Migration

**Files:**
- Modify: `pyproject.toml`
- Create: `packaging/file-tagger.spec`
- Create: `.github/workflows/ci.yml`
- Modify: `README.md`
- Test: `tests/packaging/test_installed_app.py`

**Step 1: Write the failing packaging test**

```python
def test_installed_app_uses_per_user_data_directory(subprocess_env) -> None:
    result = run_installed("--print-paths", env=subprocess_env)
    assert result.returncode == 0
    assert "file_tagger.sqlite3" in result.stdout
    assert Path.cwd().resolve().as_posix() not in result.stdout
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/packaging/test_installed_app.py -v`
Expected: FAIL because no packaged app/path command exists.

**Step 3: Implement packaging and CI**

Configure the PyInstaller/PySide6 bundle with `sqlite-vec` and Docling
resources. Add a non-UI diagnostics command. Test Python 3.13/3.14, Ruff, Mypy,
unit tests, and packaged smoke start on GitHub Actions Windows runners.

**Step 4: Run release verification**

Run:

```powershell
uv sync --locked --all-extras
ruff check src tests
mypy src
pytest -q
python -m build
```

Expected: all checks pass, wheel/sdist build, package smoke starts, and no
runtime database/log/config is created in the repository.

**Step 5: Commit**

```bash
git add pyproject.toml packaging .github README.md tests/packaging
git commit -m "build: package local-first File Tagger"
```

### Task 29: Run the Opt-In Real Ollama Quality Suite

**Files:**
- Create: `tests/integration_ollama/conftest.py`
- Create: `tests/integration_ollama/test_model_matrix.py`
- Create: `tests/integration_ollama/test_rpg_retrieval.py`
- Modify: `pyproject.toml`

**Step 1: Write the integration test with an explicit marker**

```python
pytestmark = pytest.mark.ollama


def test_installed_profiles_cover_embed_decide_and_answer(
    real_ollama, integration_profile
) -> None:
    assert real_ollama.embed(integration_profile.embedding, "hello").vector
    assert real_ollama.decide(integration_profile.decision, "hello").answer
    assert real_ollama.chat(integration_profile.generation, "hello").content
```

**Step 2: Run it without the opt-in to verify it is skipped**

Run: `pytest tests/integration_ollama -v`
Expected: SKIP because the local `ollama` marker was not enabled.

**Step 3: Implement the real-model fixture and metrics**

Require loopback Ollama, require all three roles to be installed, and record
model digests, load time, tokens/second, embedding latency, and recall@k from
the synthetic RPG corpus. Never upload results.

**Step 4: Run the opt-in suite**

Run:

```powershell
$env:FILE_TAGGER_RUN_OLLAMA_TESTS=1
pytest tests/integration_ollama -v -m ollama
```

Expected: all installed profiles pass, citations resolve, and measured
latency/memory data is saved locally for the release report.

**Step 5: Commit**

```bash
git add tests/integration_ollama pyproject.toml
git commit -m "test: validate real local Ollama profiles"
```

## Final Verification

Run this sequence from a clean checkout on Windows 10 22H2 and Windows 11:

```powershell
uv sync --locked --all-extras
ruff check src tests
ruff format --check src tests
mypy src
pytest -q
pytest tests/acceptance -v
python -m build
```

Then, on a machine with the approved Ollama profiles installed:

```powershell
$env:FILE_TAGGER_RUN_OLLAMA_TESTS=1
pytest tests/integration_ollama -v -m ollama
```

Release only when:

- No default test, runtime path, or normal model request can reach a non-loopback
  AI endpoint.
- The old SQLite data migrates with verified counts and a retained backup.
- Old Chroma vectors are never reused.
- Large files report complete chunk coverage.
- FTS5 and vector search produce cited, file-local evidence.
- Nimble/Tev1 tagging is typed and model-digest cached.
- Inkling is preferred whenever feasible, otherwise Gemma 4 is selected by the
  documented hardware benchmark.
- Missing Ollama/models block AI features with actionable manual commands.
- Packaged startup uses per-user data paths and passes the migration drill.
