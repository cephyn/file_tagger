# File Tagger

File Tagger is a Windows desktop application for browsing files, managing tags,
searching document content, and asking questions against locally indexed files.
It is local-first:

- Document and vector data are stored in one per-user SQLite database.
- Text search uses SQLite FTS5.
- Semantic search uses `sqlite-vec`.
- Document conversion uses Docling.
- Ollama is the default for chat, decisions, and embeddings.
- Gemini is available as an optional cloud provider selected explicitly in Settings.
- Direct local model loading and other cloud AI providers are not supported.

## Requirements

- Windows 10 22H2 or Windows 11
- Python 3.13 or later
- [Ollama](https://ollama.com/download) installed and running for the default
  local provider
- A Google Gemini API key for the optional Gemini provider
- `uv` for dependency and environment management

Ollama remains the default provider and AI features remain disabled until the
Ollama Readiness dialog confirms that Ollama is reachable, compatible, and the
required local models are installed. The application never downloads or pulls
Ollama models automatically.

## Start from source

From the repository root:

```powershell
uv sync --locked --all-extras
uv run file-tagger
```

You can also start the module directly:

```powershell
uv run python -m file_tagger
```

If PowerShell reports that `uv` is not recognized, use the repository launcher,
which creates or updates `.venv` and bootstraps `uv` when needed:

```powershell
.\scripts\start.ps1
```

For a permanent `uv` command on Windows, install it and reopen PowerShell:

```powershell
py -3.13 -m pip install --user uv
```

Or use the installed UV package through the virtual environment without adding
it to `PATH`:

```powershell
.\.venv\Scripts\uv.exe sync --locked --all-extras
.\.venv\Scripts\uv.exe run file-tagger
```

On first launch, create a password for the encrypted local configuration.
If an older `file_tags.db`/`.chroma` installation is detected, the application
creates a backup and migrates the available file/tag metadata. It inventories
legacy Chroma paths but does not copy old embeddings; those files must be
reindexed by the current Ollama embedding profile.

Switching providers or embedding profiles does not reuse vectors from a
different provider/profile. File Tagger queues the affected files for reindexing
before the new embeddings are used, preventing Ollama and Gemini vectors from
being mixed.

## Password recovery

Use **Forgot password** on the login screen to recover access with the recovery
key shown during first setup or after a key rotation. Recovery keys are shown
only once, are not stored in plaintext, and should be saved outside File
Tagger. A successful recovery resets the login password without re-encrypting
the existing configuration.

Successful login to an older password format migrates the encrypted
configuration to the current format and shows a new recovery key. An already
locked legacy installation cannot be recovered with its old recovery file
because that file was encrypted with the forgotten password.

If the recovery key is unavailable, **Reset configuration** requires typing
`RESET` and a new password. File Tagger quarantines the encrypted configuration
and credential files under `<per-user data>/password-reset-backups/` before
creating a new default encrypted configuration. SQLite tagging/search data,
index data, cache data, and legacy `file_tags.db`/`.chroma` data are not removed.
Treat the backup as recovery material: a manual restoration can be attempted
only after understanding the original password and credential format.

## Configure Ollama

1. Start Ollama locally.
2. Open **Settings > Ollama Readiness**.
3. Review the required embedding, decision, and generation model tags.
4. Pull only the tags suggested by the dialog:

```powershell
ollama pull <model-tag>
```

For example, a typical recommended profile may include:

```text
embeddinggemma-2:270m
nimble:9b
gemma4:e4b
```

The dialog will show the exact compatible tags for the detected hardware and
installed Ollama catalog. Model names ending in `:cloud` or remote Ollama hosts
are rejected.

## Configure Gemini

1. Open **Settings > AI Provider**.
2. Select **Gemini**.
3. Enter a Gemini API key. The key is not shown after it is saved.
4. Save and refresh Gemini readiness.

The Gemini API key is stored inside the existing encrypted per-user
configuration, never in a log or environment file by this application. Leave
the masked field blank to preserve a configured key, enter a new nonblank value
to replace it, or use **Clear Gemini Key** to remove it.

Gemini uses these fixed model IDs:

```text
gemini-4-hydrogen
gemini-decision-1
gemini-embedding-2
```

Gemini is not contacted when the Ollama provider is active. Selecting Gemini,
changing the Gemini embedding profile, or switching providers queues a reindex
of the index for that provider/profile before normal semantic search resumes.
Gemini readiness failure does not silently fall back to Ollama.

## Per-user data

Normal runtime data is stored outside the repository:

- SQLite database: `<per-user data>/file_tagger.sqlite3`
- Configuration: `<per-user config>/config.encrypted`
- Password/recovery metadata: `<per-user config>/`
- Logs: `<per-user data>/logs`
- Cache and extraction files: `<per-user cache>/`

Set `FILE_TAGGER_DATA_DIR` only for portable or test deployments:

```powershell
$env:FILE_TAGGER_DATA_DIR = "C:\portable-file-tagger"
uv run file-tagger
```

Print the resolved modern and legacy paths without starting the Qt UI:

```powershell
uv run python -m file_tagger --print-paths
uv run file-tagger --print-paths
```

## Build and package

Build the wheel and source distribution:

```powershell
uv sync --locked --all-extras
python -m build
```

Build the Windows PyInstaller bundle:

```powershell
uv run pyinstaller --clean --noconfirm packaging/file-tagger.spec
```

The packaged diagnostic executable is also available:

```powershell
.\dist\file-tagger\file-tagger.exe --print-paths
```

The package does not include Ollama or downloaded models. Package and model
installation remain separate local operations.

## Development checks

Run the default test suite:

```powershell
uv run pytest -q
```

Run lint and type checks:

```powershell
uv run ruff check src tests
uv run mypy src
```

Run a migration lifecycle smoke check:

```powershell
uv run alembic upgrade head
uv run alembic downgrade base
uv run alembic upgrade head
```

Real-model Ollama tests are opt-in and skipped by default:

```powershell
$env:FILE_TAGGER_RUN_OLLAMA_TESTS = "1"
uv run pytest tests/integration_ollama -v -m ollama
Remove-Item Env:FILE_TAGGER_RUN_OLLAMA_TESTS
```

## Main features

- File browsing, sorting, tagging, and tag search
- Typed Ollama/Nimble/Tev1 tag decisions
- Cached tag recommendations with model-digest invalidation
- Docling PDF, Office, Markdown, image, and text extraction
- OCR fallback for low-text or scanned pages
- Provenance-aware chunks and full-coverage embeddings
- SQLite FTS5 and hybrid `sqlite-vec` retrieval
- Reciprocal-rank fusion and cited search results
- Markdown-context planning and cited local RAG answers
- Resumable, transactionally swapped indexing jobs
- Ollama readiness, hardware, disk, and benchmark diagnostics
- Optional Gemini readiness and provider-specific settings
- Explicit migration backup and legacy Chroma inventory

## Project layout

```text
src/file_tagger/
  adapters/       Ollama and Gemini external adapters
  domain/         Pydantic contracts and application errors
  extraction/     Docling extraction and page/vision quality
  indexing/       Chunking, embedding coverage, and resumable jobs
  infrastructure/ SQLite, app paths, and extension setup
  migration/      Explicit legacy SQLite/config/Chroma migration
  ollama/         Hardware, recommendations, readiness, benchmarks
  rag/            Search planning, evidence, reranking, and cited answers
  search/         FTS5, sqlite-vec, RRF, and result hydration
  tagging/        Candidate generation and typed decisions
  ui/             Ollama settings, indexing status, contextual search
  file_tag_manager.py
  main.py
  login.py
```

## Security

- Ollama endpoints must resolve to loopback hosts.
- Ollama cloud model tags and remote Ollama hosts are rejected.
- Gemini is optional; its API key is stored in the encrypted per-user
  configuration and is never written to logs or status text.
- Legacy cloud credentials and provider fields are not migrated.
- Configuration is password-protected and encrypted locally.
- Recovery keys are displayed once and never written to logs or status text.
- Reset quarantines encrypted settings while preserving SQLite and index data.
- Legacy cleanup requires explicit confirmation.
- RAG answers require resolved chunk citations and do not use uncited model
  knowledge as evidence.

## License

MIT
