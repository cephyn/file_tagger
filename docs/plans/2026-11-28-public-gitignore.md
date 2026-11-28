# Public Repository Gitignore Implementation Plan

> **For Copilot:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the repository ignore appropriate generated, local, and secret-bearing artifacts without removing migration test fixtures.

**Architecture:** Consolidate `.gitignore` by concern, use root-anchored patterns for repository-root runtime files, preserve fixture exceptions, and remove only two accidentally tracked runtime artifacts from the index.

**Tech Stack:** Git, Python, PowerShell

---

### Task 1: Replace `.gitignore`

**Files:**
- Modify: `.gitignore`

**Step 1: Replace the file with grouped public-project rules**

Use the following exact content:

```gitignore
# Python bytecode and caches
__pycache__/
*.py[cod]
*$py.class
.mypy_cache/
.pytest_cache/
.ruff_cache/
.cache/

# Virtual environments
.venv/
venv/
env/
ENV/

# Distribution and build output
build/
dist/
output/
*.egg-info/
.eggs/
wheels/
sdist/

# Test and coverage output
.coverage
.coverage.*
htmlcov/
coverage.xml
.tox/
.nox/

# Logs
*.log
logs/
src/file_tagger/logs/

# Runtime databases
*.db
*.db-journal
*.db-shm
*.db-wal
*.sqlite
*.sqlite-journal
*.sqlite-shm
*.sqlite-wal
*.sqlite3
*.sqlite3-journal
*.sqlite3-shm
*.sqlite3-wal

# Runtime and secret configuration
.env
.env.*
!.env.example
/config.encrypted
/.passhash
/.salt
/.recovery

# Keep migration/security fixtures under test source trackable
!/tests/fixtures/
!/tests/fixtures/legacy/
!/tests/fixtures/legacy/.passhash
!/tests/fixtures/legacy/.recovery
!/tests/fixtures/legacy/.salt
!/tests/fixtures/legacy/config.encrypted
!/tests/fixtures/legacy/file_tags.sqlite3

# Local vector stores
.chroma/
.vectordb/
chroma_db/

# Local model artifacts
models/
*.gguf
*.bin
*.pt
*.onnx

# IDE and editor metadata
.idea/
.vscode/
.history/
.settings/
.project
.pydevproject
*.swp
*.swo

# Operating system metadata
.DS_Store
Thumbs.db
desktop.ini

```

**Step 2: Verify the diff**

Run:

```powershell
git --no-pager diff -- .gitignore
git --no-pager diff --check
```

Expected: grouped `.gitignore` changes and no whitespace errors.

### Task 2: Remove accidentally tracked runtime artifacts

**Files:**
- Modify: Git index for `file_tags.db` and `__pycache__/models.cpython-313.pyc`
- Preserve local working copies

**Step 1: Remove only the two approved paths from the index**

Run:

```powershell
git rm --cached -- file_tags.db __pycache__/models.cpython-313.pyc
```

Expected: both paths appear as deletions in `git status` while remaining on disk.

### Task 3: Verify ignore behavior

**Step 1: Verify representative runtime paths are ignored**

Run:

```powershell
git check-ignore -v --no-index __pycache__/example.cpython-313.pyc .venv/Scripts/python.exe file_tags.db file_tags.db-journal config.encrypted .passhash logs/app.log build/file-tagger.exe models/model.gguf
```

Expected: each path is matched by a rule in `.gitignore`.

**Step 2: Verify intentional fixtures remain visible**

Run:

```powershell
$paths = @(
  'tests/fixtures/legacy/.passhash',
  'tests/fixtures/legacy/.recovery',
  'tests/fixtures/legacy/.salt',
  'tests/fixtures/legacy/config.encrypted',
  'tests/fixtures/legacy/file_tags.sqlite3'
)
$failed = $false
foreach ($path in $paths) {
  git check-ignore --no-index -q -- $path
  if ($LASTEXITCODE -eq 0) { Write-Error "$path is unexpectedly ignored"; $failed = $true }
}
if ($failed) { exit 1 }
git --no-pager ls-files -- $paths
```

Expected: all five fixtures are listed and none is ignored.

**Step 3: Verify the index and diff**

Run:

```powershell
git --no-pager diff --check
git --no-pager diff --stat
git --no-pager status --short
git --no-pager ls-files -- file_tags.db __pycache__/models.cpython-313.pyc
```

Expected: no whitespace errors, no runtime artifacts remain in `git ls-files`, and `.gitignore` plus the two index removals are the only relevant changes.
