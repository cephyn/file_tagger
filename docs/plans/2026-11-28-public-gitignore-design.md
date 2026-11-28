# Public Repository Gitignore Policy

## Approved Design

The public repository will ignore generated, local, and secret-bearing
artifacts while tracking source, tests, package metadata, migration files,
and intentional test fixtures.

## Ignore Groups

- Python bytecode and cache directories.
- Virtual environments and local tool caches.
- Distribution and build output.
- Test and coverage output.
- Logs and OS/editor metadata.
- Runtime SQLite databases and SQLite sidecar files, including `*.sqlite-journal`,
  `*.sqlite-shm`, and `*.sqlite-wal`.
- Legacy and encrypted runtime configuration.
- Local vector stores and downloaded model artifacts.

Patterns for runtime secrets and databases are scoped to the repository root
where practical. Intentional files under `tests/fixtures/legacy/` remain
trackable so migration and security tests continue to work.

## Already-Tracked Artifacts

Remove these from the index without deleting the local working files:

- `/file_tags.db`
- `/__pycache__/models.cpython-313.pyc`

Do not remove the legacy files under `tests/fixtures/legacy/`; they are test
inputs and may contain placeholder credentials or configuration.

## Verification

- Representative generated and secret-like paths must be reported by
  `git check-ignore`.
- The tracked legacy fixture database and secret fixtures must not be ignored.
- `git diff --check` must pass.
- The index must no longer contain `/file_tags.db` or the root models bytecode
  file.
