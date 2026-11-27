# Task 2 Report: Gemini Gateway and Exact Model Profiles

Commit: `75b6d720abbd3ee24ec23dae847f634088988083`

## Implementation

- Added the `GeminiGateway` adapter using the pinned `google-genai==3.6.1` SDK dependency.
- Added exact model constants:
  - `gemini-4-neon`
  - `gemini-decision-1`
  - `gemini-embedding-2`
- Added deterministic synthetic SHA-256 model-profile digests and static role context/capability profiles.
- Implemented 768-dimension embedding calls with `auto_truncate=False`, single/batch vector conversion,
  wrong-dimension rejection, and `EmbeddingContextRejectedError` handling.
- Implemented Gemini SDK content conversion, structured chat JSON validation, streaming chat, and
  JSON-only decision responses compatible with `decision_service._parse_decisions()`.
- Mapped timeouts, authentication, unavailable transport failures, structured-output failures, and
  embedding context failures to existing domain errors.
- Added provider readiness validation for exact role/model mappings without API-key logging or exposure.
- Ollama behavior was not modified.

## Tests

- Added `tests/adapters/gemini/test_gateway.py` with mocked-SDK coverage for:
  - exact model names and stable profiles
  - missing API key
  - single/batch embeddings and 768 dimensions
  - truncation and dimension rejection
  - structured chat and streaming
  - decision-system-one response shape
  - malformed/empty structured output
  - timeout/authentication/unavailable transport mapping
  - close behavior
- Added adapter test-package markers to keep two `test_gateway.py` modules uniquely importable during
  the full suite.

## TDD RED/GREEN Evidence

RED:

```text
uv run pytest tests/adapters/gemini/test_gateway.py -q
ERROR: ModuleNotFoundError: No module named 'file_tagger.adapters.gemini'
```

The verified RED run stopped during collection with the missing Gemini gateway module.

GREEN:

```text
uv run pytest tests/adapters/gemini/test_gateway.py -q
14 passed
```

Full suite:

```text
uv run pytest -q
517 passed, 62 skipped, 2 warnings
```

Lint:

```text
uv run ruff check src/file_tagger/adapters/gemini tests/adapters/gemini/test_gateway.py
All checks passed!
```

## Files Changed

- `pyproject.toml`
- `uv.lock`
- `src/file_tagger/adapters/gemini/__init__.py`
- `src/file_tagger/adapters/gemini/gateway.py`
- `tests/adapters/__init__.py`
- `tests/adapters/gemini/__init__.py`
- `tests/adapters/gemini/test_gateway.py`
- `tests/adapters/ollama/__init__.py`

## Concerns

- The design leaves Gemini context lengths as provider-defined implementation details; deterministic
  static values are used for the generation, decision, and embedding profiles.
- No live Gemini API credentials or network calls were available or used in this task.

## Generation-Model Correction

Corrected the generation model to `gemini-4-hydrogen`.

Command:

```text
uv run pytest tests/adapters/gemini/test_gateway.py -q
```

Result:

```text
14 passed in 0.88s
```

## Follow-up Findings Fix

Commit: `e521088f703be9fde87717c8b3fa9718a03195bd`

### Fixes

- SDK embedding failures now map HTTP 413 and context-window/text markers to
  `EmbeddingContextRejectedError`, while timeout, authentication, and unavailable mappings remain intact.
- `GeminiGateway.system_one()` now validates exact requested answer IDs, choice answer type, accept/reject/review
  choices, exact accept/reject/review probability keys, a 1.0 probability sum within `1e-6`, and confidence
  equal to the selected probability within `1e-6`. Unknown IDs, missing IDs, duplicate JSON fields, unknown
  choices, and malformed distributions raise `GeminiStructuredOutputError`; the returned `{"answers": ...}`
  shape remains compatible with `decision_service._parse_decisions()`.
- Transport and structured-output exceptions are sanitized and raised with chaining suppressed. Public exceptions
  are generic, and SDK/HTTP exceptions containing API-key, authorization, header, or request details are not
  retained as causes or visible traceback chains.
- Syntactically malformed structured JSON is now covered and rejected, authentication by SDK status code is
  covered, returned 768-dimension vector mismatches are covered, and choice answer edge cases are covered.

### RED/GREEN Evidence

RED focused gateway run before implementation:

```text
9 failed, 16 passed
```

The failures reproduced SDK 413/context mapping, status-code authentication chaining, unknown-choice and
probability/confidence/missing/duplicate answer validation.

GREEN focused gateway run after implementation:

```text
27 passed in 1.10s
```

Relevant full suite:

```text
530 passed, 62 skipped, 2 warnings
```

Focused lint:

```text
All checks passed!
```

### Files Changed

- `src/file_tagger/adapters/gemini/gateway.py`
- `tests/adapters/gemini/test_gateway.py`
- `.superpowers/sdd/2026-11-26-gemini-cloud-provider/task-2-report.md`

### Concerns

- No live Gemini API credentials or network calls were used.
- Ollama opt-in tests remained skipped because `FILE_TAGGER_RUN_OLLAMA_TESTS=1` was not set.
