# Task 4 Report

## Status

Implemented and verified.

## Changes

- Routed `TagSuggester` modern-runtime candidate, decision, and embedding work through runtime provider gateways while preserving explicit injected gateways.
- Added separate candidate embedding-gateway support and kept direct Ollama defaults unchanged.
- Routed Gemini RAG planning, query embeddings, and answer generation through Gemini runtime gateways with exact model names.
- Prevented Gemini runtime paths from silently constructing Ollama gateway fallbacks.
- Added focused provider-routing tests for candidate, decision, planner, RAG, runtime precedence, and exact Gemini model calls.

## Verification

- RED: focused suite initially produced 3 provider-routing failures.
- GREEN: focused suite passed: 179 tests.
- Full suite passed: 546 passed, 62 skipped, 2 warnings.

## Commit

- Committed with the implementation as `feat: route tagging and RAG through Gemini`.

## Concerns

- None for the implementation. The 62 skips are opt-in Ollama integration/hardware/model checks; the 2 warnings are existing Docling deprecation warnings.

## Revision Fix Evidence

- Provider-neutral exact-Gemini routing now raises `GeminiUnavailableError` before constructing or using an Ollama gateway:
  - candidate model `gemini-4-hydrogen`
  - decision model `gemini-decision-1`
  - planner/generation model `gemini-4-hydrogen`
  - embedding model `gemini-embedding-2`
- `TagSuggester` now accepts an optional `embedding_gateway` with precedence:
  explicit embedding override > runtime embedding gateway > legacy Ollama gateway fallback.
- RED: the focused routing suite failed 5 tests before the revision:
  exact Gemini candidate, exact Gemini decision, exact Gemini planner, exact Gemini generation, and runtime candidate-vs-runtime-embedding precedence.
- GREEN: focused suite passed `186` tests.
- Full suite passed `553` tests, with `62` skipped and `2` Docling deprecation warnings.

## Re-review Fix Evidence

- `RagService` now rejects an exact `gemini-embedding-2` profile whenever no explicit
  embedding gateway is supplied, even when `gateway` is a valid Gemini generation
  gateway. It also rejects an explicitly supplied Ollama embedding gateway.
- RED: focused constructor regression test:
  `1 failed, 1 passed, 88 deselected`; the generation-gateway reuse case did not raise
  `GeminiUnavailableError`.
- GREEN: focused constructor pair passed `2 passed, 88 deselected`.
- Focused `tests/rag/test_rag_service.py` passed `90` tests.
- Full suite passed `555 passed, 62 skipped, 2 warnings` in `75.47s`.
- Added coverage for both required boundaries:
  exact Gemini generation gateway plus exact Gemini embedding profile with
  `embedding_gateway=None` raises `GeminiUnavailableError`; a distinct explicit
  embedding gateway is accepted and preserved.
