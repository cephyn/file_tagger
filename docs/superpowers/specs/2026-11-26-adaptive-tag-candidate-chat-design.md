# Adaptive Tag-Candidate Chat Context Design

**Status:** Approved

## Intent

Fix the "Local Ollama chat input exceeded the model context" failure shown when
requesting AI tag suggestions for a file.

## Goals

- Preserve full document-state coverage for candidate generation.
- On generation chat context rejection, recursively split the document state at
  a safe text boundary and retry the candidate chat request for each fragment.
- Merge and de-duplicate generated candidates while preserving the requested
  candidate limit.
- Keep provider-side truncation disabled and surface unsplittable text as an
  explicit typed context error.

## Non-goals

- Change the tag decision model or the typed System One decision contract.
- Fall back to cloud or non-Ollama providers.
- Silently discard document text before a generation retry.
- Change embedding behavior already covered by
  `docs/superpowers/plans/2026-11-26-adaptive-tag-query-embedding.md`.

## Constraints

- Candidate generation remains local and structured.
- Existing chat context rejection is represented by
  `GenerationContextRejectedError`.
- Splits must preserve UTF-8 text and avoid breaking Unicode grapheme
  sequences.
- An unsplittable fragment or exhausted retry budget must raise
  `GenerationContextRejectedError`; it must not return a success-shaped
  partial result.

## Approaches Considered

1. **Adaptive split-and-merge chat retries (selected):** retry rejected
   candidate-generation prompts with smaller document-state fragments, then
   merge unique candidates. This preserves document coverage and reuses existing
   structured output.
2. **Lower the document-state cap:** avoids overflow more often but can silently
   omit source text and reduce tag quality.
3. **Use indexed tags only:** avoids chat entirely, but cannot generate useful
   novel tags for documents outside the existing catalog.

## How It Works

`_generate_candidates(...)` will try the full document state once. When the
gateway raises `GenerationContextRejectedError`, it will split the document state
with the existing Unicode-safe boundary rules and recursively retry each
fragment. Each successful fragment contributes candidates; `_merge_candidates`
will de-duplicate the results and enforce the caller's candidate quota.

The existing `_split_document_state_text(...)` helper is appropriate for text
boundary selection but is not used for this chat path today. The implementation
must ensure that chat retries also avoid splitting a Unicode grapheme or
combining form.

If no interior boundary exists, or a one-character fragment is still rejected,
the implementation raises a new explicit generation-context message identifying
the document-state fragment as unsplittable.
