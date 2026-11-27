# Gemini Cloud Provider Design

## Intent

Add a cloud AI option for users whose PCs cannot run the local Ollama models.
The application will remain local-first, with Ollama as the default provider and
Gemini available as an explicit provider switch.

## Goals

- Support Google Gemini Developer API calls for generation, decision, and embedding.
- Use these exact model names:
  - Generation/inference: `gemini-4-hydrogen`
  - Tag decision: `gemini-decision-1`
  - Embedding: `gemini-embedding-2`
- Store the Gemini API key in the existing encrypted per-user configuration.
- Provide provider selection and Gemini key configuration in Settings.
- Route indexing, tag candidate generation, tag decisions, search planning, and
  cited RAG generation through the selected provider.
- Preserve Ollama behavior for users who do not switch providers.
- Reindex documents when the active embedding provider or profile changes so
  incompatible vectors are never mixed.

## Non-goals

- Vertex AI or Google Cloud service-account authentication.
- Automatic provider fallback from Gemini to Ollama.
- Automatic Ollama model download or Gemini model discovery.
- Supporting OpenAI or Anthropic providers in this change.
- Migrating or copying legacy cloud credentials from previous installations.

## Anti-goals

- Storing the Gemini API key in a log, environment file, README example, source
  file, or plain-text model override file.
- Sending data to Gemini while the provider is Ollama.
- Mixing Ollama and Gemini embeddings in one vector index.
- Silently replacing a user-supplied model name with a different model ID.

## Constraints

- The application must continue to run on Windows with Python 3.13.
- The existing Pydantic domain contracts, SQLite/vector index, cache invalidation,
  and Ollama gateway behavior remain the architectural baseline.
- Cloud provider imports are allowed only in the Gemini adapter and provider
  integration files; migration and legacy source audit rules must not be weakened
  to permit secret-bearing or competing provider code in unrelated modules.
- The selected provider and key must be persisted through `Config.save()` in the
  encrypted per-user configuration.
- Gemini API failures must use the existing application error taxonomy where
  possible and must not expose the API key.
- The exact model names in this document are product requirements. Do not substitute
  a generally available Gemini model name without a new approved design.

## Approaches Considered

1. **Replace Ollama with Gemini.**
   This is smaller, but it removes the local/offline workflow and contradicts the
   application's local-first design.
2. **Add Gemini as a provider alongside Ollama (approved).**
   Reuse existing role contracts and provider-neutral application services, add
   one Gemini SDK adapter, persist provider/key settings, and route each role only
   when Gemini is selected. Ollama remains the default.
3. **Build a generalized provider abstraction for several cloud vendors now.**
   This provides future flexibility but exceeds the request and introduces
   provider configuration and test surfaces that have no current requirement.

## Architecture

The application will use a two-provider `ai_provider` setting: `ollama` or
`gemini`. Ollama remains the default for installations without an explicit
Gemini selection.

A Gemini adapter at `src/file_tagger/adapters/gemini/gateway.py` will implement
the existing async gateway-style operations:

- `readiness()` for provider availability
- `model_profile(role, model)` for static provider profiles
- `embed(model, text, *, dimensions, truncate)` for indexing and search embeddings
- `chat(model, *, messages, schema)` for structured generation and planning
- `chat_stream(model, *, messages, schema)` for streaming RAG generation
- `system_one(model, *, state, questions)` for tag decisions
- `close()`

The adapter will use the official `google-genai` Python SDK and a supplied API
key, not an environment-variable side effect hidden in unrelated application
code.

The domain profile for each Gemini role will be provider-specific and stable:

- `gemini-4-hydrogen`: role `generation`, a provider-defined context length, digest
  derived from `gemini + model + profile version`
- `gemini-decision-1`: role `decision`, a provider-defined context length, digest
  derived from `gemini + model + profile version`
- `gemini-embedding-2`: role `embedding`, 768 output dimensions, digest derived
  from `gemini + model + 768`

The exact context lengths and SDK methods are implementation details, but the
profile values must be deterministic and must invalidate all derived model
caches when provider/model/profile data changes.

Tag decisions will be represented through the existing choice/probability-shaped
data contract. The Gemini adapter's `system_one()` implementation will translate
the existing `state` and `questions` payload into a structured Gemini chat
request and return the mapping shape already consumed by `decision_service.py`.

For provider switching, `ModernRuntime` will:

1. Load `ai_provider` and the Gemini key from the encrypted `Config`.
2. Select Ollama or Gemini gateway instances for readiness and role operations.
3. Publish one provider-specific `ReadinessReport`.
4. Rebuild indexing and RAG service instances after provider/model selection
   changes.
5. Queue reindexing for the active Gemini embedding profile, preventing mixed
   embeddings.

The Settings dialog will become provider-aware while remaining one dialog:

- a provider selector (`Ollama`, `Gemini`)
- an Ollama section for the existing readiness/model controls
- a Gemini section for API-key entry, key status, and the three fixed model IDs
- explicit save/refresh behavior
- provider-specific readiness copy and instructions

The API key is not shown after save. The UI may show whether a key is configured
and may require re-entry to replace it. It must not log or expose the key in
plain text.

When Settings saves provider/key changes, it will persist the provider selection
and the supplied key, then run Gemini readiness. It does not require a successful
readiness check before saving the key. If readiness fails, the provider remains
selected and Gemini-specific readiness errors are shown; there is no automatic
fallback to Ollama.

Leaving the masked key field blank preserves the existing stored key. The UI
provides a separate clear-key action for removing it. Replacing the key requires
entering a nonblank value.

## Configuration

`Config` will add helpers:

- `set_ai_provider(provider: str) -> bool`
- `get_ai_provider() -> str`
- `set_gemini_api_key(api_key: str) -> None`
- `get_gemini_api_key() -> str | None`
- `clear_gemini_api_key() -> None`

Defaults:

- `ai_provider = "ollama"`
- `gemini_api_key = null`

Values are written inside `config_data` and therefore remain encrypted through
the existing password vault. Legacy cloud fields are still removed by the
existing migration and are not imported by this feature.

## Provider Readiness

Gemini readiness requires a configured key and successful SDK client construction.
It will return a profile for every requested role and may perform provider-specific
validation. Missing, malformed, or unauthorized API responses will be reported
without recording the key.

Ollama readiness will retain loopback-host validation, version checks, model
installation checks, capability checks, and benchmarks.

## Error Handling

Add provider-neutral or Gemini-specific errors to `domain/errors.py` for:

- missing API key
- Gemini unavailable/timeout
- invalid Gemini API response
- unsupported structured output
- embedding context rejection
- model incompatibility

Existing callers may continue to catch `FileTaggerError`-derived classes.
Transport errors must not include headers or serialized SDK request objects that
could contain credentials.

## Cache and Reindex Behavior

`ModelProfile.digest` is already used to detect model and embedding changes.
Gemini profiles will therefore use stable synthetic digests. Provider switching
will change the digest and queue reindexing. Candidate and decision caches will
also receive provider/model-specific digests for cache invalidation.

## Tests

At minimum:

- Gemini gateway mocked-SDK tests for key handling, embed output shape, chat
  structured output, chat streaming, decisions, close behavior, and transport
  errors.
- Exact model-name tests for all three roles.
- Encrypted configuration tests that save/load/clear the Gemini key and default
  provider.
- `ModernRuntime` tests that choose Gemini and Ollama gateways independently.
- Tagging tests that pass Gemini gateways through candidate and decision calls.
- RAG tests that use Gemini chat and embedding gateways.
- Settings UI tests for provider switching and key status.
- Source/dependency audit tests that allow `google-genai` but continue to reject
  legacy cloud dependencies and provider dispatch outside the adapter boundary.

## Documentation

Update `README.md` with:

- Gemini setup through Settings
- the three exact model names
- API-key encryption and key replacement behavior
- reindexing impact
- Ollama as the unchanged default

