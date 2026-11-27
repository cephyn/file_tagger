from collections.abc import AsyncIterator, Mapping, Sequence
import asyncio
import hashlib
import inspect
import json
from typing import Any, Literal

import httpx
from google import genai
from google.genai import types
from ...domain.contracts import ModelProfile, OllamaReadiness
from ...domain.errors import (
    EmbeddingContextRejectedError,
    GeminiAuthenticationError,
    GeminiStructuredOutputError,
    GeminiTimeoutError,
    GeminiUnavailableError,
    ModelIncompatibleError,
)


ModelRole = Literal["embedding", "decision", "generation"]

GEMINI_GENERATION_MODEL = "gemini-4-neon"
GEMINI_DECISION_MODEL = "gemini-decision-1"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"

GEMINI_PROFILE_VERSION = "v1"
GEMINI_EMBEDDING_DIMENSIONS = 768
GEMINI_GENERATION_CONTEXT_LENGTH = 1_048_576
GEMINI_DECISION_CONTEXT_LENGTH = 128_000
GEMINI_EMBEDDING_CONTEXT_LENGTH = 2_048
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com"


def _profile_digest(model: str, canonical_profile: str) -> str:
    payload = f"gemini|{model}|{canonical_profile}"
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


class GeminiGateway:
    def __init__(
        self,
        api_key: str,
        *,
        client: Any | None = None,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise GeminiAuthenticationError("A Gemini API key is required.")
        if timeout_seconds <= 0:
            raise GeminiTimeoutError("Gemini request timeout must be positive.")

        self._api_key = api_key.strip()
        self.timeout_seconds = float(timeout_seconds)
        try:
            self._client = (
                client
                if client is not None
                else genai.Client(api_key=self._api_key)
            )
        except Exception as exc:
            raise self._translate_transport_error(exc) from exc

    def _models(self) -> Any:
        aio = getattr(self._client, "aio", None)
        aio_models = getattr(aio, "models", None)
        if aio_models is not None:
            return aio_models
        return getattr(self._client, "models")

    @staticmethod
    def _status_code(exc: BaseException) -> int | None:
        for attr in ("status_code", "code"):
            value = getattr(exc, attr, None)
            if isinstance(value, int):
                return value
        return None

    @classmethod
    def _translate_transport_error(cls, exc: BaseException) -> GeminiUnavailableError:
        if isinstance(exc, GeminiUnavailableError):
            return exc
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
            return GeminiTimeoutError("Gemini request timed out.")

        status_code = cls._status_code(exc)
        if isinstance(exc, PermissionError) or status_code in (401, 403):
            return GeminiAuthenticationError("Gemini authentication failed.")

        if status_code in (408, 504):
            return GeminiTimeoutError("Gemini request timed out.")

        message = str(exc).casefold()
        if any(marker in message for marker in ("unauthenticated", "api key", "permission denied")):
            return GeminiAuthenticationError("Gemini authentication failed.")
        if any(marker in message for marker in ("timed out", "timeout", "deadline")):
            return GeminiTimeoutError("Gemini request timed out.")

        return GeminiUnavailableError("Gemini is unavailable.")

    async def close(self) -> None:
        aio = getattr(self._client, "aio", None)
        for owner in (aio, self._client):
            if owner is None:
                continue
            close = getattr(owner, "aclose", None) or getattr(owner, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
                elif inspect.iscoroutine(result):
                    await result
                return

    async def model_profile(self, role: ModelRole, model: str) -> ModelProfile:
        if role not in ("embedding", "decision", "generation"):
            raise ModelIncompatibleError("Unknown Gemini model role.")

        expected = {
            "embedding": GEMINI_EMBEDDING_MODEL,
            "decision": GEMINI_DECISION_MODEL,
            "generation": GEMINI_GENERATION_MODEL,
        }[role]
        if model != expected:
            raise ModelIncompatibleError(
                f"The Gemini {role} role does not support model {model!r}."
            )

        if role == "embedding":
            return ModelProfile(
                role="embedding",
                model=GEMINI_EMBEDDING_MODEL,
                digest=_profile_digest(
                    GEMINI_EMBEDDING_MODEL,
                    f"{GEMINI_PROFILE_VERSION}|{GEMINI_EMBEDDING_DIMENSIONS}",
                ),
                context_length=GEMINI_EMBEDDING_CONTEXT_LENGTH,
                dimensions=GEMINI_EMBEDDING_DIMENSIONS,
                capabilities=("embedding",),
            )
        if role == "decision":
            return ModelProfile(
                role="decision",
                model=GEMINI_DECISION_MODEL,
                digest=_profile_digest(
                    GEMINI_DECISION_MODEL,
                    GEMINI_PROFILE_VERSION,
                ),
                context_length=GEMINI_DECISION_CONTEXT_LENGTH,
                capabilities=("decision", "choice", "structured-output"),
            )
        return ModelProfile(
            role="generation",
            model=GEMINI_GENERATION_MODEL,
            digest=_profile_digest(
                GEMINI_GENERATION_MODEL,
                GEMINI_PROFILE_VERSION,
            ),
            context_length=GEMINI_GENERATION_CONTEXT_LENGTH,
            capabilities=("completion", "streaming", "structured-output"),
        )

    async def readiness(
        self,
        required_models: Mapping[ModelRole, str],
    ) -> OllamaReadiness:
        incompatible: list[str] = []
        for role, model in required_models.items():
            try:
                await self.model_profile(role, model)
            except ModelIncompatibleError:
                incompatible.append(model)

        return OllamaReadiness(
            available=True,
            version=None,
            base_url=GEMINI_ENDPOINT,
            missing_models=(),
            incompatible_models=tuple(sorted(set(incompatible))),
            errors=(),
        )

    async def _invoke(self, method_name: str, **kwargs: Any) -> Any:
        method = getattr(self._models(), method_name)
        try:
            if inspect.iscoroutinefunction(method):
                result = method(**kwargs)
                return await asyncio.wait_for(
                    result,
                    timeout=self.timeout_seconds,
                )
            result = await asyncio.wait_for(
                asyncio.to_thread(method, **kwargs),
                timeout=self.timeout_seconds,
            )
            return result
        except GeminiUnavailableError:
            raise
        except BaseException as exc:
            raise self._translate_transport_error(exc) from exc

    async def embed(
        self,
        model: str,
        text: str | Sequence[str],
        *,
        dimensions: int | None = None,
        truncate: Literal[False] = False,
    ) -> tuple[tuple[float, ...], ...]:
        if truncate is not False:
            raise EmbeddingContextRejectedError(
                "Gemini embeddings must not request input truncation."
            )
        requested_dimensions = (
            GEMINI_EMBEDDING_DIMENSIONS
            if dimensions is None
            else dimensions
        )
        if requested_dimensions != GEMINI_EMBEDDING_DIMENSIONS:
            raise EmbeddingContextRejectedError(
                "Gemini embedding profile requires 768 dimensions."
            )
        if model != GEMINI_EMBEDDING_MODEL:
            raise ModelIncompatibleError(
                "Gemini embedding operations require gemini-embedding-2."
            )

        config = types.EmbedContentConfig(
            output_dimensionality=requested_dimensions,
            auto_truncate=False,
        )
        try:
            response = await self._invoke(
                "embed_content",
                model=model,
                contents=text,
                config=config,
            )
        except (GeminiUnavailableError, GeminiTimeoutError, GeminiAuthenticationError):
            raise

        raw_embeddings = getattr(response, "embeddings", None)
        if not isinstance(raw_embeddings, (list, tuple)):
            raise EmbeddingContextRejectedError(
                "Gemini returned no embedding vectors."
            )

        vectors: list[tuple[float, ...]] = []
        for raw in raw_embeddings:
            values = (
                raw.get("values")
                if isinstance(raw, Mapping)
                else getattr(raw, "values", None)
            )
            if not isinstance(values, (list, tuple)):
                raise EmbeddingContextRejectedError(
                    "Gemini returned an invalid embedding vector."
                )
            if len(values) != requested_dimensions:
                raise EmbeddingContextRejectedError(
                    "Gemini embedding vector dimensions do not match the profile."
                )
            try:
                vectors.append(tuple(float(value) for value in values))
            except (TypeError, ValueError) as exc:
                raise EmbeddingContextRejectedError(
                    "Gemini returned a non-numeric embedding vector."
                ) from exc
        if not vectors:
            raise EmbeddingContextRejectedError(
                "Gemini returned no embedding vectors."
            )
        return tuple(vectors)

    @staticmethod
    def _convert_messages(
        messages: Sequence[Mapping[str, Any]],
    ) -> tuple[list[Any], str | None]:
        contents: list[Any] = []
        system_parts: list[str] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if role == "system":
                system_parts.append(content)
                continue
            sdk_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=sdk_role,
                    parts=[types.Part(text=content)],
                )
            )
        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return contents, system_instruction

    @staticmethod
    def _chat_config(
        schema: Mapping[str, Any] | None,
        system_instruction: str | None,
    ) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {}
        if system_instruction is not None:
            kwargs["system_instruction"] = system_instruction
        if schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = dict(schema)
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise GeminiStructuredOutputError(
                "Gemini returned an empty chat response."
            )
        return text

    @staticmethod
    def _validate_structured_text(text: str, schema: Mapping[str, Any] | None) -> str:
        if schema is None:
            return text
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiStructuredOutputError(
                "Gemini returned malformed structured JSON."
            ) from exc
        expected_type = schema.get("type")
        if expected_type == "object" and not isinstance(parsed, dict):
            raise GeminiStructuredOutputError(
                "Gemini returned JSON that does not match the requested object schema."
            )
        if expected_type == "array" and not isinstance(parsed, list):
            raise GeminiStructuredOutputError(
                "Gemini returned JSON that does not match the requested array schema."
            )
        if not isinstance(parsed, (dict, list)):
            raise GeminiStructuredOutputError(
                "Gemini returned invalid structured JSON."
            )
        return text

    async def chat(
        self,
        model: str,
        *,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any] | None = None,
    ) -> str:
        if model not in (GEMINI_GENERATION_MODEL, GEMINI_DECISION_MODEL):
            raise ModelIncompatibleError(
                "Gemini chat operations require a generation or decision model."
            )
        contents, system_instruction = self._convert_messages(messages)
        config = self._chat_config(schema, system_instruction)
        response = await self._invoke(
            "generate_content",
            model=model,
            contents=contents,
            config=config,
        )
        text = self._extract_text(response)
        return self._validate_structured_text(text, schema)

    async def chat_stream(
        self,
        model: str,
        *,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        if model != GEMINI_GENERATION_MODEL:
            raise ModelIncompatibleError(
                "Gemini streaming chat operations require gemini-4-neon."
            )
        contents, system_instruction = self._convert_messages(messages)
        config = self._chat_config(schema, system_instruction)
        method = getattr(self._models(), "generate_content_stream")
        try:
            if inspect.isasyncgenfunction(method):
                stream = method(
                    model=model,
                    contents=contents,
                    config=config,
                )
            elif inspect.iscoroutinefunction(method):
                stream = await asyncio.wait_for(
                    method(
                        model=model,
                        contents=contents,
                        config=config,
                    ),
                    timeout=self.timeout_seconds,
                )
            else:
                stream = await asyncio.wait_for(
                    asyncio.to_thread(
                        method,
                        model=model,
                        contents=contents,
                        config=config,
                    ),
                    timeout=self.timeout_seconds,
                )
        except BaseException as exc:
            raise self._translate_transport_error(exc) from exc

        try:
            if hasattr(stream, "__aiter__"):
                async for chunk in stream:
                    text = getattr(chunk, "text", None)
                    if isinstance(text, str) and text:
                        yield text
            else:
                for chunk in stream:
                    text = getattr(chunk, "text", None)
                    if isinstance(text, str) and text:
                        yield text
        except BaseException as exc:
            raise self._translate_transport_error(exc) from exc

    async def system_one(
        self,
        model: str,
        *,
        state: Any,
        questions: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if model != GEMINI_DECISION_MODEL:
            raise ModelIncompatibleError(
                "Gemini decision operations require gemini-decision-1."
            )

        prompt_payload = {
            "state": state,
            "questions": dict(questions),
            "output_contract": {
                "answers": {
                    question_name: {
                        "type": "choice",
                        "choice": "accept",
                        "probabilities": {
                            "accept": 0.0,
                            "reject": 0.0,
                            "review": 0.0,
                        },
                        "confidence": 0.0,
                    }
                    for question_name in questions
                }
            },
        }
        prompt = (
            "Return only a JSON object for the following tag decisions. "
            "Every answer must use one of accept, reject, or review, and "
            "probabilities must sum to 1.0.\n"
            + json.dumps(prompt_payload, ensure_ascii=False)
        )
        schema = {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "type": {"const": "choice"},
                            "choice": {
                                "enum": ["accept", "reject", "review"],
                            },
                            "probabilities": {
                                "type": "object",
                                "properties": {
                                    "accept": {"type": "number"},
                                    "reject": {"type": "number"},
                                    "review": {"type": "number"},
                                },
                                "required": ["accept", "reject", "review"],
                            },
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "type",
                            "choice",
                            "probabilities",
                            "confidence",
                        ],
                    },
                }
            },
            "required": ["answers"],
        }
        text = await self.chat(
            model,
            messages=[{"role": "user", "content": prompt}],
            schema=schema,
        )
        parsed = json.loads(text)
        if not isinstance(parsed, Mapping):
            raise GeminiStructuredOutputError(
                "Gemini decision response must be a JSON object."
            )
        answers = parsed.get("answers")
        if not isinstance(answers, Mapping):
            raise GeminiStructuredOutputError(
                "Gemini decision response is missing the answers mapping."
            )
        return {"answers": dict(answers)}
