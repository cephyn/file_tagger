from collections.abc import AsyncIterator, Mapping, Sequence
import ipaddress
import re
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from ollama import AsyncClient, ResponseError

from ...domain.contracts import ModelProfile, OllamaReadiness
from ...domain.errors import (
    EmbeddingContextRejectedError,
    GenerationContextRejectedError,
    ModelIncompatibleError,
    ModelMissingError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    RemoteOllamaHostError,
    StructuredOutputError,
)


ModelRole = Literal["embedding", "decision", "generation"]
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
MINIMUM_OLLAMA_VERSION = (0, 35, 0)
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
CLOUD_MARKERS = (":cloud", "-cloud", "cloud:", ".cloud")
_VERSION_PATTERN = re.compile(
    r"v?(\d+)\.(\d+)\.(\d+)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?"
)


def parse_ollama_version(version: str | None) -> tuple[int, int, int] | None:
    if version is None:
        return None
    match = _VERSION_PATTERN.fullmatch(version)
    if match is None:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
    )


def validate_ollama_host(host: str) -> str:
    try:
        parsed = urlsplit(host)
        port = parsed.port
    except ValueError as exc:
        raise RemoteOllamaHostError(
            "Ollama host must be a local HTTP origin with a valid port."
        ) from exc
    if parsed.scheme != "http" or not parsed.hostname:
        raise RemoteOllamaHostError(
            "Ollama host must be an absolute local HTTP origin."
        )
    if parsed.username or parsed.password:
        raise RemoteOllamaHostError("Ollama host credentials are not supported.")
    if parsed.path or parsed.query or parsed.fragment:
        raise RemoteOllamaHostError(
            "Ollama host must not include a path, query, or fragment."
        )
    if port is None or not 1 <= port <= 65535:
        raise RemoteOllamaHostError(
            "Ollama host must include a numeric port from 1 to 65535."
        )

    hostname = parsed.hostname.lower()
    if hostname != "localhost":
        try:
            if not ipaddress.ip_address(hostname).is_loopback:
                raise RemoteOllamaHostError(
                    "Remote Ollama hosts are not supported."
                )
        except ValueError as exc:
            raise RemoteOllamaHostError(
                "Remote Ollama hosts are not supported."
            ) from exc
    return host


class OllamaGateway:
    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        *,
        client: Any | None = None,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.host = validate_ollama_host(host)
        self._client: Any = (
            client
            if client is not None
            else AsyncClient(
                host=self.host,
                timeout=timeout_seconds,
                follow_redirects=False,
            )
        )

    @staticmethod
    def _validate_host(host: str) -> str:
        return validate_ollama_host(host)

    @staticmethod
    def _validate_model(model: str) -> None:
        normalized = model.strip().lower()
        if not normalized:
            raise ModelMissingError("A local Ollama model name is required.")
        if any(marker in normalized for marker in CLOUD_MARKERS):
            raise ModelIncompatibleError("Cloud-hosted model variants are not supported.")

    @staticmethod
    def _translate_transport_error(exc: BaseException) -> OllamaUnavailableError:
        if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
            return OllamaTimeoutError(f"Local Ollama request timed out: {exc.__class__.__name__}.")
        return OllamaUnavailableError(f"Local Ollama is unavailable: {exc.__class__.__name__}.")

    @staticmethod
    def _version_tuple(version: str) -> tuple[int, int, int]:
        return parse_ollama_version(version) or (0, 0, 0)

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result

    async def server_version(self) -> str:
        try:
            request_raw = self._client._request_raw
            response = await request_raw("GET", "/api/version")
            payload = response.json()
        except (ConnectionError, TimeoutError, httpx.TimeoutException) as exc:
            raise self._translate_transport_error(exc) from exc
        except ResponseError as exc:
            raise OllamaUnavailableError(
                f"Local Ollama version probe failed with status {exc.status_code}."
            ) from exc
        except AttributeError as exc:
            raise OllamaUnavailableError(
                "The Ollama client does not support the version probe."
            ) from exc

        version = payload.get("version") if isinstance(payload, Mapping) else None
        if not isinstance(version, str) or not version:
            raise OllamaUnavailableError("Local Ollama returned no server version.")
        return version

    async def list_models(self) -> tuple[str, ...]:
        try:
            response = await self._client.list()
        except (ConnectionError, TimeoutError, httpx.TimeoutException) as exc:
            raise self._translate_transport_error(exc) from exc
        except ResponseError as exc:
            raise OllamaUnavailableError(
                f"Local Ollama model list failed with status {exc.status_code}."
            ) from exc
        return tuple(
            model.model
            for model in response.models
            if isinstance(model.model, str) and model.model
        )

    @staticmethod
    def _context_length(model_info: Mapping[str, Any] | None) -> int | None:
        if not model_info:
            return None
        for key, value in model_info.items():
            if key == "context_length" or str(key).endswith(".context_length"):
                try:
                    context_length = int(value)
                except (TypeError, ValueError):
                    return None
                return context_length if context_length > 0 else None
        return None

    @staticmethod
    def _dimensions(model_info: Mapping[str, Any] | None) -> int | None:
        if not model_info:
            return None
        for key, value in model_info.items():
            if key == "embedding_length" or str(key).endswith(".embedding_length"):
                try:
                    dimensions = int(value)
                except (TypeError, ValueError):
                    return None
                return dimensions if dimensions > 0 else None
        return None

    async def model_profile(self, role: ModelRole, model: str) -> ModelProfile:
        self._validate_model(model)
        try:
            response = await self._client.list()
            installed = {
                item.model: item.digest
                for item in response.models
                if isinstance(item.model, str) and item.model
            }
        except (ConnectionError, TimeoutError, httpx.TimeoutException) as exc:
            raise self._translate_transport_error(exc) from exc
        except ResponseError as exc:
            raise OllamaUnavailableError(
                f"Local Ollama model lookup failed with status {exc.status_code}."
            ) from exc

        installed_model = model
        if installed_model not in installed and ":" not in model.rsplit("/", 1)[-1]:
            latest_model = f"{model}:latest"
            if latest_model in installed:
                installed_model = latest_model
        if installed_model not in installed:
            raise ModelMissingError(f"Local Ollama model is not installed: {model}.")
        try:
            details = await self._client.show(installed_model)
        except ResponseError as exc:
            if exc.status_code == 404:
                raise ModelMissingError(f"Local Ollama model is not installed: {model}.") from exc
            raise OllamaUnavailableError(
                f"Local Ollama model metadata failed with status {exc.status_code}."
            ) from exc
        except (ConnectionError, TimeoutError, httpx.TimeoutException) as exc:
            raise self._translate_transport_error(exc) from exc

        capabilities = tuple(details.capabilities or ())
        required_capabilities = {
            "embedding": ("embedding",),
            "decision": ("decision", "systemone"),
            "generation": ("completion",),
        }[role]
        if not any(
            required in capabilities
            for required in required_capabilities
        ):
            required_capability = "/".join(required_capabilities)
            raise ModelIncompatibleError(
                f"Model {model} lacks the required {required_capability} capability."
            )
        model_info = details.modelinfo if isinstance(details.modelinfo, Mapping) else None
        context_length = self._context_length(model_info)
        if context_length is None:
            raise ModelIncompatibleError(f"Model {model} reports no context length.")
        digest = installed[installed_model]
        if not isinstance(digest, str) or not digest:
            raise ModelIncompatibleError(f"Model {model} reports no digest.")

        return ModelProfile(
            role=role,
            model=model,
            digest=digest,
            context_length=context_length,
            dimensions=self._dimensions(model_info) if role == "embedding" else None,
            capabilities=capabilities,
        )

    async def readiness(
        self, required_models: Mapping[ModelRole, str]
    ) -> OllamaReadiness:
        missing: list[str] = []
        incompatible: list[str] = []
        errors: list[str] = []
        version: str | None = None
        available = True

        try:
            version = await self.server_version()
        except OllamaUnavailableError as exc:
            available = False
            errors.append(str(exc))

        parsed_version = parse_ollama_version(version)
        if version is not None and parsed_version is None:
            errors.append(
                f"Ollama version {version!r} is unknown or invalid; "
                "version 0.35.0 or newer is required."
            )
        elif parsed_version is not None and parsed_version < MINIMUM_OLLAMA_VERSION:
            errors.append(
                f"Ollama {version} is too old; version 0.35.0 or newer is required."
            )

        if available:
            for role, model in required_models.items():
                try:
                    await self.model_profile(role, model)
                except ModelMissingError:
                    missing.append(model)
                except ModelIncompatibleError:
                    incompatible.append(model)
                except OllamaUnavailableError as exc:
                    available = False
                    errors.append(str(exc))

        return OllamaReadiness(
            available=available,
            version=version,
            base_url=self.host,
            missing_models=tuple(sorted(set(missing))),
            incompatible_models=tuple(sorted(set(incompatible))),
            errors=tuple(errors),
        )

    async def embed(
        self,
        model: str,
        text: str | Sequence[str],
        *,
        dimensions: int | None = None,
        truncate: Literal[False] = False,
        keep_alive: float | str | None = "5m",
    ) -> tuple[tuple[float, ...], ...]:
        if truncate is not False:
            raise EmbeddingContextRejectedError(
                "Ollama embeddings must never request input truncation."
            )
        self._validate_model(model)
        try:
            response = await self._client.embed(
                model=model,
                input=text,
                truncate=truncate,
                dimensions=dimensions,
                keep_alive=keep_alive,
            )
        except ResponseError as exc:
            message = str(exc)
            if exc.status_code == 413 or (
                exc.status_code == 400 and "context" in message.lower()
            ):
                raise EmbeddingContextRejectedError(
                    "Embedding input exceeded the model context without truncation."
                ) from exc
            if exc.status_code == 404:
                raise ModelMissingError(f"Local Ollama model is not installed: {model}.") from exc
            raise OllamaUnavailableError(
                f"Local Ollama embedding failed with status {exc.status_code}."
            ) from exc
        except (ConnectionError, TimeoutError, httpx.TimeoutException) as exc:
            raise self._translate_transport_error(exc) from exc
        return tuple(tuple(float(value) for value in vector) for vector in response.embeddings)

    async def system_one(
        self,
        model: str,
        *,
        state: Any,
        questions: Mapping[str, Any],
        keep_alive: float | str | None = "5m",
    ) -> Any:
        self._validate_model(model)
        try:
            return await self._client.systemone(
                model=model,
                state=state,
                questions=questions,
                keep_alive=keep_alive,
            )
        except (ConnectionError, TimeoutError, httpx.TimeoutException) as exc:
            raise self._translate_transport_error(exc) from exc
        except ResponseError as exc:
            if exc.status_code == 404:
                raise ModelMissingError(f"Local Ollama model is not installed: {model}.") from exc
            raise OllamaUnavailableError(
                f"Local Ollama decision failed with status {exc.status_code}."
            ) from exc

    async def chat(
        self,
        model: str,
        *,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any] | None = None,
        keep_alive: float | str | None = "5m",
    ) -> str:
        self._validate_model(model)
        try:
            response = await self._client.chat(
                model=model,
                messages=messages,
                format=dict(schema) if schema is not None else None,
                stream=False,
                keep_alive=keep_alive,
            )
        except (ConnectionError, TimeoutError, httpx.TimeoutException) as exc:
            raise self._translate_transport_error(exc) from exc
        except ResponseError as exc:
            if exc.status_code == 413:
                raise GenerationContextRejectedError(
                    "Local Ollama chat input exceeded the model context."
                ) from exc
            if exc.status_code == 404:
                raise ModelMissingError(f"Local Ollama model is not installed: {model}.") from exc
            message = str(exc).casefold()
            if any(
                marker in message
                for marker in ("context length", "context window", "too many tokens")
            ):
                raise GenerationContextRejectedError(
                    "Local Ollama chat input exceeded the model context."
                ) from exc
            if any(
                marker in message
                for marker in (
                    "json schema",
                    "structured output",
                    "response format",
                    "invalid format",
                    "unsupported format",
                )
            ):
                raise StructuredOutputError(
                    "Local Ollama model rejected structured chat output."
                ) from exc
            if any(
                marker in message
                for marker in ("capability", "does not support", "unsupported")
            ):
                raise ModelIncompatibleError(
                    "Local Ollama model is incompatible with structured chat."
                ) from exc
            raise OllamaUnavailableError(
                f"Local Ollama chat failed with status {exc.status_code}."
            ) from exc
        content = response.message.content
        if not isinstance(content, str):
            raise StructuredOutputError(
                "Local Ollama returned an empty chat response."
            )
        return content

    async def chat_stream(
        self,
        model: str,
        *,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any] | None = None,
        keep_alive: float | str | None = "5m",
    ) -> AsyncIterator[str]:
        self._validate_model(model)
        try:
            stream = await self._client.chat(
                model=model,
                messages=messages,
                format=dict(schema) if schema is not None else None,
                stream=True,
                keep_alive=keep_alive,
            )
            async for response in stream:
                content = response.message.content
                if isinstance(content, str) and content:
                    yield content
        except (ConnectionError, TimeoutError, httpx.TimeoutException) as exc:
            raise self._translate_transport_error(exc) from exc
        except ResponseError as exc:
            if exc.status_code == 404:
                raise ModelMissingError(
                    f"Local Ollama model is not installed: {model}."
                ) from exc
            message = str(exc).casefold()
            if any(
                marker in message
                for marker in ("context length", "context window", "too many tokens")
            ):
                raise GenerationContextRejectedError(
                    "Local Ollama streaming chat input exceeded the model context."
                ) from exc
            if any(
                marker in message
                for marker in (
                    "json schema",
                    "structured output",
                    "response format",
                    "invalid format",
                    "unsupported format",
                    "capability",
                    "does not support",
                    "unsupported",
                )
            ):
                raise ModelIncompatibleError(
                    "Local Ollama model is incompatible with structured streaming chat."
                ) from exc
            raise OllamaUnavailableError(
                f"Local Ollama streaming chat failed with status {exc.status_code}."
            ) from exc

