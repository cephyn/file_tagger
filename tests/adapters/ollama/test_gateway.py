import asyncio
from types import SimpleNamespace

import pytest

from file_tagger.adapters.ollama.gateway import OllamaGateway
import file_tagger.adapters.ollama.gateway as gateway_module
from file_tagger.domain.errors import (
    EmbeddingContextRejectedError,
    GenerationContextRejectedError,
    ModelIncompatibleError,
    OllamaUnavailableError,
    RemoteOllamaHostError,
)


class FakeResponse:
    def __init__(self, payload) -> None:
        self.payload = payload

    def json(self):
        return self.payload


class FakeAsyncClient:
    def __init__(self, *, version="0.35.0") -> None:
        self.version = version
        self.embed_calls = []
        self.systemone_calls = []
        self.chat_calls = []

    async def _request_raw(self, method, path):
        assert method == "GET"
        assert path == "/api/version"
        return FakeResponse({"version": self.version})

    async def list(self):
        return SimpleNamespace(
            models=[
                SimpleNamespace(
                    model="embeddinggemma-2:270m",
                    digest="sha256:embed",
                ),
                SimpleNamespace(model="nimble:9b", digest="sha256:decision"),
                SimpleNamespace(model="gemma4:e4b", digest="sha256:generation"),
            ]
        )

    async def show(self, model):
        capabilities = {
            "embeddinggemma-2:270m": ["embedding"],
            "nimble:9b": ["completion", "systemone"],
            "gemma4:e4b": ["completion"],
        }
        dimensions = {"embeddinggemma-2:270m": 768}
        family = model.split(":", 1)[0]
        return SimpleNamespace(
            capabilities=capabilities[model],
            modelinfo={
                f"{family}.context_length": 128_000,
                **(
                    {"embedding_length": dimensions[model]}
                    if model in dimensions
                    else {}
                ),
            },
        )

    async def embed(self, **kwargs):
        self.embed_calls.append(kwargs)
        return SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])

    async def systemone(self, **kwargs):
        self.systemone_calls.append(kwargs)
        return SimpleNamespace(model=kwargs["model"], answers={})

    async def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if kwargs["stream"]:
            async def chunks():
                yield SimpleNamespace(
                    message=SimpleNamespace(content='{"intents":')
                )
                yield SimpleNamespace(
                    message=SimpleNamespace(content='["locate"]}')
                )

            return chunks()
        return SimpleNamespace(
            message=SimpleNamespace(content='{"intents":["locate"]}')
        )


@pytest.mark.parametrize(
    "host",
    [
        "http://localhost:11434",
        "http://127.0.0.1:11435",
        "http://[::1]:1",
        "http://[::1]:65535",
    ],
)
def test_loopback_host_is_allowed(host) -> None:
    assert OllamaGateway(host, client=FakeAsyncClient()).host == host


def test_default_timeout_allows_large_model_cold_load(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class RecordingClient:
        def __init__(self, *, host, timeout, follow_redirects) -> None:
            captured.update(
                host=host,
                timeout=timeout,
                follow_redirects=follow_redirects,
            )

    monkeypatch.setattr(gateway_module, "AsyncClient", RecordingClient)

    OllamaGateway()

    assert float(captured["timeout"]) >= 120.0


@pytest.mark.parametrize(
    "host",
    [
        "https://localhost:11434",
        "http://localhost",
        "http://localhost:0",
        "http://localhost:65536",
        "http://localhost:not-a-port",
        "http://localhost:11434/",
        "http://localhost:11434/api",
        "http://localhost:11434?query=1",
        "http://localhost:11434#fragment",
        "http://user:pass@localhost:11434",
        "https://example.invalid:11434",
    ],
)
def test_non_local_or_non_origin_host_is_rejected(host) -> None:
    with pytest.raises(RemoteOllamaHostError):
        OllamaGateway(host, client=FakeAsyncClient())


def test_readiness_reports_local_models_and_profiles() -> None:
    gateway = OllamaGateway(client=FakeAsyncClient())

    readiness = asyncio.run(
        gateway.readiness(
            {
                "embedding": "embeddinggemma-2:270m",
                "decision": "nimble:9b",
                "generation": "gemma4:e4b",
            }
        )
    )

    assert readiness.ready
    assert readiness.version == "0.35.0"
    assert readiness.missing_models == ()


def test_model_profile_resolves_untagged_model_to_latest_metadata() -> None:
    class LatestTagClient(FakeAsyncClient):
        def __init__(self) -> None:
            super().__init__()
            self.show_calls = []

        async def list(self):
            return SimpleNamespace(
                models=[
                    SimpleNamespace(
                        model="inkling-2-mini:latest",
                        digest="sha256:inkling-latest",
                    )
                ]
            )

        async def show(self, model):
            self.show_calls.append(model)
            return SimpleNamespace(
                capabilities=["completion"],
                modelinfo={"inkling-2-mini.context_length": 32_768},
            )

    client = LatestTagClient()
    gateway = OllamaGateway(client=client)

    profile = asyncio.run(gateway.model_profile("generation", "inkling-2-mini"))
    asyncio.run(
        gateway.chat(
            "inkling-2-mini",
            messages=[{"role": "user", "content": "Create a search plan"}],
        )
    )

    assert profile.model == "inkling-2-mini"
    assert profile.digest == "sha256:inkling-latest"
    assert profile.context_length == 32_768
    assert client.show_calls == ["inkling-2-mini:latest"]
    assert client.chat_calls[0]["model"] == "inkling-2-mini"


def test_embedding_profile_resolves_suffixed_dimension_metadata() -> None:
    class SuffixedMetadataClient(FakeAsyncClient):
        async def show(self, model):
            return SimpleNamespace(
                capabilities=["embedding"],
                modelinfo={
                    "gemma-embedding2.context_length": 262_144,
                    "gemma-embedding2.embedding_length": 512,
                },
            )

    gateway = OllamaGateway(client=SuffixedMetadataClient())

    profile = asyncio.run(
        gateway.model_profile("embedding", "embeddinggemma-2:270m")
    )

    assert profile.dimensions == 512


def test_decision_profile_accepts_modern_decision_capability() -> None:
    class DecisionCapabilityClient(FakeAsyncClient):
        async def list(self):
            return SimpleNamespace(
                models=[
                    SimpleNamespace(model="tev1:4b", digest="sha256:decision"),
                ]
            )

        async def show(self, model):
            return SimpleNamespace(
                capabilities=["decision"],
                modelinfo={
                    "qwen35.context_length": 262_144,
                },
            )

    gateway = OllamaGateway(client=DecisionCapabilityClient())

    profile = asyncio.run(gateway.model_profile("decision", "tev1:4b"))

    assert profile.role == "decision"
    assert profile.capabilities == ("decision",)


def test_old_server_and_missing_model_are_not_ready() -> None:
    old = OllamaGateway(client=FakeAsyncClient(version="0.34.9"))
    readiness = asyncio.run(old.readiness({"generation": "missing:model"}))

    assert not readiness.ready
    assert readiness.missing_models == ("missing:model",)
    assert any("0.35" in error for error in readiness.errors)


def test_embedding_never_truncates_and_cloud_model_is_rejected() -> None:
    client = FakeAsyncClient()
    gateway = OllamaGateway(client=client)

    embeddings = asyncio.run(
        gateway.embed(
            "embeddinggemma-2:270m",
            "complete source",
            dimensions=3,
        )
    )

    assert embeddings == ((0.1, 0.2, 0.3),)
    assert client.embed_calls[0]["truncate"] is False
    with pytest.raises(ModelIncompatibleError, match="Cloud"):
        asyncio.run(gateway.embed("gemma4:cloud", "not sent"))


def test_gateway_delegates_typed_system_one_and_structured_chat() -> None:
    client = FakeAsyncClient()
    gateway = OllamaGateway(client=client)

    asyncio.run(
        gateway.system_one(
            "nimble:9b",
            state={"title": "Guide"},
            questions={
                "research": {
                    "type": "choice",
                    "prompt": "Apply tag?",
                    "choices": ["yes", "no"],
                }
            },
        )
    )
    response = asyncio.run(
        gateway.chat(
            "gemma4:e4b",
            messages=[{"role": "user", "content": "Find the guide"}],
            schema={"type": "object"},
        )
    )

    assert client.systemone_calls[0]["model"] == "nimble:9b"
    assert client.chat_calls[0]["format"] == {"type": "object"}
    assert client.chat_calls[0]["stream"] is False
    assert response == '{"intents":["locate"]}'


def test_gateway_streams_chat_fragments() -> None:
    gateway = OllamaGateway(client=FakeAsyncClient())

    async def collect() -> str:
        chunks = []
        async for chunk in gateway.chat_stream(
            "gemma4:e4b",
            messages=[{"role": "user", "content": "Find the guide"}],
        ):
            chunks.append(chunk)
        return "".join(chunks)

    assert asyncio.run(collect()) == '{"intents":["locate"]}'


class UnavailableClient(FakeAsyncClient):
    async def list(self):
        raise ConnectionError("offline")


def test_connection_failures_become_domain_errors() -> None:
    with pytest.raises(OllamaUnavailableError):
        asyncio.run(OllamaGateway(client=UnavailableClient()).list_models())


class ContextRejectedClient(FakeAsyncClient):
    async def embed(self, **kwargs):
        from ollama import ResponseError

        raise ResponseError("context length exceeded", status_code=400)


def test_embedding_context_rejection_is_explicit() -> None:
    gateway = OllamaGateway(client=ContextRejectedClient())

    with pytest.raises(EmbeddingContextRejectedError):
        asyncio.run(gateway.embed("embeddinggemma-2:270m", "too large"))


class PayloadTooLargeClient(FakeAsyncClient):
    async def embed(self, **kwargs):
        from ollama import ResponseError

        raise ResponseError("request body too large", status_code=413)


def test_embedding_payload_too_large_is_context_rejection() -> None:
    gateway = OllamaGateway(client=PayloadTooLargeClient())

    with pytest.raises(EmbeddingContextRejectedError):
        asyncio.run(gateway.embed("embeddinggemma-2:270m", "too large"))


class StructuredOutputRejectedClient(FakeAsyncClient):
    async def chat(self, **kwargs):
        from ollama import ResponseError

        raise ResponseError("model does not support JSON schema output", status_code=400)


def test_local_structured_output_failure_is_model_incompatible() -> None:
    gateway = OllamaGateway(client=StructuredOutputRejectedClient())

    with pytest.raises(ModelIncompatibleError):
        asyncio.run(
            gateway.chat(
                "gemma4:e4b",
                messages=[{"role": "user", "content": "Find the guide"}],
                schema={"type": "object"},
            )
        )


class ChatContextRejectedClient(FakeAsyncClient):
    async def chat(self, **kwargs):
        from ollama import ResponseError

        raise ResponseError("prompt is too long for the context window", status_code=400)


class ChatPayloadTooLargeClient(FakeAsyncClient):
    async def chat(self, **kwargs):
        from ollama import ResponseError

        raise ResponseError("request body too large", status_code=413)


def test_chat_payload_too_large_is_context_rejection() -> None:
    gateway = OllamaGateway(client=ChatPayloadTooLargeClient())

    with pytest.raises(GenerationContextRejectedError):
        asyncio.run(
            gateway.chat(
                "gemma4:e4b",
                messages=[{"role": "user", "content": "Find the guide"}],
            )
        )


def test_local_chat_context_failure_is_explicit() -> None:
    gateway = OllamaGateway(client=ChatContextRejectedClient())

    with pytest.raises(GenerationContextRejectedError):
        asyncio.run(
            gateway.chat(
                "gemma4:e4b",
                messages=[{"role": "user", "content": "Find the guide"}],
            )
        )
