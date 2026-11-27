import asyncio
from types import SimpleNamespace

import pytest

from file_tagger.adapters.gemini.gateway import (
    GEMINI_DECISION_MODEL,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_GENERATION_MODEL,
    GeminiGateway,
)
from file_tagger.domain.errors import (
    EmbeddingContextRejectedError,
    GeminiAuthenticationError,
    GeminiStructuredOutputError,
    GeminiTimeoutError,
    GeminiUnavailableError,
)


class FakeModels:
    def __init__(self) -> None:
        self.embed_calls: list[dict[str, object]] = []
        self.chat_calls: list[dict[str, object]] = []

    async def embed_content(self, **kwargs):
        self.embed_calls.append(kwargs)
        contents = kwargs["contents"]
        if isinstance(contents, (list, tuple)):
            vectors = [[0.1] * 768 for _ in contents]
        else:
            vectors = [[0.1] * 768]
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=vector) for vector in vectors]
        )

    async def generate_content(self, **kwargs):
        self.chat_calls.append(kwargs)
        return SimpleNamespace(text='{"intents":["locate"]}')

    def generate_content_stream(self, **kwargs):
        async def fragments():
            yield SimpleNamespace(text='{"intents":')
            yield SimpleNamespace(text='["locate"]}')

        return fragments()


class FakeAio:
    def __init__(self) -> None:
        self.models = FakeModels()


class FakeClient:
    def __init__(self) -> None:
        self.aio = FakeAio()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_exact_model_names_and_profiles_are_stable() -> None:
    gateway = GeminiGateway("test-key", client=FakeClient())

    generation = asyncio.run(
        gateway.model_profile("generation", GEMINI_GENERATION_MODEL)
    )
    decision = asyncio.run(
        gateway.model_profile("decision", GEMINI_DECISION_MODEL)
    )
    embedding = asyncio.run(
        gateway.model_profile("embedding", GEMINI_EMBEDDING_MODEL)
    )

    assert generation.model == "gemini-4-neon"
    assert decision.model == "gemini-decision-1"
    assert embedding.model == "gemini-embedding-2"
    assert generation.role == "generation"
    assert decision.role == "decision"
    assert embedding.role == "embedding"
    assert generation.context_length > 0
    assert decision.context_length > 0
    assert embedding.context_length > 0
    assert generation.capabilities
    assert decision.capabilities
    assert embedding.dimensions == 768
    assert generation.digest == asyncio.run(
        gateway.model_profile("generation", GEMINI_GENERATION_MODEL)
    ).digest
    assert embedding.digest == asyncio.run(
        gateway.model_profile("embedding", GEMINI_EMBEDDING_MODEL)
    ).digest
    assert "test-key" not in generation.digest
    assert "test-key" not in decision.digest
    assert "test-key" not in embedding.digest


def test_missing_key_is_rejected_without_exposure() -> None:
    with pytest.raises(GeminiAuthenticationError):
        GeminiGateway("   ")


@pytest.mark.parametrize("text", ["source text", ["one", "two"]])
def test_embed_converts_single_and_batch_vectors(text) -> None:
    client = FakeClient()
    gateway = GeminiGateway("test-key", client=client)

    vectors = asyncio.run(
        gateway.embed(
            GEMINI_EMBEDDING_MODEL,
            text,
            dimensions=768,
            truncate=False,
        )
    )

    assert len(vectors) == (1 if isinstance(text, str) else 2)
    assert all(len(vector) == 768 for vector in vectors)
    assert client.aio.models.embed_calls[0]["config"].output_dimensionality == 768


def test_embed_defaults_to_768_and_never_truncates() -> None:
    client = FakeClient()
    gateway = GeminiGateway("test-key", client=client)

    vectors = asyncio.run(
        gateway.embed(GEMINI_EMBEDDING_MODEL, "source text")
    )

    assert all(len(vector) == 768 for vector in vectors)
    with pytest.raises(EmbeddingContextRejectedError):
        asyncio.run(
            gateway.embed(
                GEMINI_EMBEDDING_MODEL,
                "source text",
                truncate=True,
            )
        )


def test_embed_rejects_wrong_requested_dimensions() -> None:
    gateway = GeminiGateway("test-key", client=FakeClient())

    with pytest.raises(EmbeddingContextRejectedError):
        asyncio.run(
            gateway.embed(
                GEMINI_EMBEDDING_MODEL,
                "source text",
                dimensions=512,
            )
        )


def test_chat_structured_output_returns_text() -> None:
    client = FakeClient()
    gateway = GeminiGateway("test-key", client=client)
    schema = {"type": "object", "properties": {"intents": {"type": "array"}}}

    response = asyncio.run(
        gateway.chat(
            GEMINI_GENERATION_MODEL,
            messages=[{"role": "user", "content": "Find the guide"}],
            schema=schema,
        )
    )

    assert response == '{"intents":["locate"]}'
    call = client.aio.models.chat_calls[0]
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_json_schema == schema


def test_chat_streams_fragments() -> None:
    gateway = GeminiGateway("test-key", client=FakeClient())

    async def collect() -> str:
        fragments = []
        async for fragment in gateway.chat_stream(
            GEMINI_GENERATION_MODEL,
            messages=[{"role": "user", "content": "Find the guide"}],
        ):
            fragments.append(fragment)
        return "".join(fragments)

    assert asyncio.run(collect()) == '{"intents":["locate"]}'


def test_system_one_maps_choice_answers_for_decision_parser() -> None:
    client = FakeClient()

    async def generate(**kwargs):
        return SimpleNamespace(
            text='{"answers":{"candidate_000":{"type":"choice","choice":"accept","probabilities":{"accept":0.8,"reject":0.1,"review":0.1},"confidence":0.8}}}'
        )

    client.aio.models.generate_content = generate
    gateway = GeminiGateway("test-key", client=client)

    answer = asyncio.run(
        gateway.system_one(
            GEMINI_DECISION_MODEL,
            state={"title": "Guide"},
            questions={
                "candidate_000": {
                    "type": "choice",
                    "instructions": "Apply the candidate tag?",
                }
            },
        )
    )

    assert answer["answers"]["candidate_000"]["choice"] == "accept"


def test_malformed_or_empty_chat_json_is_rejected() -> None:
    client = FakeClient()

    async def malformed(**kwargs):
        return SimpleNamespace(text="")

    client.aio.models.generate_content = malformed
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(GeminiStructuredOutputError):
        asyncio.run(
            gateway.chat(
                GEMINI_GENERATION_MODEL,
                messages=[{"role": "user", "content": "Find"}],
                schema={"type": "object"},
            )
        )


@pytest.mark.parametrize(
    "exception, error_type",
    [
        (TimeoutError("slow"), GeminiTimeoutError),
        (PermissionError("denied"), GeminiAuthenticationError),
        (ConnectionError("offline"), GeminiUnavailableError),
    ],
)
def test_transport_failures_are_mapped(exception, error_type) -> None:
    class FailingClient(FakeClient):
        async def generate(self, **kwargs):
            raise exception

    client = FakeClient()

    async def fail(**kwargs):
        raise exception

    client.aio.models.generate_content = fail
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(error_type) as caught:
        asyncio.run(
            gateway.chat(
                GEMINI_GENERATION_MODEL,
                messages=[{"role": "user", "content": "Find"}],
            )
        )
    assert "test-key" not in str(caught.value)


def test_close_closes_injected_client() -> None:
    client = FakeClient()
    gateway = GeminiGateway("test-key", client=client)

    asyncio.run(gateway.close())

    assert client.closed is True
