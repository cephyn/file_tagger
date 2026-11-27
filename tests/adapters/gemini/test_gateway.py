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

    assert generation.model == "gemini-4-hydrogen"
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


def test_readiness_probe_uses_exact_embedding_model_and_768_dimensions() -> None:
    client = FakeClient()
    gateway = GeminiGateway("test-key", client=client)

    report = asyncio.run(
        gateway.readiness(
            {
                "embedding": GEMINI_EMBEDDING_MODEL,
                "decision": GEMINI_DECISION_MODEL,
                "generation": GEMINI_GENERATION_MODEL,
            }
        )
    )

    assert report.available is True
    call = client.aio.models.embed_calls[0]
    assert call["model"] == "gemini-embedding-2"
    assert call["config"].output_dimensionality == 768
    assert call["config"].auto_truncate is False


def test_readiness_authentication_failure_is_sanitized_and_blocking() -> None:
    client = FakeClient()

    async def unauthorized(**kwargs):
        del kwargs
        error = RuntimeError(
            "headers={'authorization': 'Bearer test-key'}; unauthorized"
        )
        error.status_code = 401
        raise error

    client.aio.models.embed_content = unauthorized
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(GeminiAuthenticationError) as caught:
        asyncio.run(
            gateway.readiness(
                {
                    "embedding": GEMINI_EMBEDDING_MODEL,
                    "decision": GEMINI_DECISION_MODEL,
                    "generation": GEMINI_GENERATION_MODEL,
                }
            )
        )

    assert "test-key" not in str(caught.value)
    assert "authorization" not in str(caught.value).lower()
    assert caught.value.__cause__ is None


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
        return SimpleNamespace(text="{")

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


def test_empty_chat_json_is_rejected() -> None:
    client = FakeClient()

    async def empty(**kwargs):
        return SimpleNamespace(text="")

    client.aio.models.generate_content = empty
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(GeminiStructuredOutputError):
        asyncio.run(
            gateway.chat(
                GEMINI_GENERATION_MODEL,
                messages=[{"role": "user", "content": "Find"}],
                schema={"type": "object"},
            )
        )


def test_embed_maps_http_413_to_context_rejection() -> None:
    client = FakeClient()

    async def reject(**kwargs):
        error = RuntimeError("SDK request failed")
        error.status_code = 413
        raise error

    client.aio.models.embed_content = reject
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(EmbeddingContextRejectedError) as caught:
        asyncio.run(gateway.embed(GEMINI_EMBEDDING_MODEL, "source text"))

    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert caught.value.__context__ is None
    assert "test-key" not in str(caught.value)


@pytest.mark.parametrize(
    "message",
    [
        "context window exceeded",
        "maximum context length was exceeded",
        "too many tokens in the request",
    ],
)
def test_embed_maps_context_text_to_context_rejection(message: str) -> None:
    client = FakeClient()

    async def reject(**kwargs):
        raise RuntimeError(message)

    client.aio.models.embed_content = reject
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(EmbeddingContextRejectedError):
        asyncio.run(gateway.embed(GEMINI_EMBEDDING_MODEL, "source text"))


def test_embed_rejects_returned_vector_length_mismatch() -> None:
    client = FakeClient()

    async def wrong_length(**kwargs):
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1] * 767)]
        )

    client.aio.models.embed_content = wrong_length
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(EmbeddingContextRejectedError):
        asyncio.run(gateway.embed(GEMINI_EMBEDDING_MODEL, "source text"))


def test_authentication_is_mapped_by_status_code() -> None:
    client = FakeClient()

    async def unauthorized(**kwargs):
        error = RuntimeError(
            "headers={'authorization': 'Bearer test-key'}; request failed"
        )
        error.status_code = 401
        raise error

    client.aio.models.generate_content = unauthorized
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(GeminiAuthenticationError) as caught:
        asyncio.run(
            gateway.chat(
                GEMINI_GENERATION_MODEL,
                messages=[{"role": "user", "content": "Find"}],
            )
        )

    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert caught.value.__context__ is None
    assert "test-key" not in str(caught.value)
    assert "headers" not in str(caught.value)


def test_system_one_rejects_unknown_choice() -> None:
    client = FakeClient()

    async def generate(**kwargs):
        return SimpleNamespace(
            text='{"answers":{"candidate_000":{"type":"choice","choice":"maybe","probabilities":{"accept":0.4,"reject":0.3,"review":0.3},"confidence":0.4}}}'
        )

    client.aio.models.generate_content = generate
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(GeminiStructuredOutputError):
        asyncio.run(
            gateway.system_one(
                GEMINI_DECISION_MODEL,
                state={"title": "Guide"},
                questions={"candidate_000": {"instructions": "Apply?"}},
            )
        )


def test_system_one_rejects_probability_confidence_mismatch() -> None:
    client = FakeClient()

    async def generate(**kwargs):
        return SimpleNamespace(
            text='{"answers":{"candidate_000":{"type":"choice","choice":"accept","probabilities":{"accept":0.7,"reject":0.2,"review":0.1},"confidence":0.3}}}'
        )

    client.aio.models.generate_content = generate
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(GeminiStructuredOutputError):
        asyncio.run(
            gateway.system_one(
                GEMINI_DECISION_MODEL,
                state={"title": "Guide"},
                questions={"candidate_000": {"instructions": "Apply?"}},
            )
        )


def test_system_one_rejects_probability_sum_mismatch() -> None:
    client = FakeClient()

    async def generate(**kwargs):
        return SimpleNamespace(
            text='{"answers":{"candidate_000":{"type":"choice","choice":"accept","probabilities":{"accept":0.6,"reject":0.2,"review":0.1},"confidence":0.6}}}'
        )

    client.aio.models.generate_content = generate
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(GeminiStructuredOutputError):
        asyncio.run(
            gateway.system_one(
                GEMINI_DECISION_MODEL,
                state={"title": "Guide"},
                questions={"candidate_000": {"instructions": "Apply?"}},
            )
        )


def test_system_one_rejects_missing_answer() -> None:
    client = FakeClient()

    async def generate(**kwargs):
        return SimpleNamespace(text='{"answers":{}}')

    client.aio.models.generate_content = generate
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(GeminiStructuredOutputError):
        asyncio.run(
            gateway.system_one(
                GEMINI_DECISION_MODEL,
                state={"title": "Guide"},
                questions={"candidate_000": {"instructions": "Apply?"}},
            )
        )


def test_system_one_rejects_unknown_answer_id() -> None:
    client = FakeClient()

    async def generate(**kwargs):
        return SimpleNamespace(
            text='{"answers":{"unknown_000":{"type":"choice","choice":"accept","probabilities":{"accept":0.8,"reject":0.1,"review":0.1},"confidence":0.8}}}'
        )

    client.aio.models.generate_content = generate
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(GeminiStructuredOutputError):
        asyncio.run(
            gateway.system_one(
                GEMINI_DECISION_MODEL,
                state={"title": "Guide"},
                questions={"candidate_000": {"instructions": "Apply?"}},
            )
        )


def test_system_one_rejects_duplicate_answer_id() -> None:
    client = FakeClient()

    async def generate(**kwargs):
        return SimpleNamespace(
            text='{"answers":{"candidate_000":{"type":"choice","choice":"accept","probabilities":{"accept":0.8,"reject":0.1,"review":0.1},"confidence":0.8},"candidate_000":{"type":"choice","choice":"reject","probabilities":{"accept":0.1,"reject":0.8,"review":0.1},"confidence":0.8}}}'
        )

    client.aio.models.generate_content = generate
    gateway = GeminiGateway("test-key", client=client)

    with pytest.raises(GeminiStructuredOutputError):
        asyncio.run(
            gateway.system_one(
                GEMINI_DECISION_MODEL,
                state={"title": "Guide"},
                questions={"candidate_000": {"instructions": "Apply?"}},
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


def test_close_failure_is_sanitized() -> None:
    secret = "api-key-secret-request-header"

    class FailingCloseClient(FakeClient):
        async def close(self) -> None:
            raise RuntimeError(
                f"SDK close failed; api-key={secret}; "
                f"Authorization: Bearer {secret}; request={secret}"
            )

    gateway = GeminiGateway("test-key", client=FailingCloseClient())

    with pytest.raises(GeminiUnavailableError) as caught:
        asyncio.run(gateway.close())

    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert secret not in str(caught.value.__cause__)


def test_close_async_failure_still_tries_fallback_client_close() -> None:
    class FailingAio:
        models = FakeModels()

        async def close(self) -> None:
            raise RuntimeError("aio close failed")

    class FallbackClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.aio = FailingAio()

        async def close(self) -> None:
            self.closed = True

    client = FallbackClient()
    gateway = GeminiGateway("test-key", client=client)

    asyncio.run(gateway.close())

    assert client.closed is True
