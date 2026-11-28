from pathlib import Path

import pytest

from file_tagger.adapters.ollama.gateway import OllamaGateway
from file_tagger.domain.errors import RemoteOllamaHostError

pytestmark = pytest.mark.ollama


def test_remote_host_requires_loopback_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FILE_TAGGER_RUN_OLLAMA_TESTS", "1")
    monkeypatch.setenv("FILE_TAGGER_OLLAMA_HOST", "http://example.invalid:11434")
    with pytest.raises(RemoteOllamaHostError):
        OllamaGateway(host="http://example.invalid:11434")


def test_conftest_does_not_construct_raw_host_client() -> None:
    source = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")

    assert "from ollama import AsyncClient" not in source
    assert "AsyncClient(" not in source


def test_installed_profiles_cover_embed_decide_and_answer(
    real_ollama,
    integration_profile,
) -> None:
    embedding = real_ollama.embed(
        integration_profile.embedding,
        "hello",
    )
    decision = real_ollama.decide(
        integration_profile.decision,
        "Accept or reject: this is a local RPG encounter",
    )
    chat = real_ollama.chat(
        integration_profile.generation,
        "Reply with the word hello.",
    )

    assert embedding.vector
    assert embedding.latency_ms >= 0.0
    assert decision.answer
    assert chat.content

    real_ollama.record_model_metrics(integration_profile.generation)


def test_profiles_reject_cloud_variants(real_ollama, integration_profile) -> None:
    for role in ("embedding", "decision", "generation"):
        model = getattr(integration_profile, role)
        assert ":cloud" not in model.casefold()
        assert "-cloud" not in model.casefold()
        assert "cloud:" not in model.casefold()
        assert ".cloud" not in model.casefold()
