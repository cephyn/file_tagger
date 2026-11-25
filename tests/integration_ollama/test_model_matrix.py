import pytest

pytestmark = pytest.mark.ollama


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
