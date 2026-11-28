import asyncio

from file_tagger.adapters.gemini.gateway import (
    GEMINI_DECISION_MODEL,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_GENERATION_MODEL,
)
from file_tagger.domain.contracts import ModelProfile, OllamaReadiness
from file_tagger.domain.errors import GeminiUnavailableError
from file_tagger.domain.errors import GeminiAuthenticationError
from file_tagger.gemini.readiness import GeminiReadinessService
from file_tagger.ollama.readiness import RequiredModels


class FakeGeminiGateway:
    async def readiness(self, required_models):
        return OllamaReadiness(
            available=True,
            version=None,
            base_url="https://generativelanguage.googleapis.com",
        )

    async def model_profile(self, role, model):
        dimensions = 768 if role == "embedding" else None
        return ModelProfile(
            role=role,
            model=model,
            digest=f"sha256:{role}",
            context_length=128,
            dimensions=dimensions,
            capabilities=(role,),
        )


def test_missing_gemini_key_blocks_every_role_without_exposing_key() -> None:
    selected = RequiredModels(
        embedding=GEMINI_EMBEDDING_MODEL,
        decision=GEMINI_DECISION_MODEL,
        generation=GEMINI_GENERATION_MODEL,
    )

    report = asyncio.run(
        GeminiReadinessService(FakeGeminiGateway(), None).check(selected)
    )

    assert report.provider == "gemini"
    assert report.ready is False
    assert report.blocked_roles == ("indexing", "rag", "search", "tagging")
    assert report.blocked_model_roles == (
        "decision",
        "embedding",
        "generation",
    )
    assert report.ollama is None
    assert report.gemini is not None
    assert report.gemini.key_configured is False
    assert report.gemini.ready is False
    assert all("test-secret" not in item for item in report.errors)


def test_gemini_readiness_publishes_exact_three_role_profiles() -> None:
    selected = RequiredModels(
        embedding=GEMINI_EMBEDDING_MODEL,
        decision=GEMINI_DECISION_MODEL,
        generation=GEMINI_GENERATION_MODEL,
    )

    report = asyncio.run(
        GeminiReadinessService(
            FakeGeminiGateway(),
            "test-secret",
        ).check(selected)
    )

    assert report.provider == "gemini"
    assert report.ready is True
    assert report.ollama is None
    assert report.gemini is not None
    assert report.gemini.key_configured is True
    assert {
        (profile.role, profile.model)
        for profile in report.profiles
    } == {
        ("embedding", GEMINI_EMBEDDING_MODEL),
        ("decision", GEMINI_DECISION_MODEL),
        ("generation", GEMINI_GENERATION_MODEL),
    }


def test_gemini_sdk_readiness_failure_blocks_all_roles() -> None:
    class UnavailableGateway(FakeGeminiGateway):
        async def readiness(self, required_models):
            raise GeminiUnavailableError("Gemini is unavailable.")

    selected = RequiredModels(
        embedding=GEMINI_EMBEDDING_MODEL,
        decision=GEMINI_DECISION_MODEL,
        generation=GEMINI_GENERATION_MODEL,
    )

    report = asyncio.run(
        GeminiReadinessService(UnavailableGateway(), "test-secret").check(
            selected
        )
    )

    assert report.provider == "gemini"
    assert report.ready is False
    assert report.blocked_model_roles == (
        "decision",
        "embedding",
        "generation",
    )
    assert report.gemini is not None
    assert report.gemini.available is False


def test_gemini_sdk_authentication_probe_blocks_all_roles() -> None:
    class AuthenticationGateway(FakeGeminiGateway):
        async def readiness(self, required_models):
            del required_models
            raise GeminiAuthenticationError("Gemini authentication failed.")

    selected = RequiredModels(
        embedding=GEMINI_EMBEDDING_MODEL,
        decision=GEMINI_DECISION_MODEL,
        generation=GEMINI_GENERATION_MODEL,
    )

    report = asyncio.run(
        GeminiReadinessService(AuthenticationGateway(), "test-secret").check(
            selected
        )
    )

    assert report.provider == "gemini"
    assert report.ready is False
    assert report.blocked_model_roles == (
        "decision",
        "embedding",
        "generation",
    )
    assert report.gemini is not None
    assert report.gemini.available is False
    assert "authentication" in " ".join(report.errors).lower()
    assert all("test-secret" not in item for item in report.errors)
