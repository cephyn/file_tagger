import asyncio
from pathlib import Path

from sqlalchemy import func, select

from file_tagger.domain.contracts import ModelProfile, OllamaReadiness
from file_tagger.domain.errors import ModelIncompatibleError, ModelMissingError
from file_tagger.infrastructure.database import create_session_factory, upgrade_database
from file_tagger.ollama.readiness import OllamaReadinessService, RequiredModels
from file_tagger.persistence.models import ModelProfileRecord


class FakeGateway:
    def __init__(
        self,
        *,
        readiness: OllamaReadiness | None = None,
        unavailable_model: str | None = None,
        incompatible_model: str | None = None,
        latency: float = 10.0,
        installed_models: tuple[str, ...] = (),
    ) -> None:
        self.base_readiness = readiness or OllamaReadiness(
            available=True,
            version="0.35.0",
            base_url="http://127.0.0.1:11434",
        )
        self.unavailable_model = unavailable_model
        self.incompatible_model = incompatible_model
        self.latency = latency
        self.installed_models = installed_models
        self.host = self.base_readiness.base_url

    async def list_models(self):
        return self.installed_models

    async def readiness(self, required_models):
        missing = tuple(
            model
            for model in required_models.values()
            if model == self.unavailable_model
        )
        incompatible = tuple(
            model
            for model in required_models.values()
            if model == self.incompatible_model
        )
        errors = self.base_readiness.errors
        if self.base_readiness.version == "0.34.9":
            errors = errors + (
                "Ollama 0.34.9 is too old; version 0.35.0 or newer is required.",
            )
        return OllamaReadiness(
            available=self.base_readiness.available,
            version=self.base_readiness.version,
            base_url=self.host,
            missing_models=missing,
            incompatible_models=incompatible,
            errors=errors,
        )

    async def model_profile(self, role, model):
        if model == self.unavailable_model:
            raise ModelMissingError(model)
        if model == self.incompatible_model:
            raise ModelIncompatibleError(model)
        return ModelProfile(
            role=role,
            model=model,
            digest=f"sha256:{role}",
            context_length=128_000,
            dimensions=256 if role == "embedding" else None,
            capabilities=(role,),
        )

    async def embed(self, model, text, **kwargs):
        await _sleep(self.latency)
        return ((0.1, 0.2),)

    async def system_one(self, model, **kwargs):
        await _sleep(self.latency)
        return object()

    async def chat(self, model, **kwargs):
        await _sleep(self.latency)
        return '{"ok":true}'


async def _sleep(milliseconds: float) -> None:
    if milliseconds:
        await asyncio.sleep(milliseconds / 1_000_000)


def required_models() -> RequiredModels:
    return RequiredModels(
        embedding="embeddinggemma-2:270m",
        decision="nimble:9b",
        generation="gemma4:e4b",
    )


def test_missing_decision_model_blocks_tagging() -> None:
    report = asyncio.run(
        OllamaReadinessService(
            FakeGateway(unavailable_model="nimble:9b")
        ).check(required_models())
    )

    assert report.ready is False
    assert report.blocked_roles == ("tagging",)
    assert any("ollama pull nimble:9b" in item for item in report.instructions)


def test_old_server_cloud_tag_and_wrong_capability_are_distinct() -> None:
    old = asyncio.run(
        OllamaReadinessService(
            FakeGateway(
                readiness=OllamaReadiness(
                    available=True,
                    version="0.34.9",
                    base_url="http://127.0.0.1:11434",
                )
            )
        ).check(required_models())
    )
    wrong = asyncio.run(
        OllamaReadinessService(
            FakeGateway(incompatible_model="gemma4:e4b")
        ).check(required_models())
    )
    cloud = asyncio.run(
        OllamaReadinessService(FakeGateway()).check(
            RequiredModels(
                embedding="embeddinggemma-2:270m",
                decision="nimble:9b",
                generation="gemma4:cloud",
            )
        )
    )

    assert any("0.35.0" in item for item in old.instructions)
    assert "rag" in wrong.blocked_roles
    assert any("capability" in item for item in wrong.instructions)
    assert cloud.ready is False
    assert any("Cloud" in item for item in cloud.errors)


def test_slow_successful_benchmarks_receive_warning_not_failure() -> None:
    report = asyncio.run(
        OllamaReadinessService(FakeGateway(latency=12_000)).check(
            required_models(),
            slow_threshold_ms=1.0,
        )
    )

    assert report.ready is True
    assert len(report.benchmarks) == 3
    assert any(result.slow is True for result in report.benchmarks)
    assert any("slow" in warning.lower() for warning in report.warnings)


def test_successful_readiness_has_all_profiles() -> None:
    report = asyncio.run(
        OllamaReadinessService(FakeGateway()).check(required_models())
    )

    assert report.ready is True
    assert {profile.role for profile in report.profiles} == {
        "embedding",
        "decision",
        "generation",
    }


def test_readiness_includes_full_compatible_installed_catalog_by_role() -> None:
    report = asyncio.run(
        OllamaReadinessService(
            FakeGateway(
                installed_models=(
                    "embeddinggemma-2:270m",
                    "embeddinggemma-2:440m",
                    "nimble:9b",
                    "tev1:4b",
                    "gemma4:e4b",
                    "gemma4:e2b",
                )
            )
        ).check(required_models())
    )

    assert {
        (profile.role, profile.model) for profile in report.profiles
    } >= {
        ("embedding", "embeddinggemma-2:440m"),
        ("decision", "tev1:4b"),
        ("generation", "gemma4:e2b"),
    }


def test_readiness_persists_selected_model_benchmarks(tmp_path: Path) -> None:
    database = tmp_path / "readiness.sqlite3"
    upgrade_database(database)
    session_factory = create_session_factory(database)

    with session_factory.begin() as session:
        report = asyncio.run(
            OllamaReadinessService(FakeGateway(latency=2.5), session=session).check(
                required_models()
            )
        )

        records = session.scalars(select(ModelProfileRecord)).all()
        assert report.ready
        assert len(records) == 3
        assert session.scalar(
            select(func.count(ModelProfileRecord.id))
        ) == 3
        assert all(record.benchmark_data["success"] for record in records)
