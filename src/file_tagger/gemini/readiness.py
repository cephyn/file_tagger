from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..adapters.gemini.gateway import (
    GEMINI_DECISION_MODEL,
    GEMINI_EMBEDDING_DIMENSIONS,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_ENDPOINT,
    GEMINI_GENERATION_MODEL,
    GeminiGateway,
)
from ..domain.contracts import GeminiReadiness, ModelProfile
from ..domain.errors import (
    GeminiAuthenticationError,
    GeminiTimeoutError,
    GeminiUnavailableError,
    ModelIncompatibleError,
)
from ..ollama.readiness import ReadinessReport, RequiredModels
from ..persistence.models import ModelProfileRecord


ROLE_FEATURES = {
    "embedding": ("indexing", "search"),
    "decision": ("tagging",),
    "generation": ("rag", "search"),
}


def _blocked_features(roles: set[str]) -> tuple[str, ...]:
    features = {
        feature
        for role in roles
        for feature in ROLE_FEATURES.get(role, ())
    }
    return tuple(sorted(features))


class GeminiReadinessService:
    def __init__(
        self,
        gateway: Any | None,
        api_key: str | None,
        *,
        session: Session | None = None,
    ) -> None:
        self.api_key = api_key.strip() if isinstance(api_key, str) else None
        self.api_key = self.api_key or None
        self.gateway = gateway
        self.session = session
        self._construction_error: BaseException | None = None
        if self.gateway is None and self.api_key is not None:
            try:
                self.gateway = GeminiGateway(self.api_key)
            except BaseException as exc:
                self._construction_error = exc

    async def check(
        self,
        required: RequiredModels,
        *args: Any,
        **kwargs: Any,
    ) -> ReadinessReport:
        del required, args, kwargs
        exact = RequiredModels(
            embedding=GEMINI_EMBEDDING_MODEL,
            decision=GEMINI_DECISION_MODEL,
            generation=GEMINI_GENERATION_MODEL,
        )

        if self.api_key is None:
            gemini = GeminiReadiness(
                available=False,
                key_configured=False,
                api_key_configured=False,
                missing_key=True,
                base_url=GEMINI_ENDPOINT,
                errors=("Gemini API key is not configured.",),
            )
            return ReadinessReport(
                ready=False,
                provider="gemini",
                blocked_roles=("indexing", "rag", "search", "tagging"),
                blocked_model_roles=("decision", "embedding", "generation"),
                instructions=("Set a Gemini API key in Settings before using Gemini.",),
                errors=gemini.errors,
                gemini=gemini,
                selected_models=exact,
            )

        if self._construction_error is not None:
            gemini = GeminiReadiness(
                available=False,
                key_configured=True,
                api_key_configured=True,
                base_url=GEMINI_ENDPOINT,
                errors=("Gemini readiness is unavailable.",),
            )
            return ReadinessReport(
                ready=False,
                provider="gemini",
                blocked_roles=("indexing", "rag", "search", "tagging"),
                blocked_model_roles=("decision", "embedding", "generation"),
                instructions=("Check Gemini availability and API-key permissions.",),
                errors=gemini.errors,
                gemini=gemini,
                selected_models=exact,
            )

        blocked_model_roles: set[str] = set()
        errors: list[str] = []
        instructions: list[str] = []
        profiles: list[ModelProfile] = []

        try:
            readiness = getattr(self.gateway, "readiness", None)
            if callable(readiness):
                gateway_report = await readiness(exact.as_mapping())
                incompatible = set(
                    getattr(gateway_report, "incompatible_models", ()) or ()
                )
                gateway_errors = tuple(
                    getattr(gateway_report, "errors", ()) or ()
                )
                gateway_available = bool(
                    getattr(gateway_report, "available", True)
                )
                if not gateway_available:
                    errors.append("Gemini readiness is unavailable.")
                    blocked_model_roles.update(
                        {"embedding", "decision", "generation"}
                    )
                if gateway_errors:
                    errors.append("Gemini readiness returned an error.")
                    blocked_model_roles.update(
                        {"embedding", "decision", "generation"}
                    )
                if incompatible:
                    for role, model in exact.as_mapping().items():
                        if model in incompatible:
                            blocked_model_roles.add(role)
                            errors.append(
                                f"Gemini model {model} is incompatible with the {role} role."
                            )
        except (
            GeminiUnavailableError,
            GeminiTimeoutError,
            GeminiAuthenticationError,
        ) as exc:
            blocked_model_roles.update(
                {"embedding", "decision", "generation"}
            )
            if isinstance(exc, GeminiAuthenticationError):
                errors.append("Gemini authentication failed.")
            elif isinstance(exc, GeminiTimeoutError):
                errors.append("Gemini request timed out.")
            else:
                errors.append("Gemini is unavailable.")
            instructions.append(
                "Check Gemini availability and API-key permissions."
            )
        except Exception:
            blocked_model_roles.update(
                {"embedding", "decision", "generation"}
            )
            errors.append("Gemini readiness is unavailable.")
            instructions.append(
                "Check Gemini availability and API-key permissions."
            )

        for role, model in exact.as_mapping().items():
            if role in blocked_model_roles:
                continue
            try:
                profile = await self.gateway.model_profile(role, model)
            except (
                ModelIncompatibleError,
                GeminiUnavailableError,
                GeminiTimeoutError,
                GeminiAuthenticationError,
            ) as exc:
                blocked_model_roles.add(role)
                if isinstance(exc, GeminiAuthenticationError):
                    errors.append("Gemini authentication failed.")
                elif isinstance(exc, GeminiTimeoutError):
                    errors.append("Gemini request timed out.")
                elif isinstance(exc, GeminiUnavailableError):
                    errors.append("Gemini is unavailable.")
                else:
                    errors.append(f"Gemini {role} readiness is incompatible.")
                continue
            except Exception:
                blocked_model_roles.add(role)
                errors.append(f"Gemini {role} readiness failed.")
                continue

            if (
                profile.role != role
                or profile.model != model
                or (
                    role == "embedding"
                    and profile.dimensions != GEMINI_EMBEDDING_DIMENSIONS
                )
            ):
                blocked_model_roles.add(role)
                errors.append(
                    f"Gemini {role} readiness returned an incompatible profile."
                )
                continue
            profiles.append(profile)

        gemini_available = not blocked_model_roles and not errors
        gemini = GeminiReadiness(
            available=gemini_available,
            key_configured=True,
            api_key_configured=True,
            missing_key=False,
            base_url=GEMINI_ENDPOINT,
            incompatible_models=tuple(
                model
                for role, model in exact.as_mapping().items()
                if role in blocked_model_roles
            ),
            errors=tuple(dict.fromkeys(errors)),
        )

        if not gemini_available and not instructions:
            instructions.append(
                "Check Gemini availability and API-key permissions."
            )

        if self.session is not None:
            self._persist_profiles(profiles)

        return ReadinessReport(
            ready=gemini_available,
            provider="gemini",
            blocked_roles=_blocked_features(blocked_model_roles),
            blocked_model_roles=tuple(sorted(blocked_model_roles)),
            instructions=tuple(dict.fromkeys(instructions)),
            errors=tuple(dict.fromkeys(errors)),
            profiles=tuple(profiles),
            gemini=gemini,
            selected_models=exact,
        )

    def _persist_profiles(self, profiles: list[ModelProfile]) -> None:
        if self.session is None:
            return
        for profile in profiles:
            record = self.session.scalar(
                select(ModelProfileRecord).where(
                    ModelProfileRecord.role == profile.role,
                    ModelProfileRecord.model == profile.model,
                    ModelProfileRecord.digest == profile.digest,
                    ModelProfileRecord.dimensions == profile.dimensions,
                )
            )
            index_version = (
                f"{profile.model}@{profile.digest}:{profile.dimensions}"
                if profile.role == "embedding"
                else f"{profile.model}@{profile.digest}"
            )
            if record is None:
                record = ModelProfileRecord(
                    role=profile.role,
                    model=profile.model,
                    digest=profile.digest,
                    dimensions=profile.dimensions,
                    context_length=profile.context_length,
                    index_version=index_version,
                    capabilities=list(profile.capabilities),
                    benchmark_data={},
                )
                self.session.add(record)
            else:
                record.context_length = profile.context_length
                record.index_version = index_version
                record.capabilities = list(profile.capabilities)
                record.benchmark_data = {}
        self.session.flush()
