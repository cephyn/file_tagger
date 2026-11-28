from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import queue
import threading
from typing import Any, Protocol, TypeAlias

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .adapters.gemini.gateway import (
    GEMINI_DECISION_MODEL,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_GENERATION_MODEL,
    GeminiGateway,
)
from .adapters.ollama.gateway import OllamaGateway
from .extraction.docling_adapter import DoclingAdapter
from .gemini.readiness import GeminiReadinessService
from .indexing.repository import IndexingRepository
from .indexing.repository import VectorIndexReadiness
from .indexing.service import IndexingResult, IndexingService
from .infrastructure.app_paths import AppPaths
from .infrastructure.database import (
    create_session_factory,
    normalize_tag_key,
    upgrade_database,
)
from .migration.config_migration import ConfigMigrationReport, migrate_legacy_config
from .migration.sqlite_migration import LegacyMigrationReport, migrate_legacy_sqlite
from .ollama.hardware import profile_hardware
from .domain.contracts import ModelProfile
from .domain.errors import ModelRecommendationError
from .ollama.model_recommender import ModelRecommendation, recommend_models
from .ollama.readiness import OllamaReadinessService, ReadinessReport, RequiredModels
from .persistence.models import File, FileTag, Tag
from .persistence.vector_cleanup import delete_file_embeddings
from .rag.planner import SearchPlan
from .rag.citations import CitedAnswer
from .rag.service import RagService, execute_search_plan
from .search.hybrid import HybridSearch
from .search.results import SearchResult, merge_results


logger = logging.getLogger("ModernRuntime")
SessionFactory: TypeAlias = sessionmaker[Session]
TagData = tuple[str, str | None]
ReadinessCallback = Callable[[ReadinessReport | None], None]
_INDEX_RECOVERY_MAX_WAIT_SECONDS = 1.0
_CONTEXTUAL_RAG_CACHE_LIMIT = 8


@dataclass(frozen=True, slots=True)
class _IndexRequest:
    file_path: str
    generation: int
    provider: str
    provider_generation: int
    indexing_service_generation: int


class ModernIndexingUnavailable(RuntimeError):
    """Raised when a modern indexing profile has not become available."""


class ModernSearchUnavailable(RuntimeError):
    """Raised when modern content search has not become available."""


class StartupConfig(Protocol):
    paths: AppPaths

    def reload_after_migration(self) -> None: ...


def _canonical_path(file_path: str | Path) -> str:
    path = Path(file_path).expanduser().resolve()
    return os.path.normpath(str(path))


def _resolve_tag(session: Session, name: str) -> Tag | None:
    key = normalize_tag_key(name)
    if key is None:
        return None
    for tag in session.scalars(select(Tag).order_by(Tag.id)).all():
        if normalize_tag_key(tag.name) == key:
            return tag
    return None


class _FreshOllamaEmbeddingGateway:
    """Keep each asyncio invocation on a gateway bound to that event loop."""

    async def embed(self, *args: Any, **kwargs: Any) -> Any:
        gateway = OllamaGateway()
        try:
            return await gateway.embed(*args, **kwargs)
        finally:
            await gateway.close()


class _FreshGeminiEmbeddingGateway:
    """Create and close one Gemini gateway per event-loop invocation."""

    provider = "gemini"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def embed(self, *args: Any, **kwargs: Any) -> Any:
        gateway = GeminiGateway(self._api_key)
        try:
            return await gateway.embed(*args, **kwargs)
        finally:
            await gateway.close()


class ModernRuntime:
    """Modern database and local-model services used alongside the legacy UI."""

    def __init__(
        self,
        paths: AppPaths,
        session_factory: SessionFactory | None,
        *,
        sqlite_migration: LegacyMigrationReport | None = None,
        config_migration: ConfigMigrationReport | None = None,
        unavailable_reason: str | None = None,
        config: Any | None = None,
        password: str | None = None,
    ) -> None:
        del password
        self.paths = paths
        self.session_factory = session_factory
        self.sqlite_migration = sqlite_migration
        self.config_migration = config_migration
        self._config = config
        self._active_provider = "ollama"
        self._gemini_api_key: str | None = None
        get_provider = getattr(config, "get_ai_provider", None)
        if callable(get_provider):
            try:
                provider = get_provider()
                if provider in ("ollama", "gemini"):
                    self._active_provider = provider
            except Exception:
                pass
        get_key = getattr(config, "get_gemini_api_key", None)
        if callable(get_key):
            try:
                key = get_key()
                if isinstance(key, str) and key.strip():
                    self._gemini_api_key = key.strip()
            except Exception:
                pass
        self._model_readiness: ReadinessReport | None = None
        self.required_models: RequiredModels | None = None
        self._indexing_service: IndexingService | None = None
        self._indexing_repository = (
            IndexingRepository(session_factory)
            if session_factory is not None
            else None
        )
        self._embedding_gateway: Any | None = None
        self._candidate_gateway: Any | None = None
        self._decision_gateway: Any | None = None
        self._generation_gateway: Any | None = None
        self._readiness_thread: threading.Thread | None = None
        self._index_queue: queue.Queue[_IndexRequest] = queue.Queue()
        self._index_generations: dict[str, int] = {}
        self._index_worker: threading.Thread | None = None
        self._index_recovery_thread: threading.Thread | None = None
        self._indexing_service_generation = 0
        self._provider_generation = 0
        self._lock = threading.Lock()
        self._contextual_rag_lock = threading.Lock()
        self._contextual_rag_condition = threading.Condition(
            self._contextual_rag_lock
        )
        self._contextual_rag_cache_limit = _CONTEXTUAL_RAG_CACHE_LIMIT
        self._contextual_rag_services: OrderedDict[
            str,
            tuple[
                str,
                RagService,
                Any,
            ],
        ] = OrderedDict()
        self._contextual_rag_epochs: dict[str, int] = {}
        self._contextual_rag_active: dict[str, int] = {}
        self._contextual_rag_closed: set[str] = set()
        self._retired_contextual_rag: dict[
            str,
            list[tuple[int, RagService, Any]],
        ] = {}
        self._unavailable_reason = unavailable_reason
        self._readiness_generation = 0
        self._model_overrides = self._load_model_overrides()

    @property
    def readiness(self) -> ReadinessReport | None:
        report = self._model_readiness
        if report is None:
            return None
        profile = report.selected_profile("embedding")
        required = self.required_models
        selected = report.selected_models
        if (
            profile is None
            or required is None
            or selected != required
            or self._indexing_repository is None
        ):
            return report
        try:
            readiness = self._indexing_repository.vector_readiness(profile)
        except Exception:
            if not callable(
                getattr(self.session_factory, "begin", None)
            ):
                return report
            blocked_roles = tuple(
                dict.fromkeys((*report.blocked_roles, "search"))
            )
            return report.model_copy(
                update={
                    "ready": False,
                    "blocked_roles": blocked_roles,
                    "instructions": tuple(
                        dict.fromkeys(
                            (
                                *report.instructions,
                                "Semantic search is unavailable until committed "
                                "index readiness can be verified.",
                            )
                        )
                    ),
                }
            )
        if readiness.ready:
            return report
        blocked_roles = tuple(
            dict.fromkeys((*report.blocked_roles, "search"))
        )
        instructions = tuple(
            dict.fromkeys(
                (
                    *report.instructions,
                    "Semantic search requires a full reindex for the active "
                    f"embedding profile ({readiness.pending_files} file(s) pending).",
                )
            )
        )
        return report.model_copy(
            update={
                "ready": False,
                "blocked_roles": blocked_roles,
                "instructions": instructions,
            }
        )

    @readiness.setter
    def readiness(self, report: ReadinessReport | None) -> None:
        self._model_readiness = report

    @property
    def unavailable_reason(self) -> str:
        if self._unavailable_reason:
            return self._unavailable_reason
        if self._readiness_thread is not None and self._readiness_thread.is_alive():
            return "local model readiness is still being checked."
        if self.readiness is not None:
            messages = self.readiness.instructions + self.readiness.errors
            return messages[0] if messages else "local model profiles are unavailable."
        return "modern indexing has not been initialized."

    @property
    def active_provider(self) -> str:
        with self._lock:
            return self._active_provider

    @property
    def candidate_gateway(self) -> Any | None:
        with self._lock:
            return self._candidate_gateway

    @property
    def decision_gateway(self) -> Any | None:
        with self._lock:
            return self._decision_gateway

    @property
    def generation_gateway(self) -> Any | None:
        with self._lock:
            return self._generation_gateway

    @property
    def embedding_gateway(self) -> Any | None:
        with self._lock:
            return self._embedding_gateway

    @property
    def candidate_model(self) -> str | None:
        profile = self.selected_profile("generation")
        return profile.model if profile is not None else None

    @property
    def decision_model(self) -> str | None:
        profile = self.selected_profile("decision")
        return profile.model if profile is not None else None

    @property
    def embedding_profile(self) -> ModelProfile | None:
        return self.selected_profile("embedding")

    @property
    def selected_models(self) -> RequiredModels | None:
        with self._lock:
            if self.required_models is not None:
                return self.required_models
            if self.readiness is not None:
                return self.readiness.selected_models
        return None

    @property
    def vector_index_readiness(self) -> VectorIndexReadiness | None:
        report = self._model_readiness
        required = self.required_models
        profile = report.selected_profile("embedding") if report is not None else None
        repository = self._indexing_repository
        if (
            profile is None
            or required is None
            or report is None
            or report.selected_models != required
            or repository is None
        ):
            return None
        return repository.vector_readiness(profile)

    def selected_profile(self, role: str) -> ModelProfile | None:
        with self._lock:
            report = self.readiness
            required = self.required_models
        if report is None:
            return None
        selected = (
            required if required is not None else report.selected_models
        )
        if selected is None:
            return None
        if selected != report.selected_models:
            return None
        return report.selected_profile(role)

    @property
    def model_overrides(self) -> dict[str, str]:
        with self._lock:
            return dict(self._model_overrides)

    def start_readiness(
        self,
        on_complete: ReadinessCallback | None = None,
    ) -> Future[ReadinessReport | None]:
        completion: Future[ReadinessReport | None] = Future()
        if self.session_factory is None:
            completion.set_result(None)
            self._notify_readiness_complete(on_complete, None)
            return completion
        with self._lock:
            self._readiness_generation += 1
            generation = self._readiness_generation
            provider_generation = self._provider_generation
        thread = threading.Thread(
            target=self._initialize_readiness,
            args=(generation, provider_generation, completion, on_complete),
            name="file-tagger-ollama-readiness",
            daemon=True,
        )
        with self._lock:
            self._readiness_thread = thread
        thread.start()
        return completion

    def wait_for_readiness(
        self,
        timeout_seconds: float = 30.0,
    ) -> ReadinessReport | None:
        if timeout_seconds <= 0:
            return self.readiness
        with self._lock:
            thread = self._readiness_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, timeout_seconds))
        return self.readiness

    def apply_provider(
        self,
        provider: str,
        gemini_api_key: str | None = None,
    ) -> Future[ReadinessReport | None]:
        normalized = provider.strip().lower() if isinstance(provider, str) else ""
        if normalized not in ("ollama", "gemini"):
            raise ValueError("Provider must be 'ollama' or 'gemini'.")

        set_provider = getattr(self._config, "set_ai_provider", None)
        if callable(set_provider) and not set_provider(normalized):
            raise ValueError("Provider must be 'ollama' or 'gemini'.")

        key_to_store: str | None = None
        if isinstance(gemini_api_key, str) and gemini_api_key.strip():
            key_to_store = gemini_api_key.strip()
            set_key = getattr(self._config, "set_gemini_api_key", None)
            if callable(set_key):
                set_key(key_to_store)
        else:
            get_key = getattr(self._config, "get_gemini_api_key", None)
            if callable(get_key):
                try:
                    stored = get_key()
                    if isinstance(stored, str) and stored.strip():
                        key_to_store = stored.strip()
                except Exception:
                    pass
            else:
                key_to_store = self._gemini_api_key

        old_gateways: list[Any] = []
        with self._lock:
            self._provider_generation += 1
            old_gateways.extend(
                gateway
                for gateway in (
                    self._candidate_gateway,
                    self._decision_gateway,
                    self._generation_gateway,
                )
                if gateway is not None
            )
            self._active_provider = normalized
            self._gemini_api_key = key_to_store
            self.required_models = None
            self._indexing_service = None
            self._indexing_service_generation = 0
            self._embedding_gateway = None
            self._candidate_gateway = None
            self._decision_gateway = None
            self._generation_gateway = None
            self._model_readiness = None
            with self._contextual_rag_condition:
                retired_session_ids: list[str] = []
                for session_id, (_key, rag, session) in list(
                    self._contextual_rag_services.items()
                ):
                    old_epoch = self._contextual_rag_epochs.get(session_id, 0)
                    self._contextual_rag_epochs[session_id] = old_epoch + 1
                    self._contextual_rag_closed.add(session_id)
                    del self._contextual_rag_services[session_id]
                    self._retired_contextual_rag.setdefault(session_id, []).append(
                        (old_epoch, rag, session)
                    )
                    retired_session_ids.append(session_id)
                self._contextual_rag_condition.notify_all()

        for session_id in retired_session_ids:
            self._close_retired_contextual_rag(session_id)

        seen: set[int] = set()
        for old_gateway in old_gateways:
            identity = id(old_gateway)
            if identity in seen:
                continue
            seen.add(identity)
            close = getattr(old_gateway, "close", None)
            if callable(close):
                try:
                    result = close()
                    if hasattr(result, "__await__"):
                        asyncio.run(result)
                except Exception:
                    logger.debug("Closing a retired provider gateway failed.")
        return self.start_readiness()

    def apply_model_selections(
        self,
        selections: Mapping[str, str | None],
    ) -> Future[ReadinessReport | None]:
        allowed_roles = {"embedding", "decision", "generation"}
        unknown_roles = set(selections).difference(allowed_roles)
        if unknown_roles:
            raise ValueError(f"Unknown Ollama model roles: {sorted(unknown_roles)}")

        with self._lock:
            updated = dict(self._model_overrides)
            report = self.readiness
            catalog = {
                role: {
                    profile.model
                    for profile in report.profiles
                    if profile.role == role
                }
                for role in allowed_roles
            } if report is not None else {}
            required_models = self.required_models

        for role in allowed_roles:
            if role not in selections:
                continue
            model = selections[role]
            if model is None:
                updated.pop(role, None)
                continue
            normalized = model.strip()
            if not normalized:
                raise ValueError(f"A model is required for the {role} role.")
            if any(
                marker in normalized.lower()
                for marker in (":cloud", "-cloud", "cloud:", ".cloud")
            ):
                raise ValueError("Cloud-hosted model variants are not supported.")
            if role in catalog and catalog[role] and normalized not in catalog[role]:
                raise ValueError(
                    f"Model {normalized} is not a compatible installed {role} model."
                )
            updated[role] = normalized

        base_models = (
            required_models.as_mapping()
            if required_models is not None
            else {}
        )
        base_models.update(updated)
        selected = (
            RequiredModels.model_validate(base_models)
            if set(base_models) == allowed_roles
            else None
        )
        self._save_model_overrides(updated)
        with self._lock:
            self._provider_generation += 1
            self._model_overrides = updated
            self.required_models = selected
            if selected is not None and self._model_readiness is not None:
                self._model_readiness = self._model_readiness.model_copy(
                    update={"selected_models": selected}
                )
            self._indexing_service = None
            self._indexing_service_generation = 0
            self._embedding_gateway = None
        return self.start_readiness()

    def _initialize_readiness(
        self,
        generation: int,
        provider_generation: int,
        completion: Future[ReadinessReport | None],
        on_complete: ReadinessCallback | None,
    ) -> None:
        report: ReadinessReport | None = None
        gateway: Any | None = None
        close_gateway = True
        with self._lock:
            provider = self._active_provider
            gemini_api_key = self._gemini_api_key
        try:
            if provider == "gemini":
                required = RequiredModels(
                    embedding=GEMINI_EMBEDDING_MODEL,
                    decision=GEMINI_DECISION_MODEL,
                    generation=GEMINI_GENERATION_MODEL,
                )
                if gemini_api_key:
                    try:
                        gateway = GeminiGateway(gemini_api_key)
                    except Exception:
                        gateway = None
                if self.session_factory is not None:
                    with self.session_factory.begin() as session:
                        report = asyncio.run(
                            GeminiReadinessService(
                                gateway,
                                gemini_api_key,
                                session=session,
                            ).check(required)
                        )
                else:
                    report = asyncio.run(
                        GeminiReadinessService(gateway, gemini_api_key).check(required)
                    )
            else:
                gateway = OllamaGateway()
                hardware = profile_hardware()
                with self._lock:
                    overrides = dict(self._model_overrides)
                try:
                    recommendation = recommend_models(
                        hardware,
                        embedding_override=overrides.get("embedding"),
                        decision_override=overrides.get("decision"),
                        generation_override=overrides.get("generation"),
                    )
                except ModelRecommendationError:
                    recommendation = recommend_models(hardware)
                recommended = _required_models(recommendation).as_mapping()
                recommended.update(overrides)
                required = RequiredModels.model_validate(recommended)
                report = asyncio.run(
                    self._check_readiness(gateway, required)
                ).model_copy(
                    update={
                        "provider": "ollama",
                        "hardware": hardware,
                        "recommendation": recommendation,
                        "selected_models": required,
                    }
                )

            embedding_profile = report.selected_profile("embedding")
            if (
                "indexing" in report.blocked_roles
                or embedding_profile is None
                or embedding_profile.dimensions not in {256, 512, 768}
            ):
                embedding_profile = None
            indexing_service = None
            embedding_gateway: Any | None = None
            generation_profile = report.selected_profile("generation")
            generation_gateway: Any | None = None
            if report.ready:
                if provider == "gemini":
                    if gemini_api_key:
                        try:
                            generation_gateway = GeminiGateway(gemini_api_key)
                        except Exception:
                            generation_gateway = None
                else:
                    try:
                        generation_gateway = OllamaGateway()
                    except Exception:
                        generation_gateway = None
            if embedding_profile is not None and self.session_factory is not None:
                if provider == "gemini":
                    embedding_gateway = _FreshGeminiEmbeddingGateway(
                        gemini_api_key or ""
                    )
                else:
                    embedding_gateway = _FreshOllamaEmbeddingGateway()
                indexing_service = IndexingService(
                    IndexingRepository(self.session_factory),
                    DoclingAdapter(
                        vision_gateway=generation_gateway,
                        vision_profile=generation_profile,
                    ),
                    embedding_gateway=embedding_gateway,
                    embedding_profile=embedding_profile,
                    recover_on_startup=False,
                )
            candidate_gateway: Any | None = None
            decision_gateway: Any | None = None
            if report.ready:
                if provider == "gemini":
                    if gemini_api_key:
                        try:
                            candidate_gateway = GeminiGateway(gemini_api_key)
                            decision_gateway = GeminiGateway(gemini_api_key)
                        except Exception:
                            candidate_gateway = None
                            decision_gateway = None
                else:
                    try:
                        candidate_gateway = OllamaGateway()
                        decision_gateway = OllamaGateway()
                    except Exception:
                        candidate_gateway = None
                        decision_gateway = None

            published_service = False
            with self._lock:
                if (
                    generation == self._readiness_generation
                    and provider_generation == self._provider_generation
                    and provider == self._active_provider
                ):
                    if (
                        embedding_profile is not None
                        and self._indexing_repository is not None
                    ):
                        try:
                            self._indexing_repository.queue_vector_reindex(
                                embedding_profile
                            )
                        except AttributeError:
                            pass
                    self.required_models = required
                    self.readiness = report
                    self._indexing_service = indexing_service
                    self._indexing_service_generation = generation
                    self._embedding_gateway = embedding_gateway
                    self._candidate_gateway = candidate_gateway
                    self._decision_gateway = decision_gateway
                    self._generation_gateway = generation_gateway
                    self._unavailable_reason = None
                    published_service = indexing_service is not None
                else:
                    report = None
            if published_service and indexing_service is not None:
                self._recover_published_indexing_service(
                    generation,
                    indexing_service,
                    provider_generation,
                )
            with self._lock:
                if generation == self._readiness_generation:
                    report = self.readiness
            logger.info(
                "Modern %s runtime readiness initialized (ready=%s, profiles=%d).",
                provider,
                report.ready if report is not None else False,
                len(report.profiles) if report is not None else 0,
            )
        except Exception as exc:
            report = None
            with self._lock:
                if generation == self._readiness_generation:
                    self._unavailable_reason = (
                        f"{exc.__class__.__name__}: modern readiness is unavailable."
                    )
            logger.warning(
                "Modern local-model readiness is unavailable (%s).",
                exc.__class__.__name__,
            )
        finally:
            if close_gateway and gateway is not None:
                try:
                    close = getattr(gateway, "close", None)
                    if callable(close):
                        asyncio.run(close())
                except Exception:
                    logger.debug("Closing the provider readiness client failed.")
            if not completion.done():
                completion.set_result(report)
            self._notify_readiness_complete(on_complete, report)

    @staticmethod
    def _notify_readiness_complete(
        callback: ReadinessCallback | None,
        report: ReadinessReport | None,
    ) -> None:
        if callback is None:
            return
        try:
            callback(report)
        except Exception:
            logger.warning("A readiness completion callback failed.", exc_info=True)

    def _recover_published_indexing_service(
        self,
        generation: int,
        service: IndexingService,
        provider_generation: int | None = None,
    ) -> None:
        def current() -> bool:
            with self._lock:
                return (
                    generation == self._readiness_generation
                    and (
                        provider_generation is None
                        or provider_generation == self._provider_generation
                    )
                    and self._indexing_service is service
                    and self._indexing_service_generation == generation
                )

        if not current():
            return
        try:
            service.recover_abandoned(claim_guard=current)
        except Exception:
            logger.warning(
                "Initial abandoned-indexing recovery failed.",
                exc_info=True,
            )
        finally:
            self._start_indexing_recovery_scheduler()

    def _start_indexing_recovery_scheduler(self) -> None:
        with self._lock:
            service = self._indexing_service
            next_expiry = getattr(
                getattr(service, "repository", None),
                "seconds_until_next_lease_expiry",
                None,
            )
            existing = self._index_recovery_thread
            if (
                not callable(next_expiry)
                or (existing is not None and existing.is_alive())
            ):
                return
            thread = threading.Thread(
                target=self._run_indexing_recovery_scheduler,
                name="file-tagger-indexing-recovery",
                daemon=True,
            )
            self._index_recovery_thread = thread
            thread.start()

    def _run_indexing_recovery_scheduler(self) -> None:
        while True:
            with self._lock:
                service = self._indexing_service
                generation = self._readiness_generation
                published_generation = self._indexing_service_generation
                provider_generation = self._provider_generation
            if service is None:
                threading.Event().wait(_INDEX_RECOVERY_MAX_WAIT_SECONDS)
                continue
            repository = service.repository
            next_expiry = repository.seconds_until_next_lease_expiry
            try:
                delay = next_expiry()
            except Exception:
                logger.warning(
                    "Modern indexing recovery scheduling failed.",
                    exc_info=True,
                )
                delay = None
            if delay is None:
                delay = _INDEX_RECOVERY_MAX_WAIT_SECONDS
            else:
                delay = min(delay, _INDEX_RECOVERY_MAX_WAIT_SECONDS)
            if delay > 0:
                threading.Event().wait(delay)

            def current() -> bool:
                with self._lock:
                    return (
                        generation == self._readiness_generation
                        and provider_generation == self._provider_generation
                        and self._indexing_service is service
                        and published_generation == self._indexing_service_generation
                        and generation == published_generation
                    )

            if not current():
                continue
            try:
                service.recover_abandoned(claim_guard=current)
            except Exception:
                logger.warning(
                    "Modern indexing abandoned-lease recovery failed.",
                    exc_info=True,
                )

    @property
    def _model_overrides_path(self) -> Path:
        return self.paths.config_dir / "ollama-models.json"

    def _load_model_overrides(self) -> dict[str, str]:
        path = self._model_overrides_path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            models = payload.get("models", payload)
        except (OSError, AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return {}
        if not isinstance(models, dict):
            return {}
        allowed_roles = {"embedding", "decision", "generation"}
        loaded: dict[str, str] = {}
        for role, model in models.items():
            if role not in allowed_roles or not isinstance(model, str) or not model:
                continue
            if any(
                marker in model.lower()
                for marker in (":cloud", "-cloud", "cloud:", ".cloud")
            ):
                continue
            loaded[role] = model
        return loaded

    def _save_model_overrides(self, models: Mapping[str, str]) -> None:
        self.paths.config_dir.mkdir(parents=True, exist_ok=True)
        path = self._model_overrides_path
        replacement = path.with_suffix(f"{path.suffix}.tmp")
        replacement.write_text(
            json.dumps(
                {"version": 1, "models": dict(models)},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(replacement, path)

    async def _check_readiness(
        self,
        gateway: OllamaGateway,
        required: RequiredModels,
    ) -> ReadinessReport:
        if self.session_factory is None:
            raise ModernIndexingUnavailable("The modern database is unavailable.")
        with self.session_factory.begin() as session:
            return await OllamaReadinessService(gateway, session=session).check(required)

    def index_file(self, file_path: str | Path) -> IndexingResult:
        service = self._require_indexing_service()
        return service.reindex(file_path)

    def reindex_all(self) -> tuple[IndexingResult, ...]:
        """Rebuild every known file through the resumable indexing service."""
        factory = self.session_factory
        if factory is None:
            raise ModernIndexingUnavailable(
                self.unavailable_reason or "The modern database is unavailable."
            )
        service = self._require_indexing_service()
        with factory() as session:
            file_paths = tuple(
                path
                for path in session.scalars(
                    select(File.canonical_path).order_by(File.id)
                )
                if path
            )
        return tuple(service.reindex(file_path) for file_path in file_paths)

    def queue(self, file_path: str | Path) -> int:
        return self._require_indexing_service().queue(file_path)

    def queue_many(
        self,
        file_paths: Iterable[str | Path],
    ) -> tuple[int, ...]:
        return self._require_indexing_service().queue_many(file_paths)

    def resume(
        self,
        job_id: int,
        *,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> IndexingResult:
        return self._require_indexing_service().resume(
            job_id,
            cancellation_check=cancellation_check,
        )

    def cancel(self, job_id: int) -> IndexingResult:
        return self._require_indexing_service().cancel(job_id)

    def state(self, job_id: int) -> IndexingResult:
        service = self._indexing_service
        if service is not None:
            return service.state(job_id)
        repository = self._indexing_repository
        if repository is None:
            raise ModernIndexingUnavailable("The modern database is unavailable.")
        return IndexingResult.from_state(repository.state(job_id))

    def latest_resumable_job(self) -> IndexingResult | None:
        repository = self._indexing_repository
        if repository is None:
            return None
        state = repository.latest_resumable_job()
        return IndexingResult.from_state(state) if state is not None else None

    def resumable_jobs(self) -> tuple[IndexingResult, ...]:
        repository = self._indexing_repository
        if repository is None:
            return ()
        return tuple(
            IndexingResult.from_state(state)
            for state in repository.resumable_jobs()
        )

    def _require_indexing_service(self) -> IndexingService:
        with self._lock:
            service = self._indexing_service
        reason = self.unavailable_reason
        if service is None:
            raise ModernIndexingUnavailable(reason)
        return service

    def search_content(
        self,
        query: str,
        *,
        tags: Iterable[str] = (),
        use_and: bool = True,
        limit: int = 20,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> tuple[SearchResult, ...]:
        if not isinstance(query, str) or not query.strip():
            return ()
        if cancellation_check is not None and cancellation_check():
            raise asyncio.CancelledError
        with self._lock:
            gateway = self._embedding_gateway
            factory = self.session_factory
        profile = self.embedding_profile
        if profile is None or gateway is None or factory is None:
            raise ModernSearchUnavailable(self.unavailable_reason)
        repository = self._indexing_repository
        if repository is None:
            raise ModernSearchUnavailable("The modern database is unavailable.")
        try:
            repository.queue_vector_reindex(profile)
            readiness = repository.vector_readiness(profile)
        except AttributeError:
            if callable(getattr(self.session_factory, "begin", None)):
                raise ModernSearchUnavailable(
                    "the committed vector index readiness could not be verified."
                )
            readiness = None
        except Exception as exc:
            raise ModernSearchUnavailable(
                "the committed vector index readiness could not be verified."
            ) from exc
        if readiness is not None and not readiness.ready:
            raise ModernSearchUnavailable(
                "the active embedding profile requires a full reindex "
                f"({readiness.pending_files} file(s) pending)."
            )

        selected_tags = tuple(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))
        tag_groups = (
            (selected_tags,)
            if use_and or not selected_tags
            else tuple((tag,) for tag in selected_tags)
        )
        session = factory()
        try:
            search = HybridSearch(session, embedding_profile=profile)
            matches: list[SearchResult] = []
            per_query_limit = max(limit, 20)
            for tag_group in tag_groups:
                plan = SearchPlan(
                    question=query,
                    intent_names=("content_search",),
                    exact_terms=(query,),
                    paraphrase_queries=(query,),
                    filters=tuple(f"tag:{tag}" for tag in tag_group),
                    evidence_categories=("document_content",),
                )
                search_kwargs: dict[str, Any] = {
                    "embedding_gateway": gateway,
                    "embedding_profile": profile,
                    "model_digest": profile.digest,
                    "index_version": profile.default_index_version,
                    "per_query_limit": per_query_limit,
                    "limit": limit,
                    "max_sections_per_file": 1,
                }
                if cancellation_check is not None:
                    search_kwargs["cancellation_check"] = cancellation_check
                matches.extend(
                    execute_search_plan(
                        plan,
                        search,
                        **search_kwargs,
                    )
                )
                if cancellation_check is not None and cancellation_check():
                    raise asyncio.CancelledError
            return merge_results(matches, limit=limit)
        finally:
            session.close()

    def _get_contextual_rag_service(
        self,
        conversation_id: str | None,
        *,
        profile: ModelProfile,
        generation_profile: ModelProfile,
        expected_epoch: int | None = None,
    ) -> tuple[str, RagService]:
        checked_id = conversation_id or "default"
        key = "\0".join(
            (
                str(id(self.session_factory)),
                profile.digest,
                profile.default_index_version or "",
                generation_profile.model,
                generation_profile.digest,
            )
        )
        retired: list[tuple[str, int, RagService, Any]] = []
        with self._contextual_rag_condition:
            if (
                expected_epoch is not None
                and expected_epoch
                != self._contextual_rag_epochs.get(checked_id, 0)
            ):
                raise ModernSearchUnavailable(
                    "the contextual RAG session was retired."
                )
            if checked_id in self._contextual_rag_closed:
                raise ModernSearchUnavailable(
                    "the contextual RAG session was retired."
                )
            cached = self._contextual_rag_services.get(checked_id)
            if cached is not None and cached[0] == key:
                self._contextual_rag_services.move_to_end(checked_id)
                return checked_id, cached[1]
            if cached is not None:
                old_epoch = self._contextual_rag_epochs.get(checked_id, 0)
                retired.append((checked_id, old_epoch, cached[1], cached[2]))
                del self._contextual_rag_services[checked_id]
                self._contextual_rag_epochs[checked_id] = (
                    self._contextual_rag_epochs.get(checked_id, 0) + 1
                )

        if self.session_factory is None:
            raise ModernSearchUnavailable(self.unavailable_reason)
        session = self.session_factory()
        search = HybridSearch(
            session,
            model_digest=profile.digest,
            index_version=profile.default_index_version,
            embedding_profile=profile,
        )
        with self._lock:
            gateway: Any = self._generation_gateway
            embedding_gateway: Any = self._embedding_gateway
            provider = self._active_provider
            gemini_api_key = self._gemini_api_key
        if gateway is None:
            if provider == "gemini":
                if not gemini_api_key:
                    raise ModernSearchUnavailable(
                        "The Gemini generation gateway is unavailable."
                    )
                gateway = GeminiGateway(gemini_api_key)
            else:
                gateway = OllamaGateway()
        if embedding_gateway is None:
            if provider == "gemini":
                if not gemini_api_key:
                    raise ModernSearchUnavailable(
                        "The Gemini embedding gateway is unavailable."
                    )
                embedding_gateway = _FreshGeminiEmbeddingGateway(gemini_api_key)
            else:
                embedding_gateway = _FreshOllamaEmbeddingGateway()
        rag = RagService(
            search=search,
            gateway=gateway,
            embedding_gateway=embedding_gateway,
            model=generation_profile.model,
            model_digest=generation_profile.digest,
            index_version=profile.default_index_version,
            embedding_profile=profile,
        )

        with self._contextual_rag_condition:
            if checked_id in self._contextual_rag_closed:
                close_rag = getattr(rag, "close_session", None)
                if callable(close_rag):
                    try:
                        close_rag(checked_id)
                    except Exception:
                        pass
                try:
                    session.close()
                except Exception:
                    pass
                raise ModernSearchUnavailable(
                    "the contextual RAG session was retired."
                )
            self._contextual_rag_services[checked_id] = (key, rag, session)
            self._contextual_rag_services.move_to_end(checked_id)
            while len(self._contextual_rag_services) > self._contextual_rag_cache_limit:
                evicted_id: str | None = None
                evicted: tuple[str, RagService, Any] | None = None
                for candidate_id, candidate in self._contextual_rag_services.items():
                    if self._contextual_rag_active.get(candidate_id, 0) == 0:
                        evicted_id = candidate_id
                        evicted = (candidate_id, candidate[1], candidate[2])
                        break
                if evicted_id is not None:
                    assert evicted is not None
                    del self._contextual_rag_services[evicted_id]
                    retired.append(
                        (
                            evicted_id,
                            self._contextual_rag_epochs.get(evicted_id, 0),
                            evicted[1],
                            evicted[2],
                        )
                    )
                    continue
                self._contextual_rag_condition.wait(timeout=0.05)
            if retired:
                for retired_id, retired_epoch, retired_rag, retired_session in retired:
                    self._retired_contextual_rag.setdefault(
                        retired_id,
                        [],
                    ).append((retired_epoch, retired_rag, retired_session))
        if retired:
            for retired_id, _retired_epoch, _retired_rag, _retired_session in retired:
                if self._contextual_rag_active.get(retired_id, 0) == 0:
                    self._close_retired_contextual_rag(retired_id)
        return checked_id, rag

    def _begin_contextual_rag_use(
        self,
        conversation_id: str | None,
        *,
        profile: ModelProfile,
        generation_profile: ModelProfile,
    ) -> tuple[str, RagService, int]:
        checked_id = conversation_id or "default"
        with self._contextual_rag_condition:
            if checked_id in self._contextual_rag_closed:
                raise ModernSearchUnavailable(
                    "the contextual RAG session was retired."
                )
            self._contextual_rag_active[checked_id] = (
                self._contextual_rag_active.get(checked_id, 0) + 1
            )
            epoch = self._contextual_rag_epochs.get(checked_id, 0)
        try:
            checked_id, rag = self._get_contextual_rag_service(
                conversation_id,
                profile=profile,
                generation_profile=generation_profile,
                expected_epoch=epoch,
            )
        except Exception:
            is_idle = False
            with self._contextual_rag_condition:
                active = self._contextual_rag_active.get(checked_id, 0)
                if active <= 1:
                    self._contextual_rag_active.pop(checked_id, None)
                else:
                    self._contextual_rag_active[checked_id] = active - 1
                self._contextual_rag_condition.notify_all()
                is_idle = self._contextual_rag_active.get(checked_id, 0) == 0
            if is_idle:
                self._close_retired_contextual_rag(checked_id)
            raise
        with self._contextual_rag_condition:
            current_epoch = self._contextual_rag_epochs.get(checked_id, 0)
            if checked_id in self._contextual_rag_closed:
                active = self._contextual_rag_active.get(checked_id, 0)
                if active <= 1:
                    self._contextual_rag_active.pop(checked_id, None)
                else:
                    self._contextual_rag_active[checked_id] = active - 1
                self._contextual_rag_condition.notify_all()
                is_idle = self._contextual_rag_active.get(checked_id, 0) == 0
            else:
                is_idle = False
        if current_epoch != epoch:
            if is_idle:
                self._close_retired_contextual_rag(checked_id)
            raise ModernSearchUnavailable(
                "the contextual RAG session was retired."
            )
        return checked_id, rag, epoch

    def _end_contextual_rag_use(self, session_id: str, epoch: int) -> None:
        del epoch
        with self._contextual_rag_condition:
            active = self._contextual_rag_active.get(session_id, 0)
            if active <= 1:
                self._contextual_rag_active.pop(session_id, None)
            else:
                self._contextual_rag_active[session_id] = active - 1
            self._contextual_rag_condition.notify_all()
            is_idle = self._contextual_rag_active.get(session_id, 0) == 0
        if is_idle:
            self._close_retired_contextual_rag(session_id)

    def _close_retired_contextual_rag(
        self,
        session_id: str,
    ) -> None:
        with self._contextual_rag_condition:
            if self._contextual_rag_active.get(session_id, 0) != 0:
                return
            retired = self._retired_contextual_rag.pop(session_id, [])
        for _retired_epoch, rag, session in retired:
            close_rag = getattr(rag, "close_session", None)
            if callable(close_rag):
                try:
                    close_rag(session_id)
                except Exception:
                    pass
            try:
                session.close()
            except Exception:
                pass

    def close_contextual_session(
        self,
        session_id: str,
        timeout: float = 2.0,
    ) -> None:
        del timeout
        checked_id = session_id or "default"
        retired: tuple[int, RagService, Any] | None = None
        with self._contextual_rag_condition:
            old_epoch = self._contextual_rag_epochs.get(checked_id, 0)
            self._contextual_rag_epochs[checked_id] = old_epoch + 1
            self._contextual_rag_closed.add(checked_id)
            cached = self._contextual_rag_services.pop(checked_id, None)
            if cached is not None:
                retired = (old_epoch, cached[1], cached[2])
                self._retired_contextual_rag.setdefault(
                    checked_id,
                    [],
                ).append(retired)
                self._contextual_rag_condition.notify_all()
        if retired is not None:
            self._close_retired_contextual_rag(checked_id)

    def get_session_answer(self, session_id: str) -> CitedAnswer | None:
        checked_id = session_id or "default"
        with self._contextual_rag_lock:
            cached = self._contextual_rag_services.get(checked_id)
        if cached is None:
            return None
        return cached[1].get_session_answer(checked_id)

    async def answer_contextual_question_stream_async(
        self,
        question: str,
        markdown_context: str = "",
        *,
        conversation_id: str | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> AsyncIterator[str]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A nonempty search question is required.")
        profile = self.embedding_profile
        readiness = self.readiness
        generation_profile = (
            readiness.selected_profile("generation")
            if readiness is not None
            else None
        )
        if (
            readiness is None
            or not readiness.ready
            or profile is None
            or self.session_factory is None
            or generation_profile is None
        ):
            raise ModernSearchUnavailable(self.unavailable_reason)

        repository = self._indexing_repository
        if repository is not None:
            try:
                repository.queue_vector_reindex(profile)
                vector_ready = repository.vector_readiness(profile)
            except Exception as exc:
                raise ModernSearchUnavailable(
                    "the committed vector index readiness could not be verified."
                ) from exc
            if not vector_ready.ready:
                raise ModernSearchUnavailable(
                    "the active embedding profile requires a full reindex "
                    f"({vector_ready.pending_files} file(s) pending)."
                )

        checked_id, rag, epoch = self._begin_contextual_rag_use(
            conversation_id,
            profile=profile,
            generation_profile=generation_profile,
        )
        try:
            async for fragment in rag.answer_question_stream_async(
                question,
                markdown_context,
                conversation_id=checked_id,
                cancellation_check=cancellation_check,
            ):
                yield fragment
        finally:
            self._end_contextual_rag_use(checked_id, epoch)

    def answer_contextual_question(
        self,
        question: str,
        markdown_context: str = "",
        *,
        conversation_id: str | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> Any:
        """Answer a question using the shared hybrid search and RAG services."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A nonempty search question is required.")
        with self._lock:
            factory = self.session_factory
        profile = self.embedding_profile
        readiness = self.readiness
        generation_profile = (
            readiness.selected_profile("generation")
            if readiness is not None
            else None
        )
        if (
            readiness is None
            or not readiness.ready
            or profile is None
            or factory is None
            or generation_profile is None
        ):
            raise ModernSearchUnavailable(self.unavailable_reason)

        repository = self._indexing_repository
        if repository is not None:
            try:
                repository.queue_vector_reindex(profile)
                vector_ready = repository.vector_readiness(profile)
            except Exception as exc:
                raise ModernSearchUnavailable(
                    "the committed vector index readiness could not be verified."
                ) from exc
            if not vector_ready.ready:
                raise ModernSearchUnavailable(
                    "the active embedding profile requires a full reindex "
                    f"({vector_ready.pending_files} file(s) pending)."
                )

        checked_id, rag, epoch = self._begin_contextual_rag_use(
            conversation_id,
            profile=profile,
            generation_profile=generation_profile,
        )

        try:
            async def collect() -> CitedAnswer | None:
                async for _fragment in rag.answer_question_stream_async(
                    question,
                    markdown_context,
                    conversation_id=checked_id,
                    cancellation_check=cancellation_check,
                ):
                    pass
                return rag.get_session_answer(checked_id)

            answer = asyncio.run(collect())
        finally:
            self._end_contextual_rag_use(checked_id, epoch)
        if answer is None:
            raise ModernSearchUnavailable("the contextual RAG answer could not be produced.")
        return answer

    def index_file_async(self, file_path: str | Path) -> None:
        canonical_path = _canonical_path(file_path)
        with self._lock:
            service = self._indexing_service
            reason = self.unavailable_reason
        if service is None:
            raise ModernIndexingUnavailable(reason)
        with self._lock:
            request = _IndexRequest(
                canonical_path,
                self._index_generations.get(canonical_path, 0),
                self._active_provider,
                self._provider_generation,
                self._indexing_service_generation,
            )
            self._index_queue.put(request)
            if self._index_worker is None or not self._index_worker.is_alive():
                self._index_worker = threading.Thread(
                    target=self._run_index_queue,
                    name="file-tagger-modern-indexing",
                    daemon=True,
                )
                self._index_worker.start()

    def _run_index_queue(self) -> None:
        while True:
            request = self._index_queue.get()
            try:
                result = self._index_request(request)
                if result is None:
                    continue
                if result.status != "completed":
                    logger.warning(
                        "Modern indexing ended with status %s.",
                        result.status,
                    )
            except Exception as exc:
                logger.warning(
                    "Modern indexing worker failed (%s).",
                    exc.__class__.__name__,
                )
            finally:
                self._index_queue.task_done()

    def _index_request(self, request: _IndexRequest) -> IndexingResult | None:
        with self._lock:
            if (
                self._index_generations.get(request.file_path, 0)
                != request.generation
                or self._active_provider != request.provider
                or self._provider_generation != request.provider_generation
                or self._indexing_service_generation
                != request.indexing_service_generation
                or self.session_factory is None
            ):
                return None
            service = self._indexing_service
            if service is None:
                return None
            with self.session_factory() as session:
                file_exists = session.scalar(
                    select(File.id).where(
                        File.canonical_path == request.file_path
                    )
                )
            if file_exists is None:
                return None
            job_id = service.queue(request.file_path)
        with self._lock:
            if (
                self._index_generations.get(request.file_path, 0)
                != request.generation
                or self._active_provider != request.provider
                or self._provider_generation != request.provider_generation
                or self._indexing_service_generation
                != request.indexing_service_generation
                or self.session_factory is None
            ):
                return None
            with self.session_factory() as session:
                file_still_exists = session.scalar(
                    select(File.id).where(
                        File.canonical_path == request.file_path
                    )
                )
            if file_still_exists is None:
                return None
        return service.reindex(
            request.file_path,
            job_id=job_id,
            cancellation_check=lambda: self._index_request_cancelled(request),
        )

    def _index_request_cancelled(self, request: _IndexRequest) -> bool:
        with self._lock:
            if (
                self._index_generations.get(request.file_path, 0)
                != request.generation
                or self._active_provider != request.provider
                or self._provider_generation != request.provider_generation
                or self._indexing_service_generation
                != request.indexing_service_generation
                or self.session_factory is None
            ):
                return True
            with self.session_factory() as session:
                file_exists = session.scalar(
                    select(File.id).where(
                        File.canonical_path == request.file_path
                    )
                )
            return file_exists is None

    def _invalidate_index_paths_locked(self, file_paths: Iterable[str]) -> None:
        invalidated = set(file_paths)
        for file_path in invalidated:
            self._index_generations[file_path] = (
                self._index_generations.get(file_path, 0) + 1
            )
        with self._index_queue.mutex:
            pending = self._index_queue.queue
            retained = deque(
                request
                for request in pending
                if request.file_path not in invalidated
            )
            removed = len(pending) - len(retained)
            if removed:
                self._index_queue.queue = retained
                self._index_queue.unfinished_tasks -= removed
                if self._index_queue.unfinished_tasks == 0:
                    self._index_queue.all_tasks_done.notify_all()

    def sync_tag(self, name: str, color: str | None) -> None:
        if self.session_factory is None:
            return
        with self.session_factory.begin() as session:
            tag = _resolve_tag(session, name)
            if tag is None:
                tag = Tag(name=name, color=color or "#808080")
                session.add(tag)
            else:
                tag.color = color or "#808080"

    def delete_tag(self, name: str) -> None:
        if self.session_factory is None:
            return
        with self.session_factory.begin() as session:
            tag = _resolve_tag(session, name)
            if tag is not None:
                session.delete(tag)

    def sync_file(self, file_path: str, tags: Iterable[TagData]) -> None:
        if self.session_factory is None:
            return
        canonical_path = _canonical_path(file_path)
        normalized_tags = list(tags)
        with self.session_factory.begin() as session:
            file = session.scalar(
                select(File).where(File.canonical_path == canonical_path)
            )
            if file is None:
                file = File(
                    canonical_path=canonical_path,
                    indexing_state="not_indexed",
                )
                session.add(file)
                session.flush()

            selected: dict[str, Tag] = {}
            for name, color in normalized_tags:
                tag = _resolve_tag(session, name)
                if tag is None:
                    tag = Tag(name=name, color=color or "#808080")
                    session.add(tag)
                    session.flush()
                else:
                    tag.color = color or "#808080"
                key = normalize_tag_key(tag.name)
                if key is not None:
                    selected[key] = tag

            associations = {
                association.tag_id: association
                for association in session.scalars(
                    select(FileTag).where(FileTag.file_id == file.id)
                )
            }
            selected_ids = {int(tag.id) for tag in selected.values()}
            for tag_id, association in associations.items():
                if tag_id not in selected_ids:
                    session.delete(association)
            for tag in selected.values():
                if int(tag.id) not in associations:
                    session.add(FileTag(file_id=int(file.id), tag_id=int(tag.id)))

    def delete_file(self, file_path: str) -> None:
        if self.session_factory is None:
            return
        canonical_path = _canonical_path(file_path)
        with self._lock:
            self._invalidate_index_paths_locked((canonical_path,))
            with self.session_factory.begin() as session:
                file = session.scalar(
                    select(File).where(
                        File.canonical_path == canonical_path
                    )
                )
                if file is not None:
                    delete_file_embeddings(session, (int(file.id),))
                    session.delete(file)

    def sync_legacy_changes(
        self,
        *,
        tag_upserts: Iterable[TagData] = (),
        tag_deletes: Iterable[str] = (),
        file_paths: Iterable[tuple[str, Iterable[TagData]]] = (),
        file_deletes: Iterable[str] = (),
    ) -> None:
        """Mirror one successful legacy transaction as one modern transaction."""
        if self.session_factory is None:
            return
        files = [(path, list(tags)) for path, tags in file_paths]
        deleted_files = list(file_deletes)
        deleted_paths = {_canonical_path(path) for path in deleted_files}
        deleted_tags = list(tag_deletes)
        updated_tags = list(tag_upserts)
        with self._lock:
            self._invalidate_index_paths_locked(deleted_paths)
            with self.session_factory.begin() as session:
                files_to_delete: dict[int, File] = {}
                for canonical_path in deleted_paths:
                    file = session.scalar(
                        select(File).where(
                            File.canonical_path == canonical_path
                        )
                    )
                    if file is not None:
                        files_to_delete[int(file.id)] = file
                delete_file_embeddings(session, files_to_delete)
                for file in files_to_delete.values():
                    session.delete(file)
                for name in deleted_tags:
                    tag = _resolve_tag(session, name)
                    if tag is not None:
                        session.delete(tag)
                for name, color in updated_tags:
                    tag = _resolve_tag(session, name)
                    if tag is None:
                        tag = Tag(name=name, color=color or "#808080")
                        session.add(tag)
                    else:
                        tag.color = color or "#808080"
                session.flush()

                for path, tags in files:
                    canonical_path = _canonical_path(path)
                    file = session.scalar(
                        select(File).where(File.canonical_path == canonical_path)
                    )
                    if file is None:
                        file = File(
                            canonical_path=canonical_path,
                            indexing_state="not_indexed",
                        )
                        session.add(file)
                        session.flush()

                    selected: dict[str, Tag] = {}
                    for name, color in tags:
                        tag = _resolve_tag(session, name)
                        if tag is None:
                            tag = Tag(name=name, color=color or "#808080")
                            session.add(tag)
                            session.flush()
                        else:
                            tag.color = color or "#808080"
                        key = normalize_tag_key(tag.name)
                        if key is not None:
                            selected[key] = tag

                    associations = {
                        association.tag_id: association
                        for association in session.scalars(
                            select(FileTag).where(FileTag.file_id == file.id)
                        )
                    }
                    selected_ids = {int(tag.id) for tag in selected.values()}
                    for tag_id, association in associations.items():
                        if tag_id not in selected_ids:
                            session.delete(association)
                    for tag in selected.values():
                        if int(tag.id) not in associations:
                            session.add(
                                FileTag(file_id=int(file.id), tag_id=int(tag.id))
                            )


def _required_models(recommendation: ModelRecommendation) -> RequiredModels:
    return RequiredModels(
        embedding=recommendation.embedding.model,
        decision=recommendation.decision.model,
        generation=recommendation.generation.model,
    )


def initialize_modern_runtime(
    paths: AppPaths,
    config: StartupConfig,
    password: str,
    *,
    start_readiness: bool = True,
) -> ModernRuntime:
    """Upgrade and migrate the modern sidecar while preserving legacy startup."""
    paths.ensure_directories()
    backup_dir = paths.data_dir / "migration-backups"
    sqlite_report: LegacyMigrationReport | None = None
    config_report: ConfigMigrationReport | None = None

    try:
        upgrade_database(paths.database)
        session_factory = create_session_factory(paths.database)
    except Exception as exc:
        logger.warning(
            "Modern database initialization is unavailable (%s).",
            exc.__class__.__name__,
        )
        return ModernRuntime(
            paths,
            None,
            unavailable_reason=(
                f"{exc.__class__.__name__}: modern database is unavailable: {exc}"
            ),
            config=config,
            password=password,
        )

    try:
        config_report = migrate_legacy_config(
            paths.legacy_database.parent,
            paths,
            password,
            backup_dir=backup_dir / "config",
        )
        if not config_report.skipped:
            config.reload_after_migration()
    except Exception as exc:
        logger.warning(
            "Legacy configuration migration was not applied (%s).",
            exc.__class__.__name__,
        )
        if (paths.legacy_database.parent / "config.encrypted").is_file():
            raise

    legacy_database = paths.legacy_database.expanduser().resolve()
    modern_database = paths.database.expanduser().resolve()
    if legacy_database.is_file() and legacy_database != modern_database:
        try:
            with session_factory() as session:
                sqlite_report = migrate_legacy_sqlite(
                    legacy_database,
                    session,
                    backup_dir=backup_dir / "sqlite",
                )
                session.commit()
        except Exception as exc:
            logger.warning(
                "Legacy database migration was not applied (%s).",
                exc.__class__.__name__,
            )

    runtime = ModernRuntime(
        paths,
        session_factory,
        sqlite_migration=sqlite_report,
        config_migration=config_report,
        config=config,
        password=password,
    )
    if start_readiness:
        runtime.start_readiness()
    return runtime
