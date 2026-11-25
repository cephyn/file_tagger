from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Iterable, Mapping
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

from .adapters.ollama.gateway import OllamaGateway
from .extraction.docling_adapter import DoclingAdapter
from .indexing.repository import IndexingRepository
from .indexing.service import IndexingResult, IndexingService
from .infrastructure.app_paths import AppPaths
from .infrastructure.database import create_session_factory, upgrade_database
from .migration.config_migration import ConfigMigrationReport, migrate_legacy_config
from .migration.sqlite_migration import LegacyMigrationReport, migrate_legacy_sqlite
from .ollama.hardware import profile_hardware
from .domain.contracts import ModelProfile
from .domain.errors import ModelRecommendationError
from .ollama.model_recommender import ModelRecommendation, recommend_models
from .ollama.readiness import OllamaReadinessService, ReadinessReport, RequiredModels
from .persistence.models import File, FileTag, Tag
from .persistence.vector_cleanup import delete_file_embeddings


logger = logging.getLogger("ModernRuntime")
SessionFactory: TypeAlias = sessionmaker[Session]
TagData = tuple[str, str | None]
ReadinessCallback = Callable[[ReadinessReport | None], None]


@dataclass(frozen=True, slots=True)
class _IndexRequest:
    file_path: str
    generation: int


class ModernIndexingUnavailable(RuntimeError):
    """Raised when a modern indexing profile has not become available."""


class StartupConfig(Protocol):
    paths: AppPaths

    def reload_after_migration(self) -> None: ...


def _canonical_path(file_path: str | Path) -> str:
    path = Path(file_path).expanduser().resolve()
    return os.path.normpath(str(path))


class _FreshOllamaEmbeddingGateway:
    """Keep each asyncio invocation on a gateway bound to that event loop."""

    async def embed(self, *args: Any, **kwargs: Any) -> Any:
        gateway = OllamaGateway()
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
    ) -> None:
        self.paths = paths
        self.session_factory = session_factory
        self.sqlite_migration = sqlite_migration
        self.config_migration = config_migration
        self.readiness: ReadinessReport | None = None
        self.required_models: RequiredModels | None = None
        self._indexing_service: IndexingService | None = None
        self._readiness_thread: threading.Thread | None = None
        self._index_queue: queue.Queue[_IndexRequest] = queue.Queue()
        self._index_generations: dict[str, int] = {}
        self._index_worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._unavailable_reason = unavailable_reason
        self._readiness_generation = 0
        self._model_overrides = self._load_model_overrides()

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
        thread = threading.Thread(
            target=self._initialize_readiness,
            args=(generation, completion, on_complete),
            name="file-tagger-ollama-readiness",
            daemon=True,
        )
        with self._lock:
            self._readiness_thread = thread
        thread.start()
        return completion

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
            self._model_overrides = updated
            self.required_models = selected
            self._indexing_service = None
        return self.start_readiness()

    def _initialize_readiness(
        self,
        generation: int,
        completion: Future[ReadinessReport | None],
        on_complete: ReadinessCallback | None,
    ) -> None:
        report: ReadinessReport | None = None
        gateway = OllamaGateway()
        try:
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
            if embedding_profile is not None and self.session_factory is not None:
                indexing_service = IndexingService(
                    IndexingRepository(self.session_factory),
                    DoclingAdapter(),
                    embedding_gateway=_FreshOllamaEmbeddingGateway(),
                    embedding_profile=embedding_profile,
                )
            with self._lock:
                if generation == self._readiness_generation:
                    self.required_models = required
                    self.readiness = report
                    self._indexing_service = indexing_service
                    self._unavailable_reason = None
                else:
                    report = None
            logger.info(
                "Modern local-model readiness initialized (ready=%s, profiles=%d).",
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
            try:
                asyncio.run(gateway.close())
            except Exception:
                logger.debug("Closing the Ollama readiness client failed.")
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
        with self._lock:
            service = self._indexing_service
            reason = self.unavailable_reason
        if service is None:
            raise ModernIndexingUnavailable(reason)
        return service.reindex(file_path)

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
            tag = session.scalar(select(Tag).where(Tag.name == name))
            if tag is None:
                tag = Tag(name=name, color=color or "#808080")
                session.add(tag)
            else:
                tag.color = color or "#808080"

    def delete_tag(self, name: str) -> None:
        if self.session_factory is None:
            return
        with self.session_factory.begin() as session:
            tag = session.scalar(select(Tag).where(Tag.name == name))
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
                tag = session.scalar(select(Tag).where(Tag.name == name))
                if tag is None:
                    tag = Tag(name=name, color=color or "#808080")
                    session.add(tag)
                    session.flush()
                else:
                    tag.color = color or "#808080"
                selected[name] = tag

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
                    tag = session.scalar(select(Tag).where(Tag.name == name))
                    if tag is not None:
                        session.delete(tag)
                for name, color in updated_tags:
                    tag = session.scalar(select(Tag).where(Tag.name == name))
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
                        tag = session.scalar(select(Tag).where(Tag.name == name))
                        if tag is None:
                            tag = Tag(name=name, color=color or "#808080")
                            session.add(tag)
                            session.flush()
                        else:
                            tag.color = color or "#808080"
                        selected[name] = tag

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
                f"{exc.__class__.__name__}: modern database is unavailable."
            ),
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
        if (
            (paths.legacy_database.parent / "config.encrypted").is_file()
            and not paths.config_file.is_file()
        ):
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
    )
    if start_readiness:
        runtime.start_readiness()
    return runtime
