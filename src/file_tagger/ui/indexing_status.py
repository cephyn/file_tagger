from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..modern_context import ModernIndexingUnavailable


class _IndexingWorker(QThread):
    result_ready = Signal(object)
    failure = Signal(object)

    def __init__(
        self,
        service: Any,
        job_id: int | None,
        file_path: str,
        *,
        cancellation_check: Callable[[], bool],
    ) -> None:
        super().__init__()
        self.service = service
        self.job_id = job_id
        self.file_path = file_path
        self.cancellation_check = cancellation_check

    def run(self) -> None:
        try:
            if self.job_id is not None and callable(
                getattr(self.service, "resume", None)
            ):
                self.result_ready.emit(
                    self.service.resume(
                        self.job_id,
                        cancellation_check=self.cancellation_check,
                    )
                )
                return
            index_file = getattr(self.service, "index_file", None)
            if not callable(index_file):
                raise RuntimeError(
                    "The modern indexing service cannot resume this file."
                )
            self.result_ready.emit(index_file(self.file_path))
        except Exception as exc:
            self.failure.emit(exc)


class IndexingStatus(QWidget):
    """Non-blocking progress controller for resumable modern indexing."""

    status_changed = Signal(str, int)
    finished = Signal(object)

    def __init__(self, service: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self._job_id: int | None = None
        self._worker: _IndexingWorker | None = None
        self._pending: deque[str] = deque()
        self._active_path = ""
        self._active_service: Any = service
        self._result_handled = False
        self._completed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.file_label = QLabel("Modern index: idle.")
        self.phase_label = QLabel("Idle")
        self.status_label = self.phase_label
        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        controls = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        self.resume_button = QPushButton("Resume")
        self.resume_button.clicked.connect(self.resume)
        self.resume_button.setEnabled(False)
        controls.addStretch()
        controls.addWidget(self.cancel_button)
        controls.addWidget(self.resume_button)

        layout.addWidget(self.file_label)
        layout.addWidget(self.phase_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.detail_label)
        layout.addLayout(controls)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._refresh_progress)

    @property
    def previous_revision_available(self) -> bool:
        return bool(getattr(self.service, "previous_revision_available", True))

    def _resolved_service(self) -> Any:
        indexing_service = getattr(self.service, "_indexing_service", None)
        return indexing_service if indexing_service is not None else self.service

    def start(self, file_path: str | Path) -> int | None:
        path = str(Path(file_path)) if not isinstance(file_path, str) else file_path
        if self._worker is not None and self._worker.isRunning():
            self._pending.append(path)
            return None
        self._active_path = path
        self._active_service = self._resolved_service()
        if self._active_service is None:
            raise ModernIndexingUnavailable(
                "the modern indexing service is not initialized."
            )
        setattr(self._active_service, "cancel_requested", False)
        setattr(self.service, "cancel_requested", False)

        queue_index = getattr(self._active_service, "queue", None)
        if callable(queue_index):
            job_id = int(queue_index(path))
            self._run_job(job_id)
            return job_id

        if callable(getattr(self._active_service, "index_file", None)):
            self._run_job(None)
            return None
        raise ModernIndexingUnavailable("Modern indexing is not available.")

    def start_many(self, file_paths: Iterable[str | Path]) -> None:
        self._pending.extend(str(path) for path in file_paths)
        if self._worker is not None and self._worker.isRunning():
            return
        self._start_next()

    def _start_next(self) -> None:
        while self._pending:
            self.start(self._pending.popleft())
            if self._worker is not None and self._worker.isRunning():
                return

    def _run_job(self, job_id: int | None) -> None:
        if self._worker is not None and self._worker.isRunning():
            raise RuntimeError("An indexing job is already running.")
        self._job_id = job_id
        self._result_handled = False
        self._completed = False
        self._set_progress(
            "Extracting",
            3,
            f"Preparing {self._active_path}.",
        )
        self.cancel_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self._worker = _IndexingWorker(
            self._active_service,
            job_id,
            self._active_path,
            cancellation_check=lambda: bool(
                getattr(self._active_service, "cancel_requested", False)
            ),
        )
        self._worker.result_ready.connect(self._complete)
        self._worker.failure.connect(self._fail)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()
        self._refresh_progress()
        self._timer.start()

    def _refresh_progress(self) -> None:
        if self._job_id is None:
            self._set_progress(
                "Indexing",
                max(5, self.progress_bar.value()),
                "The modern indexing service is working.",
            )
            return
        state_call = getattr(self._active_service, "state", None)
        if not callable(state_call):
            return
        try:
            self._render_state(state_call(self._job_id))
        except Exception:
            return

    def _render_state(self, state: Any) -> None:
        status = str(getattr(state, "status", "running"))
        total = max(0, int(getattr(state, "total_count", 0)))
        processed = max(0, int(getattr(state, "processed_count", 0)))
        if status == "running":
            if total == 0:
                self._set_progress(
                    "Extracting",
                    5,
                    "Extracting document text and structure.",
                )
            elif processed < total:
                percent = 10 + int((processed / total) * 75)
                self._set_progress(
                    f"Embedding {processed}/{total}",
                    percent,
                    "Creating local embeddings; this checkpoint is resumable.",
                )
            else:
                self._set_progress(
                    "Committing",
                    92,
                    "Atomically replacing the indexed revision.",
                )
        elif status == "completed":
            return
        elif status in {"failed", "queued", "pending"}:
            return

    def _complete(self, result: Any) -> None:
        if self._result_handled:
            return
        self._result_handled = True
        self._timer.stop()
        self.cancel_button.setEnabled(False)
        self.finished.emit(result)
        status = str(getattr(result, "status", "completed"))
        self._completed = status == "completed"
        if status == "completed":
            self._set_progress(
                "Completed",
                100,
                "The new revision is committed; the previous revision was replaced atomically.",
            )
            self.resume_button.setEnabled(False)
            return
        self._terminal_state(result)

    def _terminal_state(self, result: Any) -> None:
        self._timer.stop()
        self.cancel_button.setEnabled(False)
        failure = getattr(result, "failure", None)
        summary = (
            str(failure.get("summary"))
            if isinstance(failure, dict) and failure.get("summary")
            else "The indexing job can be resumed from its latest checkpoint."
        )
        status = str(getattr(result, "status", "failed"))
        phase = "Cancelled (resumable)" if status == "queued" else "Failed (resumable)"
        self._set_progress(phase, self.progress_bar.value(), summary)
        self.resume_button.setEnabled(
            self._job_id is not None
            and (
                self._worker is None
                or not self._worker.isRunning()
            )
        )

    def _fail(self, exc: object) -> None:
        if self._result_handled:
            return
        self._result_handled = True
        self._timer.stop()
        self.cancel_button.setEnabled(False)
        self._completed = False
        self._set_progress(
            "Failed (resumable)",
            self.progress_bar.value(),
            f"{type(exc).__name__}: {exc}",
        )
        self.resume_button.setEnabled(False)

    def _worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        if self._completed:
            self._start_next()
        else:
            self.resume_button.setEnabled(self._job_id is not None)

    def cancel(self) -> None:
        setattr(self.service, "cancel_requested", True)
        setattr(self._active_service, "cancel_requested", True)
        cancel_job = getattr(self._active_service, "cancel", None)
        if self._job_id is not None and callable(cancel_job):
            try:
                cancel_job(self._job_id)
            except Exception:
                pass
        self._set_progress(
            "Cancel requested",
            self.progress_bar.value(),
            "The current checkpoint will remain available for resume; the previous revision is preserved.",
        )

    def resume(self) -> int | None:
        if self._job_id is None:
            self._set_progress(
                "Nothing to resume",
                self.progress_bar.value(),
                "Start indexing a file before requesting resume.",
            )
            return None
        setattr(self.service, "cancel_requested", False)
        setattr(self._active_service, "cancel_requested", False)
        self._run_job(self._job_id)
        return self._job_id

    def _set_progress(self, phase: str, value: int, detail: str) -> None:
        self.file_label.setText(f"Modern index: {self._active_path or 'idle'}")
        self.phase_label.setText(phase)
        self.progress_bar.setValue(max(0, min(100, value)))
        self.detail_label.setText(detail)
        self.status_changed.emit(
            f"{phase}: {self._active_path}" if self._active_path else phase,
            value,
        )
