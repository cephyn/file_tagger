from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import threading
import uuid
from typing import Any, Sequence, cast

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..utils import open_file


@dataclass(frozen=True, slots=True)
class ContextualSearchSubmission:
    question: str
    context_file: Path | None = None


_ACTIVE_CONTEXTUAL_SEARCH_WORKERS: set[ContextualSearchWorker] = set()


class ContextualSearchWorker(QThread):
    answer_chunk = Signal(str)
    answer_delta = Signal(str)
    answer_ready = Signal(object)
    result_ready = Signal(object)
    citations_ready = Signal(object)
    evidence_ready = Signal(object)
    failure = Signal(object)
    cancelled = Signal()

    def __init__(
        self,
        service: Any,
        submission: ContextualSearchSubmission | str,
        context_file: Path | str | None = None,
        *,
        session_id: str | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if isinstance(submission, ContextualSearchSubmission):
            checked_submission = submission
        else:
            checked_submission = ContextualSearchSubmission(
                question=str(submission).strip(),
                context_file=Path(context_file) if context_file else None,
            )
        self.submission = checked_submission
        self._service = service
        self._session_id = session_id
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            markdown_context = ""
            if self.submission.context_file is not None:
                markdown_context = self.submission.context_file.read_text(
                    encoding="utf-8"
                )

            stream_method = getattr(
                self._service,
                "answer_contextual_question_stream_async",
                None,
            )
            if not callable(stream_method):
                stream_method = getattr(
                    self._service,
                    "answer_question_stream_async",
                    None,
                )

            answer = None
            if callable(stream_method):
                async def consume_stream() -> object | None:
                    nonlocal answer
                    if hasattr(self._service, "answer_contextual_question_stream_async"):
                        try:
                            stream = stream_method(
                                self.submission.question,
                                markdown_context,
                                conversation_id=self._session_id,
                                cancellation_check=self._cancel_requested.is_set,
                            )
                        except TypeError:
                            stream = stream_method(
                                self.submission.question,
                                markdown_context,
                                conversation_id=self._session_id,
                            )
                    else:
                        stream = stream_method(
                            self.submission.question,
                            markdown_context,
                            session_id=self._session_id,
                        )
                    async for fragment in stream:
                        if self._cancel_requested.is_set():
                            return None
                        if fragment:
                            self.answer_chunk.emit(fragment)
                            self.answer_delta.emit(fragment)

                    get_answer = getattr(
                        self._service,
                        "get_session_answer",
                        None,
                    )
                    if callable(get_answer):
                        return cast(object, get_answer(self._session_id))
                    return None

                answer = asyncio.run(consume_stream())
            else:
                answer_method = getattr(
                    self._service,
                    "answer_contextual_question",
                    None,
                )
                if callable(answer_method):
                    try:
                        answer = answer_method(
                            self.submission.question,
                            markdown_context,
                            conversation_id=self._session_id,
                            cancellation_check=self._cancel_requested.is_set,
                        )
                    except TypeError:
                        answer = answer_method(
                            self.submission.question,
                            markdown_context,
                            conversation_id=self._session_id,
                        )
                else:
                    rag_method = getattr(self._service, "answer_question", None)
                    if not callable(rag_method):
                        raise RuntimeError("The contextual RAG service is not initialized.")
                    answer = rag_method(
                        self.submission.question,
                        markdown_context,
                        session_id=self._session_id,
                    )

            if self._cancel_requested.is_set():
                self.cancelled.emit()
                return

            if answer is None:
                get_answer = getattr(self._service, "get_session_answer", None)
                if callable(get_answer):
                    answer = get_answer(self._session_id)

            citations = tuple(getattr(answer, "citations", ()) or ())
            evidence = tuple(getattr(answer, "evidence", ()) or ())
            self.citations_ready.emit(citations)
            self.evidence_ready.emit(evidence)
            self.answer_ready.emit(answer)
            self.result_ready.emit(answer)
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.cancelled.emit()
        except Exception as exc:
            self.failure.emit(exc)


class ContextualSearchDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None = None,
        modern_runtime: Any | None = None,
        rag_service: Any | None = None,
        readiness: Any | None = None,
        runtime: Any | None = None,
        readiness_report: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.modern_runtime = modern_runtime or runtime
        self.rag_service = rag_service
        self._context_file: Path | None = None
        self._worker: ContextualSearchWorker | None = None
        self._worker_generation = 0
        self._session_id = f"ui:{uuid.uuid4().hex}"
        if readiness is not None:
            selected_readiness = readiness
        elif readiness_report is not None:
            selected_readiness = readiness_report
        else:
            selected_readiness = getattr(self.modern_runtime, "readiness", None)
        self._readiness = selected_readiness
        self._build_ui()
        self._apply_readiness_gate()

    def _build_ui(self) -> None:
        self.setWindowTitle("Contextual Markdown Search")
        self.resize(760, 620)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Question"))
        self.question = QLineEdit()
        self.question.setPlaceholderText("Ask a question about the indexed files")
        self.question.textChanged.connect(self._refresh_submission_state)
        layout.addWidget(self.question)

        context_layout = QHBoxLayout()
        self.context_button = QPushButton("Choose Markdown Context")
        self.context_button.clicked.connect(self._choose_context_file)
        context_layout.addWidget(self.context_button)
        self.context_file_label = QLabel("Optional context file: none")
        context_layout.addWidget(self.context_file_label, 1)
        layout.addLayout(context_layout)

        self.setup_guidance = QLabel()
        self.setup_guidance.setWordWrap(True)
        layout.addWidget(self.setup_guidance)

        self.submission_button = QPushButton("Submit")
        self.submission_button.clicked.connect(self.submit)
        layout.addWidget(self.submission_button)

        layout.addWidget(QLabel("Answer"))
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        layout.addWidget(self.answer_output)

        evidence_layout = QHBoxLayout()
        evidence_layout.addWidget(QLabel("Citations"), 1)
        evidence_layout.addWidget(QLabel("Evidence Files"), 1)
        layout.addLayout(evidence_layout)

        lists_layout = QHBoxLayout()
        self.citations_list = QListWidget()
        self.citations_list.itemDoubleClicked.connect(self._open_citation)
        lists_layout.addWidget(self.citations_list)
        self.evidence_list = QListWidget()
        self.evidence_list.itemDoubleClicked.connect(self._open_evidence)
        lists_layout.addWidget(self.evidence_list)
        layout.addLayout(lists_layout)

        layout.addWidget(QLabel("Follow-up"))
        follow_up_layout = QHBoxLayout()
        self.follow_up_input = QLineEdit()
        self.follow_up_input.setPlaceholderText("Ask a follow-up question")
        self.follow_up_input.textChanged.connect(self._refresh_submission_state)
        follow_up_layout.addWidget(self.follow_up_input)
        self.follow_up_button = QPushButton("Ask Follow-up")
        self.follow_up_button.clicked.connect(self._submit_follow_up)
        follow_up_layout.addWidget(self.follow_up_button)
        layout.addLayout(follow_up_layout)

        self.question_input = self.question
        self.submit_button = self.submission_button
        self.answer = self.answer_output
        self.citations = self.citations_list
        self.evidence_files = self.evidence_list

    def _apply_readiness_gate(self) -> None:
        report = self._readiness
        ready = bool(getattr(report, "ready", False)) if report is not None else False
        if report is None and self.rag_service is not None:
            ready = True
        modern_runtime = getattr(self, "modern_runtime", None)
        if report is None and modern_runtime is not None:
            ready = bool(getattr(modern_runtime, "readiness", None))
        messages: tuple[str, ...] = ()
        if report is not None:
            raw_messages = (
                *tuple(getattr(report, "instructions", ()) or ()),
                *tuple(getattr(report, "errors", ()) or ())
            )
            messages = tuple(str(message) for message in raw_messages)
        guidance = " ".join(str(message) for message in messages)
        if not guidance:
            guidance = (
                "Set up local Ollama models and a complete modern search index "
                "before submitting contextual search."
            )
        self.setup_guidance.setText(guidance)
        self.submission_button.setEnabled(ready)
        self.follow_up_button.setEnabled(ready)
        self._ready = ready
        self._refresh_submission_state()

    def _refresh_submission_state(self, *_args: object) -> None:
        ready = bool(getattr(self, "_ready", False))
        self.submission_button.setEnabled(
            ready and bool(self.question.text().strip())
        )
        self.follow_up_button.setEnabled(
            ready and bool(self.follow_up_input.text().strip())
            if hasattr(self, "follow_up_input")
            else ready
        )

    def set_context_file(self, path: Path | str | None) -> None:
        self._context_file = Path(path) if path else None
        if self._context_file is None:
            self.context_file_label.setText("Optional context file: none")
        else:
            self.context_file_label.setText(
                f"Optional context file: {self._context_file.name}"
            )

    def _choose_context_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Markdown Context",
            "",
            "Markdown files (*.md);;All files (*)",
        )
        if path:
            self.set_context_file(path)

    def submission(self) -> ContextualSearchSubmission:
        return ContextualSearchSubmission(
            question=self.question.text().strip(),
            context_file=self._context_file,
        )

    def submit(self) -> ContextualSearchSubmission | None:
        submission = self.submission()
        if not submission.question:
            QMessageBox.warning(self, "Question required", "Enter a search question.")
            return None
        service = self.rag_service
        if service is None:
            service = self.modern_runtime
        if service is None:
            QMessageBox.warning(
                self,
                "Contextual search unavailable",
                self.setup_guidance.text(),
            )
            return None
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self._stop_worker(self._worker)
        if self._worker is not None:
            self._close_contextual_session()
        self.answer_output.clear()
        self.citations_list.clear()
        self.evidence_list.clear()
        worker = ContextualSearchWorker(
            service,
            submission,
            session_id=self._session_id,
        )
        self._worker = worker
        self._worker_generation += 1
        generation = self._worker_generation
        _ACTIVE_CONTEXTUAL_SEARCH_WORKERS.add(worker)
        worker.answer_chunk.connect(
            lambda chunk, w=worker, g=generation: self._answer_chunk(chunk, w, g)
        )
        worker.answer_ready.connect(
            lambda answer, w=worker, g=generation: self._answer_ready(answer, w, g)
        )
        worker.citations_ready.connect(
            lambda citations, w=worker, g=generation: self._render_citations(citations, w, g)
        )
        worker.evidence_ready.connect(
            lambda evidence, w=worker, g=generation: self._render_evidence(evidence, w, g)
        )
        worker.failure.connect(
            lambda exc, w=worker, g=generation: self._search_failed(exc, w, g)
        )
        worker.cancelled.connect(
            lambda w=worker, g=generation: self._search_cancelled(w, g)
        )
        worker.finished.connect(
            lambda w=worker, g=generation: self._worker_finished(w, g)
        )
        self.submission_button.setEnabled(False)
        worker.start()
        return submission

    def _submit_follow_up(self) -> None:
        follow_up = self.follow_up_input.text().strip()
        if not follow_up:
            return
        self.question.setText(follow_up)
        self.follow_up_input.clear()
        self.submit()

    def _answer_chunk(
        self,
        chunk: str,
        worker: ContextualSearchWorker,
        generation: int,
    ) -> None:
        if not self._is_current_worker(worker, generation):
            return
        self.answer_output.insertPlainText(chunk)

    def _is_current_worker(
        self,
        worker: ContextualSearchWorker,
        generation: int,
    ) -> bool:
        return (
            worker is self._worker
            and generation == self._worker_generation
        )

    def _answer_ready(
        self,
        _answer: object,
        worker: ContextualSearchWorker | None = None,
        generation: int | None = None,
    ) -> None:
        if worker is not None and generation is not None:
            if not self._is_current_worker(worker, generation):
                return
        self.submission_button.setEnabled(True)

    def _search_failed(
        self,
        exc: object,
        worker: ContextualSearchWorker | None = None,
        generation: int | None = None,
    ) -> None:
        if worker is not None and generation is not None:
            if not self._is_current_worker(worker, generation):
                return
        QMessageBox.warning(self, "Contextual search failed", str(exc))
        self.submission_button.setEnabled(True)

    def _search_cancelled(
        self,
        worker: ContextualSearchWorker,
        generation: int,
    ) -> None:
        if not self._is_current_worker(worker, generation):
            return
        self.submission_button.setEnabled(True)

    def _worker_finished(
        self,
        worker: ContextualSearchWorker,
        generation: int | None = None,
    ) -> None:
        _ACTIVE_CONTEXTUAL_SEARCH_WORKERS.discard(worker)
        if generation is not None and self._is_current_worker(worker, generation):
            self._worker = None
            self.submission_button.setEnabled(True)
        elif worker is self._worker and generation is None:
            self._worker = None
            self.submission_button.setEnabled(True)

    def _render_citations(
        self,
        citations: Sequence[Any],
        worker: ContextualSearchWorker | None = None,
        generation: int | None = None,
    ) -> None:
        if worker is not None and generation is not None:
            if not self._is_current_worker(worker, generation):
                return
        self.citations_list.clear()
        for citation in citations:
            path = getattr(citation, "path", getattr(citation, "file_path", ""))
            excerpt = getattr(citation, "excerpt", getattr(citation, "text", ""))
            label = f"{Path(str(path)).name}: {excerpt}" if path else str(citation)
            item = self.citations_list.addItem(label)
            del item
            widget_item = self.citations_list.item(self.citations_list.count() - 1)
            if path:
                widget_item.setData(Qt.ItemDataRole.UserRole, str(path))

    def _render_evidence(
        self,
        evidence: Sequence[Any],
        worker: ContextualSearchWorker | None = None,
        generation: int | None = None,
    ) -> None:
        if worker is not None and generation is not None:
            if not self._is_current_worker(worker, generation):
                return
        self.evidence_list.clear()
        paths: list[str] = []
        for item in evidence:
            path = getattr(item, "path", getattr(item, "file_path", ""))
            if not path:
                citations = getattr(item, "citations", ())
                if citations:
                    path = getattr(citations[0], "path", "")
            if path and str(path) not in paths:
                paths.append(str(path))
        for path in paths:
            item = self.evidence_list.addItem(Path(path).name)
            del item
            widget_item = self.evidence_list.item(self.evidence_list.count() - 1)
            widget_item.setData(Qt.ItemDataRole.UserRole, path)

    def _open_citation(self, item: Any) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            open_file(str(path))

    def _open_evidence(self, item: Any) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            open_file(str(path))

    def _stop_worker(self, worker: ContextualSearchWorker) -> None:
        worker.request_cancel()
        if worker.isRunning():
            worker.wait(2000)
        if worker.isRunning():
            worker.terminate()
            worker.wait(500)

    def _close_contextual_session(self) -> None:
        close_method = getattr(self.modern_runtime, "close_contextual_session", None)
        if not callable(close_method):
            close_method = getattr(self.rag_service, "close_session", None)
        if not callable(close_method):
            close_method = getattr(self.rag_service, "close_contextual_session", None)
        if callable(close_method):
            close_method(self._session_id)

    def reject(self) -> None:
        worker = self._worker
        if worker is not None:
            self._stop_worker(worker)
        self._worker_generation += 1
        self._worker = None
        self._close_contextual_session()
        super().reject()

    def closeEvent(self, event: Any) -> None:
        worker = self._worker
        if worker is not None:
            self._stop_worker(worker)
        self._worker_generation += 1
        self._worker = None
        self._close_contextual_session()
        super().closeEvent(event)


__all__ = (
    "ContextualSearchDialog",
    "ContextualSearchSubmission",
    "ContextualSearchWorker",
)
