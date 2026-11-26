from collections.abc import Callable, Mapping
from concurrent.futures import Future
import inspect
import queue
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from ..adapters.ollama.gateway import (
    DEFAULT_OLLAMA_HOST,
    MINIMUM_OLLAMA_VERSION,
    parse_ollama_version,
    validate_ollama_host,
)
from ..domain.contracts import HardwareReport
from ..domain.errors import RemoteOllamaHostError
from ..ollama.model_recommender import ModelRecommendation
from ..ollama.readiness import ReadinessReport


GIB = 1024**3
_NO_REFRESH_RESULT = object()

RefreshProvider = Callable[[], ReadinessReport | None]
CompletionCallback = Callable[[ReadinessReport | None], None]
ModelSelectionApplier = Callable[[Mapping[str, str | None]], Any]


def _gib(byte_count: int) -> str:
    return f"{byte_count / GIB:g} GiB"


def _version_tuple(version: str | None) -> tuple[int, int, int] | None:
    return parse_ollama_version(version)


def _is_loopback_host(host: str) -> bool:
    try:
        validate_ollama_host(host)
    except RemoteOllamaHostError:
        return False
    return True


class OllamaSettingsDialog(QDialog):
    """Read-only readiness status and installed-model selection."""

    def __init__(
        self,
        report: ReadinessReport,
        hardware: HardwareReport | None = None,
        recommendation: ModelRecommendation | None = None,
        runtime_refresh_provider: Any | None = None,
        parent: QWidget | None = None,
        model_selection_applier: ModelSelectionApplier | None = None,
        *,
        hardware_report: HardwareReport | None = None,
        refresh_provider: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._report = report
        self._hardware = hardware_report or hardware or report.hardware
        self._recommendation = recommendation or report.recommendation
        self._model_selection_applier = model_selection_applier
        self._selections_dirty = False
        self._refresh_provider = (
            runtime_refresh_provider or refresh_provider
        )
        self._completion_queue: queue.Queue[
            tuple[int, ReadinessReport | None]
        ] = queue.Queue()
        self._refresh_generation = 0
        self._refreshing = False
        self._refresh_future: Future[ReadinessReport | None] | None = None
        self._refresh_provider_report: ReadinessReport | None = None
        self._refresh_observed_by_callback = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(50)
        self._refresh_timer.timeout.connect(self._poll_refresh)
        self._build_ui()
        self.set_report(report)

    def _build_ui(self) -> None:
        self.setWindowTitle("Ollama Readiness")
        self.setModal(True)
        self.resize(760, 700)

        root_layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        scroll.setWidget(content)
        root_layout.addWidget(scroll)

        ollama_group = QGroupBox("Ollama")
        ollama_layout = QFormLayout(ollama_group)
        self.host_label = QLabel()
        self.version_label = QLabel()
        self.runtime_label = QLabel()
        ollama_layout.addRow("Host:", self.host_label)
        ollama_layout.addRow("Version:", self.version_label)
        ollama_layout.addRow("Runtime:", self.runtime_label)
        self.ollama_status = QLabel()
        self.ollama_status.setWordWrap(True)
        ollama_layout.addRow(self.ollama_status)
        content_layout.addWidget(ollama_group)
        self.hardware_summary = self._summary_label(
            content_layout, "Detected hardware"
        )
        self.recommendations = self._summary_label(
            content_layout, "Model recommendations"
        )
        self.disk_fit = self._summary_label(content_layout, "Disk fit")
        self.installed_models = self._summary_label(
            content_layout, "Installed models"
        )
        self.benchmark_results = self._summary_label(
            content_layout, "Benchmark results"
        )

        selection_group = QGroupBox("Installed model selections")
        selection_layout = QFormLayout(selection_group)
        self.model_selectors: dict[str, QComboBox] = {}
        for role in ("embedding", "decision", "generation"):
            selector = QComboBox()
            selector.setObjectName(f"{role}_model_selector")
            selector.setEnabled(False)
            selector.currentTextChanged.connect(self._mark_selections_dirty)
            selection_layout.addRow(f"{role.title()}:", selector)
            self.model_selectors[role] = selector
        content_layout.addWidget(selection_group)

        self.instructions = self._summary_label(
            content_layout, "Readiness instructions"
        )
        self.manual_pull_commands = self.instructions
        content_layout.addStretch()

        button_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip(
            "Recheck the current local Ollama runtime. No models are downloaded."
        )
        self.refresh_button.setEnabled(self._refresh_provider is not None)
        self.refresh_button.clicked.connect(self.refresh)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(self.ok_button)
        root_layout.addLayout(button_layout)

    @staticmethod
    def _summary_label(parent_layout: QVBoxLayout, title: str) -> QLabel:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        label = QLabel()
        label.setWordWrap(True)
        label.setTextInteractionFlags(label.textInteractionFlags())
        layout.addWidget(label)
        parent_layout.addWidget(group)
        return label

    def set_report(self, report: ReadinessReport) -> None:
        self._report = report
        pending_selections = (
            self.selected_models()
            if self._selections_dirty
            else None
        )
        if report.hardware is not None:
            self._hardware = report.hardware
        if report.recommendation is not None:
            self._recommendation = report.recommendation

        host = (
            report.ollama.base_url
            if report.ollama is not None
            else f"{DEFAULT_OLLAMA_HOST} (default)"
        )
        version = (
            report.ollama.version
            if report.ollama is not None and report.ollama.version
            else "unknown"
        )
        availability = "available" if report.ollama is None else (
            "available" if report.ollama.available else "unavailable"
        )
        display_host = host.removesuffix(" (default)")
        self.host_label.setText(display_host)
        self.version_label.setText(version)
        self.runtime_label.setText(availability)
        self.ollama_status.setText(f"{display_host} — Ollama {version}")

        self.hardware_summary.setText(self._hardware_text())
        self.recommendations.setText(self._recommendation_text())
        self.disk_fit.setText(self._disk_fit_text())
        self.installed_models.setText(self._installed_models_text())
        self.benchmark_results.setText(self._benchmark_text())
        self._populate_selectors(preferred=pending_selections)

        valid_host = _is_loopback_host(
            report.ollama.base_url
            if report.ollama is not None
            else DEFAULT_OLLAMA_HOST
        )
        version_value = _version_tuple(
            report.ollama.version if report.ollama is not None else None
        )
        unknown_version = version_value is None
        outdated = (
            version_value is not None
            and version_value < MINIMUM_OLLAMA_VERSION
        )
        self.instructions.setText(
            self._instruction_text(
                valid_host=valid_host,
                unknown_version=unknown_version,
                outdated=outdated,
            )
        )
        self.ok_button.setEnabled(
            valid_host
            and version_value is not None
            and not outdated
            and all(selector.isEnabled() for selector in self.model_selectors.values())
        )
        self._set_refreshing(self._refreshing)

    def _hardware_text(self) -> str:
        hardware = self._hardware
        if hardware is None:
            return "Hardware details are not available."
        gpu = hardware.gpu_name or "No discrete GPU detected"
        vram = _gib(hardware.vram_bytes) if hardware.vram_bytes is not None else "unknown"
        return (
            f"{hardware.cpu_architecture}; {hardware.logical_processors} logical processors\n"
            f"RAM: {_gib(hardware.available_ram_bytes)} available / "
            f"{_gib(hardware.total_ram_bytes)} total\n"
            f"Disk: {_gib(hardware.available_disk_bytes)} available\n"
            f"GPU: {gpu}; VRAM: {vram}"
        )

    def _recommendation_text(self) -> str:
        recommendation = self._recommendation
        if recommendation is None:
            return "Recommendations are not available."
        lines: list[str] = []
        for role in ("embedding", "decision", "generation"):
            model = getattr(recommendation, role)
            prefix = f"{role.title()}: {model.model}"
            if model.is_override:
                prefix += " (user override)"
            lines.append(prefix)
            lines.append(
                f"  {model.execution_plan}; RAM {model.ram_gib:g} GiB; "
                f"disk {model.disk_gib:g} GiB"
            )
            lines.extend(f"  {reason}" for reason in model.reasons)
            lines.extend(f"  Warning: {warning}" for warning in model.warnings)
        return "\n".join(lines)

    def _disk_fit_text(self) -> str:
        hardware = self._hardware
        recommendation = self._recommendation
        if hardware is None or recommendation is None:
            return "Disk fit is not available."
        selected = (
            recommendation.embedding,
            recommendation.decision,
            recommendation.generation,
        )
        required = sum(model.disk_gib for model in selected) + 4.0
        available = hardware.available_disk_bytes / GIB
        status = "Fits" if required <= available else "Does not fit"
        return (
            f"{status}: {available:g} GiB available; {required:g} GiB required "
            "including the 4 GiB operating reserve."
        )

    def _installed_models_text(self) -> str:
        if not self._report.profiles:
            return "No compatible installed model profiles were found."
        lines = [
            f"{profile.role.title()}: {profile.model}"
            for profile in self._report.profiles
        ]
        return "\n".join(lines)

    def _benchmark_text(self) -> str:
        if not self._report.benchmarks:
            return "No benchmark results are available."
        lines = []
        for benchmark in self._report.benchmarks:
            outcome = "passed" if benchmark.success else "failed"
            slow = " (slow)" if benchmark.slow else ""
            line = (
                f"{benchmark.role.title()} {benchmark.model}: {outcome} in "
                f"{benchmark.elapsed_ms:g} ms{slow}"
            )
            if benchmark.detail:
                line += f" — {benchmark.detail}"
            lines.append(line)
        return "\n".join(lines)

    def _instruction_text(
        self,
        *,
        valid_host: bool,
        unknown_version: bool,
        outdated: bool,
    ) -> str:
        report = self._report
        lines = list(report.instructions)
        lines.extend(report.errors)
        lines.extend(f"Warning: {warning}" for warning in report.warnings)
        if report.ollama is not None:
            for model in report.ollama.missing_models:
                command = f"Install the model manually: ollama pull {model}"
                if command not in lines:
                    lines.append(command)
        if outdated:
            version = report.ollama.version if report.ollama is not None else "unknown"
            lines.append(
                f"Ollama {version} is outdated; version 0.35.0 or newer is required."
            )
        elif unknown_version:
            lines.append(
                "Ollama version is unknown or invalid; "
                "version 0.35.0 or newer is required."
            )
        if not valid_host:
            lines.append(
                "Ollama must use an absolute local loopback HTTP origin "
                "with a numeric port from 1 to 65535; paths, queries, "
                "credentials, and remote hosts are not supported."
            )
        if not lines and report.ready:
            lines.append("Ready. No installation steps are required.")
        elif not lines:
            lines.append(
                "Run Ollama locally, install the required models, and refresh."
            )
        return "\n".join(dict.fromkeys(lines))

    def _populate_selectors(
        self,
        *,
        preferred: Mapping[str, str | None] | None = None,
    ) -> None:
        for role, selector in self.model_selectors.items():
            selector.blockSignals(True)
            selector.clear()
            installed = tuple(
                dict.fromkeys(
                    profile.model
                    for profile in self._report.profiles
                    if profile.role == role
                )
            )
            selector.addItems(installed or ("No compatible installed model",))
            selector.setEnabled(bool(installed))
            selected = (
                preferred.get(role)
                if preferred is not None
                else None
            )
            if selected is None:
                selected = (
                    self._report.selected_models.as_mapping()
                    if self._report.selected_models is not None
                    else {}
                ).get(role)
            if selected is not None and installed:
                index = selector.findText(selected)
                if index >= 0:
                    selector.setCurrentIndex(index)
            elif self._recommendation is not None and installed:
                recommended = getattr(self._recommendation, role).model
                index = selector.findText(recommended)
                if index >= 0:
                    selector.setCurrentIndex(index)
            selector.blockSignals(False)

    def selected_models(self) -> dict[str, str | None]:
        selections: dict[str, str | None] = {}
        for role, selector in self.model_selectors.items():
            selections[role] = (
                selector.currentText() if selector.isEnabled() else None
            )
        return selections

    def _mark_selections_dirty(self, _text: str) -> None:
        self._selections_dirty = True

    def _apply_selection_changes(self) -> bool:
        if not self._selections_dirty or self._model_selection_applier is None:
            return True
        try:
            self._model_selection_applier(self.selected_models())
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(
                self,
                "Ollama Models",
                f"Could not apply the selected models: {exc}",
            )
            return False
        self._selections_dirty = False
        return True

    def accept(self) -> None:
        if not self._apply_selection_changes():
            return
        super().accept()

    def reject(self) -> None:
        self._selections_dirty = False
        super().reject()

    def closeEvent(self, event: Any) -> None:
        if not self._apply_selection_changes():
            event.ignore()
            return
        super().closeEvent(event)

    def refresh(self) -> None:
        provider = self._refresh_provider
        if provider is None or self._refreshing:
            return
        self._refresh_generation += 1
        generation = self._refresh_generation
        self._refresh_provider_report = self._provider_report(provider)
        self._refresh_future = None
        self._set_refreshing(True)

        def callback(report: ReadinessReport | None) -> None:
            self._completion_queue.put((generation, report))

        try:
            operation = provider
            if not callable(operation):
                operation = (
                    getattr(provider, "refresh_readiness", None)
                    or getattr(provider, "start_readiness", None)
                )
            if operation is None:
                raise TypeError("Refresh provider does not expose a refresh operation.")
            self._refresh_observed_by_callback = self._supports_completion_callback(
                operation
            )
            result = self._invoke_refresh(operation, callback)
            if isinstance(result, Future):
                self._refresh_future = result
                self._refresh_timer.start()
                return
            if isinstance(result, ReadinessReport):
                self._complete_refresh(result)
                return
            self._refresh_timer.start()
        except Exception as exc:
            self._complete_refresh(error=exc)

    @staticmethod
    def _supports_completion_callback(operation: Callable[..., Any]) -> bool:
        signature = inspect.signature(operation)
        if any(
            name in signature.parameters
            for name in ("on_complete", "callback", "completion_callback")
        ):
            return True
        parameters = signature.parameters.values()
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ):
            return True
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return (
            len(positional) == 1
            and positional[0].default is inspect.Parameter.empty
        )

    @staticmethod
    def _invoke_refresh(
        operation: Callable[..., Any],
        callback: CompletionCallback,
    ) -> Any:
        signature = inspect.signature(operation)
        parameters = signature.parameters.values()
        for name in ("on_complete", "callback", "completion_callback"):
            if name in signature.parameters:
                return operation(**{name: callback})
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ):
            return operation(on_complete=callback)
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(positional) == 1 and positional[0].default is inspect.Parameter.empty:
            return operation(callback)
        return operation()

    def _poll_refresh(self) -> None:
        queued: tuple[int, ReadinessReport | None] | None = None
        while True:
            try:
                item = self._completion_queue.get_nowait()
            except queue.Empty:
                break
            if item[0] == self._refresh_generation:
                queued = item
        if queued is not None:
            self._complete_refresh(queued[1])
            return

        future = self._refresh_future
        if future is not None and future.done():
            try:
                self._complete_refresh(future.result())
            except Exception as exc:
                self._complete_refresh(error=exc)
            return

        provider = self._refresh_provider
        latest = self._provider_report(provider)
        if (
            not self._refresh_observed_by_callback
            and latest is not None
            and latest is not self._refresh_provider_report
        ):
            self._complete_refresh(latest)
            return

        thread = (
            getattr(provider, "_readiness_thread", None)
            if not callable(provider)
            else None
        )
        if (
            future is None
            and not self._refresh_observed_by_callback
            and thread is not None
            and not thread.is_alive()
        ):
            self._complete_refresh(latest)

    @staticmethod
    def _provider_report(provider: Any) -> ReadinessReport | None:
        report = getattr(provider, "readiness", None)
        return report if isinstance(report, ReadinessReport) else None

    def _complete_refresh(
        self,
        report: ReadinessReport | None | object = _NO_REFRESH_RESULT,
        *,
        error: Exception | None = None,
    ) -> None:
        self._refresh_timer.stop()
        self._set_refreshing(False)
        self._refresh_future = None
        latest = (
            self._provider_report(self._refresh_provider)
            if report is _NO_REFRESH_RESULT
            else report
        )
        if isinstance(latest, ReadinessReport):
            self.set_report(latest)
        elif error is not None:
            self.ok_button.setEnabled(False)
            self.instructions.setText(
                f"Refresh failed: {error.__class__.__name__}: {error}"
            )
        else:
            self.ok_button.setEnabled(False)
            reason = getattr(
                self._refresh_provider,
                "unavailable_reason",
                "The readiness refresh completed without a result.",
            )
            self.instructions.setText(f"Refresh failed: {reason}")

    def _set_refreshing(self, refreshing: bool) -> None:
        self._refreshing = refreshing
        self.refresh_button.setText("Refreshing…" if refreshing else "Refresh")
        self.refresh_button.setEnabled(
            self._refresh_provider is not None and not refreshing
        )
        self.ok_button.setEnabled(
            self.ok_button.isEnabled() and not refreshing
        )
        for selector in self.model_selectors.values():
            selector.setEnabled(selector.isEnabled() and not refreshing)
        if refreshing:
            self.instructions.setText(
                "Refreshing local Ollama readiness… No models are downloaded."
            )
