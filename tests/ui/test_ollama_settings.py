from collections.abc import Callable
import logging
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QPushButton

from file_tagger.domain.contracts import HardwareReport, ModelProfile, OllamaReadiness
from file_tagger.ollama.model_recommender import (
    ModelRecommendation,
    RecommendedModel,
)
from file_tagger.ollama.readiness import ReadinessReport
from file_tagger.ui.ollama_settings import OllamaSettingsDialog


GIB = 1024**3


def _hardware() -> HardwareReport:
    return HardwareReport(
        cpu_architecture="AMD64",
        logical_processors=16,
        total_ram_bytes=32 * GIB,
        available_ram_bytes=24 * GIB,
        available_disk_bytes=100 * GIB,
        gpu_name="Test GPU",
        vram_bytes=12 * GIB,
    )


def _recommendation(*, override: bool = False) -> ModelRecommendation:
    return ModelRecommendation(
        embedding=RecommendedModel(
            role="embedding",
            model="embeddinggemma-2:270m",
            ram_gib=1.0,
            disk_gib=0.5,
            execution_plan="gpu_resident",
            quality_weight=100,
            is_override=override,
        ),
        decision=RecommendedModel(
            role="decision",
            model="nimble:9b",
            ram_gib=12.0,
            disk_gib=10.0,
            execution_plan="gpu_resident",
            quality_weight=300,
            is_override=override,
            warnings=("User override is below the available hardware gate.",)
            if override
            else (),
        ),
        generation=RecommendedModel(
            role="generation",
            model="gemma4:e4b",
            ram_gib=8.5,
            disk_gib=7.0,
            execution_plan="gpu_resident",
            quality_weight=200,
            is_override=override,
        ),
    )


def _profile(role: str, model: str) -> ModelProfile:
    return ModelProfile(
        role=role,  # type: ignore[arg-type]
        model=model,
        digest=f"{role}-digest",
        context_length=8192,
        dimensions=256 if role == "embedding" else None,
        capabilities=(role,),
    )


def _report(*, ready: bool = False) -> ReadinessReport:
    profiles = (
        (
            _profile("embedding", "embeddinggemma-2:270m"),
            _profile("decision", "nimble:9b"),
            _profile("generation", "gemma4:e4b"),
        )
        if ready
        else (_profile("embedding", "embeddinggemma-2:270m"),)
    )
    return ReadinessReport(
        ready=ready,
        blocked_roles=() if ready else ("tagging", "search"),
        instructions=()
        if ready
        else ("Install the model manually: ollama pull nimble:9b",),
        profiles=profiles,
        ollama=OllamaReadiness(
            available=True,
            version="0.35.0",
            base_url="http://127.0.0.1:11434",
            missing_models=() if ready else ("nimble:9b",),
        ),
        hardware=_hardware(),
        recommendation=_recommendation(),
    )


def test_missing_model_disables_ok_and_shows_manual_pull(qtbot) -> None:
    dialog = OllamaSettingsDialog(_report())
    qtbot.addWidget(dialog)

    assert dialog.ok_button.isEnabled() is False
    assert "ollama pull nimble:9b" in dialog.instructions.text()
    assert dialog.host_label.text() == "http://127.0.0.1:11434"
    assert dialog.version_label.text() == "0.35.0"
    assert "AMD64" in dialog.hardware_summary.text()
    assert "100 GiB" in dialog.disk_fit.text()
    assert dialog.model_selectors["embedding"].findText(
        "embeddinggemma-2:270m"
    ) >= 0
    assert "pull" in dialog.instructions.text().lower()
    assert not any(
        "download" in button.text().lower()
        for button in dialog.findChildren(QPushButton)
    )


def test_outdated_and_remote_ollama_keep_settings_disabled(qtbot) -> None:
    report = _report(
        ready=True
    ).model_copy(
        update={
            "ready": False,
            "errors": ("Ollama 0.34.0 is outdated.",),
            "ollama": OllamaReadiness(
                available=True,
                version="0.34.0",
                base_url="https://ollama.example.test",
            ),
        }
    )
    dialog = OllamaSettingsDialog(report)
    qtbot.addWidget(dialog)

    assert not dialog.ok_button.isEnabled()
    assert "0.34.0" in dialog.ollama_status.text()
    assert "loopback" in dialog.instructions.text().lower()
    assert "outdated" in dialog.instructions.text().lower()


@pytest.mark.parametrize("version", [None, "", "not-a-version", "0.35"])
def test_ready_report_without_a_valid_version_keeps_settings_disabled(
    qtbot, version
) -> None:
    report = _report(ready=True).model_copy(
        update={
            "ollama": OllamaReadiness(
                available=True,
                version=version,
                base_url="http://127.0.0.1:11434",
            )
        }
    )

    dialog = OllamaSettingsDialog(report)
    qtbot.addWidget(dialog)

    assert not dialog.ok_button.isEnabled()
    assert "unknown or invalid" in dialog.instructions.text().lower()
    assert "0.35.0" in dialog.instructions.text()


def test_ready_report_with_an_outdated_version_keeps_settings_disabled(qtbot) -> None:
    report = _report(ready=True).model_copy(
        update={
            "ollama": OllamaReadiness(
                available=True,
                version="0.34.9",
                base_url="http://127.0.0.1:11434",
            )
        }
    )

    dialog = OllamaSettingsDialog(report)
    qtbot.addWidget(dialog)

    assert not dialog.ok_button.isEnabled()
    assert "outdated" in dialog.instructions.text().lower()


@pytest.mark.parametrize(
    "host",
    [
        "http://localhost:11434",
        "http://127.0.0.1:1",
        "http://[::1]:65535",
    ],
)
def test_local_http_origins_enable_ok_for_ready_reports(qtbot, host) -> None:
    report = _report(ready=True).model_copy(
        update={
            "ollama": OllamaReadiness(
                available=True,
                version="0.35.0",
                base_url=host,
            )
        }
    )

    dialog = OllamaSettingsDialog(report)
    qtbot.addWidget(dialog)

    assert dialog.ok_button.isEnabled()


@pytest.mark.parametrize(
    "host",
    [
        "https://localhost:11434",
        "http://localhost",
        "http://localhost:0",
        "http://localhost:65536",
        "http://localhost:not-a-port",
        "http://localhost:11434/",
        "http://localhost:11434/api",
        "http://localhost:11434?query=1",
        "http://localhost:11434#fragment",
        "http://user:pass@localhost:11434",
        "http://ollama.example.test:11434",
    ],
)
def test_non_local_or_non_origin_hosts_disable_ok(qtbot, host) -> None:
    report = _report(ready=True).model_copy(
        update={
            "ollama": OllamaReadiness(
                available=True,
                version="0.35.0",
                base_url=host,
            )
        }
    )

    dialog = OllamaSettingsDialog(report)
    qtbot.addWidget(dialog)

    assert not dialog.ok_button.isEnabled()
    assert "local loopback" in dialog.instructions.text().lower()


def test_ready_report_enables_ok_and_renders_benchmarks(qtbot) -> None:
    from file_tagger.ollama.benchmarks import BenchmarkResult

    report = _report(ready=True).model_copy(
        update={
            "benchmarks": (
                BenchmarkResult(
                    role="decision",
                    model="nimble:9b",
                    elapsed_ms=125,
                    success=True,
                ),
            )
        }
    )
    dialog = OllamaSettingsDialog(report)
    qtbot.addWidget(dialog)

    assert dialog.ok_button.isEnabled()
    assert "nimble:9b" in dialog.installed_models.text()
    assert "125 ms" in dialog.benchmark_results.text()
    assert dialog.selected_models()["decision"] == "nimble:9b"


def test_override_is_explained_and_only_installed_models_are_selectable(qtbot) -> None:
    report = _report(ready=True).model_copy(
        update={"recommendation": _recommendation(override=True)}
    )
    dialog = OllamaSettingsDialog(report)
    qtbot.addWidget(dialog)

    assert "override" in dialog.recommendations.text().lower()
    assert "available hardware gate" in dialog.recommendations.text()
    selectors = dialog.model_selectors
    assert selectors["decision"].count() == 1
    assert selectors["decision"].itemText(0) == "nimble:9b"
    assert selectors["generation"].findText("inkling-2-mini") == -1


def test_refresh_provider_is_lazy_and_replaces_report(qtbot) -> None:
    refreshed = _report(ready=True)
    calls = 0

    def refresh() -> ReadinessReport:
        nonlocal calls
        calls += 1
        return refreshed

    provider: Callable[[], ReadinessReport] = refresh
    dialog = OllamaSettingsDialog(
        _report(),
        runtime_refresh_provider=provider,
    )
    qtbot.addWidget(dialog)

    assert calls == 0
    assert not dialog.ok_button.isEnabled()
    qtbot.mouseClick(dialog.refresh_button, Qt.MouseButton.LeftButton)
    assert calls == 1
    assert dialog.ok_button.isEnabled()
    assert "No installation steps are required" in dialog.instructions.text()


def test_provider_construction_does_not_start_refresh(qtbot) -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0
            self.readiness = _report(ready=True)

        def start_readiness(self) -> None:
            self.calls += 1

    provider = Provider()

    dialog = OllamaSettingsDialog(
        _report(),
        runtime_refresh_provider=provider,
    )
    qtbot.addWidget(dialog)

    assert provider.calls == 0
    assert not dialog.ok_button.isEnabled()


def test_refresh_callback_is_delivered_on_qt_event_loop(qtbot) -> None:
    refreshed = _report(ready=True)

    class Provider:
        def __init__(self) -> None:
            self.callbacks: list[Callable[[ReadinessReport], None]] = []
            self.calls = 0

        def start_readiness(
            self, on_complete: Callable[[ReadinessReport], None]
        ) -> None:
            self.calls += 1
            self.callbacks.append(on_complete)

    provider = Provider()
    dialog = OllamaSettingsDialog(
        _report(),
        runtime_refresh_provider=provider,
    )
    qtbot.addWidget(dialog)

    qtbot.mouseClick(dialog.refresh_button, Qt.MouseButton.LeftButton)

    assert provider.calls == 1
    assert not dialog.refresh_button.isEnabled()
    assert "refreshing" in dialog.refresh_button.text().lower()
    provider.callbacks[0](refreshed)
    qtbot.waitUntil(lambda: dialog.refresh_button.isEnabled())

    assert dialog.ok_button.isEnabled()
    assert dialog.selected_models()["decision"] == "nimble:9b"


def test_stale_refresh_callback_cannot_replace_latest_report(qtbot) -> None:
    class Provider:
        def __init__(self) -> None:
            self.callbacks: list[Callable[[ReadinessReport], None]] = []

        def start_readiness(
            self, on_complete: Callable[[ReadinessReport], None]
        ) -> None:
            self.callbacks.append(on_complete)

    provider = Provider()
    dialog = OllamaSettingsDialog(
        _report(),
        runtime_refresh_provider=provider,
    )
    qtbot.addWidget(dialog)

    qtbot.mouseClick(dialog.refresh_button, Qt.MouseButton.LeftButton)
    first = _report(ready=True).model_copy(
        update={"warnings": ("first result",)}
    )
    provider.callbacks[0](first)
    qtbot.waitUntil(lambda: dialog.refresh_button.isEnabled())
    qtbot.mouseClick(dialog.refresh_button, Qt.MouseButton.LeftButton)
    provider.callbacks[0](_report(ready=True).model_copy(
        update={"warnings": ("stale result",)}
    ))
    qtbot.wait(20)
    assert dialog._report is first

    latest = _report(ready=True).model_copy(
        update={"warnings": ("latest result",)}
    )
    provider.callbacks[1](latest)
    qtbot.waitUntil(lambda: dialog.refresh_button.isEnabled())
    assert dialog._report is latest


def test_failed_refresh_callback_does_not_restore_previous_report(qtbot) -> None:
    initial = _report(ready=True)

    class Provider:
        def __init__(self) -> None:
            self.readiness = initial
            self.unavailable_reason = "latest refresh failed"
            self.callbacks: list[Callable[[ReadinessReport | None], None]] = []

        def start_readiness(
            self, on_complete: Callable[[ReadinessReport | None], None]
        ) -> None:
            self.callbacks.append(on_complete)

    provider = Provider()
    dialog = OllamaSettingsDialog(initial, runtime_refresh_provider=provider)
    qtbot.addWidget(dialog)

    qtbot.mouseClick(dialog.refresh_button, Qt.MouseButton.LeftButton)
    provider.callbacks[0](None)
    qtbot.waitUntil(lambda: dialog.refresh_button.isEnabled())

    assert dialog._report is initial
    assert not dialog.ok_button.isEnabled()
    assert "latest refresh failed" in dialog.instructions.text()


def _manager_module():
    from file_tagger import file_tag_manager

    return file_tag_manager


def test_manager_settings_menu_opens_ollama_readiness(qtbot) -> None:
    module = _manager_module()
    manager = module.FileTagManager.__new__(module.FileTagManager)
    QMainWindow.__init__(manager)
    manager.logger = logging.getLogger("test-file-tagger-manager")
    manager.config = SimpleNamespace(get_pdf_extractor=lambda: "accurate")
    qtbot.addWidget(manager)

    manager.setup_menus()

    settings_actions = manager.menuBar().actions()[0].menu().actions()
    assert "Ollama Readiness" in [action.text() for action in settings_actions]
    assert "API Settings" not in [action.text() for action in settings_actions]


def test_manager_passes_runtime_readiness_to_dialog(qtbot, monkeypatch) -> None:
    module = _manager_module()
    manager = module.FileTagManager.__new__(module.FileTagManager)
    QMainWindow.__init__(manager)
    manager.logger = logging.getLogger("test-file-tagger-manager")
    qtbot.addWidget(manager)
    report = _report()
    runtime = SimpleNamespace(readiness=report, start_readiness=lambda: None)
    manager.modern_runtime = runtime
    captured: dict[str, object] = {}

    class Dialog:
        def __init__(self, readiness, **kwargs) -> None:
            captured["readiness"] = readiness
            captured.update(kwargs)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(module, "OllamaSettingsDialog", Dialog)

    manager.show_ollama_settings()

    assert captured["readiness"] is report
    assert captured["runtime_refresh_provider"] is runtime
    assert captured["parent"] is manager


def test_manager_applies_accepted_model_selections(qtbot, monkeypatch) -> None:
    module = _manager_module()
    manager = module.FileTagManager.__new__(module.FileTagManager)
    QMainWindow.__init__(manager)
    manager.logger = logging.getLogger("test-file-tagger-manager")
    qtbot.addWidget(manager)
    applied: list[dict[str, str | None]] = []
    runtime = SimpleNamespace(
        readiness=_report(ready=True),
        apply_model_selections=applied.append,
    )
    manager.modern_runtime = runtime
    selections = {
        "embedding": "embeddinggemma-2:270m",
        "decision": "nimble:9b",
        "generation": "gemma4:e4b",
    }

    class Dialog:
        def __init__(self, readiness, **kwargs) -> None:
            self.readiness = readiness

        def exec(self) -> int:
            return int(module.QDialog.DialogCode.Accepted)

        def selected_models(self) -> dict[str, str | None]:
            return selections

    monkeypatch.setattr(module, "OllamaSettingsDialog", Dialog)

    manager.show_ollama_settings()

    assert applied == [selections]


def test_manager_import_preserves_modern_tag_suggestion() -> None:
    module = _manager_module()
    from file_tagger import tag_suggestion

    assert module.TagSuggestionDialog is tag_suggestion.TagSuggestionDialog
