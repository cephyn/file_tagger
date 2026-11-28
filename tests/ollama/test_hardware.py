from types import SimpleNamespace
from typing import Any

import pytest

from file_tagger.ollama import hardware


@pytest.fixture
def local_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "file_tagger.ollama.hardware.platform.machine", lambda: "AMD64"
    )
    monkeypatch.setattr(
        "file_tagger.ollama.hardware.psutil.cpu_count",
        lambda *, logical: 12 if logical else 6,
    )
    monkeypatch.setattr(
        "file_tagger.ollama.hardware.psutil.virtual_memory",
        lambda: SimpleNamespace(total=32 * 1024**3, available=17 * 1024**3),
    )
    monkeypatch.setattr(
        "file_tagger.ollama.hardware.psutil.disk_usage",
        lambda _path: SimpleNamespace(free=480 * 1024**3),
    )


def test_profile_hardware_returns_local_metrics_without_windows_gpu_probe(
    local_metrics: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "file_tagger.ollama.hardware.platform.system", lambda: "Linux"
    )

    report = hardware.profile_hardware()

    assert report.cpu_architecture == "AMD64"
    assert report.logical_processors == 12
    assert report.total_ram_bytes == 32 * 1024**3
    assert report.available_ram_bytes == 17 * 1024**3
    assert report.available_disk_bytes == 480 * 1024**3
    assert report.gpu_name is None
    assert report.vram_bytes is None


def test_windows_gpu_probe_uses_pywin32_when_nvml_is_unavailable(
    local_metrics: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    controllers = [
        SimpleNamespace(Name="Intel GPU", AdapterRAM=2 * 1024**3),
        SimpleNamespace(Name="NVIDIA GPU", AdapterRAM=8 * 1024**3),
    ]
    client = SimpleNamespace(
        GetObject=lambda _path: SimpleNamespace(
            InstancesOf=lambda _class: controllers
        )
    )
    monkeypatch.setattr(
        "file_tagger.ollama.hardware.platform.system", lambda: "Windows"
    )
    monkeypatch.setattr(
        hardware,
        "_load_optional_module",
        lambda name: client if name == "win32com.client" else None,
    )

    report = hardware.profile_hardware()

    assert report.gpu_name == "NVIDIA GPU"
    assert report.vram_bytes == 8 * 1024**3


def test_nvml_supplies_gpu_name_and_accurate_vram_and_always_shuts_down(
    local_metrics: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def shutdown() -> None:
        calls.append("shutdown")

    nvml = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda _index: "device",
        nvmlDeviceGetName=lambda _device: b"NVIDIA RTX 4090",
        nvmlDeviceGetMemoryInfo=lambda _device: SimpleNamespace(
            total=24 * 1024**3
        ),
        nvmlShutdown=shutdown,
    )
    monkeypatch.setattr(
        "file_tagger.ollama.hardware.platform.system", lambda: "Windows"
    )
    monkeypatch.setattr(
        hardware,
        "_load_optional_module",
        lambda name: nvml if name == "pynvml" else None,
    )

    report = hardware.profile_hardware()

    assert report.gpu_name == "NVIDIA RTX 4090"
    assert report.vram_bytes == 24 * 1024**3
    assert calls == ["init", "shutdown"]


def test_optional_gpu_probe_failures_leave_gpu_fields_empty(
    local_metrics: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenNvml:
        @staticmethod
        def nvmlInit() -> None:
            raise RuntimeError("driver unavailable")

    monkeypatch.setattr(
        "file_tagger.ollama.hardware.platform.system", lambda: "Windows"
    )
    monkeypatch.setattr(
        hardware,
        "_load_optional_module",
        lambda name: BrokenNvml if name == "pynvml" else None,
    )

    report = hardware.profile_hardware()

    assert report.gpu_name is None
    assert report.vram_bytes is None


def test_nvml_name_may_be_reported_as_text(
    local_metrics: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvml: Any = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda _index: "device",
        nvmlDeviceGetName=lambda _device: "NVIDIA Test GPU",
        nvmlDeviceGetMemoryInfo=lambda _device: SimpleNamespace(total=1024),
        nvmlShutdown=lambda: None,
    )
    monkeypatch.setattr(
        "file_tagger.ollama.hardware.platform.system", lambda: "Windows"
    )
    monkeypatch.setattr(
        hardware,
        "_load_optional_module",
        lambda name: nvml if name == "pynvml" else None,
    )

    assert hardware.profile_hardware().gpu_name == "NVIDIA Test GPU"
