import importlib
import os
import platform
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import psutil  # type: ignore[import-untyped]

from ..domain.contracts import HardwareReport


def _load_optional_module(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _positive_int(value: object) -> int | None:
    try:
        result = int(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result > 0 else None


def _gpu_name(value: object) -> str | None:
    if isinstance(value, bytes):
        return value.decode(errors="replace").rstrip("\0").strip() or None
    if isinstance(value, str):
        return value.strip() or None
    return None


def _probe_errors(
    module: ModuleType | None,
    exception_name: str,
) -> tuple[type[BaseException], ...]:
    base_errors: tuple[type[BaseException], ...] = (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    )
    if module is None:
        return base_errors
    exception_type = getattr(module, exception_name, None)
    if (
        isinstance(exception_type, type)
        and issubclass(exception_type, BaseException)
        and exception_type not in base_errors
    ):
        return (*base_errors, exception_type)
    return base_errors


def _probe_wmi_gpu() -> tuple[str | None, int | None]:
    client = _load_optional_module("win32com.client")
    if client is None:
        return None, None
    pywintypes = _load_optional_module("pywintypes")
    probe_errors = _probe_errors(pywintypes, "com_error")

    try:
        get_object = getattr(client, "GetObject")
        controllers: Any = get_object("winmgmts:").InstancesOf(
            "Win32_VideoController"
        )
        detected = [
            (
                _gpu_name(getattr(controller, "Name", None)),
                _positive_int(getattr(controller, "AdapterRAM", None)),
            )
            for controller in controllers
        ]
    except probe_errors:
        return None, None

    if not detected:
        return None, None
    return max(detected, key=lambda gpu: gpu[1] if gpu[1] is not None else -1)


def _probe_nvml_gpu() -> tuple[str | None, int | None]:
    nvml = _load_optional_module("pynvml")
    if nvml is None:
        return None, None

    probe_errors = _probe_errors(nvml, "NVMLError")
    initialized = False
    try:
        getattr(nvml, "nvmlInit")()
        initialized = True
        count = int(getattr(nvml, "nvmlDeviceGetCount")())
        detected: list[tuple[str | None, int | None]] = []
        for index in range(max(count, 0)):
            device = getattr(nvml, "nvmlDeviceGetHandleByIndex")(index)
            name = _gpu_name(getattr(nvml, "nvmlDeviceGetName")(device))
            memory = getattr(nvml, "nvmlDeviceGetMemoryInfo")(device)
            detected.append(
                (name, _positive_int(getattr(memory, "total", None)))
            )
        if not detected:
            return None, None
        return max(
            detected,
            key=lambda gpu: gpu[1] if gpu[1] is not None else -1,
        )
    except probe_errors:
        return None, None
    finally:
        if initialized:
            try:
                getattr(nvml, "nvmlShutdown")()
            except probe_errors:
                pass


def _probe_windows_gpu() -> tuple[str | None, int | None]:
    wmi_name, wmi_vram = _probe_wmi_gpu()
    nvml_name, nvml_vram = _probe_nvml_gpu()
    if nvml_name is not None or nvml_vram is not None:
        return nvml_name or wmi_name, (
            nvml_vram if nvml_vram is not None else wmi_vram
        )
    return wmi_name, wmi_vram


def _logical_processors() -> int:
    logical = psutil.cpu_count(logical=True)
    if isinstance(logical, int) and logical > 0:
        return logical
    fallback = os.cpu_count()
    if isinstance(fallback, int) and fallback > 0:
        return fallback
    return 1


def profile_hardware() -> HardwareReport:
    memory = psutil.virtual_memory()
    disk_root = Path.cwd().anchor or str(Path.cwd())
    disk = psutil.disk_usage(disk_root)
    architecture = (
        platform.machine().strip()
        or platform.processor().strip()
        or platform.system().strip()
        or "unknown"
    )

    gpu_name: str | None = None
    vram_bytes: int | None = None
    if platform.system().casefold() == "windows":
        gpu_name, vram_bytes = _probe_windows_gpu()

    return HardwareReport(
        cpu_architecture=architecture,
        logical_processors=_logical_processors(),
        total_ram_bytes=int(memory.total),
        available_ram_bytes=int(memory.available),
        available_disk_bytes=int(disk.free),
        gpu_name=gpu_name,
        vram_bytes=vram_bytes,
    )


def probe_hardware() -> HardwareReport:
    return profile_hardware()
