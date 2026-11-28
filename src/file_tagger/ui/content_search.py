from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import inspect
import threading
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal


class ContentSearchWorker(QThread):
    result_ready = Signal(object)
    failure = Signal(object)
    cancelled = Signal()

    def __init__(
        self,
        service: Any,
        query: str,
        *,
        tags: Sequence[str],
        use_and: bool,
        limit: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._query = query
        self._tags = tuple(tags)
        self._use_and = use_and
        self._limit = limit
        self._cancel_requested = threading.Event()

    @property
    def cancellation_check(self) -> Callable[[], bool]:
        return self._cancel_requested.is_set

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def run(self) -> None:
        search_content = getattr(self._service, "search_content", None)
        try:
            if not callable(search_content):
                raise RuntimeError("The modern runtime is not initialized.")
            kwargs: dict[str, Any] = {
                "tags": self._tags,
                "use_and": self._use_and,
                "limit": self._limit,
            }
            try:
                parameters = inspect.signature(search_content).parameters
                supports_cancellation = "cancellation_check" in parameters or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
            except (TypeError, ValueError):
                supports_cancellation = False
            if supports_cancellation:
                kwargs["cancellation_check"] = self.cancellation_check
            results = search_content(self._query, **kwargs)
            if self._cancel_requested.is_set():
                self.cancelled.emit()
                return
            self.result_ready.emit(results)
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.cancelled.emit()
        except Exception as exc:
            self.failure.emit(exc)


__all__ = ("ContentSearchWorker",)
