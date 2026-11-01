"""Progress reports from deep inside a computation, without threading a
callback through every function on the way (Nam, 2026-09-26: "thanh tiến
trình của tất cả các quá trình").

A job (backend/jobs.py) or a worker process (backend/workers.py) binds a
reporter for the duration of one run; code anywhere below it calls
``report(done, total, stage)``. Outside a job it does nothing, so the engine,
the registry and the screener stay callable from tests and sync routes.

``stage`` is a stable code (``signals``, ``simulate``, ``symbols``…) that the
browser turns into words (§2.4). ``total == 0`` means the stage has no
measurable progress: the browser shows a moving bar and the clock, never a
percentage it would have to invent (§2.7).

A report is also where a cancelled job stops: the job's reporter raises
``JobCancelled`` from it, so a stage boundary is a cancellation point.
"""

from __future__ import annotations

import contextvars
from typing import Callable

_sink: contextvars.ContextVar["Reporter | None"] = contextvars.ContextVar(
    "qp_progress", default=None)


class Reporter:
    def __init__(self, send: Callable[[int, int, str | None], None]) -> None:
        self._send = send
        self._on_cancel: list[Callable[[], None]] = []
        self._cancelled = False

    def report(self, done: int, total: int, stage: str | None = None) -> None:
        self._send(done, total, stage)

    def on_cancel(self, fn: Callable[[], None]) -> None:
        self._on_cancel.append(fn)
        if self._cancelled:
            fn()

    def cancelled(self) -> None:
        self._cancelled = True
        for fn in list(self._on_cancel):
            try:
                fn()
            except Exception:  # noqa: BLE001 - a hook failing must not hide the cancel
                pass


def bind(reporter: Reporter | None):
    return _sink.set(reporter)


def unbind(token) -> None:
    _sink.reset(token)


def report(done: int, total: int, stage: str | None = None) -> None:
    reporter = _sink.get()
    if reporter is not None:
        reporter.report(done, total, stage)


def on_cancel(fn: Callable[[], None]) -> None:
    """Run ``fn`` when the current job is cancelled (e.g. stop a DB query)."""
    reporter = _sink.get()
    if reporter is not None:
        reporter.on_cancel(fn)


def every(n: int, parts: int = 20) -> int:
    """How many items between two reports, for about ``parts`` reports."""
    return max(1, n // parts)


class silenced:
    """Drop ``report()`` calls from the code inside, for a caller that
    reports its own coarser progress: a walk-forward reports folds, and the
    backtests inside each fold must not overwrite that with their stages."""

    def __enter__(self):
        self._token = _sink.set(None)
        return self

    def __exit__(self, *exc):
        _sink.reset(self._token)
        return False
