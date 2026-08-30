"""Structured JSON-lines logging.

Every log line carries the identifiers you need to reconstruct one call's
history across several worker processes: campaign_id, worker_id, call_id,
agent_id. Debugging distributed state without that is guesswork, and the
interview for this assignment includes a live debugging exercise.

Time comes from the injected Clock, so log timestamps in a simulation run are
virtual time and line up with the CSV the simulation writes.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from smartdialer.core.clock import Clock

# Fields promoted to top level in the JSON output when present.
CONTEXT_FIELDS = ("campaign_id", "worker_id", "call_id", "agent_id", "borrower_id", "provider")


class StructuredLogger:
    """Thin wrapper over logging that emits one JSON object per line."""

    def __init__(self, name: str, clock: Clock, **context: Any) -> None:
        self._log = logging.getLogger(name)
        self._clock = clock
        self._context = {k: v for k, v in context.items() if v is not None}

    def bind(self, **context: Any) -> "StructuredLogger":
        """Return a child logger with extra context. Used per call / per tick."""
        merged = {**self._context, **{k: v for k, v in context.items() if v is not None}}
        child = StructuredLogger(self._log.name, self._clock)
        child._context = merged
        return child

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        if not self._log.isEnabledFor(level):
            return
        record: dict[str, Any] = {
            "ts": self._clock.now().isoformat(),
            "level": logging.getLevelName(level),
            "logger": self._log.name,
            "event": event,
        }
        record.update(self._context)
        record.update(fields)
        self._log.log(level, json.dumps(record, default=str))

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)


def configure_logging(level: str = "INFO") -> None:
    """Install a bare stdout handler. The payload is already JSON, so the
    formatter must not add a prefix of its own."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(level.upper())
