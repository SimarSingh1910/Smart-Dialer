"""Async runtime setup that has to happen before any event loop is created.

One item so far, and it is a Windows problem: since Python 3.8 the default
event loop on Windows is ProactorEventLoop, and psycopg's async mode cannot run
on it -- it needs the selector loop to wait on socket readiness. Without this
every async database call fails at connect time with an InterfaceError.

This is called from the process entry points (tasks.py, the worker, the test
suite's conftest) rather than at import time, because a module that reconfigures
global interpreter state as a side effect of being imported is a trap for
whoever imports it next.
"""

from __future__ import annotations

import asyncio
import sys


def configure_event_loop() -> None:
    """Select an event loop policy the database driver can actually use.

    Safe to call more than once, and a no-op off Windows.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
