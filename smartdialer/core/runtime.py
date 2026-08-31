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
from typing import Iterable


def configure_event_loop() -> None:
    """Select an event loop policy the database driver can actually use.

    Safe to call more than once, and a no-op off Windows.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def drain_tasks(
    tasks: "Iterable[asyncio.Task]",
    *,
    settle_seconds: float = 0.02,
    max_rounds: int = 500,
) -> None:
    """Wait for background tasks that are progressing, and only those.

    TEST AND SIMULATION SCAFFOLDING. Not domain code, and the one place in the
    project that waits on real wall-clock time on purpose.

    The problem it solves is the seam between the two clocks. Under a
    VirtualClock, a task parked in `clock.sleep()` resumes only when the test
    advances time, so waiting for it here would deadlock. But a task blocked on
    a real database round trip is making progress that no amount of advancing
    virtual time will finish, and the VirtualClock's settle heuristic
    explicitly gives up on it -- see the note in clock.py.

    Polling with `timeout=0` gets this wrong in the quiet direction: it returns
    while database work is still in flight, so a test samples a half-applied
    world and fails somewhere unrelated. That is exactly how an eagerly
    released agent came to look like one that had leaked.

    So: wait a short REAL interval for something to complete, and stop as soon
    as a whole interval passes with nothing finishing. Anything still pending
    at that point is parked on the virtual clock, which is the caller's job to
    advance.
    """
    for _ in range(max_rounds):
        pending = [task for task in tasks if not task.done()]
        if not pending:
            return
        done, _ = await asyncio.wait(pending, timeout=settle_seconds)
        if not done:
            return
