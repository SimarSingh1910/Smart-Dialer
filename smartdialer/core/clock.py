"""Time injection.

Every piece of domain code takes a Clock. Nothing calls time.time(),
datetime.now() or asyncio.sleep() directly.

Why: the deliverable is a set of failure scenarios (worker crash at t=120s,
provider outage from t=200s to t=260s, 40 agents logging out inside two
seconds). Those are only reproducible if the test controls time. A test that
really sleeps for 260 seconds is not a test anybody runs.

RealClock is used in production. VirtualClock is used by the tests, the
simulation harness and the load test.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """The only way domain code is allowed to observe or wait on time."""

    def now(self) -> datetime:
        """Current time as an aware UTC datetime."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the calling task for `seconds` of this clock's time."""
        ...


class RealClock:
    """Wall-clock time. Used by the workers when they run for real."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


@dataclass(order=True)
class _Sleeper:
    """One task parked inside VirtualClock.sleep().

    `seq` breaks ties so that two tasks waking at the same virtual instant wake
    in the order they went to sleep. Without it the heap comparison would fall
    through to the Future, which is not orderable.
    """

    wake_at: datetime
    seq: int
    future: asyncio.Future = field(compare=False)


class VirtualClock:
    """A clock that only moves when the test tells it to.

    Time advances exclusively through `advance()`. Tasks that call `sleep()`
    park on a future and are woken in wake-up-time order, so a task sleeping 3s
    always resumes before a task sleeping 5s regardless of which one called
    sleep() first.
    """

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        self._sleepers: list[_Sleeper] = []
        self._seq = itertools.count()

    # -- Clock protocol ---------------------------------------------------

    def now(self) -> datetime:
        return self._now

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            # Still yield, so a `while True: await clock.sleep(0)` loop cannot
            # starve the event loop.
            await asyncio.sleep(0)
            return
        loop = asyncio.get_running_loop()
        sleeper = _Sleeper(
            wake_at=self._now + timedelta(seconds=seconds),
            seq=next(self._seq),
            future=loop.create_future(),
        )
        heapq.heappush(self._sleepers, sleeper)
        await sleeper.future

    # -- test / simulation control ---------------------------------------

    async def advance(self, seconds: float) -> None:
        """Move virtual time forward by `seconds`, waking sleepers in order.

        This does not jump straight to the target. It stops at each pending
        wake-up time in between, sets `now` to exactly that instant, wakes
        everything due, and lets the event loop run before moving on. That is
        what makes a task that sleeps again from inside its wake-up land at the
        correct virtual time instead of at the end of the whole advance.
        """
        if seconds < 0:
            raise ValueError("virtual time only moves forward")
        target = self._now + timedelta(seconds=seconds)

        await self._settle()
        while self._sleepers and self._sleepers[0].wake_at <= target:
            due_at = self._sleepers[0].wake_at
            self._now = due_at
            while self._sleepers and self._sleepers[0].wake_at <= due_at:
                sleeper = heapq.heappop(self._sleepers)
                if not sleeper.future.done():
                    sleeper.future.set_result(None)
            await self._settle()

        self._now = target
        await self._settle()

    async def advance_to(self, when: datetime) -> None:
        await self.advance((when - self._now).total_seconds())

    @property
    def pending_sleepers(self) -> int:
        return len(self._sleepers)

    async def _settle(self, max_rounds: int = 200) -> None:
        """Let every currently runnable task run to its next await point.

        `await asyncio.sleep(0)` yields one scheduling round. One round is not
        enough: a woken task typically awaits several times (query, state
        write, sleep again) before it parks. So we keep yielding until the
        observable state -- the set of pending timers and the number of live
        tasks -- stops changing across two consecutive rounds, and treat that
        as "the system has gone quiet".

        This is a heuristic, and it is worth being precise about its limits:

        * A task that loops without ever touching the clock (a busy-wait, or
          one blocked on real I/O such as a database round trip) leaves the
          observed state unchanged, so settle returns while it is still
          running. Virtual time then moves on without it, and when it does
          eventually sleep it parks at a later virtual instant than it would
          have. That is unavoidable when real I/O is mixed with virtual time;
          the simulation and the concurrency tests avoid it by driving
          everything through this clock.
        * The round cap only bites when the observed state churns on every
          round -- a task endlessly spawning tasks, say. It exists so that
          pathological case fails loudly instead of spinning forever.
        """
        previous: tuple | None = None
        stable = 0
        for _ in range(max_rounds):
            await asyncio.sleep(0)
            state = (
                len(self._sleepers),
                tuple(s.wake_at for s in self._sleepers),
                len(asyncio.all_tasks()),
            )
            if state == previous:
                stable += 1
                if stable >= 2:
                    return
            else:
                previous = state
                stable = 0
        raise RuntimeError(
            "VirtualClock could not settle: a task is runnable without ever "
            "parking on the clock. Check for a busy-wait loop."
        )
