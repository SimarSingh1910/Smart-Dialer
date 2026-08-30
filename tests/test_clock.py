"""Tests for the injected clock.

The VirtualClock is load-bearing infrastructure: if it wakes tasks in the wrong
order, every failure scenario built on top of it is quietly wrong. So it gets
tested before anything is built on it.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta, timezone

import pytest

from smartdialer.core.clock import Clock, RealClock, VirtualClock


def test_clocks_satisfy_the_protocol():
    assert isinstance(RealClock(), Clock)
    assert isinstance(VirtualClock(), Clock)


def test_virtual_clock_starts_aware_and_utc():
    clock = VirtualClock()
    assert clock.now().tzinfo is not None
    assert clock.now().utcoffset() == timedelta(0)


async def test_two_sleepers_wake_in_timestamp_order():
    """The definition of done for step 0: a task sleeping 5s and a task
    sleeping 3s must wake in the order 3s then 5s, whichever started first."""
    clock = VirtualClock()
    wake_order: list[str] = []

    async def sleeper(name: str, seconds: float) -> None:
        await clock.sleep(seconds)
        wake_order.append(name)

    # The 5s task is started first on purpose: ordering must come from the
    # wake-up time, not from the order tasks happened to be scheduled.
    tasks = [
        asyncio.create_task(sleeper("five", 5)),
        asyncio.create_task(sleeper("three", 3)),
    ]
    await clock.advance(10)
    await asyncio.gather(*tasks)

    assert wake_order == ["three", "five"]


async def test_now_is_exact_at_each_wake_up():
    """A task must observe the instant it was scheduled for, not the end of the
    advance() call. Without this, everything a woken task timestamps -- lease
    expiry, state_changed_at, wait_ms -- would be wrong."""
    clock = VirtualClock()
    start = clock.now()
    observed: dict[str, float] = {}

    async def sleeper(name: str, seconds: float) -> None:
        await clock.sleep(seconds)
        observed[name] = (clock.now() - start).total_seconds()

    tasks = [
        asyncio.create_task(sleeper("a", 3)),
        asyncio.create_task(sleeper("b", 5)),
    ]
    await clock.advance(10)
    await asyncio.gather(*tasks)

    assert observed == {"a": 3.0, "b": 5.0}
    assert (clock.now() - start).total_seconds() == 10.0


async def test_repeated_sleeps_inside_one_advance_land_correctly():
    """A worker tick loop sleeps again immediately after waking. One advance()
    of 1s must therefore produce exactly 4 ticks at 250ms, not 1."""
    clock = VirtualClock()
    ticks: list[float] = []
    start = clock.now()
    stop = False

    async def tick_loop() -> None:
        while not stop:
            await clock.sleep(0.25)
            ticks.append(round((clock.now() - start).total_seconds(), 3))

    task = asyncio.create_task(tick_loop())
    await clock.advance(1.0)
    stop = True
    await clock.advance(0.25)
    task.cancel()

    assert ticks[:4] == [0.25, 0.5, 0.75, 1.0]


async def test_ties_wake_in_the_order_they_slept():
    clock = VirtualClock()
    order: list[int] = []

    async def sleeper(index: int) -> None:
        await clock.sleep(2)
        order.append(index)

    tasks = [asyncio.create_task(sleeper(i)) for i in range(5)]
    await clock.advance(2)
    await asyncio.gather(*tasks)

    assert order == [0, 1, 2, 3, 4]


async def test_time_does_not_move_on_its_own():
    """Real time passing must not move virtual time. This is what makes a
    260-second provider outage cost nothing to test."""
    clock = VirtualClock()
    before = clock.now()
    await asyncio.sleep(0.05)  # real time, deliberately
    assert clock.now() == before


async def test_advance_rejects_negative_time():
    clock = VirtualClock()
    with pytest.raises(ValueError):
        await clock.advance(-1)


async def test_sleeper_pending_until_its_time_arrives():
    clock = VirtualClock()
    done = False

    async def sleeper() -> None:
        nonlocal done
        await clock.sleep(10)
        done = True

    task = asyncio.create_task(sleeper())
    await clock.advance(9)
    assert done is False
    assert clock.pending_sleepers == 1
    await clock.advance(1)
    assert done is True
    await task


async def test_zero_sleep_yields_without_advancing_time():
    clock = VirtualClock()
    before = clock.now()
    await clock.sleep(0)
    assert clock.now() == before


async def test_a_task_that_never_sleeps_does_not_block_advance():
    """Documents a real limit of the settle heuristic.

    A task that loops without ever touching the clock leaves the observed
    state unchanged, so settle reads the system as quiet and virtual time
    moves on without it. This is the same shape as a task blocked on real
    database I/O. It is not a hang, and asserting it here keeps the behaviour
    honest rather than surprising."""
    clock = VirtualClock()
    stop = False

    async def spinner() -> None:
        while not stop:
            await asyncio.sleep(0)  # never touches the virtual clock

    task = asyncio.create_task(spinner())
    await clock.advance(1)
    assert clock.pending_sleepers == 0
    stop = True
    await asyncio.sleep(0)
    task.cancel()


async def test_settle_raises_when_the_task_set_churns_forever():
    """The round cap turns a pathological, never-quiet system into a loud
    failure instead of an infinite loop."""
    clock = VirtualClock()
    stop = False
    spawned: list[asyncio.Task] = []

    async def spawner() -> None:
        # Each new task parks and never finishes, so the live-task count grows
        # on every round and the observed state never repeats. Settle can
        # therefore never conclude the system is quiet.
        while not stop:
            spawned.append(asyncio.create_task(asyncio.sleep(3600)))
            await asyncio.sleep(0)

    task = asyncio.create_task(spawner())
    with pytest.raises(RuntimeError, match="could not settle"):
        await clock.advance(1)
    stop = True
    await asyncio.sleep(0)
    task.cancel()
    for child in spawned:
        child.cancel()


async def test_real_clock_advances_and_sleeps():
    clock = RealClock()
    before = clock.now()
    await clock.sleep(0.01)
    assert clock.now() >= before
    assert clock.now().tzinfo == timezone.utc
