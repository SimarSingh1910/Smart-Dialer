"""A thousand agents, twenty workers, sixty seconds of virtual time.

This is not a benchmark of Postgres. It measures the two things the scale
argument in the ADR actually rests on, and it is deliberately small enough to
be read in one sitting:

    reservation latency   what it costs twenty workers to fight over the head
                          of one index. This is the first thing predicted to
                          break at a thousand agents, so it is the number that
                          has to be measured rather than asserted.

    round trips per call  the multiplier on everything above. A call that
                          costs twelve round trips instead of six halves the
                          throughput of the same database.

Everything else here is context for those two: ticks per second, and the query
plans, so that "it uses the partial index" is a plan somebody can read rather
than a claim.

Twenty workers really do run concurrently against one database, which is the
only way this measures contention rather than latency. Under a virtual clock
they tick in lockstep -- every worker reaching for the same AVAILABLE agents at
the same instant is the worst case, and the worst case is what the number is
for.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from smartdialer.core.clock import VirtualClock
from smartdialer.core.config import Settings
from smartdialer.core.db import Database
from smartdialer.core.logging import StructuredLogger
from smartdialer.domain.agents import RESERVE_AGENTS_SQL
from smartdialer.domain.snapshot import SNAPSHOT_SQL
from smartdialer.providers.mock_fast import make_fast_provider
from smartdialer.workers.dialer_worker import DialerWorker

START = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)


@dataclass
class LoadReport:
    agents: int = 0
    workers: int = 0
    ticks: int = 0
    wall_seconds: float = 0.0
    reservations: list[float] = field(default_factory=list)
    queries: int = 0
    calls_placed: int = 0
    plans: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        latencies = sorted(self.reservations)
        p50 = _percentile(latencies, 0.50)
        p99 = _percentile(latencies, 0.99)
        lines = [
            f"agents                  {self.agents}",
            f"worker coroutines       {self.workers}",
            f"ticks                   {self.ticks} in {self.wall_seconds:.1f}s wall",
            f"ticks/sec               {self.ticks / max(self.wall_seconds, 1e-9):.0f}",
            "",
            f"reservation latency p50 {p50 * 1000:.2f} ms",
            f"reservation latency p99 {p99 * 1000:.2f} ms",
            f"reservations measured   {len(latencies)}",
            "",
            f"calls placed            {self.calls_placed}",
            f"db round trips          {self.queries}",
            f"round trips per call    "
            f"{self.queries / max(self.calls_placed, 1):.1f}",
        ]
        for name, plan in self.plans.items():
            lines.append("")
            lines.append(f"EXPLAIN ANALYZE -- {name}")
            lines.extend(f"  {line}" for line in plan.splitlines())
        return "\n".join(lines)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return values[index]


# ---------------------------------------------------------------------------
# Counting the round trips
# ---------------------------------------------------------------------------


class _CountingCursor:
    """A cursor that counts what it was asked to run.

    Wrapping the cursor rather than instrumenting each call site: the number
    that matters is round trips per call across the WHOLE path, including the
    ones nobody remembers are there.
    """

    def __init__(self, cursor, counter: list[int]) -> None:
        self._cursor = cursor
        self._counter = counter

    async def execute(self, *args, **kwargs):
        self._counter[0] += 1
        return await self._cursor.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class CountingDatabase(Database):
    """The real pool, with a query counter. Used only by the load test."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.counter = [0]

    @contextlib.asynccontextmanager
    async def transaction(self):
        async with super().transaction() as cur:
            yield _CountingCursor(cur, self.counter)

    @contextlib.asynccontextmanager
    async def cursor(self):
        async with super().cursor() as cur:
            yield _CountingCursor(cur, self.counter)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


async def run_load_test(
    dsn: str,
    *,
    agents: int = 1000,
    workers: int = 20,
    borrowers: int = 20000,
    seconds: float = 60.0,
    tick_seconds: float = 0.25,
) -> LoadReport:
    pool = CountingDatabase(dsn, min_size=workers, max_size=workers + 4)
    await pool.open()
    report = LoadReport(agents=agents, workers=workers)
    campaign_id = await _setup(pool, agents=agents, borrowers=borrowers)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=11, answer_rate=0.4, talk_median_seconds=45.0)
    logger = StructuredLogger("loadtest", clock)

    running = [
        _make_worker(pool, clock, campaign_id, provider, logger, index)
        for index in range(workers)
    ]
    for worker in running:
        _instrument(worker, report)
    running[0].attach_providers()

    steps = int(seconds / tick_seconds)
    started = time.perf_counter()
    try:
        for _ in range(steps):
            # Every worker ticks at the same instant, on purpose. Staggering
            # them would measure a friendlier system than the one that runs in
            # production, where nothing coordinates the fleet.
            await asyncio.gather(*(w.tick() for w in running))
            report.ticks += workers
            await clock.advance(tick_seconds)
            await asyncio.gather(*(w.drain() for w in running))

        report.wall_seconds = time.perf_counter() - started
        report.queries = pool.counter[0]
        report.calls_placed = await _count_calls(pool, campaign_id)
        report.plans = await _plans(pool, campaign_id, clock)
    finally:
        for worker in running:
            await worker.close()
        await provider.close()
        await _teardown(pool, campaign_id)
        await pool.close()
    return report


def _make_worker(pool, clock, campaign_id, provider, logger, index: int) -> DialerWorker:
    return DialerWorker(
        db=pool,
        clock=clock,
        campaign_id=campaign_id,
        providers=[provider],
        settings=Settings(worker_id=f"load-{index}", tick_seconds=0.25),
        logger=logger,
    )


def _instrument(worker: DialerWorker, report: LoadReport) -> None:
    """Time every reservation without touching the allocator's own code."""
    allocator = worker.allocator
    original = allocator.reserve

    async def timed(**kwargs):
        started = time.perf_counter()
        try:
            return await original(**kwargs)
        finally:
            report.reservations.append(time.perf_counter() - started)

    allocator.reserve = timed  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Fixtures and plans
# ---------------------------------------------------------------------------


async def _setup(pool: Database, *, agents: int, borrowers: int) -> uuid.UUID:
    campaign_id = uuid.uuid4()
    async with pool.transaction() as cur:
        await cur.execute(
            "INSERT INTO campaigns (id, name, mode, max_concurrent, wrap_up_seconds) "
            "VALUES (%s, %s, 'PREDICTIVE', 100000, 5)",
            (campaign_id, f"load-{campaign_id}"),
        )
        await cur.execute(
            "INSERT INTO campaign_counters (campaign_id, shard) "
            "SELECT %s, generate_series(0, 15)",
            (campaign_id,),
        )
        await cur.execute(
            """
            INSERT INTO agents (id, campaign_id, state, state_changed_at,
                                last_heartbeat_at)
            SELECT gen_random_uuid(), %s, 'AVAILABLE', %s, %s
            FROM generate_series(1, %s)
            """,
            (campaign_id, START, START, agents),
        )
        await cur.execute(
            """
            INSERT INTO borrowers (id, campaign_id, phone, next_eligible_at)
            SELECT gen_random_uuid(), %s, '+9197' || lpad(i::text, 8, '0'), %s
            FROM generate_series(1, %s) AS i
            """,
            (campaign_id, START, borrowers),
        )
        await cur.execute("ANALYZE agents")
        await cur.execute("ANALYZE borrowers")
    return campaign_id


async def _count_calls(pool: Database, campaign_id: uuid.UUID) -> int:
    async with pool.transaction() as cur:
        await cur.execute(
            "SELECT count(*)::int AS n FROM calls WHERE campaign_id = %s", (campaign_id,)
        )
        return (await cur.fetchone())["n"]


async def _plans(pool: Database, campaign_id: uuid.UUID, clock) -> dict[str, str]:
    """The two queries every tick depends on, explained against real data.

    Rolled back rather than committed: EXPLAIN ANALYZE on the allocation query
    actually reserves agents, and a measurement that changed the thing it was
    measuring would be a poor way to end a load test.
    """
    plans: dict[str, str] = {}
    now = clock.now()
    try:
        async with pool.transaction() as cur:
            await cur.execute(
                "EXPLAIN (ANALYZE, BUFFERS) " + RESERVE_AGENTS_SQL,
                {
                    "campaign_id": campaign_id,
                    "worker_id": "explain",
                    "n": 20,
                    "lease_seconds": 5.0,
                    "now": now,
                },
            )
            plans["allocation (SKIP LOCKED)"] = "\n".join(
                row["QUERY PLAN"] for row in await cur.fetchall()
            )

            await cur.execute(
                "EXPLAIN (ANALYZE, BUFFERS) " + SNAPSHOT_SQL,
                _snapshot_params(campaign_id, now),
            )
            plans["snapshot"] = "\n".join(
                row["QUERY PLAN"] for row in await cur.fetchall()
            )

            # Undo the reservations the first EXPLAIN really did make.
            raise _Rollback()
    except _Rollback:
        pass
    return plans


class _Rollback(Exception):
    pass


def _snapshot_params(campaign_id: uuid.UUID, now: datetime) -> dict[str, Any]:
    from smartdialer.domain import snapshot as snap

    return {
        "campaign_id": campaign_id,
        "now": now,
        "window": snap.DEFAULT_WINDOW_SECONDS,
        "baseline_window": snap.BASELINE_WINDOW_SECONDS,
        "changepoint_span": snap.CHANGEPOINT_SPAN_SECONDS,
        "changepoint_lag": snap.CHANGEPOINT_LAG_SECONDS,
    }


async def _teardown(pool: Database, campaign_id: uuid.UUID) -> None:
    async with pool.transaction() as cur:
        await cur.execute(
            "DELETE FROM provider_events WHERE provider_call_id IN "
            "(SELECT provider_call_id FROM calls WHERE campaign_id = %s "
            " AND provider_call_id IS NOT NULL)",
            (campaign_id,),
        )
        await cur.execute(
            "DELETE FROM pacing_decisions WHERE campaign_id = %s", (campaign_id,)
        )
        await cur.execute(
            "UPDATE agents SET current_call_id = NULL WHERE campaign_id = %s",
            (campaign_id,),
        )
        await cur.execute("DELETE FROM calls WHERE campaign_id = %s", (campaign_id,))
        await cur.execute("DELETE FROM borrowers WHERE campaign_id = %s", (campaign_id,))
        await cur.execute("DELETE FROM agents WHERE campaign_id = %s", (campaign_id,))
        await cur.execute(
            "DELETE FROM campaign_counters WHERE campaign_id = %s", (campaign_id,)
        )
        await cur.execute("DELETE FROM campaigns WHERE id = %s", (campaign_id,))
