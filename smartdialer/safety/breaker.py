"""The provider circuit breaker.

A carrier that has started failing does not stop failing because we kept
dialling it. Worse, in this system the failures are not free: every call we
hand to a sick provider takes an agent out of the pool, holds a borrower's
reservation, and comes back as a shortfall a tick or two later. So the breaker
stops new calls to a provider that is failing, and lets the existing ones be
reconciled by the reaper -- cancelling in-flight calls would be the one action
guaranteed to make a bad situation worse, because some of those calls have a
real person on the other end.

DERIVED, NOT STORED. The failure rate is computed from the `calls` table on
every tick, not accumulated in a counter in memory. Three reasons, in order of
importance:

    1. Every worker computes the same answer, because they are all reading the
       same rows. An in-memory window per worker would have five workers
       holding five different opinions of one carrier, and the fleet's
       behaviour would depend on which of them happened to tick first.
    2. A worker that restarts does not start from an empty window, believing a
       dead provider to be healthy. This is exactly the "stale state" failure
       the brief asks about, and deriving the state removes the category.
    3. There is nothing to keep in sync. The rows that decide the breaker are
       the same rows that record the calls.

What is NOT derivable from the calls table is a timeout, because the
deliberate response to a provider timeout is to touch nothing at all -- see
the allocator. A call that timed out looks exactly like a call still being set
up. So the timeout rate comes from the client's own observations of the
carrier, alongside `reachable`, which is what a carrier that will not answer at
all looks like from here.

Only ONE row of state is persisted -- when the breaker opened, and who holds
the half-open probe -- and both exist because they are fleet-wide decisions.
Twenty workers independently discovering a dead carrier is twenty probe calls
at a provider that asked for one, so the probe is claimed by the same
compare-and-swap discipline that stops two workers reserving one agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence
from uuid import UUID

from psycopg import AsyncCursor

# Thresholds from the design. Both are rates over a 30-second window, and both
# are ignored below MIN_SAMPLES: three failures out of four calls is not
# evidence about a carrier, it is evidence about four calls.
FAILURE_RATE_LIMIT = 0.30
TIMEOUT_RATE_LIMIT = 0.15
MIN_SAMPLES = 10

# What counts as the CARRIER failing us. This list is the whole difference
# between a circuit breaker and a breaker that trips on a quiet afternoon: a
# borrower not picking up is a `no_answer`, and at a 50% answer rate half of
# every window is one. Counting those as failures would open the breaker on a
# perfectly healthy carrier and hold a campaign at zero for the rest of the
# run -- the numbers cannot recover while nothing is being dialled.
#
# So only the outcomes that are evidence about the CARRIER count: it refused
# the call, it was unreachable, it never took the call at all, or it took the
# call and then never told us what happened to it.
CARRIER_FAILURES = (
    "provider_rejected",
    "provider_unavailable",
    "never_placed_with_provider",
    "exceeded_max_call_lifetime",
)

WINDOW_SECONDS = 30.0
OPEN_SECONDS = 20.0
# How long a claimed probe has to produce a verdict before it is treated as a
# failure. A probe that never resolves is a carrier that never answered, which
# is the thing the breaker is for.
PROBE_TIMEOUT_SECONDS = 20.0


class BreakerState:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass(frozen=True, slots=True)
class ProviderStats:
    provider: str
    total: int = 0
    failures: int = 0
    timeouts: int = 0
    reachable: bool = True

    @property
    def failure_rate(self) -> float:
        return (self.failures / self.total) if self.total else 0.0

    @property
    def timeout_rate(self) -> float:
        return (self.timeouts / self.total) if self.total else 0.0


@dataclass(frozen=True, slots=True)
class BreakerView:
    """What the breaker permits this tick.

    `allowance` is None when the breaker is not constraining anything, which is
    different from a large number: it means this clamp did not participate in
    the decision at all, and the decision log says so.
    """

    state: str
    allowance: int | None
    provider: str | None
    terms: dict[str, Any] = field(default_factory=dict)


def is_healthy(stats: ProviderStats, *, timeout_rate: float, samples: int) -> bool:
    """Our verdict on one carrier. Deliberately our measurements, not theirs.

    A provider that graded its own health and was believed would be a way for
    an external system to move the safety boundary, which is the thing this
    whole layer exists to prevent. `reachable` and the timeout rate are
    observations of our own client -- what we saw when we called them -- not a
    status the carrier reported about itself.
    """
    if not stats.reachable:
        return False
    if stats.total >= MIN_SAMPLES and stats.failure_rate > FAILURE_RATE_LIMIT:
        return False
    if samples >= MIN_SAMPLES and timeout_rate > TIMEOUT_RATE_LIMIT:
        return False
    return True


class CircuitBreaker:
    def __init__(self, *, worker_id: str) -> None:
        self._worker_id = worker_id

    async def evaluate(
        self,
        cur: AsyncCursor,
        *,
        campaign_id: UUID,
        providers: Sequence[str],
        health: Mapping[str, Any],
        now: datetime,
    ) -> BreakerView:
        """Decide the breaker state and how many calls may be placed.

        Must run inside the same transaction that locked the campaign's safety
        row, which is what makes the probe claim below a compare-and-swap
        rather than a race.
        """
        stats = {
            name: await self._stats(
                cur, campaign_id=campaign_id, provider=name, now=now, health=health.get(name)
            )
            for name in providers
        }
        healthy = [
            name
            for name in providers
            if is_healthy(
                stats[name],
                timeout_rate=getattr(health.get(name), "timeout_rate", 0.0),
                samples=getattr(health.get(name), "samples", 0),
            )
        ]
        terms: dict[str, Any] = {
            "providers": {
                name: {
                    "total": s.total,
                    "failures": s.failures,
                    "failure_rate": round(s.failure_rate, 4),
                    "timeout_rate": round(
                        getattr(health.get(name), "timeout_rate", 0.0), 4
                    ),
                    "reachable": s.reachable,
                }
                for name, s in stats.items()
            },
            "healthy": healthy,
        }

        await cur.execute(
            "SELECT * FROM campaign_safety_state WHERE campaign_id = %s FOR UPDATE",
            (campaign_id,),
        )
        row = await cur.fetchone()
        opened_at = row["breaker_opened_at"] if row else None

        if opened_at is None:
            return await self._while_closed(
                cur, campaign_id=campaign_id, healthy=healthy, now=now, terms=terms
            )
        return await self._while_open(
            cur,
            campaign_id=campaign_id,
            row=row,
            healthy=healthy,
            providers=providers,
            now=now,
            terms=terms,
        )

    # -- the two halves of the state machine -----------------------------

    async def _while_closed(
        self, cur: AsyncCursor, *, campaign_id: UUID, healthy: list[str], now, terms: dict
    ) -> BreakerView:
        if healthy:
            # Route to a healthy carrier. With one provider configured this is
            # a no-op; with two it is the cheapest possible response to a sick
            # one, and it is why the breaker returns a provider name rather
            # than just a number.
            return BreakerView(
                state=BreakerState.CLOSED,
                allowance=None,
                provider=healthy[0],
                terms=terms,
            )

        await cur.execute(
            """
            UPDATE campaign_safety_state
            SET breaker_opened_at   = %(now)s,
                breaker_probe_owner = NULL,
                breaker_probe_at    = NULL
            WHERE campaign_id = %(campaign_id)s
            """,
            {"now": now, "campaign_id": campaign_id},
        )
        terms["opened_at"] = now
        return BreakerView(
            state=BreakerState.OPEN, allowance=0, provider=None, terms=terms
        )

    async def _while_open(
        self,
        cur: AsyncCursor,
        *,
        campaign_id: UUID,
        row: dict,
        healthy: list[str],
        providers: Sequence[str],
        now: datetime,
        terms: dict,
    ) -> BreakerView:
        opened_at: datetime = row["breaker_opened_at"]
        terms["opened_at"] = opened_at
        elapsed = (now - opened_at).total_seconds()

        if elapsed < OPEN_SECONDS:
            terms["open_for_seconds"] = elapsed
            return BreakerView(
                state=BreakerState.OPEN, allowance=0, provider=None, terms=terms
            )

        probe_owner = row["breaker_probe_owner"]
        probe_at = row["breaker_probe_at"]

        if probe_owner is None:
            return await self._claim_probe(
                cur,
                campaign_id=campaign_id,
                providers=providers,
                healthy=healthy,
                now=now,
                terms=terms,
            )

        verdict = await self._probe_verdict(
            cur, campaign_id=campaign_id, probe_at=probe_at, now=now
        )
        terms["probe_owner"] = probe_owner
        terms["probe_verdict"] = verdict

        if verdict == "PASSED":
            await self._close(cur, campaign_id=campaign_id)
            return BreakerView(
                state=BreakerState.CLOSED,
                allowance=None,
                provider=(healthy or list(providers))[0],
                terms=terms,
            )
        if verdict == "FAILED":
            # Straight back to OPEN for another full period. The probe is the
            # only call that gets to find out, so a failed one costs one
            # borrower rather than a batch.
            await cur.execute(
                """
                UPDATE campaign_safety_state
                SET breaker_opened_at   = %(now)s,
                    breaker_probe_owner = NULL,
                    breaker_probe_at    = NULL
                WHERE campaign_id = %(campaign_id)s
                """,
                {"now": now, "campaign_id": campaign_id},
            )
            return BreakerView(
                state=BreakerState.OPEN, allowance=0, provider=None, terms=terms
            )

        # PENDING: somebody's probe is still in the air. Everyone waits,
        # including the worker that placed it -- one probe means one.
        return BreakerView(
            state=BreakerState.HALF_OPEN, allowance=0, provider=None, terms=terms
        )

    # -- probe mechanics -------------------------------------------------

    async def _claim_probe(
        self,
        cur: AsyncCursor,
        *,
        campaign_id: UUID,
        providers: Sequence[str],
        healthy: list[str],
        now: datetime,
        terms: dict,
    ) -> BreakerView:
        """Try to become the one worker that dials the probe.

        The WHERE clause is the whole mechanism: the update only matches while
        the owner is still NULL, so of any number of workers arriving at the
        same instant exactly one gets a row back. Same shape as reserving an
        agent, for the same reason -- the claim and the state change are one
        write.
        """
        await cur.execute(
            """
            UPDATE campaign_safety_state
            SET breaker_probe_owner = %(worker_id)s,
                breaker_probe_at    = %(now)s
            WHERE campaign_id = %(campaign_id)s
              AND breaker_probe_owner IS NULL
            RETURNING campaign_id
            """,
            {"worker_id": self._worker_id, "now": now, "campaign_id": campaign_id},
        )
        won = await cur.fetchone() is not None
        terms["probe_claimed_by_me"] = won
        return BreakerView(
            state=BreakerState.HALF_OPEN,
            allowance=1 if won else 0,
            provider=(healthy or list(providers))[0] if won else None,
            terms=terms,
        )

    async def _probe_verdict(
        self, cur: AsyncCursor, *, campaign_id: UUID, probe_at: datetime, now: datetime
    ) -> str:
        """Did the probe call work? PASSED, FAILED or PENDING.

        Judged on the call the probe placed, which is any call created after
        the claim. Reaching RINGING is enough to pass: the question the breaker
        asks is whether the carrier is taking our calls, not whether a borrower
        chose to answer one.
        """
        if probe_at is None:
            return "FAILED"
        await cur.execute(
            """
            SELECT
              count(*) FILTER (
                  WHERE failure_reason = ANY(%(carrier_failures)s)
              )::int AS failed,
              count(*) FILTER (WHERE provider_call_id IS NOT NULL)::int AS accepted,
              count(*)::int AS total
            FROM calls
            WHERE campaign_id = %(campaign_id)s
              AND created_at >= %(probe_at)s
            """,
            {
                "campaign_id": campaign_id,
                "probe_at": probe_at,
                "carrier_failures": list(CARRIER_FAILURES),
            },
        )
        row = await cur.fetchone()
        if row["failed"]:
            return "FAILED"
        if row["accepted"]:
            return "PASSED"
        if (now - probe_at).total_seconds() > PROBE_TIMEOUT_SECONDS:
            # No answer either way for twenty seconds. Treated as a failure,
            # because the alternative is a breaker stuck half-open forever on a
            # carrier that silently swallowed the probe.
            return "FAILED"
        return "PENDING"

    async def _close(self, cur: AsyncCursor, *, campaign_id: UUID) -> None:
        await cur.execute(
            """
            UPDATE campaign_safety_state
            SET breaker_opened_at   = NULL,
                breaker_probe_owner = NULL,
                breaker_probe_at    = NULL
            WHERE campaign_id = %(campaign_id)s
            """,
            {"campaign_id": campaign_id},
        )

    async def _stats(
        self,
        cur: AsyncCursor,
        *,
        campaign_id: UUID,
        provider: str,
        now: datetime,
        health: Any,
    ) -> ProviderStats:
        await cur.execute(
            """
            SELECT
              count(*) FILTER (
                  WHERE failure_reason = ANY(%(carrier_failures)s)
              )::int AS failures,
              count(*)::int AS total
            FROM calls
            WHERE campaign_id = %(campaign_id)s
              AND provider = %(provider)s
              AND created_at > %(now)s::timestamptz
                  - make_interval(secs => %(window)s)
            """,
            {
                "campaign_id": campaign_id,
                "provider": provider,
                "now": now,
                "window": WINDOW_SECONDS,
                "carrier_failures": list(CARRIER_FAILURES),
            },
        )
        row = await cur.fetchone()
        return ProviderStats(
            provider=provider,
            total=row["total"],
            failures=row["failures"],
            timeouts=int(
                getattr(health, "timeout_rate", 0.0) * getattr(health, "samples", 0)
            ),
            reachable=getattr(health, "reachable", True),
        )
