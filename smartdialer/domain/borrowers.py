"""Borrower selection and release.

Same two patterns as agents: SKIP LOCKED to hand each borrower to exactly one
worker, compare-and-swap for everything else.

The one thing that is genuinely different, and it matters:

    A LEASE releases the WORKER'S CLAIM on a borrower.
    A TERMINAL CALL releases the BORROWER FOR REDIAL.

Those are two different events governed by two independent clocks, and
conflating them is a real bug. A call can legitimately sit non-terminal for a
long time -- the worker crashed while the call was ANSWERED, the provider is
unreachable, reconciliation is backing off. During that window the worker's
claim has expired but the borrower is still on a live call.

If lease expiry alone returned the borrower to PENDING, the dialer would
reserve them again, insert a second call, and either hit the one-live-call-per-
borrower index (a confusing failure on correct behaviour) or, without that
index, actually ring the same person twice for the same debt during a provider
outage. The second is a compliance problem, not just an untidy one.

So release is split in two, and `release_expired_leases` will not return a
borrower to PENDING while any non-terminal call for them exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from psycopg import AsyncCursor

from smartdialer.core.models import Borrower, BorrowerState

# Kept as a SQL fragment rather than a parameter list because it appears in an
# index predicate too (calls_one_live_per_borrower_idx). If these two ever
# disagree, the index stops covering the query. Changing one means changing
# both, and this comment is here to say so.
NON_TERMINAL_CALL_STATES_SQL = (
    "('QUEUED','RESERVED','INITIATED','RINGING','ANSWERED','CONNECTED')"
)


@dataclass(frozen=True, slots=True)
class BorrowerReservation:
    borrower_id: UUID
    phone: str
    version: int
    attempts: int
    dpd_bucket: str | None
    priority: int


RESERVE_BORROWERS_SQL = f"""
UPDATE borrowers b
SET state            = 'RESERVED',
    lease_owner      = %(worker_id)s,
    lease_expires_at = %(now)s::timestamptz + make_interval(secs => %(lease_seconds)s),
    version          = version + 1
WHERE b.id IN (
    SELECT id
    FROM borrowers
    WHERE campaign_id = %(campaign_id)s
      AND state = 'PENDING'
      -- Retry backoff lives in this column, so a borrower who failed recently
      -- is simply not eligible yet. No separate scheduler.
      AND next_eligible_at <= %(now)s
      AND attempts < max_attempts
    ORDER BY priority DESC, next_eligible_at
    LIMIT %(n)s
    FOR UPDATE SKIP LOCKED
)
-- Belt and braces: even though the subquery filters on PENDING, a borrower
-- with a non-terminal call must never be handed out. The subquery cannot see a
-- borrower left RESERVED by the coupling rule above, but a borrower whose
-- state was repaired by hand could slip through, and dialling somebody twice
-- for one debt is not a mistake worth risking to save a join.
  AND NOT EXISTS (
    SELECT 1 FROM calls c
    WHERE c.borrower_id = b.id
      AND c.state IN {NON_TERMINAL_CALL_STATES_SQL}
  )
RETURNING b.id, b.phone, b.version, b.attempts, b.dpd_bucket, b.priority
"""


async def reserve_borrowers(
    cur: AsyncCursor,
    *,
    campaign_id: UUID,
    worker_id: str,
    n: int,
    lease_seconds: float,
    now: datetime,
) -> list[BorrowerReservation]:
    """Reserve up to `n` dialable borrowers.

    Like agent reservation, returns fewer than asked for when fewer are
    eligible, and an empty list when the campaign has run dry. A short return
    is normal, not an error: it means stop dialling this tick.
    """
    if n <= 0:
        return []

    await cur.execute(
        RESERVE_BORROWERS_SQL,
        {
            "campaign_id": campaign_id,
            "worker_id": worker_id,
            "n": n,
            "lease_seconds": lease_seconds,
            "now": now,
        },
    )
    rows = await cur.fetchall()
    return [
        BorrowerReservation(
            borrower_id=row["id"],
            phone=row["phone"],
            version=row["version"],
            attempts=row["attempts"],
            dpd_bucket=row["dpd_bucket"],
            priority=row["priority"],
        )
        for row in rows
    ]


async def release_borrower(
    cur: AsyncCursor,
    *,
    borrower_id: UUID,
    expected_version: int,
    now: datetime,
    retry_after_seconds: float = 0.0,
) -> Borrower | None:
    """Return a borrower to the pool immediately, attempts unchanged.

    This is the path for "we reserved them but never dialled" -- no agent was
    free, the provider rejected the call before it was placed. Nothing was
    attempted, so nothing is counted against their attempt budget.

    Compare-and-swap on version, so a borrower the reaper already recovered is
    not clobbered.
    """
    await cur.execute(
        """
        UPDATE borrowers
        SET state = 'PENDING',
            lease_owner = NULL,
            lease_expires_at = NULL,
            next_eligible_at = GREATEST(
                next_eligible_at,
                %(now)s::timestamptz + make_interval(secs => %(retry_after_seconds)s)
            ),
            version = version + 1
        WHERE id = %(borrower_id)s
          AND version = %(expected_version)s
          AND state = 'RESERVED'
        RETURNING *
        """,
        {
            "borrower_id": borrower_id,
            "expected_version": expected_version,
            "now": now,
            "retry_after_seconds": retry_after_seconds,
        },
    )
    row = await cur.fetchone()
    return Borrower.from_row(row) if row else None


async def record_attempt(
    cur: AsyncCursor,
    *,
    borrower_id: UUID,
    now: datetime,
    outcome: str,
    retry_after_seconds: float,
) -> Borrower | None:
    """Count one completed attempt and decide whether the borrower comes back.

    A borrower who has used their attempt budget goes to EXHAUSTED rather than
    PENDING, which is what stops the dialer ringing the same unreachable number
    forever. Not version-guarded: this runs from the call's terminal transition,
    which already holds the call row, and the outcome is a fact about a call
    that happened rather than a competing claim on the borrower.
    """
    await cur.execute(
        """
        UPDATE borrowers
        SET attempts = attempts + 1,
            last_outcome = %(outcome)s,
            lease_owner = NULL,
            lease_expires_at = NULL,
            next_eligible_at = %(now)s::timestamptz
                               + make_interval(secs => %(retry_after_seconds)s),
            state = CASE
                        WHEN attempts + 1 >= max_attempts THEN 'EXHAUSTED'
                        ELSE 'PENDING'
                    END,
            version = version + 1
        WHERE id = %(borrower_id)s
        RETURNING *
        """,
        {
            "borrower_id": borrower_id,
            "now": now,
            "outcome": outcome,
            "retry_after_seconds": retry_after_seconds,
        },
    )
    row = await cur.fetchone()
    return Borrower.from_row(row) if row else None


async def mark_done(cur: AsyncCursor, *, borrower_id: UUID, outcome: str) -> None:
    """Terminal success: the borrower spoke to an agent. No more dialling."""
    await cur.execute(
        """
        UPDATE borrowers
        SET state = 'DONE',
            last_outcome = %(outcome)s,
            lease_owner = NULL,
            lease_expires_at = NULL,
            version = version + 1
        WHERE id = %(borrower_id)s
        """,
        {"borrower_id": borrower_id, "outcome": outcome},
    )


# ---------------------------------------------------------------------------
# Lease recovery -- the coupling described at the top of this module
# ---------------------------------------------------------------------------

RELEASE_EXPIRED_LEASES_SQL = f"""
UPDATE borrowers b
SET state            = 'PENDING',
    lease_owner      = NULL,
    lease_expires_at = NULL,
    version          = version + 1
-- UPDATE takes no LIMIT, so the batch is bounded by a subquery -- the same
-- shape as reservation, and SKIP LOCKED for the same reason: two reapers
-- running at once must not queue behind each other.
WHERE b.id IN (
    SELECT id
    FROM borrowers
    WHERE state = 'RESERVED'
      AND lease_expires_at < %(now)s
    ORDER BY lease_expires_at
    LIMIT %(limit)s
    FOR UPDATE SKIP LOCKED
)
  -- The coupling. A borrower goes back in the pool only when nothing is still
  -- live for them. If a call is still in flight -- crashed worker, unreachable
  -- provider, reconciliation backing off -- the worker's claim is gone but the
  -- borrower is not free, and the call's own recovery path releases them.
  AND NOT EXISTS (
    SELECT 1 FROM calls c
    WHERE c.borrower_id = b.id
      AND c.state IN {NON_TERMINAL_CALL_STATES_SQL}
  )
RETURNING b.id
"""

HELD_BY_LIVE_CALL_SQL = f"""
SELECT b.id AS borrower_id,
       c.id AS call_id,
       c.state AS call_state,
       c.lease_expires_at AS call_lease_expires_at
FROM borrowers b
JOIN calls c ON c.borrower_id = b.id
WHERE b.state = 'RESERVED'
  AND b.lease_expires_at < %(now)s
  AND c.state IN {NON_TERMINAL_CALL_STATES_SQL}
"""


async def release_expired_leases(
    cur: AsyncCursor, *, now: datetime, limit: int = 500
) -> list[UUID]:
    """Return borrowers whose worker died and who have nothing live.

    Attempts are deliberately NOT incremented: the worker crashed, the borrower
    was never actually reached, and charging them an attempt for our failure
    would eventually mark a perfectly reachable person EXHAUSTED.
    """
    await cur.execute(RELEASE_EXPIRED_LEASES_SQL, {"now": now, "limit": limit})
    return [row["id"] for row in await cur.fetchall()]


async def borrowers_held_by_live_call(
    cur: AsyncCursor, *, now: datetime
) -> list[dict]:
    """Borrowers whose lease expired but whose call is still in flight.

    Deliberately observable. This set is where the two clocks disagree, so it
    is exactly what to look at when a campaign stalls: if it grows and does not
    drain, call reconciliation is stuck and the borrowers behind it are frozen.
    """
    await cur.execute(HELD_BY_LIVE_CALL_SQL, {"now": now})
    return list(await cur.fetchall())
