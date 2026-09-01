"""Agent allocation and state transitions.

This module answers the first question the brief asks: two workers see the same
available agent at almost the same time, and both must not be able to reserve
it.

The answer is two SQL patterns and nothing else:

  reservation  SELECT ... FOR UPDATE SKIP LOCKED inside an UPDATE
  transitions  compare-and-swap on (id, version, state)

There is no application-level lock, no retry loop around a lock service, and no
cache. The row lock and the state write commit in the same transaction, so
there is no instant at which two workers both believe they own an agent -- and
no second source of truth that could disagree with the first.

Every function here takes a cursor and must be called inside a transaction. The
caller owns the transaction because reserving an agent and writing the call row
it is for have to commit together.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final
from uuid import UUID

from psycopg import AsyncCursor

from smartdialer.core.models import (
    Agent,
    AgentState,
    assert_legal_agent_transition,
)

# Sentinel for "do not touch this column", distinct from "set it to NULL".
UNCHANGED: Final = object()


@dataclass(frozen=True, slots=True)
class AgentReservation:
    """An agent this worker now owns, with the version it owns it at.

    The version is returned so the caller can compare-and-swap the next
    transition without re-reading the row: if anything else touched the agent
    in between, the CAS matches zero rows and the caller re-decides.
    """

    agent_id: UUID
    version: int


# ---------------------------------------------------------------------------
# Reservation
# ---------------------------------------------------------------------------

# Note on the shape of this statement: the locking clause has to come after
# LIMIT, and the SELECT has to be a subquery of an UPDATE rather than a
# separate statement. Selecting first and updating second would open a window
# between the two in which the lock exists but the state does not say so, and
# that window is exactly where a crash leaves an agent wedged.
RESERVE_AGENTS_SQL = """
UPDATE agents a
SET state            = 'RESERVED',
    lease_owner      = %(worker_id)s,
    lease_expires_at = %(now)s::timestamptz + make_interval(secs => %(lease_seconds)s),
    version          = version + 1,
    state_changed_at = %(now)s
WHERE a.id IN (
    SELECT id
    FROM agents
    WHERE campaign_id = %(campaign_id)s
      AND state = 'AVAILABLE'
    -- Longest-idle agent first. Fair distribution of work, and it keeps one
    -- unlucky agent from taking every call while another takes none.
    ORDER BY state_changed_at
    LIMIT %(n)s
    -- Note that this ORDER BY governs WHICH agents are chosen, not the order
    -- of the RETURNING rows below: the enclosing UPDATE emits them in whatever
    -- order the plan produced. Callers must not read anything into it.
    -- FOR UPDATE takes a row lock that lives until this transaction ends.
    -- SKIP LOCKED means a second worker arriving at the same instant neither
    -- blocks nor errors: it silently steps over the rows this worker holds and
    -- takes the next AVAILABLE agents instead. That is the whole mechanism.
    FOR UPDATE SKIP LOCKED
)
RETURNING a.id, a.version
"""


async def reserve_agents(
    cur: AsyncCursor,
    *,
    campaign_id: UUID,
    worker_id: str,
    n: int,
    lease_seconds: float,
    now: datetime,
) -> list[AgentReservation]:
    """Reserve up to `n` available agents for this worker.

    Returns FEWER than `n` when fewer are available, and an empty list when
    none are. Callers must treat a short return as the normal case and stop
    dialling, not assume they got what they asked for -- otherwise progressive
    mode reserves borrowers it has no agent for.
    """
    if n <= 0:
        return []

    await cur.execute(
        RESERVE_AGENTS_SQL,
        {
            "campaign_id": campaign_id,
            "worker_id": worker_id,
            "n": n,
            "lease_seconds": lease_seconds,
            "now": now,
        },
    )
    rows = await cur.fetchall()
    return [AgentReservation(agent_id=row["id"], version=row["version"]) for row in rows]


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


async def transition_agent(
    cur: AsyncCursor,
    *,
    agent_id: UUID,
    expected_version: int,
    expected_state: AgentState,
    target_state: AgentState,
    now: datetime,
    current_call_id: UUID | None | Any = UNCHANGED,
    wrap_up_ends_at: datetime | None | Any = UNCHANGED,
    lease_owner: str | None | Any = UNCHANGED,
    lease_expires_at: datetime | None | Any = UNCHANGED,
) -> Agent | None:
    """Compare-and-swap one agent from `expected_state` to `target_state`.

    Returns the updated agent, or None if the swap matched no rows -- meaning
    somebody else moved this agent first and our read was stale. The caller
    must re-read and re-decide. It must NEVER retry the write without the
    version guard: forcing the update through is how two workers end up
    believing they own the same agent.

    The transition is checked against the legal-transition table before the
    statement runs, so an impossible move raises rather than quietly matching
    zero rows and looking like a lost race.
    """
    assert_legal_agent_transition(expected_state, target_state)

    assignments = [
        "state = %(target_state)s",
        "version = version + 1",
        "state_changed_at = %(now)s",
    ]
    params: dict[str, Any] = {
        "agent_id": agent_id,
        "expected_version": expected_version,
        "expected_state": expected_state.value,
        "target_state": target_state.value,
        "now": now,
    }

    # Only touch the columns the caller named. UNCHANGED and None are
    # different instructions: leave it alone, versus clear it.
    for column, value in (
        ("current_call_id", current_call_id),
        ("wrap_up_ends_at", wrap_up_ends_at),
        ("lease_owner", lease_owner),
        ("lease_expires_at", lease_expires_at),
    ):
        if value is not UNCHANGED:
            assignments.append(f"{column} = %({column})s")
            params[column] = value

    await cur.execute(
        f"""
        UPDATE agents
        SET {', '.join(assignments)}
        WHERE id = %(agent_id)s
          AND version = %(expected_version)s
          AND state = %(expected_state)s
        RETURNING *
        """,
        params,
    )
    row = await cur.fetchone()
    return Agent.from_row(row) if row else None


async def release_agent(
    cur: AsyncCursor,
    *,
    agent_id: UUID,
    expected_version: int,
    expected_state: AgentState,
    now: datetime,
) -> Agent | None:
    """Hand a reserved or dialling agent back to the pool.

    Used when a dial fails, a borrower could not be found, or a lease expired
    with no live call behind it. Clears the lease along with the state, because
    a released agent that still carries a lease owner is a row the reaper will
    keep looking at forever.
    """
    return await transition_agent(
        cur,
        agent_id=agent_id,
        expected_version=expected_version,
        expected_state=expected_state,
        target_state=AgentState.AVAILABLE,
        now=now,
        current_call_id=None,
        lease_owner=None,
        lease_expires_at=None,
    )


async def renew_lease(
    cur: AsyncCursor,
    *,
    agent_id: UUID,
    worker_id: str,
    lease_seconds: float,
    now: datetime,
) -> bool:
    """Extend this worker's lease on an agent it already owns.

    Guarded on lease_owner, so a worker that lost its lease to the reaper
    cannot silently take the agent back: it gets False and must re-reserve.
    Deliberately does not bump `version` -- a lease renewal is not a state
    change, and bumping the version would invalidate a CAS the same worker is
    about to perform.
    """
    await cur.execute(
        """
        UPDATE agents
        SET lease_expires_at = %(now)s::timestamptz + make_interval(secs => %(lease_seconds)s)
        WHERE id = %(agent_id)s
          AND lease_owner = %(worker_id)s
        """,
        {
            "agent_id": agent_id,
            "worker_id": worker_id,
            "lease_seconds": lease_seconds,
            "now": now,
        },
    )
    return cur.rowcount == 1


# ---------------------------------------------------------------------------
# Fleet operations
# ---------------------------------------------------------------------------


async def login_agents(
    cur: AsyncCursor, *, agent_ids: list[UUID], now: datetime
) -> int:
    """Bring OFFLINE agents online. Returns how many actually moved."""
    if not agent_ids:
        return 0
    await cur.execute(
        """
        UPDATE agents
        SET state = 'AVAILABLE',
            version = version + 1,
            state_changed_at = %(now)s,
            last_heartbeat_at = %(now)s
        WHERE id = ANY(%(agent_ids)s)
          AND state = 'OFFLINE'
        """,
        {"agent_ids": agent_ids, "now": now},
    )
    return cur.rowcount


async def logout_agents(
    cur: AsyncCursor, *, agent_ids: list[UUID], now: datetime
) -> list[UUID]:
    """Take agents offline, but only the ones not mid-call.

    An agent in RESERVED, DIALING or CONNECTED is left alone: there may be a
    live call behind them, and dropping the agent would either abandon a
    borrower or leave a call bound to somebody who has gone home. Those agents
    go offline through the call's own completion path. Returns the ids that
    actually went offline, so the caller can see the difference.
    """
    if not agent_ids:
        return []
    await cur.execute(
        """
        UPDATE agents
        SET state = 'OFFLINE',
            version = version + 1,
            state_changed_at = %(now)s,
            lease_owner = NULL,
            lease_expires_at = NULL
        WHERE id = ANY(%(agent_ids)s)
          AND state IN ('AVAILABLE', 'WRAP_UP', 'PAUSED')
        RETURNING id
        """,
        {"agent_ids": agent_ids, "now": now},
    )
    return [row["id"] for row in await cur.fetchall()]


async def heartbeat(
    cur: AsyncCursor, *, agent_ids: list[UUID], now: datetime
) -> int:
    """Record that these agents are still there.

    An agent whose heartbeat goes stale is taken OFFLINE by the reaper. That is
    failure scenario 3 from the brief: 40 of 100 agents disappear within a few
    seconds, and how fast the dialer reacts is bounded by this timeout.
    """
    if not agent_ids:
        return 0
    await cur.execute(
        "UPDATE agents SET last_heartbeat_at = %(now)s WHERE id = ANY(%(agent_ids)s)",
        {"agent_ids": agent_ids, "now": now},
    )
    return cur.rowcount


async def count_agents_by_state(
    cur: AsyncCursor, *, campaign_id: UUID
) -> dict[AgentState, int]:
    """Per-state counts for the pacing snapshot. Served by
    agents_campaign_state_idx, so it is an index-only scan rather than a walk
    over the fleet."""
    await cur.execute(
        "SELECT state, count(*) AS n FROM agents WHERE campaign_id = %(campaign_id)s "
        "GROUP BY state",
        {"campaign_id": campaign_id},
    )
    rows = await cur.fetchall()
    counts = {state: 0 for state in AgentState}
    for row in rows:
        counts[AgentState(row["state"])] = row["n"]
    return counts


async def get_agent(cur: AsyncCursor, *, agent_id: UUID) -> Agent | None:
    """Read one agent, with its current version.

    Every compare-and-swap needs a version to swap against, and an event-driven
    transition (the borrower answered, the call ended) arrives without one --
    the event knows about a call, not about the agent's row. So the caller
    reads, decides, and swaps once. If the swap misses, it does NOT re-read and
    retry: something else moved the agent, and the reaper is the component that
    reconciles that, not a loop.
    """
    await cur.execute("SELECT * FROM agents WHERE id = %(id)s", {"id": agent_id})
    row = await cur.fetchone()
    return Agent.from_row(row) if row else None


async def expire_wrap_up(
    cur: AsyncCursor, *, now: datetime, limit: int = 500
) -> list[UUID]:
    """Return agents whose wrap-up timer has run out to the pool.

    Wrap-up is the one part of the agent lifecycle that is deterministic: the
    timer was set when the call ended and nothing external can change it. That
    is why wrap-up agents contribute to the pacing forecast with no variance at
    all -- we do not estimate whether they will be free in eight seconds, we
    know.

    Batched with SKIP LOCKED for the same reason as every other sweep here: two
    workers running this at once must not queue behind each other.
    """
    await cur.execute(
        """
        UPDATE agents
        SET state = 'AVAILABLE',
            version = version + 1,
            state_changed_at = %(now)s,
            wrap_up_ends_at = NULL,
            current_call_id = NULL,
            lease_owner = NULL,
            lease_expires_at = NULL
        WHERE id IN (
            SELECT id FROM agents
            WHERE state = 'WRAP_UP'
              AND wrap_up_ends_at <= %(now)s
            ORDER BY wrap_up_ends_at
            LIMIT %(limit)s
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id
        """,
        {"now": now, "limit": limit},
    )
    return [row["id"] for row in await cur.fetchall()]


async def agents_with_expired_leases(
    cur: AsyncCursor, *, now: datetime, states: tuple[AgentState, ...], limit: int = 500
) -> list[Agent]:
    """Leased agents in the given states whose worker has stopped renewing.

    Served by the partial agents_lease_idx, so the sweep is proportional to
    work in flight rather than to the size of the fleet. SKIP LOCKED so two
    reapers do not queue behind each other.
    """
    await cur.execute(
        """
        SELECT * FROM agents
        WHERE state = ANY(%(states)s)
          AND lease_expires_at IS NOT NULL
          AND lease_expires_at < %(now)s
        ORDER BY lease_expires_at
        LIMIT %(limit)s
        FOR UPDATE SKIP LOCKED
        """,
        {"states": [s.value for s in states], "now": now, "limit": limit},
    )
    return [Agent.from_row(row) for row in await cur.fetchall()]


async def expire_stale_heartbeats(
    cur: AsyncCursor, *, now: datetime, timeout_seconds: float, limit: int = 500
) -> list[UUID]:
    """Take agents offline once they stop reporting in.

    This is the brief's third failure scenario: a hundred agents available,
    forty vanish inside a few seconds. How fast the dialer notices is bounded
    entirely by this timeout, and the cost of noticing late is calls placed for
    agents who are not there -- which is to say, abandoned calls.

    Two deliberate restrictions.

    Only agents who have EVER reported in are eligible. A NULL heartbeat means
    an agent has not arrived, not that they have gone missing, and treating the
    two alike would take the whole fleet offline the moment this sweep first
    ran against seeded data.

    Only agents who are not mid-call. Somebody in DIALING or CONNECTED may have
    a live borrower on the line, and dropping them here would abandon that call
    to tidy up a session. Those agents come back through their call: it hits
    max_call_lifetime, is reconciled, settles the agent to AVAILABLE, and this
    sweep takes them offline on a later pass. Slower, and it never drops a
    stranger mid-sentence.
    """
    await cur.execute(
        """
        UPDATE agents
        SET state = 'OFFLINE',
            version = version + 1,
            state_changed_at = %(now)s,
            lease_owner = NULL,
            lease_expires_at = NULL,
            wrap_up_ends_at = NULL
        WHERE id IN (
            SELECT id FROM agents
            WHERE state IN ('AVAILABLE', 'WRAP_UP', 'PAUSED')
              AND last_heartbeat_at IS NOT NULL
              AND last_heartbeat_at < %(now)s::timestamptz
                                      - make_interval(secs => %(timeout_seconds)s)
            ORDER BY last_heartbeat_at
            LIMIT %(limit)s
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id
        """,
        {"now": now, "timeout_seconds": timeout_seconds, "limit": limit},
    )
    return [row["id"] for row in await cur.fetchall()]
