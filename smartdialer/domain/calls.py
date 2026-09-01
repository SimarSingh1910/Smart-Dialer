"""Call lifecycle and provider-event application.

This module answers the second question the brief asks: the provider sends
ANSWERED, ANSWERED, ANSWERED, COMPLETED -- or COMPLETED, ANSWERED, RINGING --
or the worker crashes in the middle, and the system still has to end up
somewhere sensible.

Three rules do all of the work.

  1. DEDUPLICATE IN THE DATABASE.
     Every event is inserted into provider_events with
     `ON CONFLICT (provider, provider_event_id) DO NOTHING RETURNING id`.
     No row back means we have seen it before, and we stop right there. There
     is no application-level "have I seen this?" lookup, because a read
     followed by a write races with itself; a unique index does not.

  2. STATE IS MONOTONIC.
     A call only ever moves to a strictly higher rank, so a late RINGING cannot
     drag a COMPLETED call backwards. All four terminal states share rank 9,
     which means the first terminal state to arrive is the one that stands --
     and that is deliberate: a COMPLETED arriving after we decided a call was
     ABANDONED must not erase the compliance event.

  3. FACTS ARE NOT MONOTONIC.
     Timestamps are absorbed unconditionally, with COALESCE so the first
     reading wins, INDEPENDENT of whether the event was too late to move the
     state. This is the part that is easy to miss. Rank monotonicity alone
     would settle COMPLETED, ANSWERED, RINGING at COMPLETED with answered_at
     still NULL -- and answered_at is what the answer rate is computed from, so
     the pacing engine would quietly be fed a lie by every provider that
     reorders its webhooks. Handling out-of-order events is not enough; you
     have to notice what naive handling costs you.

Everything here runs inside a transaction the caller owns, for the same reason
as in agents.py: the dedupe, the state change and the counter update have to
commit together or not at all.

What this module deliberately does NOT do: touch agents or borrowers. Applying
an event is a fact about a call. Deciding that an ANSWERED over-dial with no
free agent is an abandonment, releasing the agent, scheduling the borrower's
retry -- those are policy, they belong to the worker and the reaper, and they
are driven by the EventApplication result this module returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Final
from uuid import UUID

from psycopg import AsyncCursor

from smartdialer.core.models import (
    CALL_STATE_RANK,
    TERMINAL_CALL_STATES,
    Call,
    CallState,
    CampaignCounters,
    NormalisedEvent,
    ProviderEvent,
    assert_call_advances,
)

# Rank shared by COMPLETED, FAILED, CANCELLED and ABANDONED. A call at or above
# this rank is over, whichever way it ended.
TERMINAL_RANK: Final = CALL_STATE_RANK[CallState.COMPLETED]

# The timestamp columns. These are the "facts" of rule 3.
CALL_FACT_COLUMNS: Final = (
    "initiated_at",
    "ringing_at",
    "answered_at",
    "connected_at",
    "ended_at",
)


# ---------------------------------------------------------------------------
# The provider-event vocabulary
# ---------------------------------------------------------------------------
#
# Providers normalise their own webhook types into these strings inside their
# own modules. The dialer knows only this list, which is what lets one code
# path handle a tidy provider and a badly behaved one without a conditional.

EVENT_TYPE_TO_STATE: Final[dict[str, CallState]] = {
    "initiated": CallState.INITIATED,
    "ringing": CallState.RINGING,
    "answered": CallState.ANSWERED,
    # The borrower's leg was joined to an agent's leg. Only now is anybody
    # actually talking to anybody.
    "bridged": CallState.CONNECTED,
    "completed": CallState.COMPLETED,
    "failed": CallState.FAILED,
    "busy": CallState.FAILED,
    "no_answer": CallState.FAILED,
    "cancelled": CallState.CANCELLED,
}

# Which fact each event type teaches us.
EVENT_TYPE_TO_FACT: Final[dict[str, str]] = {
    "initiated": "initiated_at",
    "ringing": "ringing_at",
    "answered": "answered_at",
    "bridged": "connected_at",
    "completed": "ended_at",
    "failed": "ended_at",
    "busy": "ended_at",
    "no_answer": "ended_at",
    "cancelled": "ended_at",
}

# ABANDONED is absent from both tables on purpose. A provider cannot know
# whether an agent was on the line, so it cannot tell us a call was abandoned;
# only we know that. Abandonment is recorded by abandon_call() below, and
# because it is terminal, a "completed" event arriving afterwards is outranked
# and leaves it alone.


# Outcomes worth counting per campaign, maintained in the same transaction as
# the transition that caused them -- so the pacing snapshot never has to
# COUNT(*) over the calls table.
COUNTER_COLUMNS: Final[dict[CallState, str]] = {
    CallState.INITIATED: "calls_initiated",
    CallState.ANSWERED: "calls_answered",
    CallState.CONNECTED: "calls_connected",
    CallState.ABANDONED: "calls_abandoned",
    CallState.FAILED: "calls_failed",
}

# See the note on campaign_counters in the migration: one row per campaign
# would serialise every worker on a single row lock.
COUNTER_SHARDS: Final = 16


class EventResult:
    """The audit strings written to provider_events.apply_result.

    Every stored event ends up with exactly one of these, which makes "what did
    the system do with that webhook?" a single-row query rather than an
    archaeology exercise across log files.
    """

    DUPLICATE = "DUPLICATE"
    APPLIED = "APPLIED"
    # Stored, facts absorbed, but too late to move the state. Not an error --
    # this is the normal outcome for a provider that reorders.
    STALE = "STALE"
    # We have no call with this provider_call_id. Left unapplied for retry: it
    # is usually a race with our own INSERT rather than a genuinely unknown
    # call. See unapplied_events().
    UNKNOWN_CALL = "UNKNOWN_CALL"
    IGNORED = "IGNORED"
    # The reaper and the carrier disagreed about how a call ended. Written as
    # TERMINAL_CONFLICT:<ours><-<theirs>. See terminal_conflict() below.
    TERMINAL_CONFLICT = "TERMINAL_CONFLICT"


def terminal_conflict(ours: CallState, theirs: CallState) -> str:
    """Label a call whose ending we inferred and the carrier later contradicted.

    All terminal states share rank 9, so the first one written stands. That is
    the right rule -- it is what stops a late COMPLETED erasing an ABANDONED --
    but it has a consequence worth measuring: when the reaper force-fails a
    call at max_call_lifetime and the provider afterwards reports it COMPLETED,
    the outcome on record is now locally INFERRED rather than provider truth.

    The facts still absorb through COALESCE, so the answer rate stays honest.
    What changes is the confidence behind the final state, and the honest thing
    is to count how often it happens rather than let it hide behind a rule that
    is correct in general. "How often did my reaper guess wrong about a call
    the carrier knew about" is a number worth being able to quote.
    """
    return f"{EventResult.TERMINAL_CONFLICT}:{ours.value}<-{theirs.value}"


@dataclass(frozen=True, slots=True)
class EventApplication:
    """What happened to one event.

    Carries enough for the caller to decide policy without re-reading the call:
    the state before and after, and whether the call has an agent behind it --
    which is what turns "the borrower answered" into "the borrower answered and
    there is nobody to talk to them", the abandonment decision.
    """

    result: str
    duplicate: bool = False
    transitioned: bool = False
    event_row_id: int | None = None
    call: Call | None = None
    previous_state: CallState | None = None
    new_state: CallState | None = None

    @property
    def needs_agent(self) -> bool:
        """A borrower is on the line with no agent bound to the call.

        True only on the transition INTO answered, so it fires once per call
        however many duplicate ANSWERED events the provider sends. That is the
        whole reason abandonment is driven off `transitioned` and not off the
        arrival of an event.
        """
        return (
            self.transitioned
            and self.new_state is CallState.ANSWERED
            and self.call is not None
            and self.call.agent_id is None
        )


# ---------------------------------------------------------------------------
# Creating a call -- the intent log
# ---------------------------------------------------------------------------


async def create_call(
    cur: AsyncCursor,
    *,
    call_id: UUID,
    campaign_id: UUID,
    borrower_id: UUID,
    provider: str,
    idempotency_key: str,
    now: datetime,
    worker_id: str,
    agent_id: UUID | None = None,
    is_overdial: bool = False,
    attempt: int = 1,
    predicted_p: Decimal | float | None = None,
    lease_seconds: float | None = None,
    state: CallState = CallState.INITIATED,
) -> Call:
    """Write down the intent to place a call, BEFORE the provider is called.

    The ordering is the whole point and it is not negotiable. We generate the
    idempotency key, commit this row, and only then hand the key to the
    provider. If the worker dies in the gap, recovery finds a row that says "we
    may have placed this call" and asks the provider about that exact key
    instead of guessing whether a stranger's phone is ringing. Calling the
    provider first and recording afterwards loses the call on any crash in
    between -- and in this domain a lost call is a live one that nobody owns.

    agent_id is NULL for a predictive over-dial: the call exists before anyone
    is bound to it. That is exactly the risk predictive dialling takes on, and
    it is visible in the schema rather than buried in a flag.

    The unique partial index calls_one_live_per_borrower_idx makes a second
    live call for the same borrower raise here. That is intended: allocation is
    supposed to have prevented it, and a loud integrity error is the right way
    to find out that it did not.
    """
    await cur.execute(
        """
        INSERT INTO calls (
            id, campaign_id, borrower_id, agent_id, provider, idempotency_key,
            state, attempt, is_overdial, predicted_p,
            initiated_at, lease_owner, lease_expires_at, created_at
        )
        VALUES (
            %(call_id)s, %(campaign_id)s, %(borrower_id)s, %(agent_id)s,
            %(provider)s, %(idempotency_key)s, %(state)s, %(attempt)s,
            %(is_overdial)s, %(predicted_p)s,
            %(initiated_at)s,
            %(worker_id)s,
            CASE WHEN %(lease_seconds)s::float8 IS NULL THEN NULL
                 ELSE %(now)s::timestamptz + make_interval(secs => %(lease_seconds)s)
            END,
            %(now)s
        )
        RETURNING *
        """,
        {
            "call_id": call_id,
            "campaign_id": campaign_id,
            "borrower_id": borrower_id,
            "agent_id": agent_id,
            "provider": provider,
            "idempotency_key": idempotency_key,
            "state": state.value,
            # Decided here rather than with a CASE on %(state)s in the SQL:
            # the same parameter cannot be both a call_state and a text
            # comparison, and PostgreSQL says so rather than guessing.
            "initiated_at": now if state is CallState.INITIATED else None,
            "attempt": attempt,
            "is_overdial": is_overdial,
            "predicted_p": predicted_p,
            "now": now,
            "worker_id": worker_id,
            "lease_seconds": lease_seconds,
        },
    )
    call = Call.from_row(await cur.fetchone())
    if state is CallState.INITIATED:
        await bump_counter(
            cur, campaign_id=campaign_id, column="calls_initiated", worker_id=worker_id
        )
    return call


async def attach_provider_call_id(
    cur: AsyncCursor,
    *,
    call_id: UUID,
    provider_call_id: str,
    now: datetime,
) -> Call | None:
    """Record the provider's own id for a call we already wrote down.

    Deliberately not version-guarded and deliberately idempotent: this is
    learning a fact about a call, not competing for it. Crash recovery calls it
    too, after finding the call by idempotency key, and it has to be safe to
    run twice with the same value.

    The guard is on provider_call_id being unset or identical, so a second,
    DIFFERENT provider id for one call matches no rows and comes back None --
    which means we placed the same call twice and very much want to know.
    """
    await cur.execute(
        """
        UPDATE calls
        SET provider_call_id = %(provider_call_id)s,
            initiated_at = COALESCE(initiated_at, %(now)s::timestamptz)
        WHERE id = %(call_id)s
          AND (provider_call_id IS NULL OR provider_call_id = %(provider_call_id)s)
        RETURNING *
        """,
        {"call_id": call_id, "provider_call_id": provider_call_id, "now": now},
    )
    row = await cur.fetchone()
    return Call.from_row(row) if row else None


# ---------------------------------------------------------------------------
# Transitions we decide ourselves
# ---------------------------------------------------------------------------


async def transition_call(
    cur: AsyncCursor,
    *,
    call_id: UUID,
    expected_version: int,
    expected_state: CallState,
    target_state: CallState,
    worker_id: str,
) -> Call | None:
    """Compare-and-swap a call we believe we own, forwards only.

    Same contract as transition_agent: None means somebody else moved it first
    and the caller must re-read rather than force the write.

    This is for transitions WE decide. Provider events do not come through
    here -- they go through apply_event, which guards on rank instead of
    version, because an event that lost a race is an ordinary occurrence and
    not something to re-decide.
    """
    assert_call_advances(expected_state, target_state)

    await cur.execute(
        """
        UPDATE calls
        SET state = %(target_state)s,
            version = version + 1
        WHERE id = %(call_id)s
          AND version = %(expected_version)s
          AND state = %(expected_state)s
        RETURNING *
        """,
        {
            "call_id": call_id,
            "expected_version": expected_version,
            "expected_state": expected_state.value,
            "target_state": target_state.value,
        },
    )
    row = await cur.fetchone()
    if row is None:
        return None
    call = Call.from_row(row)
    await _count_transition(
        cur, call=call, target_state=target_state, worker_id=worker_id
    )
    return call


async def connect_call(
    cur: AsyncCursor, *, call_id: UUID, now: datetime, worker_id: str
) -> Call | None:
    """Record that the borrower's leg was bridged to an agent.

    Guarded on rank rather than version so it is safe from either direction:
    our own bridge and a "bridged" webhook can race, and whichever arrives
    first wins while the second is outranked and quietly does nothing.
    """
    return await _advance_by_rank(
        cur,
        call_id=call_id,
        target_state=CallState.CONNECTED,
        worker_id=worker_id,
        facts={"connected_at": now},
    )


async def abandon_call(
    cur: AsyncCursor, *, call_id: UUID, now: datetime, worker_id: str, reason: str
) -> Call | None:
    """Terminal: a human answered and we had nobody to give them to.

    This is the compliance event the whole predictive design exists to keep
    rare. It is never folded into COMPLETED or FAILED, it is counted against
    the campaign's abandon budget the moment it is written, and the AIMD
    controller halves the over-dial credit off the back of it. Recording it
    honestly is the point: a dialer that hides its abandons optimises the
    metric instead of the behaviour.
    """
    return await terminate_call(
        cur,
        call_id=call_id,
        target_state=CallState.ABANDONED,
        now=now,
        worker_id=worker_id,
        failure_reason=reason,
    )


async def terminate_call(
    cur: AsyncCursor,
    *,
    call_id: UUID,
    target_state: CallState,
    now: datetime,
    worker_id: str,
    failure_reason: str | None = None,
) -> Call | None:
    """End a call. The first terminal state to be written wins.

    Guarded on rank rather than on version, because ending a call is not a
    competitive act: if the reaper and a webhook both decide the call is over
    at the same moment we want one of them to win quietly, not a retry loop.
    Returns None when the call was already over.
    """
    if CALL_STATE_RANK[target_state] != TERMINAL_RANK:
        raise ValueError(f"{target_state.value} is not a terminal state")
    return await _advance_by_rank(
        cur,
        call_id=call_id,
        target_state=target_state,
        worker_id=worker_id,
        facts={"ended_at": now},
        failure_reason=failure_reason,
    )


async def renew_call_lease(
    cur: AsyncCursor,
    *,
    call_id: UUID,
    worker_id: str,
    lease_seconds: float,
    now: datetime,
) -> bool:
    """Extend this worker's claim on a call it is still driving.

    Guarded on lease_owner, so a worker the reaper has already given up on
    cannot silently take the call back. Does not bump version, for the same
    reason as renew_lease in agents.py: a lease is not a state change.
    """
    await cur.execute(
        """
        UPDATE calls
        SET lease_expires_at = %(now)s::timestamptz + make_interval(secs => %(lease_seconds)s)
        WHERE id = %(call_id)s
          AND lease_owner = %(worker_id)s
        """,
        {
            "call_id": call_id,
            "worker_id": worker_id,
            "lease_seconds": lease_seconds,
            "now": now,
        },
    )
    return cur.rowcount == 1


# ---------------------------------------------------------------------------
# Event application
# ---------------------------------------------------------------------------

INSERT_EVENT_SQL = """
INSERT INTO provider_events (
    provider, provider_event_id, provider_call_id, event_type, provider_ts,
    received_at, payload
)
VALUES (
    %(provider)s, %(provider_event_id)s, %(provider_call_id)s,
    %(event_type)s, %(provider_ts)s,
    -- received_at comes from the injected clock, NOT from the column's now()
    -- default. The retry sweep compares this against the clock, so a
    -- wall-clock value here would make an event received in a simulation
    -- running on virtual time look either decades old or not yet arrived.
    -- Time enters this system in exactly one place, and this is not an
    -- exception to that.
    %(received_at)s, %(payload)s
)
-- The deduplication mechanism, and the only one. A provider that sends
-- ANSWERED three times gets one row here and therefore causes one transition.
ON CONFLICT (provider, provider_event_id) DO NOTHING
RETURNING id
"""

# Lock the call for the rest of the transaction. Two events for the same call
# arriving on two workers at the same instant must not interleave their
# read-modify-write. Plain FOR UPDATE, not SKIP LOCKED: the second one has to
# wait its turn, not be dropped -- skipping here would lose an event.
LOCK_CALL_BY_PROVIDER_ID_SQL = """
SELECT * FROM calls
WHERE provider = %(provider)s
  AND provider_call_id = %(provider_call_id)s
LIMIT 1
FOR UPDATE
"""


async def apply_event(
    cur: AsyncCursor,
    *,
    event: NormalisedEvent,
    now: datetime,
    worker_id: str,
) -> EventApplication:
    """Store one provider event and apply it, inside the caller's transaction.

    The order below is fixed and each step depends on the one before it:

        1. deduplicate     -- unique index; a duplicate stops here
        2. lock the call   -- SELECT ... FOR UPDATE
        3. absorb facts    -- unconditional, COALESCE, order-independent
        4. advance state   -- forwards only, by rank
        5. record what happened, on the event row

    Steps 3 and 4 are separate statements on purpose. Merging them -- writing
    the timestamp only when the state also moves -- is the bug described at the
    top of this module: it silently drops answered_at for every provider that
    reorders, and corrupts the answer rate the pacing engine runs on.
    """
    row = await _insert_event(cur, event=event, received_at=now)
    if row is None:
        # Seen before. Not an error and not a transition: this is the
        # ANSWERED, ANSWERED, ANSWERED, COMPLETED case from the brief.
        return EventApplication(result=EventResult.DUPLICATE, duplicate=True)

    return await _apply_stored(
        cur, event_row_id=row["id"], event=event, now=now, worker_id=worker_id
    )


async def reapply_stored_event(
    cur: AsyncCursor,
    *,
    stored: ProviderEvent,
    now: datetime,
    worker_id: str,
) -> EventApplication:
    """Retry an event that was stored but could not be applied.

    The retry path for UNKNOWN_CALL. The dedupe insert already happened when
    the event was first received, so it is skipped here: re-running it would
    conflict with the event's own row and the retry would report itself a
    duplicate forever.
    """
    if stored.id is None:
        raise ValueError("a stored event must have been written before it is reapplied")
    event = NormalisedEvent(
        provider=stored.provider,
        provider_event_id=stored.provider_event_id,
        provider_call_id=stored.provider_call_id or "",
        event_type=stored.event_type,
        provider_ts=stored.provider_ts,
        payload=stored.payload,
    )
    return await _apply_stored(
        cur, event_row_id=stored.id, event=event, now=now, worker_id=worker_id
    )


async def _apply_stored(
    cur: AsyncCursor,
    *,
    event_row_id: int,
    event: NormalisedEvent,
    now: datetime,
    worker_id: str,
) -> EventApplication:
    # --- 2. find and lock the call ------------------------------------
    await cur.execute(
        LOCK_CALL_BY_PROVIDER_ID_SQL,
        {"provider": event.provider, "provider_call_id": event.provider_call_id},
    )
    row = await cur.fetchone()
    if row is None:
        # Usually a race: the provider's webhook beat the commit of our own
        # INSERT for the call we just placed. Left unapplied so that
        # unapplied_events() picks it up a moment later. Never a crash and
        # never discarded -- an event we cannot yet explain is still evidence.
        await _finish_event(
            cur, event_row_id, applied=False, result=EventResult.UNKNOWN_CALL
        )
        return EventApplication(
            result=EventResult.UNKNOWN_CALL, event_row_id=event_row_id
        )

    call = Call.from_row(row)
    previous_state = call.state
    target_state = EVENT_TYPE_TO_STATE.get(event.event_type)

    if target_state is None:
        # An event type we do not model. Stored, marked, ignored. Crashing on
        # an unrecognised string would let a provider take the dialer down by
        # adding a webhook type on their side.
        await _finish_event(cur, event_row_id, applied=True, result=EventResult.IGNORED)
        return EventApplication(
            result=EventResult.IGNORED,
            event_row_id=event_row_id,
            call=call,
            previous_state=previous_state,
            new_state=previous_state,
        )

    # --- 3. absorb the facts, whatever the state says -----------------
    call = await _absorb_facts(cur, call_id=call.id, facts=_facts_for(event, now=now))

    # --- 4. advance the state, forwards only --------------------------
    advanced = await _advance_state(cur, call_id=call.id, target_state=target_state)
    transitioned = advanced is not None
    if advanced is not None:
        call = advanced
        if target_state is CallState.FAILED:
            reason = None
            if isinstance(event.payload, dict):
                reason = event.payload.get("reason")
            await _set_failure_reason(
                cur, call_id=call.id, reason=reason or event.event_type
            )
        await _count_transition(
            cur, call=call, target_state=target_state, worker_id=worker_id
        )

    # --- 5. record the outcome on the event row -----------------------
    if transitioned:
        result = EventResult.APPLIED
    elif (
        CALL_STATE_RANK[target_state] == TERMINAL_RANK
        and previous_state in TERMINAL_CALL_STATES
        and previous_state is not target_state
    ):
        # Both sides think the call is over and they disagree about how. Ours
        # stands, because it was written first, but the disagreement is
        # recorded rather than flattened into an ordinary STALE.
        result = terminal_conflict(previous_state, target_state)
    else:
        result = EventResult.STALE
    await _finish_event(cur, event_row_id, applied=True, result=result)

    return EventApplication(
        result=result,
        transitioned=transitioned,
        event_row_id=event_row_id,
        call=call,
        previous_state=previous_state,
        new_state=call.state,
    )


async def _insert_event(
    cur: AsyncCursor, *, event: NormalisedEvent, received_at: datetime
) -> dict | None:
    await cur.execute(
        INSERT_EVENT_SQL,
        {
            "received_at": received_at,
            "provider": event.provider,
            "provider_event_id": event.provider_event_id,
            "provider_call_id": event.provider_call_id or None,
            "event_type": event.event_type,
            "provider_ts": event.provider_ts,
            # Serialised here rather than handed over as a dict, so the column
            # holds exactly what the provider sent even when the payload
            # contains something jsonb cannot adapt on its own.
            "payload": json.dumps(event.payload, default=str),
        },
    )
    return await cur.fetchone()


def _facts_for(event: NormalisedEvent, *, now: datetime) -> dict[str, datetime]:
    """Which timestamps this event teaches us.

    The event type implies one column; `facts` may carry more, which is how a
    reconciliation poll applies everything the provider knows in one go.

    provider_ts is used when the provider stamped the event, and our own clock
    otherwise. Not every provider stamps its webhooks, and a slightly late
    timestamp is far better than a NULL in the column the answer rate and the
    wait-time metric are computed from.
    """
    facts: dict[str, datetime] = {
        column: ts for column, ts in event.facts.items() if column in CALL_FACT_COLUMNS
    }
    column = EVENT_TYPE_TO_FACT.get(event.event_type)
    if column is not None:
        facts.setdefault(column, event.provider_ts or now)
    return facts


ABSORB_FACTS_SQL = """
UPDATE calls
SET initiated_at = COALESCE(initiated_at, %(initiated_at)s::timestamptz),
    ringing_at   = COALESCE(ringing_at,   %(ringing_at)s::timestamptz),
    answered_at  = COALESCE(answered_at,  %(answered_at)s::timestamptz),
    connected_at = COALESCE(connected_at, %(connected_at)s::timestamptz),
    ended_at     = COALESCE(ended_at,     %(ended_at)s::timestamptz),
    -- The headline metric: how long a human waited between saying hello and
    -- hearing an agent. Computed from whichever readings we hold after this
    -- statement rather than at the moment of connection, so it still comes out
    -- right when the two events that produce it arrive in the wrong order.
    -- GREATEST(0, ...) because provider and local clocks disagree by small
    -- amounts and a negative wait is meaningless rather than informative.
    --
    -- The explicit NULL test is load-bearing and not defensive noise:
    -- GREATEST IGNORES nulls in PostgreSQL, so GREATEST(0, NULL) is 0, not
    -- NULL. Without the CASE, every call that had not connected yet would be
    -- stamped with a wait of zero milliseconds -- a metric that reads
    -- perfectly while measuring nothing.
    wait_ms = COALESCE(
        wait_ms,
        CASE
            WHEN COALESCE(connected_at, %(connected_at)s::timestamptz) IS NOT NULL
             AND COALESCE(answered_at,  %(answered_at)s::timestamptz) IS NOT NULL
            THEN GREATEST(
                0,
                (EXTRACT(EPOCH FROM (
                    COALESCE(connected_at, %(connected_at)s::timestamptz)
                    - COALESCE(answered_at, %(answered_at)s::timestamptz)
                )) * 1000)::int
            )
        END
    )
WHERE id = %(call_id)s
RETURNING *
"""


async def _absorb_facts(
    cur: AsyncCursor, *, call_id: UUID, facts: dict[str, datetime]
) -> Call:
    params: dict[str, Any] = {column: facts.get(column) for column in CALL_FACT_COLUMNS}
    params["call_id"] = call_id
    await cur.execute(ABSORB_FACTS_SQL, params)
    return Call.from_row(await cur.fetchone())


ADVANCE_STATE_SQL = """
UPDATE calls
SET state = %(target_state)s,
    version = version + 1
WHERE id = %(call_id)s
  -- Forwards only. A late RINGING for a COMPLETED call matches no rows, and a
  -- second terminal state cannot displace the first -- so an ABANDONED call
  -- stays ABANDONED when the provider's COMPLETED turns up behind it.
  AND state_rank < %(target_rank)s
RETURNING *
"""


async def _advance_state(
    cur: AsyncCursor, *, call_id: UUID, target_state: CallState
) -> Call | None:
    await cur.execute(
        ADVANCE_STATE_SQL,
        {
            "call_id": call_id,
            "target_state": target_state.value,
            "target_rank": CALL_STATE_RANK[target_state],
        },
    )
    row = await cur.fetchone()
    return Call.from_row(row) if row else None


async def _advance_by_rank(
    cur: AsyncCursor,
    *,
    call_id: UUID,
    target_state: CallState,
    worker_id: str,
    facts: dict[str, datetime],
    failure_reason: str | None = None,
) -> Call | None:
    """Absorb facts, then advance by rank.

    Our own transitions follow exactly the same two rules as a provider event,
    which is why they share this helper: the timestamp is recorded even when
    the state has already moved past this point. Returns None when the call had
    already gone at least this far.
    """
    await _absorb_facts(cur, call_id=call_id, facts=facts)
    call = await _advance_state(cur, call_id=call_id, target_state=target_state)
    if call is None:
        return None
    if failure_reason is not None:
        await _set_failure_reason(cur, call_id=call_id, reason=failure_reason)
    await _count_transition(
        cur, call=call, target_state=target_state, worker_id=worker_id
    )
    return call


async def _set_failure_reason(cur: AsyncCursor, *, call_id: UUID, reason: str) -> None:
    """First explanation wins, for the same reason the first terminal state
    does: later events describe the consequence, not the cause."""
    await cur.execute(
        "UPDATE calls SET failure_reason = COALESCE(failure_reason, %(reason)s) "
        "WHERE id = %(call_id)s",
        {"call_id": call_id, "reason": reason[:500]},
    )


async def _finish_event(
    cur: AsyncCursor, event_row_id: int, *, applied: bool, result: str
) -> None:
    await cur.execute(
        "UPDATE provider_events SET applied = %(applied)s, apply_result = %(result)s "
        "WHERE id = %(id)s",
        {"id": event_row_id, "applied": applied, "result": result},
    )


# ---------------------------------------------------------------------------
# The retry worklist
# ---------------------------------------------------------------------------


async def unapplied_events(
    cur: AsyncCursor,
    *,
    now: datetime,
    older_than_seconds: float = 1.0,
    limit: int = 200,
) -> list[ProviderEvent]:
    """Events stored but not yet applied, oldest first.

    Served by the partial index provider_events_unapplied_idx, so the sweep
    costs what the backlog costs rather than what the table costs.

    `older_than_seconds` gives our own INSERT time to commit before we decide
    an event names a call we do not have; without it the sweeper spins on
    events that are about to become applicable anyway. SKIP LOCKED so two
    ingesters can sweep at once without queueing behind each other.
    """
    await cur.execute(
        """
        SELECT * FROM provider_events
        WHERE NOT applied
          AND received_at <= %(now)s::timestamptz - make_interval(secs => %(age)s)
        ORDER BY id
        LIMIT %(limit)s
        FOR UPDATE SKIP LOCKED
        """,
        {"now": now, "age": older_than_seconds, "limit": limit},
    )
    return [ProviderEvent.from_row(row) for row in await cur.fetchall()]


async def abandon_unmatched_event(
    cur: AsyncCursor, *, event_row_id: int, result: str = "UNMATCHED"
) -> None:
    """Stop retrying an event whose call never appeared.

    Marked applied so it leaves the worklist, with a result that says it was
    never matched. The raw payload stays: a call we never recorded but the
    provider believes in is exactly the sort of thing worth being able to find
    afterwards.
    """
    await _finish_event(cur, event_row_id, applied=True, result=result)


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


async def bump_counter(
    cur: AsyncCursor, *, campaign_id: UUID, column: str, worker_id: str, n: int = 1
) -> None:
    """Add to one sharded campaign counter.

    The shard is chosen by hashing the worker id, so each worker keeps to its
    own row and workers do not serialise on a single hot one. Interpolating
    `column` into the SQL is safe because it can only come from
    COUNTER_COLUMNS -- which is asserted here rather than assumed.

    An upsert rather than a plain UPDATE so a campaign whose counter shards
    were never pre-created still counts correctly; the alternative is losing
    numbers silently, which is the worst way to lose them.
    """
    if column not in COUNTER_COLUMNS.values():
        raise ValueError(f"unknown counter column {column!r}")
    await cur.execute(
        f"""
        INSERT INTO campaign_counters (campaign_id, shard, {column})
        -- hashtext can return a negative number; the double modulo keeps the
        -- shard inside [0, {COUNTER_SHARDS}).
        VALUES (
            %(campaign_id)s,
            ((hashtext(%(worker_id)s) %% {COUNTER_SHARDS}) + {COUNTER_SHARDS})
                %% {COUNTER_SHARDS},
            %(n)s
        )
        ON CONFLICT (campaign_id, shard)
        DO UPDATE SET {column} = campaign_counters.{column} + EXCLUDED.{column}
        """,
        {"campaign_id": campaign_id, "worker_id": worker_id, "n": n},
    )


async def _count_transition(
    cur: AsyncCursor, *, call: Call, target_state: CallState, worker_id: str
) -> None:
    """Count an outcome once, at the moment it actually happened.

    Only ever called after an UPDATE that matched a row, so duplicated and
    reordered events cannot inflate the numbers: what is being counted is the
    transition, not the arrival of an event.
    """
    column = COUNTER_COLUMNS.get(target_state)
    if column is None:
        return
    await bump_counter(
        cur, campaign_id=call.campaign_id, column=column, worker_id=worker_id
    )


async def read_counters(cur: AsyncCursor, *, campaign_id: UUID) -> CampaignCounters:
    """Sum the shards. One index scan over at most COUNTER_SHARDS rows."""
    await cur.execute(
        """
        -- sum() over bigint returns numeric, which arrives in Python as a
        -- Decimal and poisons every arithmetic expression downstream. Cast
        -- back to bigint here so the dataclass really does hold ints.
        SELECT campaign_id,
               COALESCE(sum(calls_initiated), 0)::bigint AS calls_initiated,
               COALESCE(sum(calls_answered),  0)::bigint AS calls_answered,
               COALESCE(sum(calls_connected), 0)::bigint AS calls_connected,
               COALESCE(sum(calls_abandoned), 0)::bigint AS calls_abandoned,
               COALESCE(sum(calls_failed),    0)::bigint AS calls_failed
        FROM campaign_counters
        WHERE campaign_id = %(campaign_id)s
        GROUP BY campaign_id
        """,
        {"campaign_id": campaign_id},
    )
    row = await cur.fetchone()
    if row is None:
        return CampaignCounters(campaign_id=campaign_id)
    return CampaignCounters.from_row(row)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


async def get_call(cur: AsyncCursor, *, call_id: UUID) -> Call | None:
    await cur.execute("SELECT * FROM calls WHERE id = %(id)s", {"id": call_id})
    row = await cur.fetchone()
    return Call.from_row(row) if row else None


async def get_call_by_idempotency_key(cur: AsyncCursor, *, key: str) -> Call | None:
    """The crash-recovery lookup, served by the UNIQUE constraint on the
    column. This is what makes "did we already place this call?" answerable
    after a worker dies mid-flight."""
    await cur.execute(
        "SELECT * FROM calls WHERE idempotency_key = %(key)s", {"key": key}
    )
    row = await cur.fetchone()
    return Call.from_row(row) if row else None


async def get_call_by_provider_call_id(
    cur: AsyncCursor, *, provider: str, provider_call_id: str
) -> Call | None:
    await cur.execute(
        "SELECT * FROM calls WHERE provider = %(provider)s "
        "AND provider_call_id = %(provider_call_id)s LIMIT 1",
        {"provider": provider, "provider_call_id": provider_call_id},
    )
    row = await cur.fetchone()
    return Call.from_row(row) if row else None


async def count_calls_by_state(
    cur: AsyncCursor, *, campaign_id: UUID
) -> dict[CallState, int]:
    """In-flight counts for the pacing snapshot.

    Restricted to non-terminal calls so it is served by calls_inflight_idx and
    costs what is happening now rather than everything that ever happened.
    """
    await cur.execute(
        """
        SELECT state, count(*) AS n
        FROM calls
        WHERE campaign_id = %(campaign_id)s
          AND state IN ('RESERVED','INITIATED','RINGING','ANSWERED','CONNECTED')
        GROUP BY state
        """,
        {"campaign_id": campaign_id},
    )
    counts = {state: 0 for state in CallState}
    for row in await cur.fetchall():
        counts[CallState(row["state"])] = row["n"]
    return counts


# ---------------------------------------------------------------------------
# The reaper's worklist
# ---------------------------------------------------------------------------


async def calls_with_expired_leases(
    cur: AsyncCursor, *, now: datetime, limit: int = 200
) -> list[Call]:
    """Non-terminal calls whose worker has stopped renewing their lease.

    Served by the partial calls_lease_idx, so the sweep costs what is in flight
    rather than what the table holds. SKIP LOCKED so two reapers can run at
    once without queueing -- and, more importantly, so a call already being
    reconciled by one reaper is not picked up by another and reconciled twice
    against the carrier.
    """
    await cur.execute(
        """
        SELECT * FROM calls
        WHERE state IN ('QUEUED','RESERVED','INITIATED','RINGING','ANSWERED','CONNECTED')
          AND lease_expires_at IS NOT NULL
          AND lease_expires_at < %(now)s
        ORDER BY lease_expires_at
        LIMIT %(limit)s
        FOR UPDATE SKIP LOCKED
        """,
        {"now": now, "limit": limit},
    )
    return [Call.from_row(row) for row in await cur.fetchall()]


async def calls_over_lifetime(
    cur: AsyncCursor, *, now: datetime, max_seconds: float, limit: int = 200
) -> list[Call]:
    """Calls that have been non-terminal for implausibly long.

    The backstop for a call the carrier has simply forgotten about: no events,
    no status change, nothing. Reconciliation should have caught it, so
    anything here means reconciliation itself is failing, and forcing the call
    closed is a last resort that must be alarmed rather than done quietly.
    """
    await cur.execute(
        """
        SELECT * FROM calls
        WHERE state IN ('QUEUED','RESERVED','INITIATED','RINGING','ANSWERED','CONNECTED')
          AND created_at < %(now)s::timestamptz - make_interval(secs => %(max_seconds)s)
        ORDER BY created_at
        LIMIT %(limit)s
        FOR UPDATE SKIP LOCKED
        """,
        {"now": now, "max_seconds": max_seconds, "limit": limit},
    )
    return [Call.from_row(row) for row in await cur.fetchall()]


async def extend_call_lease(
    cur: AsyncCursor, *, call_id: UUID, seconds: float, now: datetime, owner: str
) -> None:
    """Give a call another lease period, and take ownership of it.

    Used by the reaper when it has reconciled a call and found it genuinely
    still live. Without this the same call would be re-reconciled on every
    single sweep -- one provider round trip per second, per stuck call, which
    is how a struggling carrier gets a retry storm on top of its problems.
    """
    await cur.execute(
        """
        UPDATE calls
        SET lease_owner = %(owner)s,
            lease_expires_at = %(now)s::timestamptz + make_interval(secs => %(seconds)s)
        WHERE id = %(call_id)s
        """,
        {"call_id": call_id, "seconds": seconds, "now": now, "owner": owner},
    )


async def count_terminal_conflicts(cur: AsyncCursor, *, campaign_id: UUID) -> int:
    """How often our inferred ending contradicted the carrier's.

    A metric, not an error count. A handful means the reaper is doing its job
    on a flaky carrier; a lot means max_call_lifetime is too aggressive and the
    reaper is closing calls that were merely slow.
    """
    await cur.execute(
        """
        SELECT count(*)::int AS n
        FROM provider_events e
        WHERE e.apply_result LIKE 'TERMINAL_CONFLICT:%%'
          AND e.provider_call_id IN (
              SELECT provider_call_id FROM calls
              WHERE campaign_id = %(campaign_id)s AND provider_call_id IS NOT NULL
          )
        """,
        {"campaign_id": campaign_id},
    )
    return (await cur.fetchone())["n"]
