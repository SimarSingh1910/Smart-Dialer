"""Call lifecycle and provider-event tests.

These are the tests for the brief's second hard question: the provider sends
ANSWERED, ANSWERED, ANSWERED, COMPLETED, or sends everything backwards, or the
worker dies halfway through.

Like test_allocation.py, these use real committed connections from a pool
rather than the rolled-back `conn` fixture. Two reasons here: the functions
under test are async and take an AsyncCursor, and the atomicity test needs a
transaction it can genuinely abort and then inspect from outside.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from smartdialer.core.db import Database
from smartdialer.core.models import CallState, NormalisedEvent
from smartdialer.domain.calls import (
    EventResult,
    abandon_call,
    abandon_unmatched_event,
    apply_event,
    attach_provider_call_id,
    connect_call,
    count_calls_by_state,
    create_call,
    get_call,
    get_call_by_idempotency_key,
    read_counters,
    reapply_stored_event,
    terminate_call,
    unapplied_events,
)

NOW = datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc)
WORKER = "worker-events"
PROVIDER = "mock_fast"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Wide enough for every worker in the concurrency tests below to hold an open
# transaction at once; see the note on MAX_CONNECTIONS in test_allocation.py.
MAX_CONNECTIONS = 14


@pytest.fixture()
async def pool(dsn: str):
    database = Database(dsn, min_size=2, max_size=MAX_CONNECTIONS)
    await database.open()
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture()
async def campaign(pool: Database):
    """A committed campaign, with its children deleted afterwards.

    provider_events is cleaned by provider_call_id rather than by campaign,
    because the table deliberately has no campaign column: it records what a
    provider told us, which is true whether or not we can attribute it.
    """
    campaign_id = uuid.uuid4()
    async with pool.transaction() as cur:
        await cur.execute(
            "INSERT INTO campaigns (id, name) VALUES (%s, %s)",
            (campaign_id, f"call-test-{campaign_id}"),
        )
    try:
        yield campaign_id
    finally:
        async with pool.transaction() as cur:
            await cur.execute(
                "DELETE FROM provider_events WHERE provider_call_id IN "
                "(SELECT provider_call_id FROM calls WHERE campaign_id = %s "
                " AND provider_call_id IS NOT NULL)",
                (campaign_id,),
            )
            await cur.execute(
                "DELETE FROM provider_events WHERE provider_event_id LIKE %s",
                (f"{campaign_id}:%",),
            )
            await cur.execute("DELETE FROM calls WHERE campaign_id = %s", (campaign_id,))
            await cur.execute(
                "DELETE FROM borrowers WHERE campaign_id = %s", (campaign_id,)
            )
            await cur.execute("DELETE FROM agents WHERE campaign_id = %s", (campaign_id,))
            await cur.execute(
                "DELETE FROM campaign_counters WHERE campaign_id = %s", (campaign_id,)
            )
            await cur.execute("DELETE FROM campaigns WHERE id = %s", (campaign_id,))


async def make_borrower(pool: Database, campaign_id) -> uuid.UUID:
    borrower_id = uuid.uuid4()
    async with pool.transaction() as cur:
        await cur.execute(
            "INSERT INTO borrowers (id, campaign_id, phone) VALUES (%s, %s, %s)",
            (borrower_id, campaign_id, f"+9190{borrower_id.int % 1000000:06d}"),
        )
    return borrower_id


async def make_agent(pool: Database, campaign_id) -> uuid.UUID:
    agent_id = uuid.uuid4()
    async with pool.transaction() as cur:
        await cur.execute(
            "INSERT INTO agents (id, campaign_id, state) VALUES (%s, %s, 'RESERVED')",
            (agent_id, campaign_id),
        )
    return agent_id


async def make_call(
    pool: Database,
    campaign_id,
    *,
    provider_call_id: str | None = "pc-1",
    agent_id: uuid.UUID | None = None,
    is_overdial: bool = False,
):
    """A call in INITIATED with the provider's id attached, as it exists a
    moment after a successful place_call()."""
    borrower_id = await make_borrower(pool, campaign_id)
    call_id = uuid.uuid4()
    async with pool.transaction() as cur:
        call = await create_call(
            cur,
            call_id=call_id,
            campaign_id=campaign_id,
            borrower_id=borrower_id,
            provider=PROVIDER,
            idempotency_key=f"idem-{call_id}",
            now=NOW,
            worker_id=WORKER,
            agent_id=agent_id,
            is_overdial=is_overdial,
        )
        if provider_call_id is not None:
            call = await attach_provider_call_id(
                cur,
                call_id=call.id,
                provider_call_id=f"{campaign_id}:{provider_call_id}",
                now=NOW,
            )
    return call


def event(
    campaign_id,
    event_type: str,
    *,
    event_id: str,
    provider_call_id: str = "pc-1",
    ts: datetime | None = None,
    payload: dict | None = None,
) -> NormalisedEvent:
    """Build an event the way a provider module would.

    Ids are namespaced by campaign so that a test's rows are its own even
    though provider_events has no campaign column.
    """
    return NormalisedEvent(
        provider=PROVIDER,
        provider_event_id=f"{campaign_id}:{event_id}",
        provider_call_id=f"{campaign_id}:{provider_call_id}",
        event_type=event_type,
        provider_ts=ts,
        payload=payload or {},
    )


# ---------------------------------------------------------------------------
# Deduplication -- ANSWERED, ANSWERED, ANSWERED, COMPLETED
# ---------------------------------------------------------------------------


async def test_duplicate_answered_produces_one_transition(pool: Database, campaign):
    """The brief's exact sequence. Three identical ANSWERED deliveries and one
    COMPLETED must leave one answered call that then completes -- and, just as
    importantly, must count one answer rather than three."""
    call = await make_call(pool, campaign)
    answered = event(campaign, "answered", event_id="e1", ts=NOW)

    results = []
    for _ in range(3):
        async with pool.transaction() as cur:
            results.append(
                await apply_event(cur, event=answered, now=NOW, worker_id=WORKER)
            )

    assert results[0].result == EventResult.APPLIED
    assert results[0].transitioned is True
    # The second and third deliveries never even reach the call row.
    for repeat in results[1:]:
        assert repeat.duplicate is True
        assert repeat.transitioned is False

    async with pool.transaction() as cur:
        after = await get_call(cur, call_id=call.id)
        counters = await read_counters(cur, campaign_id=campaign)
    assert after.state is CallState.ANSWERED
    # One transition, so one answer counted. A counter driven by event arrival
    # instead of by transition would read 3 here.
    assert counters.calls_answered == 1

    async with pool.transaction() as cur:
        completed = await apply_event(
            cur,
            event=event(campaign, "completed", event_id="e2", ts=NOW + timedelta(seconds=90)),
            now=NOW,
            worker_id=WORKER,
        )
    assert completed.transitioned is True
    assert completed.new_state is CallState.COMPLETED


async def test_a_duplicate_is_recorded_only_once(pool: Database, campaign):
    """The dedupe is the unique index, so the second delivery leaves no second
    row to reason about later."""
    await make_call(pool, campaign)
    answered = event(campaign, "answered", event_id="e1", ts=NOW)
    for _ in range(2):
        async with pool.transaction() as cur:
            await apply_event(cur, event=answered, now=NOW, worker_id=WORKER)

    async with pool.transaction() as cur:
        await cur.execute(
            "SELECT count(*) AS n FROM provider_events "
            "WHERE provider = %s AND provider_event_id = %s",
            (PROVIDER, answered.provider_event_id),
        )
        assert (await cur.fetchone())["n"] == 1


# ---------------------------------------------------------------------------
# Out of order -- COMPLETED, ANSWERED, RINGING
# ---------------------------------------------------------------------------


async def test_completed_then_answered_then_ringing_settles_completed(
    pool: Database, campaign
):
    """The brief's reversed sequence. State must not walk backwards."""
    call = await make_call(pool, campaign)

    async with pool.transaction() as cur:
        first = await apply_event(
            cur,
            event=event(campaign, "completed", event_id="e1", ts=NOW + timedelta(seconds=90)),
            now=NOW,
            worker_id=WORKER,
        )
        second = await apply_event(
            cur,
            event=event(campaign, "answered", event_id="e2", ts=NOW + timedelta(seconds=8)),
            now=NOW,
            worker_id=WORKER,
        )
        third = await apply_event(
            cur,
            event=event(campaign, "ringing", event_id="e3", ts=NOW + timedelta(seconds=2)),
            now=NOW,
            worker_id=WORKER,
        )

    assert first.transitioned is True
    # Not errors. A reordering provider is a supported provider, and both of
    # these are stored, absorbed and marked STALE.
    assert second.transitioned is False
    assert second.result == EventResult.STALE
    assert third.transitioned is False

    async with pool.transaction() as cur:
        after = await get_call(cur, call_id=call.id)
    assert after.state is CallState.COMPLETED


async def test_late_answered_still_records_answered_at(pool: Database, campaign):
    """THE facts-versus-state test.

    The ANSWERED event lost the race to move the state, but it still carries
    the only reading we will ever get of when the borrower picked up. If the
    timestamp were written only alongside a successful transition, answered_at
    would be NULL here -- and the answer rate the pacing engine runs on is
    computed from exactly this column. Every reordered call would silently
    look like a call nobody answered, and the engine would under-dial forever
    without a single error anywhere.
    """
    call = await make_call(pool, campaign)
    answered_ts = NOW + timedelta(seconds=8)
    ringing_ts = NOW + timedelta(seconds=2)

    async with pool.transaction() as cur:
        await apply_event(
            cur,
            event=event(campaign, "completed", event_id="e1", ts=NOW + timedelta(seconds=90)),
            now=NOW,
            worker_id=WORKER,
        )
        await apply_event(
            cur,
            event=event(campaign, "answered", event_id="e2", ts=answered_ts),
            now=NOW,
            worker_id=WORKER,
        )
        await apply_event(
            cur,
            event=event(campaign, "ringing", event_id="e3", ts=ringing_ts),
            now=NOW,
            worker_id=WORKER,
        )
        after = await get_call(cur, call_id=call.id)

    assert after.state is CallState.COMPLETED
    assert after.answered_at == answered_ts, "a late ANSWERED must still record the fact"
    assert after.ringing_at == ringing_ts
    assert after.ended_at == NOW + timedelta(seconds=90)


async def test_the_first_reading_of_a_fact_wins(pool: Database, campaign):
    """Two events carrying the same fact: the earlier delivery stands.

    COALESCE, not overwrite. A provider that re-sends ANSWERED with a fresh
    timestamp must not be able to rewrite when the borrower picked up.
    """
    call = await make_call(pool, campaign)
    async with pool.transaction() as cur:
        await apply_event(
            cur,
            event=event(campaign, "answered", event_id="e1", ts=NOW + timedelta(seconds=5)),
            now=NOW,
            worker_id=WORKER,
        )
        await apply_event(
            cur,
            event=event(campaign, "answered", event_id="e2", ts=NOW + timedelta(seconds=50)),
            now=NOW,
            worker_id=WORKER,
        )
        after = await get_call(cur, call_id=call.id)
    assert after.answered_at == NOW + timedelta(seconds=5)


async def test_wait_ms_is_computed_even_when_events_arrive_backwards(
    pool: Database, campaign
):
    """The headline customer metric survives reordering.

    Bridged arrives before answered. wait_ms is derived from whichever
    timestamps are held after each absorb, so it comes out at the real 3
    seconds rather than NULL.
    """
    call = await make_call(pool, campaign)
    answered_ts = NOW + timedelta(seconds=7)
    connected_ts = NOW + timedelta(seconds=10)

    async with pool.transaction() as cur:
        await apply_event(
            cur,
            event=event(campaign, "bridged", event_id="e1", ts=connected_ts),
            now=NOW,
            worker_id=WORKER,
        )
        mid = await get_call(cur, call_id=call.id)
        # Only one of the two timestamps is known, so there is no wait to
        # report yet. Zero would be a lie that reads like success.
        assert mid.wait_ms is None

        await apply_event(
            cur,
            event=event(campaign, "answered", event_id="e2", ts=answered_ts),
            now=NOW,
            worker_id=WORKER,
        )
        after = await get_call(cur, call_id=call.id)

    assert after.state is CallState.CONNECTED
    assert after.wait_ms == 3000


# ---------------------------------------------------------------------------
# Terminal states
# ---------------------------------------------------------------------------


async def test_abandoned_is_not_overwritten_by_a_later_completed(
    pool: Database, campaign
):
    """The compliance case.

    We decide a call was abandoned; the provider then reports the same call as
    completed, because from its side the leg simply ended. Terminal states
    share rank 9, so the first one written stands and the abandon stays
    visible and counted. A design where COMPLETED outranked ABANDONED would
    quietly erase the one outcome a regulator asks about.
    """
    call = await make_call(pool, campaign, is_overdial=True)
    async with pool.transaction() as cur:
        await apply_event(
            cur,
            event=event(campaign, "answered", event_id="e1", ts=NOW),
            now=NOW,
            worker_id=WORKER,
        )
        abandoned = await abandon_call(
            cur, call_id=call.id, now=NOW, worker_id=WORKER, reason="no_agent_available"
        )
        assert abandoned is not None

        late = await apply_event(
            cur,
            event=event(campaign, "completed", event_id="e2", ts=NOW + timedelta(seconds=4)),
            now=NOW,
            worker_id=WORKER,
        )
        after = await get_call(cur, call_id=call.id)
        counters = await read_counters(cur, campaign_id=campaign)

    assert late.transitioned is False
    assert after.state is CallState.ABANDONED
    assert after.failure_reason == "no_agent_available"
    assert counters.calls_abandoned == 1
    # 1 answer, 1 abandon: the rate is per answered call, not per dial.
    assert counters.abandon_rate_pct == 100.0


async def test_terminating_twice_has_one_winner(pool: Database, campaign):
    """The reaper and a webhook can both decide a call is over. One wins
    quietly and the other gets None -- no exception, no retry loop."""
    call = await make_call(pool, campaign)
    async with pool.transaction() as cur:
        first = await terminate_call(
            cur,
            call_id=call.id,
            target_state=CallState.FAILED,
            now=NOW,
            worker_id=WORKER,
            failure_reason="provider_timeout",
        )
        second = await terminate_call(
            cur,
            call_id=call.id,
            target_state=CallState.CANCELLED,
            now=NOW,
            worker_id=WORKER,
        )
        counters = await read_counters(cur, campaign_id=campaign)

    assert first is not None and first.state is CallState.FAILED
    assert second is None
    assert counters.calls_failed == 1


async def test_terminate_call_refuses_a_non_terminal_state(pool: Database, campaign):
    call = await make_call(pool, campaign)
    async with pool.transaction() as cur:
        with pytest.raises(ValueError):
            await terminate_call(
                cur,
                call_id=call.id,
                target_state=CallState.RINGING,
                now=NOW,
                worker_id=WORKER,
            )


async def test_needs_agent_flags_an_answered_overdial(pool: Database, campaign):
    """The abandonment decision the worker acts on.

    An over-dial call has no agent bound to it, so the moment the borrower
    answers there is a human on the line and possibly nobody to talk to. The
    flag fires on the transition, so duplicate ANSWERED deliveries cannot make
    the worker act twice.
    """
    await make_call(pool, campaign, is_overdial=True)
    answered = event(campaign, "answered", event_id="e1", ts=NOW)
    async with pool.transaction() as cur:
        first = await apply_event(cur, event=answered, now=NOW, worker_id=WORKER)
        repeat = await apply_event(cur, event=answered, now=NOW, worker_id=WORKER)
    assert first.needs_agent is True
    assert repeat.needs_agent is False


async def test_a_call_with_an_agent_does_not_need_one(pool: Database, campaign):
    agent_id = await make_agent(pool, campaign)
    await make_call(pool, campaign, agent_id=agent_id)
    async with pool.transaction() as cur:
        applied = await apply_event(
            cur,
            event=event(campaign, "answered", event_id="e1", ts=NOW),
            now=NOW,
            worker_id=WORKER,
        )
    assert applied.needs_agent is False


async def test_connect_call_is_safe_against_a_racing_bridged_event(
    pool: Database, campaign
):
    """Our own bridge and the provider's confirmation of it. Whichever lands
    first sets the state; the other is outranked and changes nothing, so the
    connect is counted once."""
    call = await make_call(pool, campaign)
    async with pool.transaction() as cur:
        await apply_event(
            cur,
            event=event(campaign, "answered", event_id="e1", ts=NOW),
            now=NOW,
            worker_id=WORKER,
        )
        ours = await connect_call(
            cur, call_id=call.id, now=NOW + timedelta(seconds=1), worker_id=WORKER
        )
        theirs = await apply_event(
            cur,
            event=event(campaign, "bridged", event_id="e2", ts=NOW + timedelta(seconds=1)),
            now=NOW,
            worker_id=WORKER,
        )
        counters = await read_counters(cur, campaign_id=campaign)

    assert ours is not None
    assert theirs.transitioned is False
    assert counters.calls_connected == 1


# ---------------------------------------------------------------------------
# Events we cannot place
# ---------------------------------------------------------------------------


async def test_unknown_provider_call_id_is_stored_not_crashed(pool: Database, campaign):
    """An event for a call we have never heard of.

    Almost always a race: the webhook beat the commit of our own INSERT. So it
    is stored, left unapplied, and picked up by the retry sweep. Discarding it
    would lose the only notification we get that somebody's phone is ringing.
    """
    ghost = event(campaign, "answered", event_id="e1", provider_call_id="never-placed")
    async with pool.transaction() as cur:
        applied = await apply_event(cur, event=ghost, now=NOW, worker_id=WORKER)

    assert applied.result == EventResult.UNKNOWN_CALL
    assert applied.transitioned is False

    async with pool.transaction() as cur:
        await cur.execute(
            "SELECT applied, apply_result, payload FROM provider_events "
            "WHERE provider = %s AND provider_event_id = %s",
            (PROVIDER, ghost.provider_event_id),
        )
        row = await cur.fetchone()
    assert row is not None, "the event must be stored even though it made no sense"
    assert row["applied"] is False, "left on the worklist for retry"
    assert row["apply_result"] == EventResult.UNKNOWN_CALL


async def test_an_unmatched_event_is_applied_once_its_call_appears(
    pool: Database, campaign
):
    """The retry path, end to end: webhook first, our own INSERT second."""
    early = event(campaign, "answered", event_id="e1", ts=NOW)
    async with pool.transaction() as cur:
        first = await apply_event(cur, event=early, now=NOW, worker_id=WORKER)
    assert first.result == EventResult.UNKNOWN_CALL

    call = await make_call(pool, campaign)

    async with pool.transaction() as cur:
        # `now` is pushed past the settle window so the sweep considers it.
        # Filtered to this test's own event: the worklist is global by
        # design, and a leftover from an earlier run must not decide whether
        # this assertion passes.
        pending = await unapplied_events(cur, now=NOW + timedelta(seconds=5))
        mine = [p for p in pending if p.provider_event_id == early.provider_event_id]
        assert len(mine) == 1
        retried = await reapply_stored_event(
            cur, stored=mine[0], now=NOW, worker_id=WORKER
        )
        after = await get_call(cur, call_id=call.id)

    assert retried.transitioned is True
    assert after.state is CallState.ANSWERED
    assert after.answered_at == NOW


async def test_a_fresh_unmatched_event_is_not_swept_immediately(
    pool: Database, campaign
):
    """The settle window. Sweeping instantly would spin on events that are
    about to become applicable when our own INSERT commits."""
    ev = event(campaign, "answered", event_id="e1")
    await apply_event_alone(pool, ev)
    async with pool.transaction() as cur:
        pending = await unapplied_events(cur, now=NOW, older_than_seconds=60.0)
    assert ev.provider_event_id not in [p.provider_event_id for p in pending]


async def apply_event_alone(pool: Database, ev: NormalisedEvent) -> None:
    async with pool.transaction() as cur:
        await apply_event(cur, event=ev, now=NOW, worker_id=WORKER)


async def test_giving_up_on_an_event_takes_it_off_the_worklist(
    pool: Database, campaign
):
    ghost = event(campaign, "answered", event_id="e1", provider_call_id="never-placed")
    async with pool.transaction() as cur:
        applied = await apply_event(cur, event=ghost, now=NOW, worker_id=WORKER)
        await abandon_unmatched_event(cur, event_row_id=applied.event_row_id)
        pending = await unapplied_events(cur, now=NOW + timedelta(seconds=60))
    assert ghost.provider_event_id not in [p.provider_event_id for p in pending]


async def test_unknown_event_type_is_ignored_not_fatal(pool: Database, campaign):
    """A provider adding a webhook type must not be able to take the dialer
    down. Stored, marked IGNORED, no transition."""
    call = await make_call(pool, campaign)
    async with pool.transaction() as cur:
        applied = await apply_event(
            cur,
            event=event(campaign, "machine_detected", event_id="e1", ts=NOW),
            now=NOW,
            worker_id=WORKER,
        )
        after = await get_call(cur, call_id=call.id)

    assert applied.result == EventResult.IGNORED
    assert applied.transitioned is False
    assert after.state is CallState.INITIATED


# ---------------------------------------------------------------------------
# Atomicity and the intent log
# ---------------------------------------------------------------------------


async def test_event_application_is_atomic(pool: Database, campaign):
    """A failure part-way through leaves nothing behind -- including the
    dedupe row.

    That last part matters more than it looks. If the event row survived a
    rolled-back application, the event would be permanently marked as seen
    while never having been applied, and a redelivery would be dismissed as a
    duplicate. The call would sit in the wrong state forever. Storing the
    dedupe key in the same transaction as the state change is what makes a
    crash mid-apply recoverable by simply receiving the event again.
    """
    call = await make_call(pool, campaign)
    answered = event(campaign, "answered", event_id="e1", ts=NOW)

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        async with pool.transaction() as cur:
            applied = await apply_event(cur, event=answered, now=NOW, worker_id=WORKER)
            assert applied.transitioned is True
            raise Boom()

    async with pool.transaction() as cur:
        after = await get_call(cur, call_id=call.id)
        await cur.execute(
            "SELECT count(*) AS n FROM provider_events WHERE provider_event_id = %s",
            (answered.provider_event_id,),
        )
        events = (await cur.fetchone())["n"]
        counters = await read_counters(cur, campaign_id=campaign)

    assert after.state is CallState.INITIATED, "the transition must have rolled back"
    assert after.answered_at is None
    assert events == 0, "the dedupe row must roll back too, or redelivery is lost"
    assert counters.calls_answered == 0

    # And the redelivery, which is the entire point, works.
    async with pool.transaction() as cur:
        retried = await apply_event(cur, event=answered, now=NOW, worker_id=WORKER)
    assert retried.transitioned is True


async def test_the_call_row_is_written_before_the_provider_is_called(
    pool: Database, campaign
):
    """The intent log. After create_call, and before any provider id exists,
    recovery can already find the call by the key it is about to send."""
    borrower_id = await make_borrower(pool, campaign)
    call_id = uuid.uuid4()
    key = f"idem-{call_id}"
    async with pool.transaction() as cur:
        await create_call(
            cur,
            call_id=call_id,
            campaign_id=campaign,
            borrower_id=borrower_id,
            provider=PROVIDER,
            idempotency_key=key,
            now=NOW,
            worker_id=WORKER,
            lease_seconds=30.0,
        )

    async with pool.transaction() as cur:
        found = await get_call_by_idempotency_key(cur, key=key)
    assert found is not None
    assert found.provider_call_id is None, "no provider id yet -- that is the gap"
    assert found.state is CallState.INITIATED
    assert found.lease_owner == WORKER


async def test_adopting_a_provider_call_id_is_idempotent(pool: Database, campaign):
    """Recovery re-attaches the same id after a crash; a different id means we
    placed the call twice and must not be papered over."""
    call = await make_call(pool, campaign, provider_call_id=None)
    async with pool.transaction() as cur:
        first = await attach_provider_call_id(
            cur, call_id=call.id, provider_call_id="pc-adopted", now=NOW
        )
        again = await attach_provider_call_id(
            cur, call_id=call.id, provider_call_id="pc-adopted", now=NOW
        )
        conflicting = await attach_provider_call_id(
            cur, call_id=call.id, provider_call_id="pc-different", now=NOW
        )
    assert first is not None and again is not None
    assert conflicting is None


async def test_a_second_live_call_for_one_borrower_is_rejected(
    pool: Database, campaign
):
    """The database backstop for the invariant borrower allocation enforces.

    Reservation is what prevents this; the unique partial index is what makes
    it impossible rather than merely unlikely. It firing means allocation has
    a bug, so it must be loud.
    """
    import psycopg

    call = await make_call(pool, campaign)
    with pytest.raises(psycopg.errors.UniqueViolation):
        async with pool.transaction() as cur:
            await create_call(
                cur,
                call_id=uuid.uuid4(),
                campaign_id=campaign,
                borrower_id=call.borrower_id,
                provider=PROVIDER,
                idempotency_key=f"idem-second-{call.borrower_id}",
                now=NOW,
                worker_id=WORKER,
            )


# ---------------------------------------------------------------------------
# Counters and snapshot reads
# ---------------------------------------------------------------------------


async def test_counters_are_summed_across_shards(pool: Database, campaign):
    """Different workers hash to different shards; the read sums them.

    This is the hot-row fix from the scale analysis, tested at small scale: the
    total has to be right regardless of how the writes were spread.
    """
    borrower_ids = [await make_borrower(pool, campaign) for _ in range(3)]
    for index, borrower_id in enumerate(borrower_ids):
        async with pool.transaction() as cur:
            await create_call(
                cur,
                call_id=uuid.uuid4(),
                campaign_id=campaign,
                borrower_id=borrower_id,
                provider=PROVIDER,
                idempotency_key=f"idem-shard-{index}-{borrower_id}",
                now=NOW,
                worker_id=f"worker-{index}",
            )

    async with pool.transaction() as cur:
        await cur.execute(
            "SELECT count(*) AS n FROM campaign_counters WHERE campaign_id = %s "
            "AND calls_initiated > 0",
            (campaign,),
        )
        touched = (await cur.fetchone())["n"]
        counters = await read_counters(cur, campaign_id=campaign)

    assert counters.calls_initiated == 3
    assert touched >= 1, "writes must land in the sharded table, not a single row"


async def test_in_flight_counts_exclude_finished_calls(pool: Database, campaign):
    call_a = await make_call(pool, campaign, provider_call_id="pc-a")
    call_b = await make_call(pool, campaign, provider_call_id="pc-b")

    async with pool.transaction() as cur:
        await apply_event(
            cur,
            event=event(campaign, "ringing", event_id="e1", provider_call_id="pc-a", ts=NOW),
            now=NOW,
            worker_id=WORKER,
        )
        await terminate_call(
            cur,
            call_id=call_b.id,
            target_state=CallState.COMPLETED,
            now=NOW,
            worker_id=WORKER,
        )
        counts = await count_calls_by_state(cur, campaign_id=campaign)

    assert counts[CallState.RINGING] == 1
    assert counts[CallState.COMPLETED] == 0, "terminal calls are not in flight"
    assert call_a.id != call_b.id


async def test_a_failed_event_records_the_providers_reason(pool: Database, campaign):
    call = await make_call(pool, campaign)
    async with pool.transaction() as cur:
        await apply_event(
            cur,
            event=event(
                campaign,
                "failed",
                event_id="e1",
                ts=NOW,
                payload={"reason": "circuit_congestion"},
            ),
            now=NOW,
            worker_id=WORKER,
        )
        after = await get_call(cur, call_id=call.id)
        counters = await read_counters(cur, campaign_id=campaign)

    assert after.state is CallState.FAILED
    assert after.failure_reason == "circuit_congestion"
    assert counters.calls_failed == 1


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


async def test_the_same_event_delivered_concurrently_transitions_once(
    pool: Database, campaign
):
    """The duplicate case under a real race, on separate connections.

    A provider retrying a webhook does not politely wait for the first delivery
    to finish; both can be in flight at the same instant. The single-connection
    version of this test proves nothing, because the dedupe would then be
    ordinary sequential execution.

    The barrier holds all twelve deliveries inside an open transaction until
    every one has arrived, so they genuinely collide on the unique index. Ten
    of them block there, and when the winner commits they find the row present
    and return no id. Exactly one transition, exactly one answer counted.
    """
    call = await make_call(pool, campaign)
    answered = event(campaign, "answered", event_id="e1", ts=NOW)
    workers = 12
    barrier = asyncio.Barrier(workers)

    async def deliver() -> bool:
        async with pool.transaction() as cur:
            await barrier.wait()
            applied = await apply_event(cur, event=answered, now=NOW, worker_id=WORKER)
            return applied.transitioned

    results = await asyncio.gather(*(deliver() for _ in range(workers)))

    assert sum(results) == 1, "exactly one delivery may move the call"

    async with pool.transaction() as cur:
        after = await get_call(cur, call_id=call.id)
        counters = await read_counters(cur, campaign_id=campaign)
        await cur.execute(
            "SELECT count(*) AS n FROM provider_events WHERE provider_event_id = %s",
            (answered.provider_event_id,),
        )
        stored = (await cur.fetchone())["n"]

    assert after.state is CallState.ANSWERED
    assert stored == 1
    assert counters.calls_answered == 1


async def test_concurrent_distinct_events_do_not_interleave(pool: Database, campaign):
    """Different events for one call, racing.

    ringing, answered and completed arrive on three workers at once. They are
    distinct events, so none of them is deduplicated -- what protects the call
    row is the SELECT ... FOR UPDATE, which makes each worker's read-modify-
    write run to completion before the next one starts.

    Whatever the order, the call must end COMPLETED (the highest rank) and hold
    all three timestamps: the state settles at the furthest point reached, and
    no fact is lost on the way.
    """
    call = await make_call(pool, campaign)
    barrier = asyncio.Barrier(3)
    specs = [
        ("ringing", "e1", NOW + timedelta(seconds=2)),
        ("answered", "e2", NOW + timedelta(seconds=8)),
        ("completed", "e3", NOW + timedelta(seconds=95)),
    ]

    async def deliver(event_type: str, event_id: str, ts: datetime) -> None:
        async with pool.transaction() as cur:
            await barrier.wait()
            await apply_event(
                cur,
                event=event(campaign, event_type, event_id=event_id, ts=ts),
                now=NOW,
                worker_id=WORKER,
            )

    await asyncio.gather(*(deliver(*spec) for spec in specs))

    async with pool.transaction() as cur:
        after = await get_call(cur, call_id=call.id)

    assert after.state is CallState.COMPLETED
    assert after.ringing_at == NOW + timedelta(seconds=2)
    assert after.answered_at == NOW + timedelta(seconds=8)
    assert after.ended_at == NOW + timedelta(seconds=95)
