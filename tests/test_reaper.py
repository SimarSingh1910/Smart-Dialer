"""Crash recovery: the reaper, and the brief's failure scenarios.

HOW A CRASH IS SIMULATED HERE.

Not with os.kill. A killed process takes the test runner's database pool and
event loop with it, and what it leaves behind is a machine-dependent mess
rather than a state you can assert on. What we actually need to reproduce is
narrower and more precise: a worker that stopped renewing its leases at an
exact point in the call's life.

So a "crash" is: drive the worker to the point of interest, then stop calling
it and advance the VirtualClock past the lease. The rows it left behind are
byte-identical to what a SIGKILL would leave, because leases are the only thing
that made those rows live in the first place. And because time is injected, the
crash lands at exactly the intended instant on every run instead of wherever
the scheduler happened to be.

WHAT EACH TEST IS REALLY ASSERTING.

The interesting cases are not "does recovery happen" but "does recovery decline
to guess". Three of these assert that the reaper leaves things ALONE -- holds an
agent, keeps a call in flight, refuses to invent a call id -- because every
available action is wrong when you do not know whether a stranger is on the
line.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from smartdialer.core.clock import VirtualClock
from smartdialer.core.config import Settings
from smartdialer.core.db import Database
from smartdialer.core.logging import StructuredLogger
from smartdialer.core.models import CallState, NormalisedEvent
from smartdialer.domain.calls import count_terminal_conflicts
from smartdialer.providers.mock_fast import make_fast_provider
from smartdialer.workers.dialer_worker import DialerWorker
from smartdialer.workers.reaper import Reaper

START = datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
async def pool(dsn: str):
    database = Database(dsn, min_size=2, max_size=12)
    await database.open()
    try:
        yield database
    finally:
        await database.close()


async def make_campaign(pool: Database, *, agents: int, borrowers: int, **columns):
    campaign_id = uuid.uuid4()
    defaults = {"wrap_up_seconds": 5}
    defaults.update(columns)
    async with pool.transaction() as cur:
        await cur.execute(
            "INSERT INTO campaigns (id, name, wrap_up_seconds) VALUES (%s, %s, %s)",
            (campaign_id, f"reaper-{campaign_id}", defaults["wrap_up_seconds"]),
        )
        for _ in range(agents):
            await cur.execute(
                "INSERT INTO agents (id, campaign_id, state, state_changed_at, "
                "last_heartbeat_at) VALUES (%s, %s, 'AVAILABLE', %s, %s)",
                (uuid.uuid4(), campaign_id, START, START),
            )
        for index in range(borrowers):
            await cur.execute(
                "INSERT INTO borrowers (id, campaign_id, phone, next_eligible_at) "
                "VALUES (%s, %s, %s, %s)",
                (uuid.uuid4(), campaign_id, f"+9197{index:08d}", START),
            )
    return campaign_id


async def cleanup(pool: Database, campaign_id) -> None:
    async with pool.transaction() as cur:
        await cur.execute(
            "DELETE FROM provider_events WHERE provider_call_id IN "
            "(SELECT provider_call_id FROM calls WHERE campaign_id = %s "
            " AND provider_call_id IS NOT NULL)",
            (campaign_id,),
        )
        await cur.execute("DELETE FROM pacing_decisions WHERE campaign_id = %s", (campaign_id,))
        await cur.execute("UPDATE agents SET current_call_id = NULL WHERE campaign_id = %s", (campaign_id,))
        await cur.execute("DELETE FROM calls WHERE campaign_id = %s", (campaign_id,))
        await cur.execute("DELETE FROM borrowers WHERE campaign_id = %s", (campaign_id,))
        await cur.execute("DELETE FROM agents WHERE campaign_id = %s", (campaign_id,))
        await cur.execute("DELETE FROM campaign_counters WHERE campaign_id = %s", (campaign_id,))
        await cur.execute("DELETE FROM campaigns WHERE id = %s", (campaign_id,))


def settings(**overrides) -> Settings:
    base = dict(
        worker_id=overrides.pop("worker_id", "doomed-worker"),
        tick_seconds=0.25,
        reserve_lease_seconds=5.0,
        lease_seconds=30.0,
        heartbeat_timeout_seconds=30.0,
        max_call_lifetime_seconds=900.0,
    )
    base.update(overrides)
    return Settings(**base)


def make_worker(pool, clock, campaign_id, provider, **overrides):
    return DialerWorker(
        db=pool,
        clock=clock,
        campaign_id=campaign_id,
        providers=[provider],
        settings=settings(**overrides),
        logger=StructuredLogger("worker", clock),
    )


def make_reaper(pool, clock, campaign_id, provider, **overrides):
    return Reaper(
        db=pool,
        clock=clock,
        campaign_id=campaign_id,
        providers=[provider],
        settings=settings(worker_id="reaper", **overrides),
        logger=StructuredLogger("reaper", clock),
    )


async def state_of(pool: Database, campaign_id) -> dict:
    async with pool.transaction() as cur:
        await cur.execute(
            "SELECT state::text s, count(*)::int n FROM agents WHERE campaign_id=%s GROUP BY 1",
            (campaign_id,),
        )
        agents = {r["s"]: r["n"] for r in await cur.fetchall()}
        await cur.execute(
            "SELECT state::text s, count(*)::int n FROM calls WHERE campaign_id=%s GROUP BY 1",
            (campaign_id,),
        )
        calls = {r["s"]: r["n"] for r in await cur.fetchall()}
        await cur.execute(
            "SELECT state, count(*)::int n FROM borrowers WHERE campaign_id=%s GROUP BY 1",
            (campaign_id,),
        )
        borrowers = {r["state"]: r["n"] for r in await cur.fetchall()}
    return {"agents": agents, "calls": calls, "borrowers": borrowers}


async def sweep(reaper: Reaper, clock: VirtualClock, *, max_advance: float = 20.0):
    """Run one sweep while driving the clock underneath it.

    Reconciliation talks to the carrier, and a bridge takes real time -- which
    on a VirtualClock means it parks until somebody advances. Awaiting
    reaper.sweep() directly therefore deadlocks: the sweep waits for the clock
    and the test waits for the sweep.

    So the sweep runs as a task and time moves on without it, which is exactly
    the shape of the production loop -- the reaper is a background task and the
    world does not stop while it reconciles.
    """
    task = asyncio.ensure_future(reaper.sweep())
    advanced = 0.0
    while not task.done() and advanced <= max_advance:
        # Wait a REAL moment first. Most of a sweep is database round trips,
        # and advancing virtual time to wait for those would burn seconds of
        # simulated clock on work that takes microseconds -- which is how a
        # sweep silently expires the very lease or heartbeat the test is
        # asserting about. Virtual time only moves when the sweep is genuinely
        # parked on it, which happens for exactly one thing: a bridge.
        done, _ = await asyncio.wait([task], timeout=0.02)
        if done:
            break
        await clock.advance(0.25)
        advanced += 0.25
    return await task


async def one_call(pool: Database, campaign_id) -> dict:
    async with pool.transaction() as cur:
        await cur.execute(
            "SELECT * FROM calls WHERE campaign_id = %s ORDER BY created_at LIMIT 1",
            (campaign_id,),
        )
        return await cur.fetchone()


# ---------------------------------------------------------------------------
# Crash before the carrier was ever called
# ---------------------------------------------------------------------------


async def test_crash_after_agent_reserved_releases_agent_on_short_lease(pool: Database):
    """The batch-reservation window, and why there are two lease tiers.

    The worker reserves agents and dies before writing any call row. Those
    agents have nothing behind them -- no call, no borrower on a line, nothing
    to reconcile -- so they must come back in SECONDS, not after the long lease
    an in-flight call would justify. Thirty seconds of idle capacity per agent,
    every time a worker restarts, is a real cost at any fleet size.
    """
    campaign_id = await make_campaign(pool, agents=3, borrowers=10)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=1)

    # Reserve directly, exactly as the allocator's first statement does, then
    # stop -- the crash lands between reservation and the call row.
    from smartdialer.domain.agents import reserve_agents

    async with pool.transaction() as cur:
        reserved = await reserve_agents(
            cur,
            campaign_id=campaign_id,
            worker_id="doomed-worker",
            n=3,
            lease_seconds=5.0,
            now=START,
        )
    assert len(reserved) == 3

    reaper = make_reaper(pool, clock, campaign_id, provider)
    try:
        # Four seconds in: still inside the short lease, so nothing moves.
        await clock.advance(4.0)
        early = await sweep(reaper, clock)
        assert early.agents_released == 0
        assert (await state_of(pool, campaign_id))["agents"].get("RESERVED") == 3

        # Past it.
        await clock.advance(2.0)
        report = await sweep(reaper, clock)
        assert report.agents_released == 3
        assert (await state_of(pool, campaign_id))["agents"].get("AVAILABLE") == 3
    finally:
        await provider.close()
        await cleanup(pool, campaign_id)


async def test_crash_after_provider_call_placed_is_adopted_via_idempotency_key(
    pool: Database,
):
    """The orphan case, and the reason the key is written before the dial.

    The carrier accepted the call and the worker died before recording the
    provider's id. All we hold is the key we generated ourselves -- and that is
    enough, because we sent it and the carrier kept it.
    """
    campaign_id = await make_campaign(pool, agents=1, borrowers=5)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=3, answer_rate=1.0, reject_rate=0.0)
    worker = make_worker(pool, clock, campaign_id, provider)
    worker.attach_providers()

    try:
        await worker.tick()
        # Let the carrier actually accept the call before injecting the crash.
        # place_call sleeps for the setup delay on the virtual clock, so
        # draining alone leaves it still in flight -- and nulling an id that
        # has not been written yet tests nothing at all.
        for _ in range(8):
            await clock.advance(0.5)
            await worker.drain()
            if (await one_call(pool, campaign_id))["provider_call_id"]:
                break
        assert (await one_call(pool, campaign_id))["provider_call_id"], (
            "the carrier must have accepted before the crash is injected"
        )

        # The crash: erase the provider id the worker had just learned, exactly
        # as if it had died between place_call returning and that write. Note
        # what this also costs us -- with no id, the carrier's webhooks for
        # this call no longer correlate to anything, so they pile up
        # unapplied. The idempotency key is the only thread left.
        async with pool.transaction() as cur:
            await cur.execute(
                "UPDATE calls SET provider_call_id = NULL WHERE campaign_id = %s",
                (campaign_id,),
            )
        orphan = await one_call(pool, campaign_id)
        assert orphan["provider_call_id"] is None
        assert orphan["idempotency_key"]

        await clock.advance(40.0)  # past the long lease
        reaper = make_reaper(pool, clock, campaign_id, provider)
        report = await sweep(reaper, clock)

        assert report.calls_adopted == 1, "the carrier knew this key; adopt it"
        adopted = await one_call(pool, campaign_id)
        assert adopted["provider_call_id"] is not None
        assert adopted["state"] not in ("CANCELLED", "FAILED"), (
            "a call the carrier confirms exists must never be cancelled"
        )
    finally:
        await worker.close()
        await provider.close()
        await cleanup(pool, campaign_id)


async def test_unknown_idempotency_key_resolves_to_cancelled(pool: Database):
    """The other half of adoption, and the one that keeps it honest.

    A key the carrier has genuinely never seen must resolve to CANCELLED. This
    is the test that stops anybody reconstructing the provider call id locally
    -- the mock derives its ids deterministically from the key, so a local
    reconstruction would produce a plausible-looking id and this path would
    silently "adopt" calls that were never placed. Only the carrier's own
    key -> id map may answer this question, and for an unknown key it answers
    None.
    """
    campaign_id = await make_campaign(pool, agents=1, borrowers=5)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=3)

    # An intent row for a call the carrier was never actually asked to place.
    from smartdialer.domain.calls import create_call

    async with pool.transaction() as cur:
        await cur.execute(
            "SELECT id FROM borrowers WHERE campaign_id = %s LIMIT 1", (campaign_id,)
        )
        borrower_id = (await cur.fetchone())["id"]
        await cur.execute(
            "SELECT id, version FROM agents WHERE campaign_id = %s LIMIT 1",
            (campaign_id,),
        )
        agent = await cur.fetchone()
        await cur.execute(
            "UPDATE agents SET state='DIALING', current_call_id=NULL, "
            "lease_owner='doomed-worker', lease_expires_at=%s WHERE id=%s",
            (START, agent["id"]),
        )
        call = await create_call(
            cur,
            call_id=uuid.uuid4(),
            campaign_id=campaign_id,
            borrower_id=borrower_id,
            provider=provider.name,
            idempotency_key="a-key-the-carrier-never-saw",
            now=START,
            worker_id="doomed-worker",
            agent_id=agent["id"],
            lease_seconds=30.0,
        )
        await cur.execute(
            "UPDATE agents SET current_call_id=%s WHERE id=%s", (call.id, agent["id"])
        )

    try:
        assert await provider.find_by_idempotency_key("a-key-the-carrier-never-saw") is None

        await clock.advance(40.0)
        # The heartbeat timeout is widened for this test only. Forty seconds of
        # crash window also exceeds the default thirty-second heartbeat, so the
        # freed agent would correctly be taken OFFLINE in the same sweep -- two
        # mechanisms firing at once, and this test is about the orphan path.
        reaper = make_reaper(
            pool, clock, campaign_id, provider, heartbeat_timeout_seconds=600.0
        )
        report = await sweep(reaper, clock)

        assert report.calls_cancelled == 1
        assert report.calls_adopted == 0
        after = await one_call(pool, campaign_id)
        assert after["state"] == "CANCELLED"
        assert after["provider_call_id"] is None, "no id may be invented"

        state = await state_of(pool, campaign_id)
        assert state["agents"].get("AVAILABLE") == 1, "the agent goes back"
        assert state["borrowers"].get("PENDING") == 5, "and so does the borrower"
    finally:
        await provider.close()
        await cleanup(pool, campaign_id)


# ---------------------------------------------------------------------------
# Crash after ANSWERED -- the headline case
# ---------------------------------------------------------------------------


async def _crash_mid_answer(pool: Database, clock: VirtualClock, provider, campaign_id):
    """Drive a call to ANSWERED, then stop the worker there.

    The worker is never ticked again, so nothing renews the lease -- which is
    precisely and only what a crash means to the rest of the system.
    """
    worker = make_worker(pool, clock, campaign_id, provider)

    async def sink(event: NormalisedEvent) -> None:
        # The worker applies the event but is not allowed to act on it: this is
        # the process dying in the microsecond between the webhook landing and
        # the bridge going out.
        from smartdialer.domain.calls import apply_event

        async with pool.transaction() as cur:
            await apply_event(
                cur, event=event, now=clock.now(), worker_id="doomed-worker"
            )

    provider.set_event_sink(sink)
    await worker.tick()
    await worker.drain()
    for _ in range(120):
        await clock.advance(0.5)
        await worker.drain()
        call = await one_call(pool, campaign_id)
        if call and call["state"] == "ANSWERED":
            break
    await worker.close()
    return call


async def test_crash_after_answered_bridges_when_agent_free(pool: Database):
    """The recoverable version: the borrower is still there and so is an agent.

    The reaper asks the carrier, learns the call is live, and completes the
    bridge the dead worker never got to. Nobody is dropped.
    """
    campaign_id = await make_campaign(pool, agents=1, borrowers=5)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(
        clock, seed=5, answer_rate=1.0, reject_rate=0.0,
        talk_median_seconds=600.0, talk_sigma=0.02,
        # A patient borrower: still on the line when the reaper runs, which
        # is the situation these tests are about.
        answer_patience_seconds=120.0,
    )

    try:
        call = await _crash_mid_answer(pool, clock, provider, campaign_id)
        assert call["state"] == "ANSWERED", call["state"]

        await clock.advance(40.0)
        reaper = make_reaper(pool, clock, campaign_id, provider)
        report = await sweep(reaper, clock)

        assert report.calls_bridged == 1, report.as_dict()
        assert report.calls_abandoned == 0
        after = await one_call(pool, campaign_id)
        assert after["state"] == "CONNECTED"
        assert (await state_of(pool, campaign_id))["agents"].get("CONNECTED") == 1
    finally:
        await provider.close()
        await cleanup(pool, campaign_id)


async def test_crash_after_answered_abandons_when_no_agent_free(pool: Database):
    """The unrecoverable version, and it must be called what it is.

    A borrower is on the line, the worker is dead, and there is no agent to
    give them to. That is an ABANDONED call -- a compliance event -- and the
    only honest thing to do is record it as one, count it against the budget,
    and hang up. Recording it as FAILED would be more comfortable and would
    understate the single number this campaign is judged on.
    """
    campaign_id = await make_campaign(pool, agents=1, borrowers=5)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(
        clock, seed=5, answer_rate=1.0, reject_rate=0.0,
        talk_median_seconds=600.0, talk_sigma=0.02,
        answer_patience_seconds=120.0,
    )

    try:
        call = await _crash_mid_answer(pool, clock, provider, campaign_id)
        assert call["state"] == "ANSWERED"

        # The one agent goes offline while the borrower waits -- so there is
        # genuinely nobody to bridge to.
        async with pool.transaction() as cur:
            await cur.execute(
                "UPDATE agents SET state='OFFLINE', current_call_id=NULL, "
                "lease_owner=NULL, lease_expires_at=NULL WHERE campaign_id=%s",
                (campaign_id,),
            )

        await clock.advance(40.0)
        reaper = make_reaper(pool, clock, campaign_id, provider)
        report = await sweep(reaper, clock)

        assert report.calls_abandoned == 1, report.as_dict()
        after = await one_call(pool, campaign_id)
        assert after["state"] == "ABANDONED", (
            "an answered call with nobody to take it is ABANDONED, never FAILED"
        )

        # And it is counted where the abandon budget will look for it.
        from smartdialer.domain.calls import read_counters

        async with pool.transaction() as cur:
            counters = await read_counters(cur, campaign_id=campaign_id)
        assert counters.calls_abandoned == 1
    finally:
        await provider.close()
        await cleanup(pool, campaign_id)


async def test_crash_after_answered_holds_agent_reserved_while_provider_unreachable(
    pool: Database,
):
    """Fail closed, and pay for it.

    The worker is dead and the carrier will not answer either. We do not know
    whether a human is on this line. Every available action is wrong: releasing
    the agent risks bridging them to a second borrower while the first is live,
    cancelling the call risks re-dialling somebody already talking to us.

    So the reaper does NOTHING -- and this test asserts nothing happened, which
    is the point. The cost is an agent held idle during an outage, which is
    exactly when utilisation hurts most. That is the trade, taken deliberately.
    """
    campaign_id = await make_campaign(pool, agents=1, borrowers=5)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(
        clock, seed=5, answer_rate=1.0, reject_rate=0.0,
        talk_median_seconds=600.0, talk_sigma=0.02,
        answer_patience_seconds=120.0,
    )

    try:
        call = await _crash_mid_answer(pool, clock, provider, campaign_id)
        assert call["state"] == "ANSWERED"
        before = await state_of(pool, campaign_id)

        provider.start_outage(600.0)  # the carrier goes dark too
        await clock.advance(40.0)
        reaper = make_reaper(pool, clock, campaign_id, provider)
        report = await sweep(reaper, clock)

        assert report.calls_unreachable == 1
        assert report.calls_settled == 0
        assert report.calls_abandoned == 0
        assert report.agents_released == 0

        after = await state_of(pool, campaign_id)
        assert after["calls"] == before["calls"], "the call must not move"
        assert after["agents"] == before["agents"], "the agent must not be released"
        assert after["borrowers"] == before["borrowers"]

        # And once the carrier comes back, recovery proceeds normally.
        provider.end_outage()
        recovered = await sweep(reaper, clock)
        assert recovered.calls_bridged == 1, recovered.as_dict()
    finally:
        await provider.close()
        await cleanup(pool, campaign_id)


# ---------------------------------------------------------------------------
# Idempotence, heartbeats, and the terminal conflict metric
# ---------------------------------------------------------------------------


async def test_reaper_is_idempotent(pool: Database):
    """Run it twice; the second pass must do nothing at all.

    Not a nicety. A sweep that keeps finding the same work makes "is recovery
    keeping up?" unanswerable, which is the one question worth asking at three
    in the morning. Every query in the sweep is written so that acting on a row
    removes it from its own worklist -- an adopted call has an id, a reconciled
    live call has a fresh lease, a detached borrower has none.

    The scenario deliberately contains no call needing a BRIDGE. Bridging is
    the one part of a sweep that waits on the carrier, so it is the one part
    that must advance the virtual clock -- and a sweep that moves time forward
    lets the world evolve underneath it, so the second pass legitimately finds
    new work. That would be a real property of the system, not a bug, but it is
    not the property this test is about: idempotence here means running twice
    at the SAME instant changes nothing the second time. Nobody answering
    (answer_rate=0) gives a sweep plenty to reconcile with no bridge in it.
    """
    campaign_id = await make_campaign(pool, agents=3, borrowers=10)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=7, answer_rate=0.0, reject_rate=0.0)
    worker = make_worker(pool, clock, campaign_id, provider)
    worker.attach_providers()

    try:
        for _ in range(30):
            await worker.tick()
            await clock.advance(0.25)
            await worker.drain()

        # The crash. Detaching the sink as well as stopping the worker is what
        # makes this a dead process rather than a paused one: a worker that has
        # died is not still quietly applying webhooks.
        provider.set_event_sink(None)
        await worker.close()

        await clock.advance(60.0)
        reaper = make_reaper(pool, clock, campaign_id, provider)

        first = await sweep(reaper, clock)
        before = await state_of(pool, campaign_id)
        second = await sweep(reaper, clock)
        after = await state_of(pool, campaign_id)

        assert first.changes > 0, "the first pass should have had work to do"
        assert second.changes == 0, f"second pass was not a no-op: {second.as_dict()}"
        assert after == before, "the second sweep changed the world"

        # And a third, to catch anything that alternates rather than settles.
        third = await sweep(reaper, clock)
        assert third.changes == 0, third.as_dict()
    finally:
        await worker.close()
        await provider.close()
        await cleanup(pool, campaign_id)


async def test_stale_heartbeat_marks_agent_offline(pool: Database):
    """Failure scenario 3: agents disappear, and the dialer has to notice.

    How fast it notices is bounded entirely by the heartbeat timeout, and the
    cost of noticing late is calls placed for agents who are not there.
    """
    campaign_id = await make_campaign(pool, agents=5, borrowers=10)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=1)
    reaper = make_reaper(pool, clock, campaign_id, provider, heartbeat_timeout_seconds=30.0)

    try:
        # Two keep reporting in; three vanish.
        from smartdialer.domain.agents import heartbeat

        async with pool.transaction() as cur:
            await cur.execute(
                "SELECT id FROM agents WHERE campaign_id = %s ORDER BY id LIMIT 2",
                (campaign_id,),
            )
            survivors = [r["id"] for r in await cur.fetchall()]

        await clock.advance(20.0)
        async with pool.transaction() as cur:
            await heartbeat(cur, agent_ids=survivors, now=clock.now())

        assert (await sweep(reaper, clock)).heartbeats_expired == 0, "not stale yet"

        await clock.advance(15.0)  # 35s since the vanished three last reported
        report = await sweep(reaper, clock)

        assert report.heartbeats_expired == 3
        state = await state_of(pool, campaign_id)
        assert state["agents"].get("OFFLINE") == 3
        assert state["agents"].get("AVAILABLE") == 2
    finally:
        await provider.close()
        await cleanup(pool, campaign_id)


async def test_an_agent_who_never_reported_in_is_not_reaped(pool: Database):
    """A NULL heartbeat means "has not arrived", not "has gone missing".

    Conflating the two would take an entire seeded fleet offline the first time
    this sweep ran.
    """
    campaign_id = await make_campaign(pool, agents=2, borrowers=2)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=1)
    async with pool.transaction() as cur:
        await cur.execute(
            "UPDATE agents SET last_heartbeat_at = NULL WHERE campaign_id = %s",
            (campaign_id,),
        )

    reaper = make_reaper(pool, clock, campaign_id, provider)
    try:
        await clock.advance(600.0)
        assert (await sweep(reaper, clock)).heartbeats_expired == 0
        assert (await state_of(pool, campaign_id))["agents"].get("AVAILABLE") == 2
    finally:
        await provider.close()
        await cleanup(pool, campaign_id)


async def test_terminal_conflict_is_recorded_in_apply_result(pool: Database):
    """When the reaper's guess and the carrier's truth disagree.

    The reaper force-fails a call past max_call_lifetime. The carrier then
    reports it COMPLETED. First-terminal-wins keeps FAILED -- which is the
    right rule, it is what stops a late COMPLETED erasing an ABANDONED -- but
    the outcome on record is now locally INFERRED rather than provider truth.

    That disagreement is recorded rather than flattened into an ordinary STALE,
    so "how often did my reaper guess wrong about a call the carrier knew
    about" is a number. The facts still absorb through COALESCE, so the answer
    rate stays honest either way.

    The worker deliberately never attaches its event sink here. With no
    webhooks flowing, the call sits in INITIATED and ages out -- which is
    exactly the situation the lifetime backstop exists for: a call the carrier
    has silently stopped telling us anything about.
    """
    campaign_id = await make_campaign(pool, agents=1, borrowers=5)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=3, answer_rate=1.0, reject_rate=0.0)
    worker = make_worker(pool, clock, campaign_id, provider)

    try:
        await worker.tick()
        for _ in range(8):
            await clock.advance(0.5)
            await worker.drain()
            if (await one_call(pool, campaign_id))["provider_call_id"]:
                break
        call = await one_call(pool, campaign_id)
        provider_call_id = call["provider_call_id"]
        assert provider_call_id
        assert call["state"] == "INITIATED", "no events were delivered"

        # Well past max_call_lifetime, with the carrier unreachable so ordinary
        # reconciliation cannot resolve it either.
        provider.start_outage(5000.0)
        await clock.advance(1000.0)
        reaper = make_reaper(
            pool,
            clock,
            campaign_id,
            provider,
            max_call_lifetime_seconds=900.0,
            heartbeat_timeout_seconds=5000.0,
        )
        report = await sweep(reaper, clock)
        assert report.calls_unreachable == 1, "the carrier could not be asked"
        assert report.calls_force_failed == 1, report.as_dict()

        # The carrier surfaces later with a different story.
        from smartdialer.domain.calls import apply_event

        async with pool.transaction() as cur:
            applied = await apply_event(
                cur,
                event=NormalisedEvent(
                    provider=provider.name,
                    provider_event_id=f"{provider_call_id}:completed",
                    provider_call_id=provider_call_id,
                    event_type="completed",
                    provider_ts=clock.now(),
                    payload={},
                ),
                now=clock.now(),
                worker_id="ingester",
            )

        assert applied.transitioned is False, "the first terminal state stands"
        assert applied.result == "TERMINAL_CONFLICT:FAILED<-COMPLETED"

        after = await one_call(pool, campaign_id)
        assert after["state"] == "FAILED"
        assert after["ended_at"] is not None, "facts absorb regardless of rank"

        async with pool.transaction() as cur:
            assert await count_terminal_conflicts(cur, campaign_id=campaign_id) == 1
    finally:
        await worker.close()
        await provider.close()
        await cleanup(pool, campaign_id)


async def test_borrower_with_non_terminal_call_is_not_returned_to_pending(
    pool: Database,
):
    """The two-clock rule, exercised through the reaper rather than the query.

    A crashed worker leaves a borrower whose lease is dead and whose call is
    still live. The reaper must detach the claim without freeing the borrower;
    freeing them would let the dialer insert a second call and ring one person
    twice about one debt while the first call is still up.
    """
    campaign_id = await make_campaign(pool, agents=1, borrowers=5)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(
        clock, seed=5, answer_rate=1.0, reject_rate=0.0,
        talk_median_seconds=600.0, talk_sigma=0.02,
        answer_patience_seconds=120.0,
    )

    try:
        call = await _crash_mid_answer(pool, clock, provider, campaign_id)
        assert call["state"] == "ANSWERED"
        borrower_id = call["borrower_id"]

        provider.start_outage(600.0)  # so reconciliation cannot finish the call
        await clock.advance(40.0)
        reaper = make_reaper(pool, clock, campaign_id, provider)
        report = await sweep(reaper, clock)

        assert report.borrowers_detached == 1
        assert report.borrowers_released == 0

        async with pool.transaction() as cur:
            await cur.execute(
                "SELECT state, lease_owner, attempts FROM borrowers WHERE id = %s",
                (borrower_id,),
            )
            row = await cur.fetchone()
        assert row["state"] == "RESERVED", "still held by a live call"
        assert row["lease_owner"] is None, "but the dead worker's claim is gone"
        assert row["attempts"] == 0, "our crash is not the borrower's fault"
    finally:
        await provider.close()
        await cleanup(pool, campaign_id)
