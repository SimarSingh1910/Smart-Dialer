"""Progressive dialer, end to end.

The headline is test_progressive_never_exceeds_available_agents, which is the
rule the brief states outright: one available agent, one outbound call.

These run the real worker, the real allocator, the real safety controller and a
real database, against a mock carrier on a virtual clock. The only thing
substituted is the telephone network, which is the point -- if the invariant
held only because the test drove the components by hand, it would say nothing
about the system.

On driving the clock: the worker's tick does database I/O, and the carrier's
setup delays are virtual. So a step is tick, advance, drain. `drain` waits for
the place_call tasks and event deliveries that are ready to finish rather than
for wall-clock time, so the sequence is deterministic. The invariant is
asserted after EVERY step, not once at the end: a dialer that overshoots for
one tick and recovers has still overshot.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from smartdialer.core.clock import VirtualClock
from smartdialer.core.config import Settings
from smartdialer.core.db import Database
from smartdialer.core.logging import StructuredLogger
from smartdialer.core.models import AgentState, CallState, CampaignMode
from smartdialer.providers.mock_fast import make_fast_provider
from smartdialer.providers.mock_flaky import make_flaky_provider
from smartdialer.safety.controller import Reason
from smartdialer.workers.dialer_worker import DialerWorker

START = datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def pool(dsn: str):
    database = Database(dsn, min_size=2, max_size=12)
    await database.open()
    try:
        yield database
    finally:
        await database.close()


async def make_campaign(
    pool: Database,
    *,
    agents: int,
    borrowers: int,
    mode: CampaignMode = CampaignMode.PROGRESSIVE,
    wrap_up_seconds: int = 5,
    max_concurrent: int = 1000,
) -> uuid.UUID:
    campaign_id = uuid.uuid4()
    async with pool.transaction() as cur:
        await cur.execute(
            "INSERT INTO campaigns (id, name, mode, wrap_up_seconds, max_concurrent) "
            "VALUES (%s, %s, %s, %s, %s)",
            (campaign_id, f"prog-{campaign_id}", mode.value, wrap_up_seconds, max_concurrent),
        )
        for index in range(agents):
            await cur.execute(
                "INSERT INTO agents (id, campaign_id, state, state_changed_at) "
                "VALUES (%s, %s, 'AVAILABLE', %s)",
                (uuid.uuid4(), campaign_id, START),
            )
        for index in range(borrowers):
            await cur.execute(
                "INSERT INTO borrowers (id, campaign_id, phone, next_eligible_at) "
                "VALUES (%s, %s, %s, %s)",
                (uuid.uuid4(), campaign_id, f"+9198{index:08d}", START),
            )
    return campaign_id


async def cleanup(pool: Database, campaign_id: uuid.UUID) -> None:
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


def make_worker(pool: Database, clock: VirtualClock, campaign_id, provider, **overrides):
    settings = Settings(
        worker_id=overrides.pop("worker_id", "worker-progressive"),
        tick_seconds=0.25,
        lease_seconds=30.0,
        **overrides,
    )
    return DialerWorker(
        db=pool,
        clock=clock,
        campaign_id=campaign_id,
        providers=[provider],
        settings=settings,
        logger=StructuredLogger("test", clock),
    )


async def agent_bound_calls_in_flight(pool: Database, campaign_id) -> int:
    async with pool.transaction() as cur:
        await cur.execute(
            """
            SELECT count(*)::int AS n FROM calls
            WHERE campaign_id = %s
              AND agent_id IS NOT NULL
              AND state IN ('RESERVED','INITIATED','RINGING','ANSWERED','CONNECTED')
            """,
            (campaign_id,),
        )
        return (await cur.fetchone())["n"]


async def counts(pool: Database, campaign_id) -> dict:
    async with pool.transaction() as cur:
        await cur.execute(
            "SELECT state::text AS s, count(*)::int AS n FROM calls "
            "WHERE campaign_id = %s GROUP BY 1",
            (campaign_id,),
        )
        calls = {row["s"]: row["n"] for row in await cur.fetchall()}
        await cur.execute(
            "SELECT state::text AS s, count(*)::int AS n FROM agents "
            "WHERE campaign_id = %s GROUP BY 1",
            (campaign_id,),
        )
        agents = {row["s"]: row["n"] for row in await cur.fetchall()}
    return {"calls": calls, "agents": agents}


async def run_ticks(worker: DialerWorker, clock: VirtualClock, steps: int, check=None):
    """tick, advance, drain -- and check the invariant after every step."""
    for step in range(steps):
        await worker.tick()
        await clock.advance(worker._settings.tick_seconds)
        await worker.drain()
        if check is not None:
            await check(step)


# ---------------------------------------------------------------------------
# The headline test
# ---------------------------------------------------------------------------


async def test_progressive_never_exceeds_available_agents(pool: Database):
    """50 agents, 500 borrowers, and never a 51st live call.

    The invariant is not enforced by an arithmetic check anywhere in the
    dialer, which is the interesting part. It holds because reserving an agent
    moves them out of AVAILABLE inside the same transaction that writes the
    call row, so the pool the next tick sees has already shrunk. There is no
    counter to get wrong and no window in which two workers both see the same
    free agent.
    """
    campaign_id = await make_campaign(pool, agents=50, borrowers=500)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=4, answer_rate=0.5, talk_median_seconds=20.0)
    worker = make_worker(pool, clock, campaign_id, provider)
    worker.attach_providers()

    peak = 0

    async def check(step: int) -> None:
        nonlocal peak
        live = await agent_bound_calls_in_flight(pool, campaign_id)
        peak = max(peak, live)
        assert live <= 50, f"tick {step}: {live} agent-bound calls for 50 agents"

    try:
        await run_ticks(worker, clock, steps=240, check=check)

        # The test has to prove the dialer was actually working, or a dialer
        # that placed no calls at all would pass it.
        state = await counts(pool, campaign_id)
        assert peak > 10, f"the dialer barely dialled; peak in flight was {peak}"
        assert state["calls"].get("COMPLETED", 0) > 0
        assert not provider.sink_errors, provider.sink_errors
        assert not worker.event_errors, worker.event_errors
    finally:
        await worker.close()
        await provider.close()
        await cleanup(pool, campaign_id)


async def test_no_borrower_is_dialled_twice_at_once(pool: Database):
    """The other allocation invariant, over a whole run.

    The unique partial index would raise if this were violated, so the test
    would fail loudly anyway -- this asserts it directly so the failure names
    the actual problem rather than surfacing as an integrity error.
    """
    campaign_id = await make_campaign(pool, agents=20, borrowers=200)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=8, answer_rate=0.4, talk_median_seconds=15.0)
    worker = make_worker(pool, clock, campaign_id, provider)
    worker.attach_providers()

    async def check(step: int) -> None:
        async with pool.transaction() as cur:
            await cur.execute(
                """
                SELECT borrower_id, count(*)::int AS n FROM calls
                WHERE campaign_id = %s
                  AND state IN ('RESERVED','INITIATED','RINGING','ANSWERED','CONNECTED')
                GROUP BY borrower_id HAVING count(*) > 1
                """,
                (campaign_id,),
            )
            assert await cur.fetchall() == []

    try:
        await run_ticks(worker, clock, steps=160, check=check)
    finally:
        await worker.close()
        await provider.close()
        await cleanup(pool, campaign_id)


# ---------------------------------------------------------------------------
# The agent lifecycle actually happens
# ---------------------------------------------------------------------------


async def test_an_agent_walks_the_whole_lifecycle(pool: Database):
    """AVAILABLE -> RESERVED -> DIALING -> CONNECTED -> WRAP_UP -> AVAILABLE.

    One agent, one borrower who always answers. Every state in the brief's
    machine is visited in order, and the agent is back in the pool at the end
    rather than stranded in wrap-up.
    """
    campaign_id = await make_campaign(pool, agents=1, borrowers=1, wrap_up_seconds=5)
    clock = VirtualClock(start=START)
    # This test asserts a SEQUENCE OF STATES, so every source of randomness in
    # the carrier is turned off -- otherwise it is a coin flip wearing an
    # assertion.
    #
    # reject_rate=0: the fast carrier rejects about 2% of calls, and a single
    # rejection puts this campaign's only borrower into a five-minute backoff,
    # well past the window. The agent then correctly returns to AVAILABLE
    # having never connected, and the test fails while the dialer is behaving
    # perfectly.
    #
    # talk_sigma≈0: talk duration is drawn from the call's UUID, so with the
    # realistic spread roughly one run in thirty draws a conversation longer
    # than the window.
    #
    # Both behaviours are real and both are covered by their own tests -- the
    # rejection path in test_a_borrower_who_never_answers..., the timing spread
    # in the invariant tests, which run the carrier fully unmuzzled.
    provider = make_fast_provider(
        clock,
        seed=2,
        answer_rate=1.0,
        reject_rate=0.0,
        talk_median_seconds=20.0,
        talk_sigma=0.02,
    )
    worker = make_worker(pool, clock, campaign_id, provider)
    worker.attach_providers()

    seen: list[str] = []

    async def record() -> None:
        async with pool.transaction() as cur:
            await cur.execute(
                "SELECT state::text AS s FROM agents WHERE campaign_id = %s", (campaign_id,)
            )
            state = (await cur.fetchone())["s"]
        if not seen or seen[-1] != state:
            seen.append(state)

    try:
        for _ in range(300):
            await worker.tick()
            await clock.advance(0.25)
            await worker.drain()
            await record()

        assert "DIALING" in seen, seen
        assert "CONNECTED" in seen, seen
        assert "WRAP_UP" in seen, seen
        assert seen[-1] == "AVAILABLE", seen
        assert seen.index("DIALING") < seen.index("CONNECTED") < seen.index("WRAP_UP")

        state = await counts(pool, campaign_id)
        assert state["calls"].get("COMPLETED") == 1
        async with pool.transaction() as cur:
            await cur.execute(
                "SELECT b.state AS borrower_state, c.wait_ms AS wait_ms "
                "FROM borrowers b JOIN calls c ON c.borrower_id = b.id "
                "WHERE b.campaign_id = %s",
                (campaign_id,),
            )
            row = await cur.fetchone()
        assert row["borrower_state"] == "DONE", "a borrower who spoke to an agent is finished"
        assert row["wait_ms"] is not None, "customer wait must be measured"
    finally:
        await worker.close()
        await provider.close()
        await cleanup(pool, campaign_id)


async def test_a_borrower_who_never_answers_comes_back_with_an_attempt_spent(
    pool: Database,
):
    campaign_id = await make_campaign(pool, agents=2, borrowers=2)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=2, answer_rate=0.0)
    worker = make_worker(pool, clock, campaign_id, provider)
    worker.attach_providers()

    try:
        await run_ticks(worker, clock, steps=200)

        async with pool.transaction() as cur:
            await cur.execute(
                "SELECT attempts, state, next_eligible_at FROM borrowers "
                "WHERE campaign_id = %s ORDER BY phone",
                (campaign_id,),
            )
            borrowers = await cur.fetchall()

        assert all(b["attempts"] >= 1 for b in borrowers), borrowers
        # Backoff, not an immediate redial: the borrower is PENDING again but
        # not eligible yet.
        assert all(b["next_eligible_at"] > START for b in borrowers)
        # And the agents are not stranded on a call that never connected.
        state = await counts(pool, campaign_id)
        assert state["agents"].get("AVAILABLE") == 2, state
    finally:
        await worker.close()
        await provider.close()
        await cleanup(pool, campaign_id)


# ---------------------------------------------------------------------------
# The safety boundary
# ---------------------------------------------------------------------------


async def test_an_inactive_campaign_dials_nothing(pool: Database):
    """The kill switch, via the campaign's own row. An operator flips `active`
    and the very next tick stops dialling, with a reason code that says why."""
    campaign_id = await make_campaign(pool, agents=10, borrowers=50)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=1)
    worker = make_worker(pool, clock, campaign_id, provider)
    worker.attach_providers()

    try:
        first = await worker.tick()
        assert first.approved > 0

        async with pool.transaction() as cur:
            await cur.execute(
                "UPDATE campaigns SET active = false WHERE id = %s", (campaign_id,)
            )

        second = await worker.tick()
        assert second.approved == 0
        assert second.reason_code == Reason.KILL_SWITCH
    finally:
        await worker.close()
        await provider.close()
        await cleanup(pool, campaign_id)


async def test_the_concurrency_cap_binds_and_says_so(pool: Database):
    campaign_id = await make_campaign(pool, agents=40, borrowers=200, max_concurrent=5)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=1, answer_rate=0.5)
    worker = make_worker(pool, clock, campaign_id, provider)
    worker.attach_providers()

    try:
        decisions = []
        for _ in range(20):
            decisions.append(await worker.tick())
            await clock.advance(0.25)
            await worker.drain()
            live = await agent_bound_calls_in_flight(pool, campaign_id)
            assert live <= 5, f"{live} calls in flight with max_concurrent=5"

        assert any(d.reason_code == Reason.CAMPAIGN_CONCURRENCY for d in decisions)
    finally:
        await worker.close()
        await provider.close()
        await cleanup(pool, campaign_id)


async def test_a_controller_failure_approves_nothing(pool: Database):
    """Fail closed. The allocator is replaced with one that raises, and the
    controller must return zero rather than let the exception escape into the
    tick loop."""
    campaign_id = await make_campaign(pool, agents=10, borrowers=50)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=1)
    worker = make_worker(pool, clock, campaign_id, provider)

    class Exploding:
        async def dial(self, **kwargs):
            raise RuntimeError("allocator is on fire")

    worker.controller._allocator = Exploding()

    try:
        decision = await worker.tick()
        assert decision.approved == 0
        assert decision.reason_code == Reason.CONTROLLER_ERROR

        # And it was written down. A safety event that leaves no record is
        # indistinguishable from a quiet campaign.
        async with pool.transaction() as cur:
            await cur.execute(
                "SELECT reason_code FROM pacing_decisions WHERE campaign_id = %s "
                "ORDER BY id DESC LIMIT 1",
                (campaign_id,),
            )
            assert (await cur.fetchone())["reason_code"] == Reason.CONTROLLER_ERROR
    finally:
        await worker.close()
        await provider.close()
        await cleanup(pool, campaign_id)


async def test_every_tick_is_logged_with_its_inputs(pool: Database):
    """The "why 17 and not 10" requirement, at progressive scale.

    Every tick writes a row, including the ones that dialled nothing, and the
    row carries the whole snapshot -- so the proposal can be recomputed from
    it, the engine being a pure function.
    """
    campaign_id = await make_campaign(pool, agents=4, borrowers=20)
    clock = VirtualClock(start=START)
    provider = make_fast_provider(clock, seed=1, answer_rate=0.5)
    worker = make_worker(pool, clock, campaign_id, provider)
    worker.attach_providers()

    try:
        await run_ticks(worker, clock, steps=12)
        async with pool.transaction() as cur:
            await cur.execute(
                "SELECT proposed, approved, reason_code, inputs FROM pacing_decisions "
                "WHERE campaign_id = %s ORDER BY id",
                (campaign_id,),
            )
            rows = await cur.fetchall()

        assert len(rows) == 12, "one row per tick, busy or not"
        first = rows[0]
        assert first["proposed"] == 4
        snapshot = first["inputs"]["snapshot"]
        # All eight signals present, not just the ones progressive reads.
        for key in (
            "agents_available",
            "calls_connected",
            "calls_ringing",
            "historical_answer_rate",
            "call_setup_time_p95",
            "avg_call_duration",
            "provider_health",
            "recent_campaign_behaviour",
        ):
            assert key in snapshot, f"{key} missing from the decision log"
    finally:
        await worker.close()
        await provider.close()
        await cleanup(pool, campaign_id)


# ---------------------------------------------------------------------------
# The badly behaved carrier, through the whole stack
# ---------------------------------------------------------------------------


async def test_the_invariant_holds_on_the_flaky_provider_too(pool: Database):
    """Provider B: timeouts, duplicates, reordering, rejections.

    The invariant must hold with no code anywhere in the dialer that knows
    which carrier it is talking to. Note what a timeout does here -- the agent
    stays reserved and the call stays in flight, deliberately, because we do
    not know whether a phone is ringing. That makes the dialer SLOWER on this
    carrier, and it must never make it unsafe.
    """
    campaign_id = await make_campaign(pool, agents=25, borrowers=300)
    clock = VirtualClock(start=START)
    provider = make_flaky_provider(clock, seed=6, answer_rate=0.5, talk_median_seconds=20.0)
    worker = make_worker(pool, clock, campaign_id, provider)
    worker.attach_providers()

    async def check(step: int) -> None:
        live = await agent_bound_calls_in_flight(pool, campaign_id)
        assert live <= 25, f"tick {step}: {live} agent-bound calls for 25 agents"

    try:
        await run_ticks(worker, clock, steps=240, check=check)

        state = await counts(pool, campaign_id)
        assert sum(state["calls"].values()) > 0, "nothing was dialled at all"
        assert not worker.event_errors, worker.event_errors
        assert not provider.sink_errors, provider.sink_errors

        # Duplicates reached the dialer and were absorbed: more events stored
        # than transitions caused.
        async with pool.transaction() as cur:
            await cur.execute(
                """
                SELECT count(*) FILTER (WHERE apply_result = 'DUPLICATE')::int AS dupes,
                       count(*) FILTER (WHERE apply_result = 'STALE')::int     AS stale,
                       count(*)::int AS total
                FROM provider_events
                WHERE provider_call_id IN (
                    SELECT provider_call_id FROM calls
                    WHERE campaign_id = %s AND provider_call_id IS NOT NULL
                )
                """,
                (campaign_id,),
            )
            events = await cur.fetchone()
        assert events["total"] > 0
    finally:
        await worker.close()
        await provider.close()
        await cleanup(pool, campaign_id)
