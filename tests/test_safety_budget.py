"""The AIMD over-dial budget and the provider circuit breaker.

Six tests, one per behaviour that the safety argument actually rests on. They
run against a real database because both components exist precisely to be
shared between workers -- an in-memory version of either would pass a test like
`test_breaker_half_open_allows_exactly_one_probe_across_workers` by not having
the problem, which is the wrong reason to pass.

The clock is virtual throughout, so the twenty-second open period and the
sixty-second cooldown cost nothing to test and land on exactly the boundary
they are supposed to.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from smartdialer.core.db import Database
from smartdialer.core.models import Campaign, CampaignMode
from smartdialer.pacing.engine import ProviderHealthSignal
from smartdialer.safety.breaker import BreakerState, CircuitBreaker
from smartdialer.safety.budget import AbandonBudget, CreditReason

START = datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
async def pool(dsn: str):
    database = Database(dsn, min_size=1, max_size=6)
    await database.open()
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture()
async def campaign(pool: Database):
    """A campaign of its own, deleted afterwards.

    campaign_safety_state cascades with it, so the budget row never outlives
    the test that created it.
    """
    campaign_id = uuid.uuid4()
    async with pool.transaction() as cur:
        await cur.execute(
            "INSERT INTO campaigns (id, name, mode, abandon_budget_pct, "
            "max_overdial_ratio) VALUES (%s, %s, 'PREDICTIVE', 3.0, 2.0)",
            (campaign_id, f"budget-{campaign_id}"),
        )
    row = Campaign(
        id=campaign_id,
        name="budget-test",
        mode=CampaignMode.PREDICTIVE,
        max_concurrent=1000,
        abandon_budget_pct=3.0,
        target_shortfall_eps=0.02,
        max_overdial_ratio=2.0,
        active=True,
        wrap_up_seconds=10,
    )
    try:
        yield row
    finally:
        async with pool.transaction() as cur:
            await cur.execute("DELETE FROM calls WHERE campaign_id = %s", (campaign_id,))
            await cur.execute(
                "DELETE FROM borrowers WHERE campaign_id = %s", (campaign_id,)
            )
            await cur.execute("DELETE FROM campaigns WHERE id = %s", (campaign_id,))


async def tick(pool, budget, campaign, now, *, agents=10, changepoint=False):
    async with pool.transaction() as cur:
        return await budget.evaluate(
            cur,
            campaign=campaign,
            agents_available=agents,
            changepoint=changepoint,
            now=now,
        )


async def add_call(
    pool,
    campaign,
    *,
    state: str,
    now: datetime,
    provider: str = "mock_fast",
    failure_reason: str | None = None,
    answered: bool = False,
    provider_call_id: str | None = None,
):
    """One call row, the way the dialer would have written it."""
    borrower_id = uuid.uuid4()
    call_id = uuid.uuid4()
    async with pool.transaction() as cur:
        await cur.execute(
            "INSERT INTO borrowers (id, campaign_id, phone) VALUES (%s, %s, %s)",
            (borrower_id, campaign.id, f"+9199{uuid.uuid4().int % 10**8:08d}"),
        )
        await cur.execute(
            """
            INSERT INTO calls (id, campaign_id, borrower_id, provider,
                               idempotency_key, state, created_at,
                               answered_at, ended_at, failure_reason,
                               provider_call_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                call_id,
                campaign.id,
                borrower_id,
                provider,
                str(call_id),
                state,
                now,
                now if (answered or state == "ABANDONED") else None,
                now,
                failure_reason,
                provider_call_id,
            ),
        )
    return call_id


# ---------------------------------------------------------------------------
# The AIMD budget
# ---------------------------------------------------------------------------


async def test_clean_ticks_increase_credit_additively(pool, campaign):
    """Additive increase: one call per clean tick, and no faster.

    The credit is the whole over-dial allowance, so how quickly it grows is
    how quickly the system is willing to bet again after it last got burned.
    One per tick means recovery from a halving is measured in seconds, which
    is the deliberately slow half of AIMD.
    """
    budget = AbandonBudget()
    seen = []
    for step in range(5):
        state = await tick(pool, budget, campaign, START + timedelta(seconds=step))
        seen.append(state.credit)

    assert seen == [1, 2, 3, 4, 5]
    assert state.reason == CreditReason.INCREASE

    # And it stops at the pool-scaled ceiling rather than growing forever.
    # ratio 2.0 with 3 agents leaves headroom for exactly 3 more calls.
    small = await tick(pool, budget, campaign, START + timedelta(seconds=6), agents=3)
    assert small.credit == 3


async def test_abandon_halves_credit(pool, campaign):
    """Multiplicative decrease: one abandoned call halves the allowance.

    Not a decrement. An abandoned call is a compliance event that cannot be
    undone after the fact, so the system gives back half its allowance at once
    and earns it back one clean tick at a time. The asymmetry is the entire
    safety property of the controller.
    """
    budget = AbandonBudget()
    for step in range(8):
        state = await tick(pool, budget, campaign, START + timedelta(seconds=step))
    assert state.credit == 8

    await add_call(pool, campaign, state="ABANDONED", now=START + timedelta(seconds=8))

    after = await tick(pool, budget, campaign, START + timedelta(seconds=9))
    assert after.credit == 4
    assert after.reason == CreditReason.ABANDON
    assert after.abandons_charged == 1

    # The same abandoned call must not be charged twice. `updated_at` moved
    # forward in the same write that halved the credit, so the next tick sees
    # a clean window and resumes climbing.
    again = await tick(pool, budget, campaign, START + timedelta(seconds=10))
    assert again.credit == 5
    assert again.abandons_charged == 0


async def test_changepoint_halves_credit(pool, campaign):
    """The engine reporting that its own model just broke is a reason to
    shrink the bet immediately, without waiting for an abandon to prove it."""
    budget = AbandonBudget()
    for step in range(6):
        state = await tick(pool, budget, campaign, START + timedelta(seconds=step))
    assert state.credit == 6

    after = await tick(
        pool, budget, campaign, START + timedelta(seconds=6), changepoint=True
    )
    assert after.credit == 3
    assert after.reason == CreditReason.CHANGEPOINT


async def test_abandon_rate_over_budget_forces_progressive(pool, campaign):
    """Over the campaign's abandon budget, the credit is zero and stays zero
    for the cooldown -- which is exactly progressive dialling, arrived at by
    the safety system rather than by configuration."""
    budget = AbandonBudget(cooldown_seconds=60.0)
    for step in range(6):
        await tick(pool, budget, campaign, START + timedelta(seconds=step))

    # 12 answers, 2 of them abandoned: 16.7% against a 3% budget.
    for index in range(10):
        await add_call(
            pool, campaign, state="COMPLETED", now=START + timedelta(seconds=6), answered=True
        )
    for index in range(2):
        await add_call(
            pool, campaign, state="ABANDONED", now=START + timedelta(seconds=6)
        )

    breached = await tick(pool, budget, campaign, START + timedelta(seconds=7))
    assert breached.credit == 0
    assert breached.reason == CreditReason.COOLDOWN

    # Still zero half a minute later, even though nothing has been abandoned
    # since: recovery is not allowed to begin in the same minute as the breach.
    during = await tick(pool, budget, campaign, START + timedelta(seconds=40))
    assert during.credit == 0
    assert during.reason == CreditReason.COOLDOWN

    # The cooldown is extended for as long as the breach is still visible in
    # the 60s window -- a rate that is still over budget restarts the clock
    # rather than letting it run out underneath an ongoing problem. Only once
    # the bad minute has aged out entirely does the additive increase resume,
    # from zero.
    after = await tick(pool, budget, campaign, START + timedelta(seconds=130))
    assert after.credit == 1
    assert after.reason == CreditReason.INCREASE


# ---------------------------------------------------------------------------
# The circuit breaker
# ---------------------------------------------------------------------------


def health(name="mock_fast", *, reachable=True, timeout_rate=0.0, samples=0):
    return ProviderHealthSignal(
        name=name,
        reachable=reachable,
        failure_rate=0.0,
        timeout_rate=timeout_rate,
        samples=samples,
    )


async def evaluate(pool, breaker, campaign, now, *, providers=("mock_fast",), health_map=None):
    async with pool.transaction() as cur:
        await cur.execute(
            "INSERT INTO campaign_safety_state (campaign_id, updated_at) "
            "VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (campaign.id, now),
        )
        return await breaker.evaluate(
            cur,
            campaign_id=campaign.id,
            providers=list(providers),
            health=health_map or {name: health(name) for name in providers},
            now=now,
        )


async def test_breaker_opens_on_failure_rate_and_halts_new_calls(pool, campaign):
    """Above 30% failures over the window, the breaker opens and approves
    nothing new. Existing calls are untouched -- some of them have a person on
    the line, and cancelling those is the one action that makes a provider
    outage worse than it already is."""
    breaker = CircuitBreaker(worker_id="worker-a")

    # Healthy to begin with: no samples at all is not evidence of a problem.
    assert (await evaluate(pool, breaker, campaign, START)).state == BreakerState.CLOSED

    # 12 calls, 5 of them failed: 41%.
    for index in range(7):
        await add_call(pool, campaign, state="COMPLETED", now=START, answered=True)
    for index in range(5):
        await add_call(
            pool, campaign, state="FAILED", now=START, failure_reason="provider_rejected"
        )

    opened = await evaluate(pool, breaker, campaign, START + timedelta(seconds=1))
    assert opened.state == BreakerState.OPEN
    assert opened.allowance == 0
    assert opened.provider is None

    # Still open ten seconds later. The open period is a fixed twenty seconds,
    # not "until the numbers look better" -- the numbers cannot improve while
    # nothing is being dialled.
    later = await evaluate(pool, breaker, campaign, START + timedelta(seconds=11))
    assert later.state == BreakerState.OPEN
    assert later.allowance == 0


async def test_breaker_half_open_allows_exactly_one_probe_across_workers(pool, campaign):
    """Twenty workers reaching half-open at the same instant place ONE probe.

    This is the same compare-and-swap that stops two workers reserving one
    agent, applied to a different resource. The alternative -- each worker
    keeping its own breaker -- would pass every single-process test and put
    twenty probe calls into a carrier that asked for one.
    """
    for index in range(5):
        await add_call(
            pool, campaign, state="FAILED", now=START, failure_reason="provider_rejected"
        )
    for index in range(5):
        await add_call(pool, campaign, state="COMPLETED", now=START, answered=True)

    workers = [CircuitBreaker(worker_id=f"worker-{i}") for i in range(20)]
    await evaluate(pool, workers[0], campaign, START + timedelta(seconds=1))

    # Past the open period, every worker asks at once.
    half_open = START + timedelta(seconds=22)
    views = [await evaluate(pool, w, campaign, half_open) for w in workers]

    allowed = [v for v in views if v.allowance == 1]
    assert len(allowed) == 1, [v.allowance for v in views]
    assert allowed[0].state == BreakerState.HALF_OPEN
    assert allowed[0].provider == "mock_fast"
    assert all(v.allowance == 0 for v in views if v is not allowed[0])

    # The probe call goes out and the carrier accepts it: back to CLOSED, for
    # every worker, because the state is one row rather than twenty opinions.
    await add_call(
        pool,
        campaign,
        state="RINGING",
        now=half_open + timedelta(seconds=1),
        provider_call_id="probe-1",
    )
    closed = await evaluate(pool, workers[7], campaign, half_open + timedelta(seconds=2))
    assert closed.state == BreakerState.CLOSED
    assert closed.allowance is None
