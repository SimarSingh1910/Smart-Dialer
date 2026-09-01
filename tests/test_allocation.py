"""Concurrency tests for agent and borrower allocation.

READ THIS BEFORE CHANGING THESE TESTS.

They deliberately do NOT use the rolled-back `conn` fixture the schema tests
use. That fixture wraps one connection in one transaction, and a single
connection cannot skip its own locks -- SKIP LOCKED would never do anything,
and every assertion here would pass while proving nothing at all. That is the
single most common way this test is green on a broken system.

So these tests:

  * take real, separate connections from a pool, one per simulated worker;
  * commit their starting state so every connection can see it;
  * clean up explicitly in a fixture teardown.

They are also written to FAIL if SKIP LOCKED is removed from the query, not
merely to pass when it is present. That was verified by actually deleting it
and watching them go red -- see the comment on
test_concurrent_batch_reservation_never_overlaps for the numbers.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from smartdialer.core.db import Database
from smartdialer.core.models import AgentState, BorrowerState, IllegalTransition
from smartdialer.domain.agents import (
    count_agents_by_state,
    heartbeat,
    login_agents,
    logout_agents,
    release_agent,
    renew_lease,
    reserve_agents,
    transition_agent,
)
from smartdialer.domain.borrowers import (
    borrowers_held_by_live_call,
    detach_expired_leases,
    release_borrower,
    release_expired_leases,
    reserve_borrowers,
)

NOW = datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc)
LEASE_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Fixtures: committed state, explicit teardown
# ---------------------------------------------------------------------------


# Wide enough that every simulated worker in the biggest race gets its own
# connection at the same time. That is not a detail: the barrier below makes
# all workers hold an open transaction simultaneously, so a pool smaller than
# the worker count would deadlock waiting for a connection that never frees.
MAX_CONNECTIONS = 52


@pytest.fixture()
async def pool(dsn: str):
    """A real connection pool. Each simulated worker gets its own connection,
    which is the only way row locks between them mean anything."""
    database = Database(dsn, min_size=2, max_size=MAX_CONNECTIONS)
    await database.open()
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture()
async def campaign(pool: Database):
    """A committed campaign, deleted afterwards.

    Committed on purpose: the whole point is that other connections can see it.
    Teardown deletes children before parents to respect the foreign keys.
    """
    campaign_id = uuid.uuid4()
    async with pool.transaction() as cur:
        await cur.execute(
            "INSERT INTO campaigns (id, name) VALUES (%s, %s)",
            (campaign_id, f"alloc-test-{campaign_id}"),
        )
    try:
        yield campaign_id
    finally:
        async with pool.transaction() as cur:
            await cur.execute(
                "DELETE FROM calls WHERE campaign_id = %s", (campaign_id,)
            )
            await cur.execute(
                "DELETE FROM borrowers WHERE campaign_id = %s", (campaign_id,)
            )
            await cur.execute(
                "DELETE FROM agents WHERE campaign_id = %s", (campaign_id,)
            )
            await cur.execute("DELETE FROM campaigns WHERE id = %s", (campaign_id,))


async def make_agents(pool: Database, campaign_id, count: int) -> list[uuid.UUID]:
    """Create `count` AVAILABLE agents, committed.

    state_changed_at is staggered so the longest-idle-first ordering is
    deterministic instead of depending on insertion timing.
    """
    ids = [uuid.uuid4() for _ in range(count)]
    async with pool.transaction() as cur:
        for index, agent_id in enumerate(ids):
            await cur.execute(
                "INSERT INTO agents (id, campaign_id, state, state_changed_at) "
                "VALUES (%s, %s, 'AVAILABLE', %s)",
                (agent_id, campaign_id, NOW - timedelta(seconds=count - index)),
            )
    return ids


async def make_borrowers(pool: Database, campaign_id, count: int) -> list[uuid.UUID]:
    ids = [uuid.uuid4() for _ in range(count)]
    async with pool.transaction() as cur:
        for index, borrower_id in enumerate(ids):
            await cur.execute(
                "INSERT INTO borrowers (id, campaign_id, phone, next_eligible_at) "
                "VALUES (%s, %s, %s, %s)",
                (borrower_id, campaign_id, f"+9190{index:06d}", NOW - timedelta(hours=1)),
            )
    return ids


# ---------------------------------------------------------------------------
# The headline test
# ---------------------------------------------------------------------------


async def test_two_workers_cannot_reserve_same_agent(pool: Database, campaign):
    """50 workers race for 10 agents, one each.

    Exactly 10 must win, and every winner must hold a different agent.

    The barrier is load-bearing. asyncio.gather() alone does NOT guarantee the
    transactions overlap -- each worker can open, reserve and commit before the
    next one even starts, in which case no lock is ever contested. The barrier
    holds every worker inside an open transaction until all 50 have arrived, so
    the reservations genuinely collide.

    What this test does NOT prove is that SKIP LOCKED is present, and it is
    worth being exact about why, because it is tempting to assume otherwise.
    Plain FOR UPDATE gives the same answer here. In READ COMMITTED a blocked
    lock waiter re-checks the row when the lock releases (EvalPlanQual); the
    row no longer matches `state = 'AVAILABLE'` and is dropped -- but LockRows
    sits BELOW Limit in the plan, so the Limit node just pulls a replacement
    row. Both variants end up with ten distinct agents. The difference is that
    without SKIP LOCKED the workers serialise behind each other first.

    So this test guards correctness -- no agent reserved twice, none stranded.
    test_reservation_never_blocks_on_another_workers_lock below is the one that
    fails when the clause is deleted.
    """
    agent_ids = await make_agents(pool, campaign, 10)
    workers = 50
    barrier = asyncio.Barrier(workers)

    async def worker(index: int) -> list[uuid.UUID]:
        async with pool.transaction() as cur:
            # Everyone is now inside an open transaction. Nobody proceeds until
            # everyone is here, so the statements below really do race.
            await barrier.wait()
            reservations = await reserve_agents(
                cur,
                campaign_id=campaign,
                worker_id=f"worker-{index}",
                n=1,
                lease_seconds=LEASE_SECONDS,
                now=NOW,
            )
            return [r.agent_id for r in reservations]

    results = await asyncio.gather(*(worker(i) for i in range(workers)))

    won = [agent_id for result in results for agent_id in result]
    assert len(won) == 10, f"expected 10 reservations, got {len(won)}"
    assert len(set(won)) == 10, "an agent was reserved by two workers"
    assert set(won) == set(agent_ids)

    # And the database agrees: no agent is left AVAILABLE.
    async with pool.transaction() as cur:
        counts = await count_agents_by_state(cur, campaign_id=campaign)
    assert counts[AgentState.RESERVED] == 10
    assert counts[AgentState.AVAILABLE] == 0


async def test_concurrent_batch_reservation_never_overlaps(pool: Database, campaign):
    """10 workers each ask for 20 agents from a pool of 50.

    All 50 must be handed out, each exactly once. Nobody gets a duplicate and
    nothing is stranded.

    Same barrier, same reason as the test above, and the same caveat: this is a
    correctness assertion, not proof that SKIP LOCKED is present.
    """
    await make_agents(pool, campaign, 50)
    workers = 10
    barrier = asyncio.Barrier(workers)

    async def worker(index: int) -> list[uuid.UUID]:
        async with pool.transaction() as cur:
            await barrier.wait()
            reservations = await reserve_agents(
                cur,
                campaign_id=campaign,
                worker_id=f"batch-worker-{index}",
                n=20,
                lease_seconds=LEASE_SECONDS,
                now=NOW,
            )
            return [r.agent_id for r in reservations]

    results = await asyncio.gather(*(worker(i) for i in range(workers)))
    won = [agent_id for result in results for agent_id in result]

    assert len(won) == 50, f"expected all 50 agents handed out, got {len(won)}"
    assert len(set(won)) == 50, "an agent was handed to two workers"


async def test_reservation_never_blocks_on_another_workers_lock(
    pool: Database, campaign
):
    """THE test for SKIP LOCKED. Delete the clause and this one goes red.

    Worker A reserves an agent and holds its transaction open, the way a worker
    does while it talks to the telecom provider. Worker B then asks for an
    agent. It must step over A's locked row and take the next one immediately.

    Without SKIP LOCKED, B's scan hits A's locked row first (ORDER BY puts it
    there) and blocks until A commits, so B waits on a network call it has
    nothing to do with. The timeout below turns that into a failure instead of
    a hang.

    This is what SKIP LOCKED actually buys, and it is a throughput property
    rather than a correctness one -- the correctness tests above pass either
    way. Under a real tick loop it is the difference between workers running
    independently and every worker queueing behind the slowest provider call in
    the fleet.
    """
    agent_ids = await make_agents(pool, campaign, 2)
    holder_ready = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with pool.transaction() as cur:
            await reserve_agents(
                cur, campaign_id=campaign, worker_id="holder", n=1,
                lease_seconds=LEASE_SECONDS, now=NOW,
            )
            holder_ready.set()
            # Transaction stays open, so the lock stays held.
            await release_holder.wait()

    holder_task = asyncio.create_task(holder())
    try:
        await holder_ready.wait()
        async with pool.transaction() as cur:
            reservations = await asyncio.wait_for(
                reserve_agents(
                    cur, campaign_id=campaign, worker_id="passer-by", n=1,
                    lease_seconds=LEASE_SECONDS, now=NOW,
                ),
                timeout=5.0,
            )
        # It skipped the locked agent and took the other one.
        assert [r.agent_id for r in reservations] == [agent_ids[1]]
    finally:
        release_holder.set()
        await holder_task


async def test_a_fully_locked_pool_returns_empty_rather_than_waiting(
    pool: Database, campaign
):
    """The degenerate case of the same property. One agent, already locked by
    somebody else: the right answer is "nothing available right now", returned
    immediately, so the tick ends and the next one re-reads. Waiting would
    block the whole worker on another worker's provider call."""
    await make_agents(pool, campaign, 1)
    holder_ready = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with pool.transaction() as cur:
            await reserve_agents(
                cur, campaign_id=campaign, worker_id="holder", n=1,
                lease_seconds=LEASE_SECONDS, now=NOW,
            )
            holder_ready.set()
            await release_holder.wait()

    holder_task = asyncio.create_task(holder())
    try:
        await holder_ready.wait()
        async with pool.transaction() as cur:
            reservations = await asyncio.wait_for(
                reserve_agents(
                    cur, campaign_id=campaign, worker_id="passer-by", n=1,
                    lease_seconds=LEASE_SECONDS, now=NOW,
                ),
                timeout=5.0,
            )
        assert reservations == []
    finally:
        release_holder.set()
        await holder_task


async def test_each_worker_sees_its_own_reservation_only(pool: Database, campaign):
    """Two workers, ten agents, five each. Their sets must be disjoint, and the
    lease owner recorded in the database must match who actually won."""
    await make_agents(pool, campaign, 10)

    async def worker(name: str) -> set[uuid.UUID]:
        async with pool.transaction() as cur:
            reservations = await reserve_agents(
                cur,
                campaign_id=campaign,
                worker_id=name,
                n=5,
                lease_seconds=LEASE_SECONDS,
                now=NOW,
            )
            return {r.agent_id for r in reservations}

    first, second = await asyncio.gather(worker("worker-a"), worker("worker-b"))
    assert first.isdisjoint(second)

    async with pool.cursor() as cur:
        await cur.execute(
            "SELECT id, lease_owner FROM agents WHERE campaign_id = %s", (campaign,)
        )
        owner_of = {row["id"]: row["lease_owner"] for row in await cur.fetchall()}
    for agent_id in first:
        assert owner_of[agent_id] == "worker-a"
    for agent_id in second:
        assert owner_of[agent_id] == "worker-b"


# ---------------------------------------------------------------------------
# Short and empty returns -- the trap Step 5's loop must respect
# ---------------------------------------------------------------------------


async def test_reserving_more_than_available_returns_what_exists(
    pool: Database, campaign
):
    """LIMIT n with fewer than n rows returns fewer rows, silently.

    The dialer loop has to treat that as "stop", not as "I got 20". Asserting
    on the count rather than only on distinctness is what makes this visible:
    a version that silently returned an empty list would pass a
    distinctness-only assertion.
    """
    await make_agents(pool, campaign, 3)
    async with pool.transaction() as cur:
        reservations = await reserve_agents(
            cur,
            campaign_id=campaign,
            worker_id="greedy",
            n=20,
            lease_seconds=LEASE_SECONDS,
            now=NOW,
        )
    assert len(reservations) == 3


async def test_reserving_from_an_empty_pool_returns_empty(pool: Database, campaign):
    """No agents logged in at all. Must be an empty list, not an error and not
    a None the caller will iterate over."""
    async with pool.transaction() as cur:
        reservations = await reserve_agents(
            cur,
            campaign_id=campaign,
            worker_id="lonely",
            n=5,
            lease_seconds=LEASE_SECONDS,
            now=NOW,
        )
    assert reservations == []


async def test_offline_agents_are_never_reserved(pool: Database, campaign):
    """Only AVAILABLE agents are dialable. An OFFLINE agent has gone home."""
    await make_agents(pool, campaign, 2)
    async with pool.transaction() as cur:
        await cur.execute(
            "UPDATE agents SET state = 'OFFLINE' WHERE campaign_id = %s", (campaign,)
        )
    async with pool.transaction() as cur:
        reservations = await reserve_agents(
            cur,
            campaign_id=campaign,
            worker_id="w",
            n=5,
            lease_seconds=LEASE_SECONDS,
            now=NOW,
        )
    assert reservations == []


async def test_requesting_zero_or_negative_does_not_touch_the_database(
    pool: Database, campaign
):
    """The safety controller can approve zero calls, and it does so often.
    That must be a cheap no-op, not a query that reserves something."""
    await make_agents(pool, campaign, 5)
    async with pool.transaction() as cur:
        assert await reserve_agents(
            cur, campaign_id=campaign, worker_id="w", n=0,
            lease_seconds=LEASE_SECONDS, now=NOW,
        ) == []
        assert await reserve_agents(
            cur, campaign_id=campaign, worker_id="w", n=-5,
            lease_seconds=LEASE_SECONDS, now=NOW,
        ) == []
        counts = await count_agents_by_state(cur, campaign_id=campaign)
    assert counts[AgentState.AVAILABLE] == 5


async def test_longest_idle_agent_is_reserved_first(pool: Database, campaign):
    """Fair distribution. Without the ORDER BY, one agent could take every call
    while another sits idle all shift.

    Compares SETS, not sequences, and that is not laziness. The ORDER BY sits
    in the subquery that decides WHICH rows to lock; the RETURNING clause of
    the enclosing UPDATE has no defined order at all, and PostgreSQL is free to
    emit the updated rows in whatever order the plan produced them. Asserting
    the returned sequence tests the planner's mood -- it passed for a while and
    then started failing once unrelated churn changed the table's physical
    layout. The set is what the query actually guarantees, so the set is what
    this asserts.
    """
    agent_ids = await make_agents(pool, campaign, 5)  # staggered, oldest first
    async with pool.transaction() as cur:
        reservations = await reserve_agents(
            cur, campaign_id=campaign, worker_id="w", n=2,
            lease_seconds=LEASE_SECONDS, now=NOW,
        )
    assert {r.agent_id for r in reservations} == set(agent_ids[:2])


# ---------------------------------------------------------------------------
# Compare-and-swap
# ---------------------------------------------------------------------------


async def test_cas_rejects_stale_version(pool: Database, campaign):
    """A worker holding an out-of-date read must lose, quietly and safely.

    It gets None back, which means "somebody moved this, re-read and decide
    again" -- never "force the write through".
    """
    agent_ids = await make_agents(pool, campaign, 1)
    agent_id = agent_ids[0]

    async with pool.transaction() as cur:
        reservations = await reserve_agents(
            cur, campaign_id=campaign, worker_id="w1", n=1,
            lease_seconds=LEASE_SECONDS, now=NOW,
        )
    current_version = reservations[0].version

    # Someone else moves the agent on, bumping the version.
    async with pool.transaction() as cur:
        moved = await transition_agent(
            cur,
            agent_id=agent_id,
            expected_version=current_version,
            expected_state=AgentState.RESERVED,
            target_state=AgentState.DIALING,
            now=NOW,
        )
    assert moved is not None

    # Our stale write must not land.
    async with pool.transaction() as cur:
        stale = await transition_agent(
            cur,
            agent_id=agent_id,
            expected_version=current_version,  # stale
            expected_state=AgentState.RESERVED,
            target_state=AgentState.AVAILABLE,
            now=NOW,
        )
    assert stale is None

    async with pool.cursor() as cur:
        await cur.execute("SELECT state FROM agents WHERE id = %s", (agent_id,))
        row = await cur.fetchone()
    assert row["state"] == "DIALING", "a stale write overwrote a newer state"


async def test_cas_rejects_a_wrong_expected_state(pool: Database, campaign):
    """Version alone is not enough. The state guard catches the case where the
    version happens to line up but the agent is not where we think."""
    agent_ids = await make_agents(pool, campaign, 1)
    async with pool.cursor() as cur:
        await cur.execute("SELECT version FROM agents WHERE id = %s", (agent_ids[0],))
        version = (await cur.fetchone())["version"]

    async with pool.transaction() as cur:
        result = await transition_agent(
            cur,
            agent_id=agent_ids[0],
            expected_version=version,
            expected_state=AgentState.RESERVED,  # it is actually AVAILABLE
            target_state=AgentState.DIALING,
            now=NOW,
        )
    assert result is None


async def test_illegal_transition_raises(pool: Database, campaign):
    """Refused before it reaches the database, and loudly.

    If it merely matched zero rows it would be indistinguishable from a lost
    race, and a lost race is normal while an impossible transition is a bug.
    """
    agent_ids = await make_agents(pool, campaign, 1)
    async with pool.transaction() as cur:
        with pytest.raises(IllegalTransition):
            await transition_agent(
                cur,
                agent_id=agent_ids[0],
                expected_version=0,
                expected_state=AgentState.AVAILABLE,
                target_state=AgentState.CONNECTED,  # never without dialling
                now=NOW,
            )


async def test_concurrent_transitions_of_one_agent_have_exactly_one_winner(
    pool: Database, campaign
):
    """20 workers all try the same transition from the same version.

    Exactly one may succeed. This is the CAS equivalent of the SKIP LOCKED
    test: the guarantee is not "the fastest wins" but "only one wins".
    """
    agent_ids = await make_agents(pool, campaign, 1)
    async with pool.transaction() as cur:
        reservations = await reserve_agents(
            cur, campaign_id=campaign, worker_id="w", n=1,
            lease_seconds=LEASE_SECONDS, now=NOW,
        )
    version = reservations[0].version

    async def contender() -> bool:
        async with pool.transaction() as cur:
            result = await transition_agent(
                cur,
                agent_id=agent_ids[0],
                expected_version=version,
                expected_state=AgentState.RESERVED,
                target_state=AgentState.DIALING,
                now=NOW,
            )
            return result is not None

    outcomes = await asyncio.gather(*(contender() for _ in range(20)))
    assert sum(outcomes) == 1, f"{sum(outcomes)} workers all thought they won"


async def test_release_clears_the_lease(pool: Database, campaign):
    """A released agent must carry no lease. One that does is a row the reaper
    keeps looking at forever."""
    agent_ids = await make_agents(pool, campaign, 1)
    async with pool.transaction() as cur:
        reservations = await reserve_agents(
            cur, campaign_id=campaign, worker_id="w", n=1,
            lease_seconds=LEASE_SECONDS, now=NOW,
        )
        released = await release_agent(
            cur,
            agent_id=agent_ids[0],
            expected_version=reservations[0].version,
            expected_state=AgentState.RESERVED,
            now=NOW,
        )
    assert released is not None
    assert released.state is AgentState.AVAILABLE
    assert released.lease_owner is None
    assert released.lease_expires_at is None


async def test_lease_renewal_requires_still_owning_the_lease(pool: Database, campaign):
    """A worker that lost its agent to the reaper cannot quietly take it back
    by renewing. It gets False and must re-reserve."""
    agent_ids = await make_agents(pool, campaign, 1)
    async with pool.transaction() as cur:
        await reserve_agents(
            cur, campaign_id=campaign, worker_id="owner", n=1,
            lease_seconds=LEASE_SECONDS, now=NOW,
        )
    async with pool.transaction() as cur:
        assert await renew_lease(
            cur, agent_id=agent_ids[0], worker_id="owner",
            lease_seconds=LEASE_SECONDS, now=NOW,
        ) is True
        assert await renew_lease(
            cur, agent_id=agent_ids[0], worker_id="impostor",
            lease_seconds=LEASE_SECONDS, now=NOW,
        ) is False


# ---------------------------------------------------------------------------
# Fleet operations
# ---------------------------------------------------------------------------


async def test_logout_leaves_agents_who_are_mid_call_alone(pool: Database, campaign):
    """Failure scenario 3 in miniature. An agent on a live call cannot simply
    vanish: dropping them would abandon the borrower or strand the call."""
    agent_ids = await make_agents(pool, campaign, 3)
    async with pool.transaction() as cur:
        await reserve_agents(
            cur, campaign_id=campaign, worker_id="w", n=1,
            lease_seconds=LEASE_SECONDS, now=NOW,
        )
    async with pool.transaction() as cur:
        went_offline = await logout_agents(cur, agent_ids=agent_ids, now=NOW)
    assert len(went_offline) == 2  # the reserved one stays put

    async with pool.transaction() as cur:
        counts = await count_agents_by_state(cur, campaign_id=campaign)
    assert counts[AgentState.OFFLINE] == 2
    assert counts[AgentState.RESERVED] == 1


async def test_login_only_moves_offline_agents(pool: Database, campaign):
    agent_ids = await make_agents(pool, campaign, 2)
    async with pool.transaction() as cur:
        await cur.execute(
            "UPDATE agents SET state = 'OFFLINE' WHERE id = %s", (agent_ids[0],)
        )
    async with pool.transaction() as cur:
        moved = await login_agents(cur, agent_ids=agent_ids, now=NOW)
    assert moved == 1


async def test_heartbeat_records_liveness(pool: Database, campaign):
    agent_ids = await make_agents(pool, campaign, 2)
    async with pool.transaction() as cur:
        assert await heartbeat(cur, agent_ids=agent_ids, now=NOW) == 2
    async with pool.cursor() as cur:
        # Scoped to this campaign. The unscoped version counted every agent in
        # the database that happened to share the timestamp, so it depended on
        # what other test files had left lying around -- it passed until a new
        # fixture seeded agents at the same instant.
        await cur.execute(
            "SELECT count(*) AS n FROM agents "
            "WHERE campaign_id = %s AND last_heartbeat_at = %s",
            (campaign, NOW),
        )
        assert (await cur.fetchone())["n"] == 2


# ---------------------------------------------------------------------------
# Borrowers
# ---------------------------------------------------------------------------


async def test_borrower_not_double_dialed(pool: Database, campaign):
    """The borrower-side equivalent of the headline test. 30 workers, 8
    borrowers: exactly 8 reservations, all distinct. Ringing the same person
    twice for one debt is a compliance problem, not an inefficiency."""
    await make_borrowers(pool, campaign, 8)

    async def worker(index: int) -> list[uuid.UUID]:
        async with pool.transaction() as cur:
            reservations = await reserve_borrowers(
                cur,
                campaign_id=campaign,
                worker_id=f"w{index}",
                n=1,
                lease_seconds=LEASE_SECONDS,
                now=NOW,
            )
            return [r.borrower_id for r in reservations]

    results = await asyncio.gather(*(worker(i) for i in range(30)))
    won = [borrower_id for result in results for borrower_id in result]
    assert len(won) == 8
    assert len(set(won)) == 8


async def test_borrowers_not_yet_eligible_are_skipped(pool: Database, campaign):
    """Retry backoff. A borrower dialled a moment ago is not eligible yet, and
    that is what stops a failing number being redialled in a tight loop."""
    borrower_ids = await make_borrowers(pool, campaign, 2)
    async with pool.transaction() as cur:
        await cur.execute(
            "UPDATE borrowers SET next_eligible_at = %s WHERE id = %s",
            (NOW + timedelta(minutes=5), borrower_ids[0]),
        )
    async with pool.transaction() as cur:
        reservations = await reserve_borrowers(
            cur, campaign_id=campaign, worker_id="w", n=10,
            lease_seconds=LEASE_SECONDS, now=NOW,
        )
    assert [r.borrower_id for r in reservations] == [borrower_ids[1]]


async def test_borrowers_out_of_attempts_are_skipped(pool: Database, campaign):
    borrower_ids = await make_borrowers(pool, campaign, 2)
    async with pool.transaction() as cur:
        await cur.execute(
            "UPDATE borrowers SET attempts = max_attempts WHERE id = %s",
            (borrower_ids[0],),
        )
    async with pool.transaction() as cur:
        reservations = await reserve_borrowers(
            cur, campaign_id=campaign, worker_id="w", n=10,
            lease_seconds=LEASE_SECONDS, now=NOW,
        )
    assert [r.borrower_id for r in reservations] == [borrower_ids[1]]


async def test_releasing_an_undialled_borrower_does_not_spend_an_attempt(
    pool: Database, campaign
):
    """Reserved but never dialled -- no agent was free. Nothing was attempted,
    so nothing is charged against their budget."""
    borrower_ids = await make_borrowers(pool, campaign, 1)
    async with pool.transaction() as cur:
        reservations = await reserve_borrowers(
            cur, campaign_id=campaign, worker_id="w", n=1,
            lease_seconds=LEASE_SECONDS, now=NOW,
        )
        released = await release_borrower(
            cur,
            borrower_id=borrower_ids[0],
            expected_version=reservations[0].version,
            now=NOW,
        )
    assert released is not None
    assert released.state is BorrowerState.PENDING
    assert released.attempts == 0
    assert released.lease_owner is None


# ---------------------------------------------------------------------------
# The lease / terminality coupling
# ---------------------------------------------------------------------------


async def _reserve_one_borrower_with_call(pool: Database, campaign, call_state: str):
    """Reserve a borrower and put a call in `call_state` behind them, the way a
    worker would just before it crashed."""
    borrower_ids = await make_borrowers(pool, campaign, 1)
    async with pool.transaction() as cur:
        reservations = await reserve_borrowers(
            cur, campaign_id=campaign, worker_id="doomed-worker", n=1,
            lease_seconds=LEASE_SECONDS, now=NOW,
        )
        await cur.execute(
            "INSERT INTO calls (id, campaign_id, borrower_id, provider, "
            "idempotency_key, state) VALUES (%s,%s,%s,'mock_fast',%s,%s)",
            (uuid.uuid4(), campaign, borrower_ids[0], f"key-{uuid.uuid4()}", call_state),
        )
    return borrower_ids[0], reservations[0]


async def test_expired_lease_with_a_live_call_does_not_free_the_borrower(
    pool: Database, campaign
):
    """The coupling, stated as a test.

    The worker died with the call in ANSWERED and the provider unreachable. The
    worker's CLAIM has expired, but the borrower is not free: somebody may be
    on the phone. Returning them to PENDING here would let the dialer ring the
    same person a second time for the same debt during an outage.
    """
    borrower_id, _ = await _reserve_one_borrower_with_call(pool, campaign, "ANSWERED")
    after_expiry = NOW + timedelta(seconds=LEASE_SECONDS + 1)

    async with pool.transaction() as cur:
        freed = await release_expired_leases(cur, now=after_expiry)
    assert borrower_id not in freed

    async with pool.cursor() as cur:
        await cur.execute("SELECT state FROM borrowers WHERE id = %s", (borrower_id,))
        assert (await cur.fetchone())["state"] == "RESERVED"


async def test_expired_lease_with_no_live_call_frees_the_borrower(
    pool: Database, campaign
):
    """The other half. The worker died before placing anything, so there is
    nothing live and the borrower goes straight back in the pool -- attempts
    untouched, because our crash is not their failure to answer."""
    borrower_ids = await make_borrowers(pool, campaign, 1)
    async with pool.transaction() as cur:
        await reserve_borrowers(
            cur, campaign_id=campaign, worker_id="doomed", n=1,
            lease_seconds=LEASE_SECONDS, now=NOW,
        )
    after_expiry = NOW + timedelta(seconds=LEASE_SECONDS + 1)

    async with pool.transaction() as cur:
        freed = await release_expired_leases(cur, now=after_expiry)
    assert borrower_ids[0] in freed

    async with pool.cursor() as cur:
        await cur.execute(
            "SELECT state, attempts FROM borrowers WHERE id = %s", (borrower_ids[0],)
        )
        row = await cur.fetchone()
    assert row["state"] == "PENDING"
    assert row["attempts"] == 0


async def test_borrower_is_freed_once_the_call_reaches_a_terminal_state(
    pool: Database, campaign
):
    """The held borrower is not stuck forever: as soon as reconciliation ends
    the call, the next reaper pass releases them."""
    borrower_id, _ = await _reserve_one_borrower_with_call(pool, campaign, "ANSWERED")
    after_expiry = NOW + timedelta(seconds=LEASE_SECONDS + 1)

    async with pool.transaction() as cur:
        assert borrower_id not in await release_expired_leases(cur, now=after_expiry)

    async with pool.transaction() as cur:
        await cur.execute(
            "UPDATE calls SET state = 'COMPLETED' WHERE borrower_id = %s", (borrower_id,)
        )
    async with pool.transaction() as cur:
        assert borrower_id in await release_expired_leases(cur, now=after_expiry)


async def test_held_borrowers_are_observable(pool: Database, campaign):
    """Where the two clocks disagree has to be visible. If this set grows and
    never drains, call reconciliation is stuck and the borrowers behind it are
    frozen -- which is the first thing to look at when a campaign stalls."""
    borrower_id, _ = await _reserve_one_borrower_with_call(pool, campaign, "RINGING")
    after_expiry = NOW + timedelta(seconds=LEASE_SECONDS + 1)

    async with pool.transaction() as cur:
        held = await borrowers_held_by_live_call(cur, campaign_id=campaign)
    assert [row["borrower_id"] for row in held] == [borrower_id]
    assert held[0]["call_state"] == "RINGING"


async def test_a_borrower_with_a_live_call_is_never_reserved(pool: Database, campaign):
    """Defence in depth against the same compliance failure.

    Even if a borrower's row were repaired to PENDING by hand while a call was
    still live, reservation refuses to hand them out.
    """
    borrower_id, _ = await _reserve_one_borrower_with_call(pool, campaign, "RINGING")
    async with pool.transaction() as cur:
        await cur.execute(
            "UPDATE borrowers SET state = 'PENDING', lease_owner = NULL "
            "WHERE id = %s",
            (borrower_id,),
        )
    async with pool.transaction() as cur:
        reservations = await reserve_borrowers(
            cur, campaign_id=campaign, worker_id="w", n=10,
            lease_seconds=LEASE_SECONDS, now=NOW,
        )
    assert reservations == []


# ---------------------------------------------------------------------------
# The lease / terminality coupling (the two-clock rule)
# ---------------------------------------------------------------------------


async def test_borrower_with_non_terminal_call_is_not_returned_to_pending(
    pool: Database, campaign
):
    """The rule that keeps the one-live-call index from firing on correct
    behaviour.

    A worker crashes while a call is ANSWERED and the provider is unreachable,
    so the call stays non-terminal for a long time. The worker's CLAIM on the
    borrower has expired; the borrower has NOT become free. If lease expiry
    alone returned them to PENDING, the dialer would reserve them, insert a
    second call, and hit the unique partial index -- a confusing integrity
    error raised by allocation working exactly as designed. Worse, without that
    index it would ring one person twice about one debt during an outage.

    So: the lease is detached, the state stays RESERVED, and only the call
    reaching a terminal state frees them.
    """
    borrower_id, _ = await _reserve_one_borrower_with_call(pool, campaign, "ANSWERED")
    expired = NOW + timedelta(seconds=LEASE_SECONDS + 1)

    async with pool.transaction() as cur:
        freed = await release_expired_leases(cur, now=expired)
        detached = await detach_expired_leases(cur, now=expired)

    assert borrower_id not in freed, "a borrower on a live call must not be freed"
    assert borrower_id in detached, "the dead worker's claim must still be dropped"

    async with pool.transaction() as cur:
        await cur.execute(
            "SELECT state, lease_owner, lease_expires_at FROM borrowers WHERE id = %s",
            (borrower_id,),
        )
        row = await cur.fetchone()

    assert row["state"] == "RESERVED", "still held by the live call"
    assert row["lease_owner"] is None, "but no worker claims them any more"
    assert row["lease_expires_at"] is None


async def test_detaching_the_lease_makes_the_sweep_idempotent(pool: Database, campaign):
    """Run the pair twice; the second pass finds nothing.

    This is why the lease is cleared rather than left expired. A borrower whose
    lease stays in the past is rediscovered by every sweep forever, so the
    reaper reports work on every pass and "did anything change?" stops being a
    question the logs can answer.
    """
    borrower_id, _ = await _reserve_one_borrower_with_call(pool, campaign, "ANSWERED")
    expired = NOW + timedelta(seconds=LEASE_SECONDS + 1)

    async with pool.transaction() as cur:
        first = await detach_expired_leases(cur, now=expired)
    async with pool.transaction() as cur:
        second = await detach_expired_leases(cur, now=expired)

    assert first == [borrower_id]
    assert second == [], "the second pass must be a no-op"


async def test_a_detached_borrower_is_freed_once_the_call_ends(pool: Database, campaign):
    """The other half of the rule. The borrower has no lease at all now, so the
    release sweep has to accept a NULL lease or they are stranded forever."""
    borrower_id, _ = await _reserve_one_borrower_with_call(pool, campaign, "ANSWERED")
    expired = NOW + timedelta(seconds=LEASE_SECONDS + 1)

    async with pool.transaction() as cur:
        await detach_expired_leases(cur, now=expired)
        # Reconciliation finishes the call.
        await cur.execute(
            "UPDATE calls SET state = 'COMPLETED' WHERE borrower_id = %s",
            (borrower_id,),
        )
        freed = await release_expired_leases(cur, now=expired)

    assert borrower_id in freed

    async with pool.transaction() as cur:
        await cur.execute("SELECT state, attempts FROM borrowers WHERE id = %s", (borrower_id,))
        row = await cur.fetchone()
    assert row["state"] == "PENDING"
    # The worker crashed; the borrower was never actually reached. Charging an
    # attempt for our failure would eventually mark a reachable person
    # EXHAUSTED.
    assert row["attempts"] == 0


async def test_held_borrowers_stay_observable_after_their_lease_is_detached(
    pool: Database, campaign
):
    """The stall diagnostic has to keep working once the lease is gone --
    otherwise clearing the lease would hide exactly the set you need to see
    when reconciliation is stuck."""
    borrower_id, _ = await _reserve_one_borrower_with_call(pool, campaign, "ANSWERED")
    expired = NOW + timedelta(seconds=LEASE_SECONDS + 1)

    async with pool.transaction() as cur:
        await detach_expired_leases(cur, now=expired)
        held = await borrowers_held_by_live_call(cur, campaign_id=campaign)

    assert borrower_id in [row["borrower_id"] for row in held]
