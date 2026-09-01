"""Running a scenario end to end, on a virtual clock.

The point of this harness is what it does NOT substitute. It runs the real
worker, the real allocator, the real safety controller, the real reaper and the
real database. The only thing replaced is the telephone network, and even that
is replaced by a carrier that misbehaves in the specific ways the brief
describes. If the simulation ran a simplified dialer, a good result would mean
nothing about the system.

Time is virtual, so a 60-second provider outage and a 420-second campaign cost
no wall-clock at all, and every run is reproducible from its seed. This is the
entire return on the injected-clock rule: without it, scenario E would take a
minute of real time to observe, so nobody would run it, so the recovery path
would be tested once and then never again.

The tick sequence is deliberate:

    tick -> advance the clock -> drain -> sweep

`drain` waits for the carrier tasks that are ready to finish rather than for
wall-clock time, which is what makes the run deterministic instead of merely
seeded. The reaper sweeps on its own schedule, exactly as in production, so
recovery is part of what is being measured rather than an idealisation.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from smartdialer.core.clock import VirtualClock
from smartdialer.core.config import Settings
from smartdialer.core.db import Database
from smartdialer.core.logging import StructuredLogger
from smartdialer.core.models import CampaignMode
from smartdialer.providers.mock_fast import make_fast_provider
from smartdialer.providers.mock_flaky import make_flaky_provider
from smartdialer.sim.scenarios import EventKind, Scenario, ScenarioEvent
from smartdialer.workers.dialer_worker import DialerWorker
from smartdialer.workers.reaper import Reaper

START = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)

# The columns of the per-tick CSV, in order.
COLUMNS = (
    "t",
    "mode",
    "agents_available",
    "agents_connected",
    "utilization_pct",
    "calls_initiated",
    "calls_ringing",
    "calls_connected",
    "calls_abandoned",
    "abandon_rate_pct",
    "p_hat",
    "mu_A",
    "sigma_A",
    "mu_G",
    "sigma_G",
    "proposed",
    "approved",
    "dialed",
    "reason_code",
    "shortfall_reason",
    "overdial_credit",
    "breaker_state",
    "wait_ms_p50",
    "wait_ms_p95",
)


@dataclass
class RunResult:
    scenario: str
    mode: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    # One narrative per tick, for --explain. Kept beside the numbers rather
    # than in the CSV: it is a sentence for a human, not a column to plot.
    narratives: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


async def run_scenario(
    pool: Database,
    scenario: Scenario,
    mode: CampaignMode,
    *,
    seed: int = 7,
    tick_seconds: float = 0.25,
) -> RunResult:
    campaign_id = await _create_campaign(pool, scenario, mode)
    clock = VirtualClock(start=START)
    provider = _make_provider(scenario, clock, seed=seed)

    settings = Settings(
        worker_id=f"sim-{scenario.key}-{mode.value.lower()}",
        tick_seconds=tick_seconds,
        lease_seconds=30.0,
        reserve_lease_seconds=5.0,
    )
    logger = StructuredLogger("sim", clock)
    worker = DialerWorker(
        db=pool,
        clock=clock,
        campaign_id=campaign_id,
        providers=[provider],
        settings=settings,
        logger=logger,
    )
    worker.attach_providers()
    reaper = Reaper(
        db=pool,
        clock=clock,
        campaign_id=campaign_id,
        providers=[provider],
        settings=settings,
        logger=logger,
    )

    result = RunResult(scenario=scenario.key, mode=mode.value)
    pending = list(scenario.events)
    steps = int(scenario.duration_seconds / tick_seconds)
    next_sweep = 0.0
    # The sweep runs as a task rather than inline, and this is not an
    # optimisation. Reconciliation asks the carrier what happened to a call,
    # and the carrier's latency is virtual -- so an awaited sweep parks on a
    # clock that only this loop advances, and the run deadlocks. As a task it
    # progresses while time moves, which is also what it does in production,
    # where the reaper is its own loop and nothing waits for it.
    sweeping: asyncio.Task | None = None

    try:
        for step in range(steps):
            t = step * tick_seconds

            # Whatever the scenario does to the world, it does before the tick
            # that has to cope with it.
            while pending and pending[0].at_seconds <= t:
                await _apply(pending.pop(0), pool, campaign_id, provider, clock)

            # The agents' softphones checking in. Without this the reaper
            # correctly takes the entire workforce OFFLINE after thirty
            # seconds of silence -- which is the right behaviour for a real
            # agent whose browser died, and a simulation artefact here.
            if step % 4 == 0:
                await _heartbeat(pool, campaign_id, clock.now())

            execution = await worker.tick()
            metrics = await _metrics(pool, campaign_id, scenario.agents)
            result.rows.append(_row(t, mode, execution, metrics, worker.last_proposal))
            result.narratives.append(
                _narrative(step, t, execution, worker.last_proposal)
            )

            await clock.advance(tick_seconds)
            await worker.drain()

            if t >= next_sweep and (sweeping is None or sweeping.done()):
                # Never two sweeps at once. One reaper loop per worker is the
                # production shape, and overlapping sweeps would have them
                # reconciling the same call against the same carrier.
                sweeping = asyncio.ensure_future(reaper.sweep())
                next_sweep = t + settings.reaper_seconds

        result.summary = await _summarise(pool, campaign_id, scenario, result)
    finally:
        if sweeping is not None and not sweeping.done():
            # Cancel and then let the cancellation actually land. Dropping a
            # task mid-transaction leaves its connection in the pool INTRANS,
            # and the pool then discards it noisily on the way out.
            sweeping.cancel()
            await asyncio.gather(sweeping, return_exceptions=True)
        await worker.close()
        await provider.close()
        await _cleanup(pool, campaign_id)

    return result


# ---------------------------------------------------------------------------
# Setup and teardown
# ---------------------------------------------------------------------------


def _make_provider(scenario: Scenario, clock: VirtualClock, *, seed: int):
    factory = make_fast_provider if scenario.provider == "fast" else make_flaky_provider
    return factory(
        clock,
        seed=seed,
        answer_rate=scenario.answer_rate,
        talk_median_seconds=scenario.talk_median_seconds,
    )


async def _create_campaign(
    pool: Database, scenario: Scenario, mode: CampaignMode
) -> uuid.UUID:
    campaign_id = uuid.uuid4()
    async with pool.transaction() as cur:
        await cur.execute(
            """
            -- target_shortfall_eps is 0.005 here rather than the 0.02
            -- default. The bound is a per-TICK probability and the dialer
            -- ticks four times a second, so 0.02 per tick is a great many
            -- chances to be unlucky over a campaign, while the number that
            -- has to stay under the abandon budget is the total. This is the
            -- one parameter tuned against the simulation, and it lives in the
            -- campaign row rather than in the engine precisely so that tuning
            -- it needs no code change.
            INSERT INTO campaigns (id, name, mode, max_concurrent,
                                   abandon_budget_pct, target_shortfall_eps,
                                   max_overdial_ratio, wrap_up_seconds)
            VALUES (%s, %s, %s, 5000, 3.0, 0.005, 2.0, 5)
            """,
            (campaign_id, f"sim-{scenario.key}-{mode.value}", mode.value),
        )
        await cur.execute(
            "INSERT INTO campaign_counters (campaign_id, shard) "
            "SELECT %s, generate_series(0, 15)",
            (campaign_id,),
        )
        # One statement each rather than a loop of inserts: six thousand round
        # trips would dominate the runtime of the scenario they are setting up.
        await cur.execute(
            """
            INSERT INTO agents (id, campaign_id, state, state_changed_at,
                                last_heartbeat_at)
            SELECT gen_random_uuid(), %s, 'AVAILABLE', %s, %s
            FROM generate_series(1, %s)
            """,
            (campaign_id, START, START, scenario.agents),
        )
        await cur.execute(
            """
            INSERT INTO borrowers (id, campaign_id, phone, next_eligible_at,
                                   dpd_bucket)
            SELECT gen_random_uuid(), %s, '+9198' || lpad(i::text, 8, '0'), %s,
                   (ARRAY['0-30','31-60','61-90','90+'])[1 + (i %% 4)]
            FROM generate_series(1, %s) AS i
            """,
            (campaign_id, START, scenario.borrowers),
        )
    return campaign_id


async def _heartbeat(pool: Database, campaign_id: uuid.UUID, now: datetime) -> None:
    """Every agent who is still logged in says so.

    One statement for the whole workforce: the alternative is a heartbeat
    endpoint per agent, which would measure the simulation's own plumbing
    rather than the dialer's.
    """
    async with pool.transaction() as cur:
        await cur.execute(
            "UPDATE agents SET last_heartbeat_at = %s "
            "WHERE campaign_id = %s AND state <> 'OFFLINE'",
            (now, campaign_id),
        )


async def _cleanup(pool: Database, campaign_id: uuid.UUID) -> None:
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


async def _apply(
    event: ScenarioEvent,
    pool: Database,
    campaign_id: uuid.UUID,
    provider,
    clock: VirtualClock,
) -> None:
    if event.kind == EventKind.ANSWER_RATE:
        provider.controls.answer_rate = event.value
        if event.talk_median_seconds is not None:
            provider.controls.talk_median_seconds = event.talk_median_seconds
    elif event.kind == EventKind.OUTAGE:
        provider.controls.outage_until = clock.now() + timedelta(
            seconds=event.duration_seconds
        )
    elif event.kind == EventKind.AGENTS_OFFLINE:
        # Agents vanish the way they actually vanish: they stop being
        # available, whatever they happened to be doing. Only the idle ones
        # can be taken instantly -- an agent on a call goes offline when the
        # call ends, which is what makes this a gradual drop rather than a
        # cliff, and what the dialer has to track.
        async with pool.transaction() as cur:
            await cur.execute(
                """
                UPDATE agents SET state = 'OFFLINE', state_changed_at = %s,
                                  version = version + 1
                WHERE id IN (
                    SELECT id FROM agents
                    WHERE campaign_id = %s AND state = 'AVAILABLE'
                    LIMIT %s
                )
                """,
                (clock.now(), campaign_id, int(event.value)),
            )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


async def _metrics(pool: Database, campaign_id: uuid.UUID, agents: int) -> dict:
    """One query per tick for everything the CSV needs.

    One, not six. The snapshot query already costs a round trip per tick, and
    a measurement harness that doubled the tick's database work would be
    measuring itself as much as the dialer.
    """
    async with pool.transaction() as cur:
        await cur.execute(
            """
            WITH agent_states AS (
                SELECT
                  count(*) FILTER (WHERE state = 'AVAILABLE')::int AS available,
                  count(*) FILTER (WHERE state = 'CONNECTED')::int AS connected,
                  count(*) FILTER (WHERE state = 'OFFLINE')::int   AS offline
                FROM agents WHERE campaign_id = %(campaign_id)s
            ),
            call_states AS (
                SELECT
                  count(*) FILTER (WHERE state = 'INITIATED')::int AS initiated,
                  count(*) FILTER (WHERE state = 'RINGING')::int   AS ringing,
                  count(*) FILTER (WHERE state = 'CONNECTED')::int AS connected,
                  count(*) FILTER (WHERE state = 'ABANDONED')::int AS abandoned,
                  count(*) FILTER (WHERE answered_at IS NOT NULL)::int AS answered,
                  percentile_cont(0.5) WITHIN GROUP (ORDER BY wait_ms)
                    FILTER (WHERE wait_ms IS NOT NULL) AS wait_p50,
                  percentile_cont(0.95) WITHIN GROUP (ORDER BY wait_ms)
                    FILTER (WHERE wait_ms IS NOT NULL) AS wait_p95
                FROM calls WHERE campaign_id = %(campaign_id)s
            )
            SELECT * FROM agent_states, call_states
            """,
            {"campaign_id": campaign_id},
        )
        row = await cur.fetchone()

    # Utilization is against the agents who are actually logged in. Counting
    # agents who went offline would make scenario F look like a collapse in
    # utilization when what happened is that the workforce shrank.
    online = max(1, agents - row["offline"])
    row["utilization_pct"] = 100.0 * row["connected"] / online
    row["online"] = online
    return row


def _row(t: float, mode: CampaignMode, execution, metrics: dict, proposal) -> dict:
    terms = execution.decision.terms
    answered = metrics["answered"] or 0
    abandoned = metrics["abandoned"] or 0
    return {
        "t": round(t, 3),
        "mode": mode.value,
        "agents_available": metrics["available"],
        "agents_connected": metrics["connected"],
        "utilization_pct": round(metrics["utilization_pct"], 2),
        "calls_initiated": metrics["initiated"],
        "calls_ringing": metrics["ringing"],
        "calls_connected": metrics["connected"],
        "calls_abandoned": abandoned,
        "abandon_rate_pct": round(100.0 * abandoned / answered, 3) if answered else 0.0,
        "p_hat": round(getattr(proposal, "p_hat", 0.0), 4),
        "mu_A": round(getattr(proposal, "mu_A", 0.0), 3),
        "sigma_A": round(getattr(proposal, "sigma_A", 0.0), 3),
        "mu_G": round(getattr(proposal, "mu_G", 0.0), 3),
        "sigma_G": round(getattr(proposal, "sigma_G", 0.0), 3),
        "proposed": execution.proposed,
        "approved": execution.approved,
        "dialed": execution.dialed,
        "reason_code": execution.reason_code,
        "shortfall_reason": execution.shortfall_reason,
        "overdial_credit": terms.get("overdial_credit", 0),
        "breaker_state": terms.get("breaker_state", "CLOSED"),
        "wait_ms_p50": round(metrics["wait_p50"] or 0.0, 1),
        "wait_ms_p95": round(metrics["wait_p95"] or 0.0, 1),
    }


def _narrative(step: int, t: float, execution, proposal) -> dict:
    """Everything --explain needs for one tick, and nothing more."""
    decision = execution.decision
    return {
        "tick": step,
        "t": round(t, 3),
        "proposed": execution.proposed,
        "approved": execution.approved,
        "dialed": execution.dialed,
        "reason_code": execution.reason_code,
        "shortfall_reason": execution.shortfall_reason,
        "clamps": list(decision.clamps),
        "overdial": decision.overdial,
        "terms": {
            key: value
            for key, value in decision.terms.items()
            if key not in ("budget", "breaker")
        },
        "budget": decision.terms.get("budget", {}),
        "breaker": decision.terms.get("breaker", {}),
        "engine": {
            "explanation": proposal.explain() if proposal else "",
            "reason": getattr(proposal, "reason", ""),
            "mu_A": getattr(proposal, "mu_A", 0.0),
            "sigma_A": getattr(proposal, "sigma_A", 0.0),
            "mu_G": getattr(proposal, "mu_G", 0.0),
            "sigma_G": getattr(proposal, "sigma_G", 0.0),
            "p_hat": getattr(proposal, "p_hat", 0.0),
            "epsilon": getattr(proposal, "epsilon", 0.0),
            "window_seconds": getattr(proposal, "window_seconds", 0.0),
            "changepoint_detected": getattr(proposal, "changepoint_detected", False),
            "used_exact_dp": getattr(proposal, "used_exact_dp", False),
            "search_trace": [list(pair) for pair in getattr(proposal, "search_trace", ())],
            "wrap_up_freeing": getattr(proposal, "terms", {}).get("wrap_up_freeing"),
            "ring_hazards": getattr(proposal, "terms", {}).get("ring_hazards"),
        },
    }


async def _summarise(
    pool: Database, campaign_id: uuid.UUID, scenario: Scenario, result: RunResult
) -> dict:
    """The five numbers the comparison table is made of."""
    async with pool.transaction() as cur:
        await cur.execute(
            """
            SELECT
              count(*) FILTER (WHERE connected_at IS NOT NULL)::int AS connects,
              count(*) FILTER (WHERE answered_at IS NOT NULL)::int  AS answers,
              count(*) FILTER (WHERE state = 'ABANDONED')::int      AS abandoned,
              count(*)::int                                         AS placed,
              percentile_cont(0.5) WITHIN GROUP (ORDER BY wait_ms)
                FILTER (WHERE wait_ms IS NOT NULL) AS wait_p50,
              percentile_cont(0.95) WITHIN GROUP (ORDER BY wait_ms)
                FILTER (WHERE wait_ms IS NOT NULL) AS wait_p95
            FROM calls WHERE campaign_id = %s
            """,
            (campaign_id,),
        )
        row = await cur.fetchone()

    utilization = (
        sum(r["utilization_pct"] for r in result.rows) / len(result.rows)
        if result.rows
        else 0.0
    )
    agent_hours = scenario.agents * scenario.duration_seconds / 3600.0
    answers = row["answers"] or 0
    return {
        "scenario": scenario.key,
        "mode": result.mode,
        "utilization_pct": round(utilization, 1),
        "connects_per_agent_hour": round((row["connects"] or 0) / agent_hours, 1),
        "abandon_pct": round(100.0 * (row["abandoned"] or 0) / answers, 2)
        if answers
        else 0.0,
        "wait_p50_ms": round(row["wait_p50"] or 0.0, 0),
        "wait_p95_ms": round(row["wait_p95"] or 0.0, 0),
        "calls_placed": row["placed"],
        "connects": row["connects"],
        "answers": answers,
        "abandoned": row["abandoned"],
    }
