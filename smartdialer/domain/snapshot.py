"""Building the pacing snapshot: one query per tick.

Reading the campaign's state is the only thing the dialer does on every single
tick regardless of whether it dials anything, so its cost is the cost of
running at all. At 250ms ticks and several workers, a snapshot that issued a
handful of round trips would spend most of the tick budget waiting on the
network for data that all comes from one database.

So it is one statement built from CTEs. Each branch is a small indexed
aggregate -- the partial indexes on AVAILABLE agents and in-flight calls exist
for exactly these -- and the result is a single row.

The direction of the dependency matters as much as the cost: this module
imports the pure dataclasses from `pacing` and fills them in. `pacing` does not
import this. Data flows towards the engine and nothing flows back.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg import AsyncCursor

from smartdialer.core.models import AgentState, Campaign, CallState, CampaignMode
from smartdialer.pacing.engine import (
    PacingSnapshot,
    ProviderHealthSignal,
    RecentBehaviour,
)

# How far back the rolling statistics look. Long enough that a quiet few
# seconds does not erase the campaign's history, short enough that a genuine
# change in behaviour -- the answer rate collapsing -- shows up quickly.
DEFAULT_WINDOW_SECONDS = 60.0

# The changepoint check looks at calls placed between LAG and SPAN seconds ago:
# old enough to have finished ringing, recent enough to describe the present.
# Without the lag the answer would always be "hardly any were answered",
# because they are still ringing.
CHANGEPOINT_SPAN_SECONDS = 90.0
CHANGEPOINT_LAG_SECONDS = 30.0
# What the campaign normally does, for the collapse to be measured against.
BASELINE_WINDOW_SECONDS = 1800.0

# Non-terminal call states, as a SQL literal. Kept next to the query that uses
# it and matched to calls_inflight_idx; see the same note in borrowers.py.
IN_FLIGHT_SQL = "('RESERVED','INITIATED','RINGING','ANSWERED','CONNECTED')"

SNAPSHOT_SQL = f"""
WITH agent_counts AS (
    -- Index-only over agents_campaign_state_idx.
    SELECT state::text AS state, count(*)::int AS n
    FROM agents
    WHERE campaign_id = %(campaign_id)s
    GROUP BY 1
),
call_counts AS (
    -- Served by the partial calls_inflight_idx, so this costs what is
    -- happening now rather than everything that ever happened.
    SELECT state::text AS state, count(*)::int AS n
    FROM calls
    WHERE campaign_id = %(campaign_id)s
      AND state IN {IN_FLIGHT_SQL}
    GROUP BY 1
),
ringing AS (
    -- How long each call has been ringing. The count alone is not enough for
    -- the hazard model: a call ringing 3 seconds and one ringing 18 have very
    -- different chances of being answered in the next two.
    SELECT COALESCE(
        array_agg(
            EXTRACT(EPOCH FROM (%(now)s::timestamptz - COALESCE(ringing_at, initiated_at)))::float8
        ),
        '{{}}'::float8[]
    ) AS ages
    FROM calls
    WHERE campaign_id = %(campaign_id)s
      AND state IN ('INITIATED','RINGING')
      AND COALESCE(ringing_at, initiated_at) IS NOT NULL
),
talking AS (
    -- Same idea on the agent side: talk time is roughly log-normal, so how
    -- long this call has already run says a lot about when it will end.
    SELECT COALESCE(
        array_agg(EXTRACT(EPOCH FROM (%(now)s::timestamptz - connected_at))::float8),
        '{{}}'::float8[]
    ) AS ages
    FROM calls
    WHERE campaign_id = %(campaign_id)s
      AND state = 'CONNECTED'
      AND connected_at IS NOT NULL
),
wrapping AS (
    -- Deterministic free capacity: these timers were set when the call ended
    -- and nothing external can move them.
    SELECT COALESCE(
        array_agg(EXTRACT(EPOCH FROM (wrap_up_ends_at - %(now)s::timestamptz))::float8),
        '{{}}'::float8[]
    ) AS remaining
    FROM agents
    WHERE campaign_id = %(campaign_id)s
      AND state = 'WRAP_UP'
      AND wrap_up_ends_at IS NOT NULL
),
recent AS (
    -- Rolling behaviour over the window. Counts, not rates: the engine needs
    -- to know that "80%%" came from four calls out of five.
    SELECT
        count(*) FILTER (WHERE initiated_at  IS NOT NULL)::int AS initiated,
        count(*) FILTER (WHERE answered_at   IS NOT NULL)::int AS answered,
        count(*) FILTER (WHERE connected_at  IS NOT NULL)::int AS connected,
        count(*) FILTER (WHERE state = 'ABANDONED')::int       AS abandoned,
        count(*) FILTER (WHERE state = 'FAILED')::int          AS failed
    FROM calls
    WHERE campaign_id = %(campaign_id)s
      AND created_at >= %(now)s::timestamptz - make_interval(secs => %(window)s)
),
setup AS (
    -- Post-dial delay. p95 sets the forecast window, because what hurts is the
    -- tail: exposure lasts as long as the slowest calls take to connect. p50
    -- is carried too -- the engine needs to know how much of that window a
    -- call placed right now would actually spend ringing rather than
    -- connecting.
    SELECT
        COALESCE(percentile_disc(0.95) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (ringing_at - initiated_at))
        ), 0)::float8 AS p95,
        COALESCE(percentile_disc(0.50) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (ringing_at - initiated_at))
        ), 0)::float8 AS p50
    FROM calls
    WHERE campaign_id = %(campaign_id)s
      AND created_at >= %(now)s::timestamptz - make_interval(secs => %(window)s)
      AND ringing_at IS NOT NULL
      AND initiated_at IS NOT NULL
),
talk_history AS (
    SELECT COALESCE(avg(EXTRACT(EPOCH FROM (ended_at - connected_at))), 0)::float8 AS avg_seconds
    FROM calls
    WHERE campaign_id = %(campaign_id)s
      AND created_at >= %(now)s::timestamptz - make_interval(secs => %(window)s)
      AND connected_at IS NOT NULL
      AND ended_at IS NOT NULL
),
changepoint AS (
    -- Calls old enough to have resolved, but recent enough to describe now.
    -- The engine compares how many of these were answered against how many
    -- the long-run rate said would be, and flags the gap. The lag is
    -- deliberate: asking "how many of the calls placed in the last five
    -- seconds were answered?" always answers "very few", because they have
    -- not finished ringing yet.
    SELECT
        count(*)::int AS resolved,
        count(*) FILTER (WHERE answered_at IS NOT NULL)::int AS answered
    FROM calls
    WHERE campaign_id = %(campaign_id)s
      AND created_at >= %(now)s::timestamptz - make_interval(secs => %(changepoint_span)s)
      AND created_at <  %(now)s::timestamptz - make_interval(secs => %(changepoint_lag)s)
      AND (answered_at IS NOT NULL OR ended_at IS NOT NULL)
),
baseline AS (
    -- The long-run answer rate the changepoint check measures against. A wider
    -- window than `recent`, so a genuine collapse is compared with normality
    -- rather than with itself.
    SELECT
        count(*)::int AS dialled,
        count(*) FILTER (WHERE answered_at IS NOT NULL)::int AS answered
    FROM calls
    WHERE campaign_id = %(campaign_id)s
      AND created_at >= %(now)s::timestamptz - make_interval(secs => %(baseline_window)s)
      AND initiated_at IS NOT NULL
)
SELECT
    (SELECT json_object_agg(state, n) FROM agent_counts) AS agent_counts,
    (SELECT json_object_agg(state, n) FROM call_counts)  AS call_counts,
    (SELECT ages FROM ringing)      AS ring_seconds,
    (SELECT ages FROM talking)      AS talk_seconds,
    (SELECT remaining FROM wrapping) AS wrap_up_remaining,
    (SELECT p95 FROM setup)         AS setup_p95,
    (SELECT p50 FROM setup)         AS setup_p50,
    (SELECT avg_seconds FROM talk_history) AS avg_call_duration,
    (SELECT resolved FROM changepoint)     AS changepoint_resolved,
    (SELECT answered FROM changepoint)     AS changepoint_answered,
    (SELECT dialled  FROM baseline)        AS baseline_dialled,
    (SELECT answered FROM baseline)        AS baseline_answered,
    recent.initiated, recent.answered, recent.connected,
    recent.abandoned, recent.failed
FROM recent
"""


@dataclass(frozen=True, slots=True)
class RawSnapshot:
    """The database half of a pacing snapshot.

    Kept separate from PacingSnapshot because provider health does not come
    from the database -- it comes from the carrier objects the worker holds --
    and stitching the two together is the worker's job, not this module's.
    """

    campaign_id: UUID
    snapshot_taken_at: datetime
    agents: dict[AgentState, int]
    calls: dict[CallState, int]
    ring_seconds: tuple[float, ...]
    talk_seconds: tuple[float, ...]
    wrap_up_remaining: tuple[float, ...]
    setup_p95: float
    setup_p50: float
    avg_call_duration: float
    recent: RecentBehaviour
    # Inputs to the changepoint check: what resolved recently, and what the
    # campaign's long-run rate says should have.
    changepoint_resolved: int = 0
    changepoint_answered: int = 0
    baseline_dialled: int = 0
    baseline_answered: int = 0

    @property
    def baseline_rate(self) -> float:
        return (
            self.baseline_answered / self.baseline_dialled
            if self.baseline_dialled
            else 0.0
        )


async def build_raw_snapshot(
    cur: AsyncCursor,
    *,
    campaign_id: UUID,
    now: datetime,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    changepoint_span_seconds: float = CHANGEPOINT_SPAN_SECONDS,
    changepoint_lag_seconds: float = CHANGEPOINT_LAG_SECONDS,
    baseline_window_seconds: float = BASELINE_WINDOW_SECONDS,
) -> RawSnapshot:
    await cur.execute(
        SNAPSHOT_SQL,
        {
            "campaign_id": campaign_id,
            "now": now,
            "window": window_seconds,
            "changepoint_span": changepoint_span_seconds,
            "changepoint_lag": changepoint_lag_seconds,
            "baseline_window": baseline_window_seconds,
        },
    )
    row = await cur.fetchone()
    # `recent` always yields exactly one row (it is an ungrouped aggregate), so
    # a missing row would mean the query changed shape, not that the campaign
    # is quiet.
    assert row is not None, "the snapshot query must always return one row"

    agent_counts = _counts(row["agent_counts"], AgentState)
    call_counts = _counts(row["call_counts"], CallState)

    return RawSnapshot(
        campaign_id=campaign_id,
        snapshot_taken_at=now,
        agents=agent_counts,
        calls=call_counts,
        ring_seconds=tuple(row["ring_seconds"] or ()),
        talk_seconds=tuple(row["talk_seconds"] or ()),
        wrap_up_remaining=tuple(row["wrap_up_remaining"] or ()),
        setup_p95=float(row["setup_p95"] or 0.0),
        setup_p50=float(row["setup_p50"] or 0.0),
        avg_call_duration=float(row["avg_call_duration"] or 0.0),
        changepoint_resolved=row["changepoint_resolved"] or 0,
        changepoint_answered=row["changepoint_answered"] or 0,
        baseline_dialled=row["baseline_dialled"] or 0,
        baseline_answered=row["baseline_answered"] or 0,
        recent=RecentBehaviour(
            window_seconds=window_seconds,
            initiated=row["initiated"],
            answered=row["answered"],
            connected=row["connected"],
            abandoned=row["abandoned"],
            failed=row["failed"],
        ),
    )


def _counts(raw: dict[str, int] | None, enum_type) -> dict[Any, int]:
    """json_object_agg returns NULL when there are no rows, which is the normal
    state of a campaign that has not started. Every member is present in the
    result so callers never have to guard a lookup."""
    counts = {member: 0 for member in enum_type}
    for state, n in (raw or {}).items():
        counts[enum_type(state)] = n
    return counts


def to_pacing_snapshot(
    raw: RawSnapshot,
    *,
    campaign: Campaign,
    now: datetime,
    provider_health: ProviderHealthSignal,
    history: "CampaignHistory | None" = None,
    overdial_credit: int = 0,
) -> PacingSnapshot:
    """Assemble the snapshot the engine sees.

    `now` is passed separately from raw.snapshot_taken_at on purpose: the
    difference between them is how stale the reading is by the time a decision
    is made, and the safety controller forces progressive behaviour once that
    gap gets large. Collapsing the two would make every snapshot look perfectly
    fresh and quietly disable that clamp.
    """
    from smartdialer.domain.history import empty_history

    history = history or empty_history(now, talk_prior_median=max(1.0, raw.avg_call_duration or 120.0))

    # What the campaign's long-run rate says should have been answered among
    # the calls that recently resolved, and how much that count would naturally
    # vary. The engine compares the observation against this and reports a
    # changepoint; acting on it is the safety controller's job.
    baseline = raw.baseline_rate
    predicted_mean = raw.changepoint_resolved * baseline
    predicted_variance = raw.changepoint_resolved * baseline * (1.0 - baseline)

    return PacingSnapshot(
        mode=campaign.mode,
        snapshot_taken_at=raw.snapshot_taken_at,
        now=now,
        max_concurrent=campaign.max_concurrent,
        max_overdial_ratio=float(campaign.max_overdial_ratio),
        target_shortfall_eps=float(campaign.target_shortfall_eps),
        wrap_up_seconds=campaign.wrap_up_seconds,
        agents_available=raw.agents[AgentState.AVAILABLE],
        calls_connected=raw.calls[CallState.CONNECTED],
        calls_ringing=raw.ring_seconds,
        historical_answer_rate=raw.recent.answer_rate,
        call_setup_time_p95=raw.setup_p95,
        call_setup_time_p50=raw.setup_p50,
        avg_call_duration=raw.avg_call_duration,
        provider_health=provider_health,
        recent_campaign_behaviour=raw.recent,
        talk_seconds=raw.talk_seconds,
        wrap_up_remaining=raw.wrap_up_remaining,
        candidate_propensities=history.candidate_probabilities(),
        ring_hazard=history.ring_hazard,
        talk_hazard=history.talk_hazard,
        agents_reserved=raw.agents[AgentState.RESERVED],
        agents_dialing=raw.agents[AgentState.DIALING],
        agents_wrap_up=raw.agents[AgentState.WRAP_UP],
        agents_paused=raw.agents[AgentState.PAUSED],
        agents_offline=raw.agents[AgentState.OFFLINE],
        calls_reserved=raw.calls[CallState.RESERVED],
        calls_initiated=raw.calls[CallState.INITIATED],
        calls_answered=raw.calls[CallState.ANSWERED],
        overdial_credit=overdial_credit,
        observed_answers_30s=float(raw.changepoint_answered),
        predicted_answers_30s=predicted_mean,
        predicted_answers_30s_variance=predicted_variance,
    )


async def load_campaign(cur: AsyncCursor, *, campaign_id: UUID) -> Campaign | None:
    """Read the campaign's policy row.

    Re-read every tick rather than cached in the process, so an operator
    tightening the abandon budget or flipping `active` on a live campaign takes
    effect on the next tick without a redeploy. It is one primary-key lookup;
    caching it would save nothing and cost the ability to stop a campaign.
    """
    await cur.execute(
        "SELECT * FROM campaigns WHERE id = %(id)s", {"id": campaign_id}
    )
    row = await cur.fetchone()
    return Campaign.from_row(row) if row else None


def is_within_dialing_window(campaign: Campaign, now: datetime) -> bool:
    """Whether the campaign may dial at this moment.

    A campaign with no window set may dial at any time. A window that wraps
    past midnight is treated as two ranges, which is why this is not simply
    `start <= t <= end`.
    """
    start, end = campaign.dialing_window_start, campaign.dialing_window_end
    if start is None or end is None:
        return True
    current = now.timetz().replace(tzinfo=None)
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def describe_mode(mode: CampaignMode) -> str:
    return mode.value
