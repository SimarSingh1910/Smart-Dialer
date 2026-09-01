"""The learned distributions: hazard tables and answer propensities.

Separated from the per-tick snapshot because they change on a completely
different timescale. Agent counts and ringing calls change every tick and are
read every tick; the shape of the ring-to-answer distribution changes over
minutes and is built from thousands of calls. Rebuilding it four times a second
would spend most of the tick budget re-deriving a curve that has not moved.

So this is refreshed on its own schedule and handed to the engine as data. The
cost of the staleness is small and bounded: a hazard curve a few seconds out of
date is still the right curve, whereas an agent count a few seconds out of date
is how a dialer abandons calls -- which is why the snapshot's freshness has a
safety clamp and this does not.

Everything here produces PURE objects from `pacing/`. This module does the I/O;
the engine receives tables and knows nothing about where they came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from psycopg import AsyncCursor

from smartdialer.pacing.hazard import (
    DEFAULT_RING_MEDIAN,
    DEFAULT_RING_SIGMA,
    DEFAULT_TALK_MEDIAN,
    DEFAULT_TALK_SIGMA,
    HazardTable,
)
from smartdialer.pacing.propensity import BorrowerFeatures, PropensityTable

# How far back the distributions are learned from. Long enough to fill 2-second
# buckets with real counts, short enough that yesterday's behaviour does not
# outvote this afternoon's.
HISTORY_WINDOW_SECONDS = 3600.0
# A cap on rows pulled per rebuild. The tables converge long before this, and an
# unbounded scan on a busy campaign is a latency spike waiting to happen.
HISTORY_LIMIT = 2000
# How many upcoming borrowers to price. The engine only needs propensities for
# the calls it is about to authorise.
CANDIDATE_LIMIT = 64


@dataclass(frozen=True, slots=True)
class CampaignHistory:
    """Everything learned rather than observed."""

    built_at: datetime
    ring_hazard: HazardTable
    talk_hazard: HazardTable
    propensity: PropensityTable
    candidates: tuple[BorrowerFeatures, ...] = ()

    def candidate_probabilities(self) -> tuple[float, ...]:
        return tuple(self.propensity.probability(c) for c in self.candidates)

    def age_seconds(self, now: datetime) -> float:
        return (now - self.built_at).total_seconds()


RING_OBSERVATIONS_SQL = """
-- Ring-to-answer durations, split into the two populations survival analysis
-- needs. A call that was ANSWERED is an event at that duration. A call that
-- ended some other way is CENSORED: it was genuinely at risk up to the moment
-- it left, and it says nothing about whether it would have been answered
-- later. Treating those as non-answers across the whole horizon would drag
-- every later bucket towards zero and teach the model that long-ringing calls
-- never answer -- which is precisely when it would then over-dial.
SELECT
    EXTRACT(EPOCH FROM (answered_at - COALESCE(ringing_at, initiated_at)))::float8
        AS duration,
    true AS answered
FROM calls
WHERE campaign_id = %(campaign_id)s
  AND created_at >= %(now)s::timestamptz - make_interval(secs => %(window)s)
  AND answered_at IS NOT NULL
  AND COALESCE(ringing_at, initiated_at) IS NOT NULL
  AND answered_at > COALESCE(ringing_at, initiated_at)

UNION ALL

SELECT
    EXTRACT(EPOCH FROM (ended_at - COALESCE(ringing_at, initiated_at)))::float8
        AS duration,
    false AS answered
FROM calls
WHERE campaign_id = %(campaign_id)s
  AND created_at >= %(now)s::timestamptz - make_interval(secs => %(window)s)
  AND answered_at IS NULL
  AND ended_at IS NOT NULL
  AND COALESCE(ringing_at, initiated_at) IS NOT NULL
  AND ended_at > COALESCE(ringing_at, initiated_at)

LIMIT %(limit)s
"""

TALK_OBSERVATIONS_SQL = """
-- Completed conversations. Every one is an event: the agent did become free.
SELECT EXTRACT(EPOCH FROM (ended_at - connected_at))::float8 AS duration
FROM calls
WHERE campaign_id = %(campaign_id)s
  AND created_at >= %(now)s::timestamptz - make_interval(secs => %(window)s)
  AND connected_at IS NOT NULL
  AND ended_at IS NOT NULL
  AND ended_at > connected_at
LIMIT %(limit)s
"""

PROPENSITY_SQL = """
-- Answer rates per (hour, attempt, prior outcome, days-past-due) cell.
-- Aggregated in the database rather than by pulling raw calls, because the
-- cells are few and the calls are many.
SELECT
    EXTRACT(HOUR FROM c.initiated_at)::int AS hour_of_day,
    LEAST(c.attempt, 3)                    AS attempt_bucket,
    COALESCE(b.last_outcome, 'none')       AS prior_outcome,
    COALESCE(b.dpd_bucket, 'unknown')      AS dpd_bucket,
    count(*) FILTER (WHERE c.answered_at IS NOT NULL)::int AS answered,
    count(*)::int                                         AS dialled
FROM calls c
JOIN borrowers b ON b.id = c.borrower_id
WHERE c.campaign_id = %(campaign_id)s
  AND c.created_at >= %(now)s::timestamptz - make_interval(secs => %(window)s)
  AND c.initiated_at IS NOT NULL
GROUP BY 1, 2, 3, 4
"""

CANDIDATES_SQL = """
-- The borrowers the allocator would take next, in the order it would take
-- them. Priced so the engine can reason about the calls it is actually about
-- to authorise rather than about an average borrower who does not exist.
--
-- Mirrors the ORDER BY in reserve_borrowers deliberately. If the two ever
-- disagree, the engine is pricing a different set of people than the allocator
-- dials, and the propensity term becomes decoration.
SELECT
    %(hour)s::int                     AS hour_of_day,
    LEAST(attempts + 1, 3)            AS attempt_bucket,
    COALESCE(last_outcome, 'none')    AS prior_outcome,
    COALESCE(dpd_bucket, 'unknown')   AS dpd_bucket
FROM borrowers
WHERE campaign_id = %(campaign_id)s
  AND state = 'PENDING'
  AND next_eligible_at <= %(now)s
  AND attempts < max_attempts
ORDER BY priority DESC, next_eligible_at
LIMIT %(limit)s
"""


async def build_history(
    cur: AsyncCursor,
    *,
    campaign_id: UUID,
    now: datetime,
    campaign_answer_rate: float,
    talk_prior_median: float = DEFAULT_TALK_MEDIAN,
    window_seconds: float = HISTORY_WINDOW_SECONDS,
    limit: int = HISTORY_LIMIT,
    candidate_limit: int = CANDIDATE_LIMIT,
) -> CampaignHistory:
    """Rebuild the learned distributions from recent calls.

    Four small queries rather than one large one. Unlike the snapshot, this is
    not on the tick path, so the round trips do not compete with pacing -- and
    keeping them separate means each is readable and individually explicable,
    which matters more here than shaving a millisecond.
    """
    params = {
        "campaign_id": campaign_id,
        "now": now,
        "window": window_seconds,
        "limit": limit,
    }

    await cur.execute(RING_OBSERVATIONS_SQL, params)
    ring_rows = await cur.fetchall()
    ring_events = [r["duration"] for r in ring_rows if r["answered"]]
    ring_censored = [r["duration"] for r in ring_rows if not r["answered"]]

    await cur.execute(TALK_OBSERVATIONS_SQL, params)
    talk_durations = [r["duration"] for r in await cur.fetchall()]

    await cur.execute(
        PROPENSITY_SQL,
        {"campaign_id": campaign_id, "now": now, "window": window_seconds},
    )
    propensity_rows = await cur.fetchall()

    await cur.execute(
        CANDIDATES_SQL,
        {
            "campaign_id": campaign_id,
            "now": now,
            "hour": now.hour,
            "limit": candidate_limit,
        },
    )
    candidates = tuple(
        BorrowerFeatures(
            hour_of_day=row["hour_of_day"],
            attempt_number=row["attempt_bucket"],
            prior_outcome=row["prior_outcome"],
            dpd_bucket=row["dpd_bucket"],
        )
        for row in await cur.fetchall()
    )

    return CampaignHistory(
        built_at=now,
        ring_hazard=HazardTable.from_observations(
            event_times=ring_events,
            censored_times=ring_censored,
            prior_median=DEFAULT_RING_MEDIAN,
            prior_sigma=DEFAULT_RING_SIGMA,
        ),
        talk_hazard=HazardTable.from_observations(
            event_times=talk_durations,
            prior_median=talk_prior_median,
            prior_sigma=DEFAULT_TALK_SIGMA,
        ),
        propensity=PropensityTable.from_rows(
            propensity_rows, campaign_mean=campaign_answer_rate
        ),
        candidates=candidates,
    )


def empty_history(now: datetime, *, talk_prior_median: float = DEFAULT_TALK_MEDIAN) -> CampaignHistory:
    """The honest starting state: priors only, no cells, no candidates.

    A campaign that has never dialled has no distributions, and pretending
    otherwise would be the worst kind of confidence -- the engine's caution at
    cold start comes from the Wilson upper bound rather than from here, but
    this must not quietly supply numbers it does not have.
    """
    return CampaignHistory(
        built_at=now,
        ring_hazard=HazardTable.prior_only(
            prior_median=DEFAULT_RING_MEDIAN, prior_sigma=DEFAULT_RING_SIGMA
        ),
        talk_hazard=HazardTable.prior_only(
            prior_median=talk_prior_median, prior_sigma=DEFAULT_TALK_SIGMA
        ),
        propensity=PropensityTable(cells={}, campaign_mean=0.2),
    )
