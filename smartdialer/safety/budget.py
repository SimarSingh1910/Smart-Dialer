"""The abandon budget: an AIMD over-dial credit, shared by every worker.

This is the number that makes predictive dialling different from progressive
dialling, and it is deliberately the most boring component in the project.

    progressive floor   n <= agents_available
    predictive ceiling  n <= agents_available + credit

The credit is the entire over-dial allowance. Everything else about predictive
mode -- the hazard curves, the Poisson-binomial tail, the propensity table --
decides how much of the credit to spend and when. None of it can widen the
credit, because none of it is an input here.

WHAT MOVES THE CREDIT. Only measured outcomes:

    a tick with no new abandons        credit + 1      (additive increase)
    an abandoned call                  credit / 2      (multiplicative decrease)
    abandon rate over the campaign's   credit = 0 for the cooldown period
      budget
    the changepoint detector firing    credit / 2

Additive up, multiplicative down. The asymmetry is the point: an abandoned call
is a compliance event that cannot be undone after the fact, so the system gives
back its allowance in one step and earns it back one call at a time. Recovery
from a bad minute takes tens of seconds, and that is the correct price.

Note what is NOT in that list: model confidence. A pacing engine that becomes
very sure of itself cannot buy credit with certainty, only with a run of ticks
that abandoned nobody. This is what the brief's final question is really
asking -- the credit is a measured budget, not a prediction -- and it is why
the failure mode of the whole predictive path is "degrades to progressive"
rather than "degrades to whatever the model believes".

WHY IT LIVES IN POSTGRES. Several workers dial the same campaign. A credit held
in each worker's memory would grant N times the campaign's budget while every
worker reported itself compliant, and the number a regulator cares about is the
campaign total. One row, locked and updated in one transaction, has the same
single-source-of-truth property as every other piece of state here.

DOUBLE-COUNTING. `updated_at` is the high-water mark for abandons already
charged. The transaction that halves the credit moves the mark forward in the
same write, so two workers ticking either side of one abandoned call halve the
credit once between them, not once each.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from psycopg import AsyncCursor

from smartdialer.core.models import Campaign

# How long the credit stays pinned at zero after the abandon rate breaches the
# campaign's budget. Long enough that recovery does not begin in the same
# minute as the breach -- an AIMD that starts climbing again immediately is an
# oscillator, not a controller.
DEFAULT_COOLDOWN_SECONDS = 60.0

# An absolute ceiling on the credit, on top of the pool-scaled one below. It
# exists so a campaign with thousands of agents cannot accumulate an allowance
# large enough that a single bad tick is a mass event.
MAX_CREDIT_CEILING = 64


class CreditReason:
    """Why the credit is what it is. Lands in the decision log."""

    INCREASE = "INCREASE"
    ABANDON = "ABANDON"
    CHANGEPOINT = "CHANGEPOINT"
    COOLDOWN = "COOLDOWN"
    STEADY = "STEADY"


@dataclass(frozen=True, slots=True)
class BudgetState:
    """The allowance for this tick, and the arithmetic behind it."""

    credit: int
    reason: str
    abandons_charged: int = 0
    cooldown_until: datetime | None = None
    terms: dict[str, Any] = field(default_factory=dict)


def max_credit_for(*, agents_available: int, max_overdial_ratio: float) -> int:
    """The ceiling on the credit, scaled by the size of the agent pool.

    Scaled rather than constant, and that is not a detail. Answers arrive as a
    binomial, so the spread around the expected number grows with the square
    root of the pool while the pool grows linearly: the same over-dial ratio
    that is reckless with 20 agents is conservative with 2,000. A fixed credit
    would therefore be simultaneously too large for the small campaign and too
    small for the large one.

    Tied to the campaign's own hard ratio so the two clamps cannot disagree --
    the credit can never authorise a call that clamp 2 would then refuse, which
    would show up in the logs as a budget that was never spendable.
    """
    headroom = (max_overdial_ratio - 1.0) * max(0, agents_available)
    return max(0, min(MAX_CREDIT_CEILING, int(headroom)))


def next_credit(
    *,
    current: int,
    max_credit: int,
    abandons: int,
    over_budget: bool,
    in_cooldown: bool,
    changepoint: bool,
) -> tuple[int, str]:
    """The AIMD step itself. Pure, so the control law can be tested as maths.

    Order matters: a tick that both breached the budget and saw an abandon is a
    cooldown, not a halving. The strictest applicable rule wins, always.
    """
    if over_budget or in_cooldown:
        return 0, CreditReason.COOLDOWN

    if changepoint or abandons > 0:
        # Multiplicative decrease. floor(), so a credit of 1 goes to 0 rather
        # than lingering at 1 forever -- the floor of this controller is
        # progressive, and it has to be reachable.
        halved = int(math.floor(current * 0.5))
        reason = CreditReason.ABANDON if abandons > 0 else CreditReason.CHANGEPOINT
        return min(halved, max_credit), reason

    if current < max_credit:
        return current + 1, CreditReason.INCREASE
    return min(current, max_credit), CreditReason.STEADY


class AbandonBudget:
    """Reads, applies and writes back the credit for one campaign."""

    def __init__(self, *, cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS) -> None:
        self._cooldown = cooldown_seconds

    async def evaluate(
        self,
        cur: AsyncCursor,
        *,
        campaign: Campaign,
        agents_available: int,
        changepoint: bool,
        now: datetime,
    ) -> BudgetState:
        """Advance the credit by one tick and return the allowance.

        The caller must be inside a transaction: the row is locked, read,
        stepped and written here, so two workers ticking at the same instant
        serialise on this row rather than both reading the pre-tick value.
        """
        row = await self._lock_row(cur, campaign_id=campaign.id, now=now)

        # Abandons that ended after the last time anybody charged the budget.
        # `created_at` is the fallback for the case where a call was abandoned
        # before an end timestamp was recorded -- an abandon with a missing
        # timestamp must still cost the credit.
        await cur.execute(
            """
            SELECT count(*)::int AS n
            FROM calls
            WHERE campaign_id = %(campaign_id)s
              AND state = 'ABANDONED'
              AND coalesce(ended_at, created_at) > %(since)s
            """,
            {"campaign_id": campaign.id, "since": row["updated_at"]},
        )
        abandons = (await cur.fetchone())["n"]

        rate, answered = await self._abandon_rate(cur, campaign_id=campaign.id, now=now)
        budget = float(campaign.abandon_budget_pct) / 100.0
        # A thin sample cannot trigger a cooldown on its own: one abandon out
        # of two answers is 50% and means nothing. The halving above has
        # already responded to that abandon; the cooldown is for a rate that
        # has actually established itself.
        over_budget = answered >= 10 and rate > budget

        cooldown_until = row["cooldown_until"]
        in_cooldown = cooldown_until is not None and now < cooldown_until

        credit, reason = next_credit(
            current=row["overdial_credit"],
            max_credit=max_credit_for(
                agents_available=agents_available,
                max_overdial_ratio=float(campaign.max_overdial_ratio),
            ),
            abandons=abandons,
            over_budget=over_budget,
            in_cooldown=in_cooldown,
            changepoint=changepoint,
        )

        if over_budget:
            cooldown_until = now + timedelta(seconds=self._cooldown)

        await cur.execute(
            """
            UPDATE campaign_safety_state
            SET overdial_credit = %(credit)s,
                cooldown_until  = %(cooldown_until)s,
                updated_at      = %(now)s
            WHERE campaign_id = %(campaign_id)s
            """,
            {
                "credit": credit,
                "cooldown_until": cooldown_until,
                "now": now,
                "campaign_id": campaign.id,
            },
        )

        return BudgetState(
            credit=credit,
            reason=reason,
            abandons_charged=abandons,
            cooldown_until=cooldown_until,
            terms={
                "credit_before": row["overdial_credit"],
                "credit_after": credit,
                "abandons_charged": abandons,
                "abandon_rate": rate,
                "abandon_budget": budget,
                "answers_in_window": answered,
                "over_budget": over_budget,
                "in_cooldown": in_cooldown,
                "changepoint": changepoint,
                "credit_reason": reason,
            },
        )

    async def read_credit(self, cur: AsyncCursor, *, campaign_id: UUID) -> int:
        """The credit as it stands, without advancing it.

        For the decision log and the simulation report only. The authoritative
        read is the locked one in evaluate().
        """
        await cur.execute(
            "SELECT overdial_credit FROM campaign_safety_state WHERE campaign_id = %s",
            (campaign_id,),
        )
        row = await cur.fetchone()
        return int(row["overdial_credit"]) if row else 0

    # -- helpers --------------------------------------------------------

    async def _lock_row(self, cur: AsyncCursor, *, campaign_id: UUID, now: datetime) -> dict:
        """Get this campaign's safety row, creating it if this is the first tick.

        ON CONFLICT DO NOTHING rather than a migration-time backfill, because a
        campaign created at runtime must not have to remember to create its own
        safety state -- forgetting would mean a campaign silently exempt from
        the budget.
        """
        await cur.execute(
            """
            INSERT INTO campaign_safety_state (campaign_id, updated_at)
            VALUES (%(campaign_id)s, %(now)s)
            ON CONFLICT (campaign_id) DO NOTHING
            """,
            {"campaign_id": campaign_id, "now": now},
        )
        await cur.execute(
            "SELECT * FROM campaign_safety_state WHERE campaign_id = %s FOR UPDATE",
            (campaign_id,),
        )
        return await cur.fetchone()

    async def _abandon_rate(
        self, cur: AsyncCursor, *, campaign_id: UUID, now: datetime
    ) -> tuple[float, int]:
        """Abandons as a share of answers, over the last 60 seconds.

        Denominated in ANSWERS, not in calls placed. A borrower who was never
        answered was never in a position to be abandoned, so including them
        would quietly deflate the rate exactly when the answer rate drops --
        which is the moment the number needs to be honest.
        """
        await cur.execute(
            """
            SELECT
              count(*) FILTER (WHERE answered_at IS NOT NULL)::int   AS answered,
              count(*) FILTER (WHERE state = 'ABANDONED')::int       AS abandoned
            FROM calls
            WHERE campaign_id = %(campaign_id)s
              AND created_at > %(now)s::timestamptz - interval '60 seconds'
            """,
            {"campaign_id": campaign_id, "now": now},
        )
        row = await cur.fetchone()
        answered = row["answered"] or 0
        abandoned = row["abandoned"] or 0
        return ((abandoned / answered) if answered else 0.0), answered
