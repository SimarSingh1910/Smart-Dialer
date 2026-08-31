"""The pacing engine. A pure function from a snapshot to a proposal.

THIS MODULE IS THE SAFETY BOUNDARY, and the boundary is structural rather than
defensive. `propose()` takes a dataclass and returns a dataclass. It holds no
database handle, no provider handle and no reference to the allocator, and it
imports nothing from `providers`, `allocator`, `workers` or `psycopg`. It
cannot place a call because it has nothing to place a call with.

That is the whole mechanism, and it is deliberately not a permission check.
Signed tokens or a runtime "am I allowed?" flag would be theatre: whatever
grants permission can be made to grant it, and a reviewer reading that code has
to trace every path to be convinced. A dependency that does not exist needs no
tracing. test_pacing_engine_has_no_forbidden_imports parses this package's AST
and fails the build if anyone adds one.

The brief asks that the predictive algorithm not have a way to switch the
safety mechanism off. Here it cannot even reach it: the engine says what it
would like, the Safety Controller decides what actually happens, and the
controller's clamps are computed from measured state that the engine has no
way to influence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from smartdialer.core.models import CampaignMode

__all__ = [
    "PacingProposal",
    "PacingSnapshot",
    "ProviderHealthSignal",
    "RecentBehaviour",
    "propose",
]


@dataclass(frozen=True, slots=True)
class ProviderHealthSignal:
    """Provider health reduced to plain numbers.

    Deliberately not providers.base.ProviderHealth. Importing that type would
    put `providers` in this module's import graph and break the boundary above
    for no benefit -- the engine needs four floats, not a carrier object. The
    worker does the conversion on the way in.
    """

    name: str = ""
    reachable: bool = True
    failure_rate: float = 0.0
    timeout_rate: float = 0.0
    avg_setup_seconds: float = 0.0
    samples: int = 0


@dataclass(frozen=True, slots=True)
class RecentBehaviour:
    """Rolling campaign counts -- the eighth signal.

    Raw counts rather than derived rates, so the engine can decide what to do
    with a thin sample. Four answers out of five is not a 80% answer rate worth
    acting on, and only the counts let the engine know that.
    """

    window_seconds: float = 60.0
    initiated: int = 0
    answered: int = 0
    connected: int = 0
    abandoned: int = 0
    failed: int = 0

    @property
    def answer_rate(self) -> float:
        return (self.answered / self.initiated) if self.initiated else 0.0

    @property
    def abandon_rate(self) -> float:
        """Per answered call, matching the regulatory definition."""
        return (self.abandoned / self.answered) if self.answered else 0.0


@dataclass(frozen=True, slots=True)
class PacingSnapshot:
    """Everything the engine is allowed to know.

    The eight signals the brief lists are all here, named, populated on every
    tick and written into the decision log -- even the ones progressive mode
    does not read, because a signal that is only wired up when it is first used
    is a signal nobody has ever checked the units of.

        1. agents_available          current agent availability
        2. calls_connected           calls already connected
        3. calls_ringing             calls currently ringing
        4. historical_answer_rate    historical answer rate
        5. call_setup_time_p95       call setup time
        6. avg_call_duration         average call duration
        7. provider_health           provider health
        8. recent_campaign_behaviour recent campaign behaviour

    The per-call detail below the counts (how long each call has been ringing,
    how long each agent has been talking) is what step 7's hazard model needs:
    a call ringing for 3 seconds and one ringing for 18 have very different
    chances of being answered in the next 2, and a single `calls_ringing` count
    cannot tell them apart.
    """

    # --- identity and campaign policy ---------------------------------
    mode: CampaignMode
    taken_at: datetime
    now: datetime
    max_concurrent: int = 1000
    max_overdial_ratio: float = 2.0
    target_shortfall_eps: float = 0.02
    wrap_up_seconds: int = 10

    # --- the eight signals --------------------------------------------
    agents_available: int = 0
    calls_connected: int = 0
    calls_ringing: int = 0
    historical_answer_rate: float = 0.0
    call_setup_time_p95: float = 0.0
    avg_call_duration: float = 0.0
    provider_health: ProviderHealthSignal = field(default_factory=ProviderHealthSignal)
    recent_campaign_behaviour: RecentBehaviour = field(default_factory=RecentBehaviour)

    # --- supporting detail --------------------------------------------
    agents_reserved: int = 0
    agents_dialing: int = 0
    agents_wrap_up: int = 0
    agents_paused: int = 0
    agents_offline: int = 0
    calls_reserved: int = 0
    calls_initiated: int = 0
    calls_answered: int = 0
    # Per-call ages, in seconds. Empty tuples are normal, not missing data.
    ring_seconds: tuple[float, ...] = ()
    talk_seconds: tuple[float, ...] = ()
    wrap_up_remaining: tuple[float, ...] = ()
    # Granted by the safety controller's AIMD budget, reported back to the
    # engine as an input. The engine may spend it; it can never widen it.
    overdial_credit: int = 0

    @property
    def calls_in_flight(self) -> int:
        return (
            self.calls_reserved
            + self.calls_initiated
            + self.calls_ringing
            + self.calls_answered
            + self.calls_connected
        )

    @property
    def age_seconds(self) -> float:
        """How stale this reading is.

        The safety controller forces progressive behaviour past a threshold.
        Predicting from state that is seconds old is how a dialer abandons
        calls while every component reports itself healthy.
        """
        return (self.now - self.taken_at).total_seconds()


@dataclass(frozen=True, slots=True)
class PacingProposal:
    """What the engine would like to do, and every number behind it.

    `terms` is copied verbatim into pacing_decisions.inputs, which is what
    makes "why did it dial 17 and not 10?" answerable months later instead of
    reconstructible at best.
    """

    n: int
    mode: CampaignMode
    reason: str
    terms: dict[str, Any] = field(default_factory=dict)


def propose(snapshot: PacingSnapshot) -> PacingProposal:
    """How many calls the engine would like to start this tick.

    Pure and total: the same snapshot always produces the same proposal, and
    there is no input for which this raises. Determinism is not decoration --
    a decision log is only evidence if the decision can be recomputed from the
    inputs that were recorded next to it.
    """
    if snapshot.mode is CampaignMode.PREDICTIVE:
        return _propose_predictive(snapshot)
    return _propose_progressive(snapshot)


def _propose_progressive(snapshot: PacingSnapshot) -> PacingProposal:
    """One available agent, one call. The deterministic floor.

    There is no cleverness here and there is not supposed to be any. An agent
    who is AVAILABLE is not reserved, not dialling and not talking, so a call
    started for them has somebody waiting for it by construction. The invariant
    "agent-bound calls in flight never exceeds the agent pool" is not enforced
    by this line -- it is enforced by reservation moving the agent out of
    AVAILABLE before the call row is written. This just declines to ask for
    more than exists.
    """
    n = max(0, snapshot.agents_available)
    return PacingProposal(
        n=n,
        mode=snapshot.mode,
        reason="PROGRESSIVE_ONE_TO_ONE",
        terms={
            "agents_available": snapshot.agents_available,
            "calls_in_flight": snapshot.calls_in_flight,
            "historical_answer_rate": snapshot.historical_answer_rate,
            "call_setup_time_p95": snapshot.call_setup_time_p95,
            "avg_call_duration": snapshot.avg_call_duration,
            "provider_failure_rate": snapshot.provider_health.failure_rate,
            "provider_timeout_rate": snapshot.provider_health.timeout_rate,
            "recent_answer_rate": snapshot.recent_campaign_behaviour.answer_rate,
            "recent_abandon_rate": snapshot.recent_campaign_behaviour.abandon_rate,
        },
    )


def _propose_predictive(snapshot: PacingSnapshot) -> PacingProposal:
    """Progressive, plus the over-dial credit the controller has granted.

    The shape of predictive dialling in this design is deliberately this
    modest: it is progressive with a bounded, separately accounted extra. The
    credit is NOT a model output. It is set by the AIMD controller in
    safety/budget.py from observed abandons, so a confident model cannot widen
    it and a wrong one cannot spend more than the measurements allow.

    What is not here yet is the part that decides how much of an available
    credit is worth spending on THIS tick: the Poisson-binomial tail bound over
    the ringing calls' answer hazards and the connected agents' hang-up
    hazards, searching for the largest n with P(answers > free agents) <= eps.
    That lands in step 7 and it will read the ring_seconds and talk_seconds
    already carried on the snapshot.

    Until then predictive spends the credit flat. That is conservative in the
    right direction: the credit is bounded by measured abandons regardless of
    what this function does with it, so the worst case here is a coarser
    decision, never an unsafe one.
    """
    floor = max(0, snapshot.agents_available)
    credit = max(0, snapshot.overdial_credit)
    return PacingProposal(
        n=floor + credit,
        mode=snapshot.mode,
        reason="PREDICTIVE_FLOOR_PLUS_CREDIT",
        terms={
            "agents_available": snapshot.agents_available,
            "progressive_floor": floor,
            "overdial_credit": credit,
            "calls_ringing": snapshot.calls_ringing,
            "calls_connected": snapshot.calls_connected,
            "ring_seconds": list(snapshot.ring_seconds),
            "talk_seconds": list(snapshot.talk_seconds),
            "wrap_up_remaining": list(snapshot.wrap_up_remaining),
            "historical_answer_rate": snapshot.historical_answer_rate,
            "call_setup_time_p95": snapshot.call_setup_time_p95,
            "avg_call_duration": snapshot.avg_call_duration,
            "target_shortfall_eps": snapshot.target_shortfall_eps,
            "provider_failure_rate": snapshot.provider_health.failure_rate,
            "provider_timeout_rate": snapshot.provider_health.timeout_rate,
            "recent_answer_rate": snapshot.recent_campaign_behaviour.answer_rate,
            "recent_abandon_rate": snapshot.recent_campaign_behaviour.abandon_rate,
        },
    )
