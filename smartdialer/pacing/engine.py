"""The pacing engine. A pure function from a snapshot to a proposal.

THIS MODULE IS THE SAFETY BOUNDARY, and the boundary is structural rather than
defensive. `propose()` takes a dataclass and returns a dataclass. It holds no
database handle, no provider handle and no reference to the allocator, and its
entire transitive import closure is an allowlist of stdlib plus two pure `core`
modules. It cannot place a call because it has nothing to place a call with.

That is deliberately not a permission check. Signed tokens or a runtime "am I
allowed?" flag would be theatre: whatever grants permission can be made to
grant it, and a reviewer has to trace every path to be convinced. A dependency
that does not exist needs no tracing, and
tests/test_safety_boundary.py walks the closure to keep it that way.

THE DECISION, AND WHY IT IS A QUANTILE.

The naive predictive dialer estimates an answer rate p, divides the agent
deficit by p, and dials that many. That optimises a MEAN. But customer wait and
abandonment are TAIL events -- a borrower waits only when the number of
simultaneous answers happens to exceed the number of free agents -- and the
mean of a distribution says very little about its tail.

So the question is not "how many calls will be answered?" but:

    what is the largest N such that P(answers > free agents) <= epsilon?

Both sides are sums of independent Bernoulli trials with DIFFERENT
probabilities: each ringing call has its own hazard of answering in the window,
each connected agent its own hazard of hanging up. That is a Poisson-binomial
distribution, whose mean and variance are cheap sums, and whose exact
distribution is a short dynamic program when the pool is small.

WHY THE VARIANCE MATTERS MORE THAN THE ESTIMATE.

Dialling N calls at probability p gives mean Np and standard deviation
sqrt(Np(1-p)), so the coefficient of variation falls as 1/sqrt(N). The
consequence is that the same over-dial RATIO which is reckless with 20 agents
is conservative with 2,000. Any pacing constant not scaled by pool size is
simply wrong, and this is also why splitting a campaign across shards costs
utilisation: fragmenting the pool raises the relative variance of every piece.

Nothing here reads the abandon budget or the circuit breaker, and nothing here
clamps itself. The engine proposes; the Safety Controller disposes. Those
ceilings are computed from measured outcomes the engine cannot influence, which
is the entire point of putting them somewhere the engine cannot reach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from smartdialer.core.models import CampaignMode
from smartdialer.pacing.hazard import (
    normal_cdf,
    DEFAULT_RING_MEDIAN,
    DEFAULT_RING_SIGMA,
    DEFAULT_TALK_MEDIAN,
    DEFAULT_TALK_SIGMA,
    HazardTable,
)
from smartdialer.pacing.stats import (
    MIN_ANSWER_RATE,
    changepoint_detected,
    RateWindow,
)

__all__ = [
    "PacingProposal",
    "PacingSnapshot",
    "ProviderHealthSignal",
    "RecentBehaviour",
    "propose",
]

# The forecast window. Everything is asked "within the next Delta seconds", and
# Delta is how long a call placed now takes to become somebody saying hello --
# so p95 setup time, floored so a suspiciously fast carrier or an empty history
# cannot collapse the window to nothing.
MIN_WINDOW_SECONDS = 5.0

# Below this spread the normal approximation is untrustworthy and the exact
# Poisson-binomial is computed instead. See _shortfall_probability.
EXACT_SIGMA_THRESHOLD = 2.0

# A ceiling on the binary search, so a pathological snapshot cannot make the
# engine spend real time exploring absurd numbers of calls.
MAX_SEARCH = 512


@dataclass(frozen=True, slots=True)
class ProviderHealthSignal:
    """Provider health reduced to plain numbers.

    Deliberately not providers.base.ProviderHealth. Importing that type would
    put `providers` in this module's import closure and break the boundary for
    no benefit -- the engine needs four floats, not a carrier object.
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

    Raw counts rather than derived rates, so the engine can tell that "80%"
    came from four calls out of five. That distinction is what the Wilson bound
    in stats.py exists to act on.
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

    @property
    def rate_window(self) -> RateWindow:
        return RateWindow(successes=self.answered, trials=self.initiated)


@dataclass(frozen=True, slots=True)
class PacingSnapshot:
    """Everything the engine is allowed to know.

    The eight signals the brief lists are all here, named, populated on every
    tick and written to the decision log even where the current logic barely
    uses them -- a signal that is only wired up when it is first needed is a
    signal nobody has ever checked the units of.

        1. agents_available            current agent availability
        2. calls_connected             calls already connected
        3. calls_ringing               calls currently ringing
        4. historical_answer_rate      historical answer rate
        5. call_setup_time_p95         call setup time
        6. avg_call_duration           average call duration
        7. provider_health             provider health
        8. recent_campaign_behaviour   recent campaign behaviour

    Note that `calls_ringing` is a tuple of PER-CALL RING DURATIONS, not a
    count. That is the whole point of the hazard model: a count cannot
    distinguish six calls that just started ringing from six that are about to
    be given up on, and those two pools have completely different chances of
    delivering simultaneous answers.
    """

    # --- identity and campaign policy ---------------------------------
    mode: CampaignMode
    snapshot_taken_at: datetime
    now: datetime
    max_concurrent: int = 1000
    max_overdial_ratio: float = 2.0
    target_shortfall_eps: float = 0.02
    wrap_up_seconds: int = 10

    # --- the eight signals --------------------------------------------
    agents_available: int = 0
    calls_connected: int = 0
    # Per-call ring durations, in seconds. Empty is normal, not missing.
    calls_ringing: tuple[float, ...] = ()
    historical_answer_rate: float = 0.0
    call_setup_time_p95: float = 0.0
    avg_call_duration: float = 0.0
    provider_health: ProviderHealthSignal = field(default_factory=ProviderHealthSignal)
    recent_campaign_behaviour: RecentBehaviour = field(default_factory=RecentBehaviour)

    # --- detail the hazard maths needs --------------------------------
    # How long each CONNECTED agent has been talking.
    talk_seconds: tuple[float, ...] = ()
    # How long each WRAP_UP agent has left. Deterministic: these timers were
    # set when a call ended and nothing external can move them, which is why
    # they contribute to the forecast with no variance at all.
    wrap_up_remaining: tuple[float, ...] = ()
    # Median setup time, used to work out how much of the window a call placed
    # now would actually spend ringing.
    call_setup_time_p50: float = 0.0
    # Answer probabilities for the borrowers most likely to be dialled next.
    # Heterogeneous on purpose -- see propensity.py on why this is a control
    # lever and not just an accuracy improvement.
    candidate_propensities: tuple[float, ...] = ()

    # --- learned distributions ----------------------------------------
    ring_hazard: HazardTable = field(
        default_factory=lambda: HazardTable.prior_only(
            prior_median=DEFAULT_RING_MEDIAN, prior_sigma=DEFAULT_RING_SIGMA
        )
    )
    talk_hazard: HazardTable = field(
        default_factory=lambda: HazardTable.prior_only(
            prior_median=DEFAULT_TALK_MEDIAN, prior_sigma=DEFAULT_TALK_SIGMA
        )
    )

    # --- supporting counts --------------------------------------------
    agents_reserved: int = 0
    agents_dialing: int = 0
    agents_wrap_up: int = 0
    agents_paused: int = 0
    agents_offline: int = 0
    calls_reserved: int = 0
    calls_initiated: int = 0
    calls_answered: int = 0
    # Granted by the safety controller's AIMD budget. Carried so it appears in
    # the decision log, and deliberately NOT read by the engine: the credit is
    # a ceiling the controller applies, not an allowance the engine spends.
    overdial_credit: int = 0
    # Answers actually observed in the last 30s, and what the model predicted
    # for that same period. The changepoint check compares them.
    observed_answers_30s: float = 0.0
    predicted_answers_30s: float = 0.0
    predicted_answers_30s_variance: float = 0.0

    @property
    def n_ringing(self) -> int:
        return len(self.calls_ringing)

    @property
    def calls_in_flight(self) -> int:
        return (
            self.calls_reserved
            + self.calls_initiated
            + self.n_ringing
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
        return (self.now - self.snapshot_taken_at).total_seconds()


@dataclass(frozen=True, slots=True)
class PacingProposal:
    """What the engine wants, and every number behind it.

    Carries its own explanation so the decision log can reconstruct the
    sentence "N=18 gives P(shortfall) = 2.3% > eps; N=17 gives 1.9% <= eps;
    proposed 17" months later, rather than leaving somebody to guess.
    """

    n: int
    mode: CampaignMode
    reason: str
    # The forecast, both sides.
    mu_A: float = 0.0
    sigma_A: float = 0.0
    mu_G: float = 0.0
    sigma_G: float = 0.0
    p_hat: float = 0.0
    epsilon: float = 0.02
    window_seconds: float = MIN_WINDOW_SECONDS
    changepoint_detected: bool = False
    used_exact_dp: bool = False
    # (N, P(shortfall)) for every candidate the search evaluated, in order.
    search_trace: tuple[tuple[int, float], ...] = ()
    terms: dict[str, Any] = field(default_factory=dict)

    def explain(self) -> str:
        """One human sentence. Used by the simulation report."""
        if self.mode is CampaignMode.PROGRESSIVE:
            return f"progressive: {self.n} free agents, {self.n} calls"
        trace = ", ".join(f"N={n} -> {p:.1%}" for n, p in self.search_trace[-4:])
        return (
            f"mu_G={self.mu_G:.1f} (sd {self.sigma_G:.1f}), "
            f"mu_A={self.mu_A:.1f} (sd {self.sigma_A:.1f}), "
            f"p_hat={self.p_hat:.2f}, eps={self.epsilon:.1%}; "
            f"{trace}; proposed {self.n}"
        )


def propose(snapshot: PacingSnapshot) -> PacingProposal:
    """How many calls the engine would like to start this tick.

    Pure and total: the same snapshot always produces the same proposal, and
    there is no input for which this raises. Determinism is not decoration -- a
    decision log is only evidence if the decision can be recomputed from the
    inputs recorded beside it.
    """
    if snapshot.mode is CampaignMode.PREDICTIVE:
        return _propose_predictive(snapshot)
    return _propose_progressive(snapshot)


# ---------------------------------------------------------------------------
# Progressive
# ---------------------------------------------------------------------------


def _propose_progressive(snapshot: PacingSnapshot) -> PacingProposal:
    """One available agent, one call. The deterministic floor.

    There is no cleverness here and there is not supposed to be any. An agent
    who is AVAILABLE is not reserved, not dialling and not talking, so a call
    started for them has somebody waiting for it by construction. The invariant
    is not enforced by this line -- it is enforced by reservation moving the
    agent out of AVAILABLE before the call row is written. This just declines
    to ask for more than exists.
    """
    n = max(0, snapshot.agents_available)
    return PacingProposal(
        n=n,
        mode=snapshot.mode,
        reason="PROGRESSIVE_ONE_TO_ONE",
        p_hat=snapshot.recent_campaign_behaviour.rate_window.believed,
        epsilon=snapshot.target_shortfall_eps,
        mu_G=float(n),
        terms=_common_terms(snapshot),
    )


# ---------------------------------------------------------------------------
# Predictive: the Poisson-binomial tail
# ---------------------------------------------------------------------------


def _propose_predictive(snapshot: PacingSnapshot) -> PacingProposal:
    window = max(MIN_WINDOW_SECONDS, snapshot.call_setup_time_p95)
    epsilon = _clamp_epsilon(snapshot.target_shortfall_eps)

    # --- the agent side ------------------------------------------------
    #
    # Free agents are certain, so they carry no variance. Wrap-up agents whose
    # timer expires inside the window are also certain -- the timer was set
    # when their call ended and nothing external can move it, which makes them
    # free capacity that costs nothing to predict. Only the connected agents
    # are a gamble, and each has their own probability of hanging up based on
    # how long they have already been talking.
    free_probabilities = [
        snapshot.talk_hazard.hazard(elapsed, window)
        for elapsed in snapshot.talk_seconds
    ]
    wrap_up_freeing = sum(1 for left in snapshot.wrap_up_remaining if left <= window)
    mu_G = float(snapshot.agents_available) + wrap_up_freeing + sum(free_probabilities)
    var_G = sum(q * (1.0 - q) for q in free_probabilities)

    # --- the answers already in flight ---------------------------------
    ring_hazards = [
        snapshot.ring_hazard.hazard(elapsed, window)
        for elapsed in snapshot.calls_ringing
    ]
    mu_A0 = sum(ring_hazards)
    var_A0 = sum(h * (1.0 - h) for h in ring_hazards)

    # --- what one NEW call contributes ---------------------------------
    #
    # Two different estimates of the same rate, and the difference matters.
    # `believed` is the Wilson LOWER bound and is what gets reported: it is the
    # right number for "how many dials do I need for k connects". `risk` is the
    # UPPER bound and is what the tail bound consumes, because here p is the
    # probability a new call becomes a risk. Using the lower bound for that
    # would make a campaign with no history conclude that dialling is free.
    # See wilson_upper_bound in stats.py.
    rate_window = snapshot.recent_campaign_behaviour.rate_window
    p_hat = rate_window.believed
    p_risk = rate_window.risk
    p_new = _new_call_answer_probability(snapshot, window=window, p_hat=p_risk)

    # Heterogeneous candidates, highest first: the engine assumes the allocator
    # dials the most likely answerers first, so adding the Nth call adds the
    # Nth-best probability rather than an average. Being wrong about that in
    # this direction is safe -- it over-states the risk of each extra call.
    candidates = sorted(snapshot.candidate_propensities, reverse=True)
    scale = p_new / p_risk if p_risk > 0 else 0.0
    per_call = [min(1.0, p * scale) for p in candidates]

    def probability_of_call(index: int) -> float:
        return per_call[index] if index < len(per_call) else p_new

    # --- the search ----------------------------------------------------
    #
    # P(shortfall) is monotonically increasing in N -- every extra call can
    # only add answers, never agents -- so binary search finds the largest N
    # that still satisfies the bound.
    trace: list[tuple[int, float]] = []
    used_exact = [False]

    def shortfall(n: int) -> float:
        probability, exact = _shortfall_probability(
            n=n,
            mu_A0=mu_A0,
            var_A0=var_A0,
            mu_G=mu_G,
            var_G=var_G,
            ring_hazards=ring_hazards,
            free_probabilities=free_probabilities,
            deterministic_agents=float(snapshot.agents_available) + wrap_up_freeing,
            probability_of_call=probability_of_call,
        )
        used_exact[0] = used_exact[0] or exact
        trace.append((n, probability))
        return probability

    upper = min(
        MAX_SEARCH,
        max(1, int(snapshot.max_overdial_ratio * max(1, snapshot.agents_available)) + 8),
    )

    best = 0
    if shortfall(0) <= epsilon:
        low, high = 0, upper
        while low <= high:
            middle = (low + high) // 2
            if shortfall(middle) <= epsilon:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
    # If even zero new calls breaches the bound, the pool is already
    # over-committed: the honest proposal is nothing, and the progressive floor
    # is restored by the safety controller rather than smuggled in here.

    # Make the trace bracket the answer explicitly. Binary search usually
    # evaluates the boundary anyway, but "usually" is not good enough for the
    # one line that has to justify the decision.
    evaluated = {n for n, _ in trace}
    if best not in evaluated:
        shortfall(best)
    if best + 1 <= upper and best + 1 not in evaluated:
        shortfall(best + 1)

    sigma_A = math.sqrt(max(0.0, var_A0 + _new_call_variance(best, probability_of_call)))
    mu_A = mu_A0 + sum(probability_of_call(i) for i in range(best))

    changepoint = changepoint_detected(
        observed=snapshot.observed_answers_30s,
        expected_mean=snapshot.predicted_answers_30s,
        expected_variance=snapshot.predicted_answers_30s_variance,
    )

    terms = _common_terms(snapshot)
    terms.update(
        {
            "window_seconds": window,
            "wrap_up_freeing": wrap_up_freeing,
            "ring_hazards": [round(h, 4) for h in ring_hazards],
            "free_probabilities": [round(q, 4) for q in free_probabilities],
            "p_new_call": p_new,
            "p_hat_lower": p_hat,
            "p_hat_risk": p_risk,
            "candidate_propensities": [round(p, 4) for p in per_call[:20]],
            "mu_A0": mu_A0,
            "var_A0": var_A0,
            "mu_G": mu_G,
            "var_G": var_G,
            "observed_answers_30s": snapshot.observed_answers_30s,
            "predicted_answers_30s": snapshot.predicted_answers_30s,
            "confidence_gap": snapshot.recent_campaign_behaviour.rate_window.confidence_gap,
        }
    )

    return PacingProposal(
        n=best,
        mode=snapshot.mode,
        reason="PREDICTIVE_TAIL_BOUND",
        mu_A=mu_A,
        sigma_A=sigma_A,
        mu_G=mu_G,
        sigma_G=math.sqrt(max(0.0, var_G)),
        p_hat=p_hat,
        epsilon=epsilon,
        window_seconds=window,
        changepoint_detected=changepoint,
        used_exact_dp=used_exact[0],
        search_trace=tuple(trace),
        terms=terms,
    )


def _new_call_answer_probability(
    snapshot: PacingSnapshot, *, window: float, p_hat: float
) -> float:
    """P(a call placed now is answered inside the window).

    Much smaller than the answer rate, and that gap is what makes over-dialling
    possible at all. A call placed now spends most of the window in setup --
    the window IS the p95 setup time -- so it only starts ringing near the end
    and has very little chance of landing an answer inside it. The calls that
    threaten the agent pool in the next few seconds are the ones already
    ringing, not the ones about to be dialled.

    Getting this wrong in the optimistic direction would be the classic
    predictive-dialer failure: assume every new call might answer immediately,
    conclude that almost no over-dialling is safe, and never beat progressive.
    Getting it wrong pessimistically just costs utilisation.
    """
    # The chance the call is even connected before the window closes. The
    # window IS the p95 of setup time, so by construction about 95% make it.
    connected_in_time = 0.95

    # Deliberately NOT multiplied by the probability that the answer also lands
    # inside the remaining sliver of the window. That refinement is defensible
    # and it is also how this model would talk itself into recklessness: with
    # the window equal to setup time, a new call barely starts ringing before
    # the window closes, so its in-window answer probability rounds to nothing
    # and the engine concludes that any number of new calls is free. It is not
    # free -- those calls land moments later, and the bound is supposed to be
    # about the exposure they create, not about a horizon chosen to exclude it.
    #
    # Charging each new call its full answer probability keeps the bound
    # conservative in the direction that costs utilisation rather than the one
    # that costs abandoned calls.
    return max(0.0, min(1.0, p_hat * connected_in_time))


def _new_call_variance(n: int, probability_of_call) -> float:
    return sum(
        probability_of_call(i) * (1.0 - probability_of_call(i)) for i in range(n)
    )


def _shortfall_probability(
    *,
    n: int,
    mu_A0: float,
    var_A0: float,
    mu_G: float,
    var_G: float,
    ring_hazards: Sequence[float],
    free_probabilities: Sequence[float],
    deterministic_agents: float,
    probability_of_call,
) -> tuple[float, bool]:
    """P(answers > free agents) if we add `n` calls. Returns (p, used_exact).

    Two regimes, and the switch between them is the interesting part.

    When the combined spread is comfortable, the normal approximation with a
    continuity correction is accurate and costs nothing.

    When it is small -- few ringing calls, few connected agents, a small
    campaign -- the normal approximation is at its worst precisely where the
    consequences are at their highest. A ten-agent campaign has no room to
    absorb a mistake, and approximating a handful of Bernoullis by a smooth
    curve can be off by several percentage points on a bound whose entire
    budget is two. So below the threshold the exact Poisson-binomial is
    computed by dynamic programming. The pool is small by definition in exactly
    that case, which is what makes the exact route affordable.
    """
    new_probabilities = [probability_of_call(i) for i in range(n)]
    mu_A = mu_A0 + sum(new_probabilities)
    var_A = var_A0 + sum(p * (1.0 - p) for p in new_probabilities)

    mu_D = mu_A - mu_G
    sigma_D = math.sqrt(max(0.0, var_A + var_G))

    if sigma_D >= EXACT_SIGMA_THRESHOLD:
        # Continuity correction: the shortfall is an integer count, so
        # "greater than zero" means "at least one", and the half-unit shift is
        # what keeps a discrete quantity from being systematically
        # mis-estimated by a continuous curve.
        return 1.0 - normal_cdf((0.5 - mu_D) / sigma_D), False

    answers = list(ring_hazards) + new_probabilities
    return (
        _exact_shortfall(
            answer_probabilities=answers,
            agent_probabilities=list(free_probabilities),
            deterministic_agents=deterministic_agents,
        ),
        True,
    )


def _poisson_binomial_pmf(probabilities: Sequence[float]) -> list[float]:
    """Exact distribution of a sum of independent, non-identical Bernoullis.

    The textbook O(n^2) convolution. Affordable here because it is only ever
    reached when the pool is small, which is exactly when it is needed.
    """
    pmf = [1.0]
    for p in probabilities:
        p = max(0.0, min(1.0, p))
        updated = [0.0] * (len(pmf) + 1)
        for k, weight in enumerate(pmf):
            if weight == 0.0:
                continue
            updated[k] += weight * (1.0 - p)
            updated[k + 1] += weight * p
        pmf = updated
    return pmf


def _exact_shortfall(
    *,
    answer_probabilities: Sequence[float],
    agent_probabilities: Sequence[float],
    deterministic_agents: float,
) -> float:
    """P(A > G) exactly, by convolving both sides.

    The deterministic agents -- currently free, plus wrap-up timers expiring
    inside the window -- are a constant offset rather than part of the
    convolution. They have no variance, and pretending otherwise would inflate
    the spread and make the engine needlessly timid on precisely the campaigns
    with the least room to spare.
    """
    answers = _poisson_binomial_pmf(answer_probabilities)
    agents = _poisson_binomial_pmf(agent_probabilities)
    base = int(math.floor(deterministic_agents))

    # Cumulative answers, so the inner loop is a lookup rather than a sum.
    cumulative: list[float] = []
    running = 0.0
    for weight in answers:
        running += weight
        cumulative.append(running)

    def p_answers_at_most(k: int) -> float:
        if k < 0:
            return 0.0
        if k >= len(cumulative):
            return 1.0
        return cumulative[k]

    total = 0.0
    for extra, weight in enumerate(agents):
        if weight == 0.0:
            continue
        available = base + extra
        total += weight * (1.0 - p_answers_at_most(available))
    return max(0.0, min(1.0, total))


def _clamp_epsilon(value: float) -> float:
    """Keep the target inside a sane range.

    An epsilon of zero would make the search propose nothing forever; one of
    one would make it propose everything. Both are configuration mistakes
    rather than intentions, and the campaign row's CHECK constraint already
    rejects them -- this is the belt to that pair of braces.
    """
    if value <= 0.0:
        return 0.001
    if value >= 1.0:
        return 0.999
    return value


def _common_terms(snapshot: PacingSnapshot) -> dict[str, Any]:
    """The inputs that go into every decision row, whatever the mode.

    All eight signals appear here, including the ones progressive mode never
    reads, so the log is comparable across modes and across the two halves of
    a simulation run.
    """
    return {
        "agents_available": snapshot.agents_available,
        "calls_connected": snapshot.calls_connected,
        "calls_ringing": [round(t, 2) for t in snapshot.calls_ringing],
        "n_ringing": snapshot.n_ringing,
        "historical_answer_rate": snapshot.historical_answer_rate,
        "call_setup_time_p95": snapshot.call_setup_time_p95,
        "call_setup_time_p50": snapshot.call_setup_time_p50,
        "avg_call_duration": snapshot.avg_call_duration,
        "provider_health": {
            "name": snapshot.provider_health.name,
            "reachable": snapshot.provider_health.reachable,
            "failure_rate": snapshot.provider_health.failure_rate,
            "timeout_rate": snapshot.provider_health.timeout_rate,
            "avg_setup_seconds": snapshot.provider_health.avg_setup_seconds,
        },
        "recent_campaign_behaviour": {
            "window_seconds": snapshot.recent_campaign_behaviour.window_seconds,
            "initiated": snapshot.recent_campaign_behaviour.initiated,
            "answered": snapshot.recent_campaign_behaviour.answered,
            "connected": snapshot.recent_campaign_behaviour.connected,
            "abandoned": snapshot.recent_campaign_behaviour.abandoned,
            "failed": snapshot.recent_campaign_behaviour.failed,
            "answer_rate": snapshot.recent_campaign_behaviour.answer_rate,
            "abandon_rate": snapshot.recent_campaign_behaviour.abandon_rate,
        },
        "talk_seconds": [round(t, 2) for t in snapshot.talk_seconds],
        "wrap_up_remaining": [round(t, 2) for t in snapshot.wrap_up_remaining],
        "overdial_credit": snapshot.overdial_credit,
        "ring_hazard_samples": snapshot.ring_hazard.samples,
        "talk_hazard_samples": snapshot.talk_hazard.samples,
        "min_answer_rate": MIN_ANSWER_RATE,
        "snapshot_age_seconds": snapshot.age_seconds,
    }
