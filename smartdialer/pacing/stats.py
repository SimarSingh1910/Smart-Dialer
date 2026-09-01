"""The estimators. All pure, all stdlib.

Three ideas, and the first is the one that does the most work.

WILSON, NOT THE RAW RATE.

Four answers out of five calls is a raw answer rate of 80%. Acting on 80% at
the start of a campaign is how a predictive dialer abandons a room full of
people in its first minute: the estimate is not wrong so much as unsupported,
and over-dialling multiplies the consequences of being unsupported.

The Wilson score lower bound at 95% turns those same four-of-five into about
38%. It is not a pessimistic fudge -- it is the honest statement that the data
is consistent with a rate that low. As calls accumulate the bound rises towards
the observed rate on its own, so the system becomes aggressive only in
proportion to what it actually knows. This is the single most effective defence
against early-campaign over-dialling, and it costs one function.

EWMA, WEIGHTED BY AGE.

Recent behaviour matters more than old behaviour, and "recent" has to mean
elapsed time rather than position in a list -- a hundred calls in the last ten
seconds and a hundred spread over an hour are not the same evidence. So the
weights decay with a half-life in seconds.

CHANGEPOINT, NOT JUST DRIFT.

An EWMA notices a collapsing answer rate eventually, and eventually is too
late: an abandon is not recoverable after the fact. So there is a separate,
blunter check for "the last thirty seconds are not consistent with the model at
all", which the safety controller's budget reacts to immediately. The engine
only reports it -- see the note in engine.py about who is allowed to act.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "wilson_lower_bound",
    "wilson_upper_bound",
    "answer_rate_risk",
    "ewma",
    "changepoint_detected",
    "answer_rate_estimate",
    "MIN_ANSWER_RATE",
    "Z_95",
    "Z_05_ONE_SIDED",
]

# 95% two-sided normal quantile, for the Wilson interval.
Z_95 = 1.959963984540054
# 5th percentile of the standard normal, for the changepoint check.
Z_05_ONE_SIDED = -1.6448536269514722

# A floor under the answer-rate estimate. The rate appears in denominators
# elsewhere, and more importantly an estimate of literally zero would tell the
# engine that dialling is free -- no call will ever be answered, so no call can
# ever overshoot. That is the most dangerous possible conclusion to draw from a
# quiet minute.
MIN_ANSWER_RATE = 0.05


def wilson_lower_bound(successes: int, trials: int, z: float = Z_95) -> float:
    """Lower end of the Wilson score interval for a proportion.

    Chosen over the textbook normal interval because that one is badly behaved
    exactly where this matters: with few trials, or a proportion near 0 or 1,
    it produces bounds outside [0, 1] and intervals far too narrow. Wilson
    stays sensible at n = 1.
    """
    if trials <= 0:
        return 0.0
    if successes <= 0:
        return 0.0

    phat = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = phat + z2 / (2.0 * trials)
    margin = z * math.sqrt(
        (phat * (1.0 - phat) + z2 / (4.0 * trials)) / trials
    )
    lower = (centre - margin) / denominator
    return max(0.0, min(1.0, lower))


def wilson_upper_bound(successes: int, trials: int, z: float = Z_95) -> float:
    """Upper end of the Wilson score interval.

    WHICH END YOU WANT DEPENDS ON WHICH SIDE OF THE FRACTION p IS ON, and
    getting this backwards is a subtle, dangerous mistake.

    In the naive formulation -- "I need k connects, dial k/p calls" -- p is a
    DENOMINATOR. A lower p means a bigger N, so the LOWER bound is the cautious
    choice and a thin sample cannot make the system aggressive.

    In the tail bound this engine actually uses, p is a NUMERATOR: it is the
    probability each new call turns into somebody saying hello, which is to say
    the probability each new call is a risk. A lower p makes over-dialling look
    SAFER, so using the lower bound there would mean a campaign with no history
    concluded that dialling was free -- maximum aggression at the moment of
    maximum ignorance, which is precisely backwards.

    So the risk term uses this bound, and it returns 1.0 for an empty sample:
    knowing nothing, assume every call you place will be answered. That makes a
    cold-started predictive campaign behave like a progressive one and become
    aggressive only in proportion to what it has actually learned.
    """
    if trials <= 0:
        # No evidence at all. The only safe assumption about a risk you cannot
        # bound is that it is certain.
        return 1.0

    phat = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = phat + z2 / (2.0 * trials)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * trials)) / trials)
    upper = (centre + margin) / denominator
    return max(0.0, min(1.0, upper))


def answer_rate_risk(successes: int, trials: int) -> float:
    """The answer rate to use where p multiplies RISK. See wilson_upper_bound."""
    return wilson_upper_bound(successes, trials)


def answer_rate_estimate(
    successes: int, trials: int, *, floor: float = MIN_ANSWER_RATE
) -> float:
    """The answer rate the engine is allowed to believe.

    The Wilson lower bound, floored. Deliberately NOT the observed mean: this
    number decides how many extra calls it is safe to place, and a mean is a
    statement about the past while a lower bound is a statement about how
    little we actually know.
    """
    return max(floor, wilson_lower_bound(successes, trials))


def ewma(
    samples: Sequence[tuple[float, float]],
    *,
    half_life_seconds: float = 60.0,
    default: float = 0.0,
) -> float:
    """Exponentially weighted mean of (age_seconds, value) pairs.

    Weighted by AGE rather than by position, because a hundred calls in the
    last ten seconds and a hundred spread over an hour carry very different
    information about what is happening now, and a positional EWMA cannot tell
    them apart.
    """
    if not samples or half_life_seconds <= 0:
        return default
    decay = math.log(2.0) / half_life_seconds
    total_weight = 0.0
    total = 0.0
    for age, value in samples:
        weight = math.exp(-decay * max(0.0, age))
        total_weight += weight
        total += weight * value
    if total_weight <= 0.0:
        return default
    return total / total_weight


def changepoint_detected(
    *,
    observed: float,
    expected_mean: float,
    expected_variance: float,
    z: float = Z_05_ONE_SIDED,
) -> bool:
    """Has reality fallen out of the bottom of what the model predicted?

    True when the observed count is below the 5th percentile of the model's own
    predictive distribution. Deliberately ONE-SIDED: more answers than expected
    is a pleasant surprise that costs nothing, while fewer means the model is
    wrong about a campaign we are actively over-dialling on.

    The asymmetry runs through the whole design. Adapt downwards fast, upwards
    slowly. Waiting for an EWMA to drift is fine for a metric and useless for a
    safety response, because the abandoned calls have already happened by the
    time a smoothed average notices.
    """
    if expected_mean <= 0.0:
        return False
    sigma = math.sqrt(max(expected_variance, 0.0))
    if sigma < 1e-9:
        # No spread to speak of, so any real shortfall is a changepoint.
        return observed < expected_mean
    return observed < expected_mean + z * sigma


@dataclass(frozen=True, slots=True)
class RateWindow:
    """A rolling count, and what the engine should believe about it."""

    successes: int = 0
    trials: int = 0

    @property
    def observed(self) -> float:
        return (self.successes / self.trials) if self.trials else 0.0

    @property
    def believed(self) -> float:
        """What the engine acts on. See answer_rate_estimate."""
        return answer_rate_estimate(self.successes, self.trials)

    @property
    def risk(self) -> float:
        """The rate to use where p multiplies risk rather than dividing it.

        The upper bound, so that ignorance produces caution. See
        wilson_upper_bound for why this is not the same number as `believed`.
        """
        return answer_rate_risk(self.successes, self.trials)

    @property
    def confidence_gap(self) -> float:
        """How much the estimate is being discounted for thin data.

        Logged on every decision. A large gap is the visible reason a
        well-performing campaign is still being dialled conservatively, and
        without it that looks like the engine ignoring its own numbers.
        """
        return max(0.0, self.observed - self.believed)
