"""Empirical hazard tables: when will this call be answered, when will that
agent be free.

WHY HAZARDS AND NOT A RATE.

The naive predictive dialer treats the ringing pool with a single answer
probability p. That is wrong in a way that produces exactly the failure the
whole design is trying to avoid.

A call that has been ringing for 3 seconds and one that has been ringing for 18
have very different chances of being answered in the next 2 seconds. Ring-to-
answer is roughly log-normal: almost nobody picks up in the first second, most
people who are going to answer do so between 5 and 15 seconds, and a call still
ringing at 25 seconds is probably never going to be answered at all. Collapsing
that into one number means the model cannot tell a pool about to deliver six
simultaneous answers from a pool that has already given up -- and it is
precisely the first case that overshoots the agents and abandons calls.

So what we want is the HAZARD: the probability of the event happening in the
next window GIVEN that it has not happened yet. That is a function of how long
the call has already been ringing, and it is what makes the tail bound in the
engine meaningful.

The same reasoning applies on the agent side. Talk time is also roughly
log-normal, so an agent 20 seconds into a call and one 4 minutes in have
different probabilities of being free within the window.

HOW THE TABLE IS BUILT.

Empirically, from completed calls, in 2-second buckets, using the standard
survival-analysis counts: at_risk[k] is how many calls were still ringing when
they entered bucket k, and events[k] is how many of those were answered during
it. Calls that ended some other way (the borrower gave up, the carrier failed
it) are CENSORED -- they were at risk up to the point they left, and then they
stop counting. Treating them as non-answers for the whole horizon instead would
bias every later bucket towards zero.

Cold start is a log-normal prior, and every bucket is shrunk towards that prior
in proportion to how little data it has. A campaign in its first minute has no
history, and a table that answered "hazard is zero, we have never seen an
answer at 7 seconds" would make the engine wildly over-confident exactly when
it knows least.

Pure. No I/O, no clock, stdlib only -- see the boundary test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "HazardTable",
    "lognormal_cdf",
    "normal_cdf",
    "BUCKET_SECONDS",
    "DEFAULT_RING_MEDIAN",
    "DEFAULT_RING_SIGMA",
    "DEFAULT_TALK_MEDIAN",
    "DEFAULT_TALK_SIGMA",
]

# Two seconds. Small enough that the shape of the ring-to-answer curve is
# visible -- the whole distribution lives between about 3 and 25 seconds -- and
# large enough that individual buckets accumulate samples in minutes rather
# than hours.
BUCKET_SECONDS = 2.0

# Cold-start priors. Ring-to-answer around 9 seconds median matches the 5-25s
# range the brief describes; the talk prior is overridden per campaign, since
# a collections call and a survey call have nothing in common.
DEFAULT_RING_MEDIAN = 9.0
DEFAULT_RING_SIGMA = 0.45
DEFAULT_TALK_MEDIAN = 120.0
DEFAULT_TALK_SIGMA = 0.6

# Beyond this the table stops modelling and the prior takes over entirely.
# Nothing useful is learned about a call that has rung for five minutes.
MAX_BUCKETS = 256


def normal_cdf(x: float) -> float:
    """Standard normal CDF, via erf. No scipy, no table."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def lognormal_cdf(t: float, median: float, sigma: float) -> float:
    """P(T <= t) for a log-normal with the given median and log-scale sigma."""
    if t <= 0.0:
        return 0.0
    if median <= 0.0 or sigma <= 0.0:
        return 1.0
    return normal_cdf((math.log(t) - math.log(median)) / sigma)


@dataclass(frozen=True, slots=True)
class HazardTable:
    """Discrete hazards over fixed-width buckets, shrunk towards a prior."""

    bucket_seconds: float = BUCKET_SECONDS
    # Survival-analysis counts, per bucket.
    at_risk: tuple[int, ...] = ()
    events: tuple[int, ...] = ()
    # The cold-start distribution, also used wherever a bucket is thin.
    prior_median: float = DEFAULT_RING_MEDIAN
    prior_sigma: float = DEFAULT_RING_SIGMA
    # Pseudo-count controlling how fast the empirical data displaces the prior.
    # A bucket with this many observations sits halfway between the two.
    min_samples: int = 20

    # -- construction ---------------------------------------------------

    @classmethod
    def from_observations(
        cls,
        *,
        event_times: Sequence[float],
        censored_times: Sequence[float] = (),
        prior_median: float = DEFAULT_RING_MEDIAN,
        prior_sigma: float = DEFAULT_RING_SIGMA,
        bucket_seconds: float = BUCKET_SECONDS,
        min_samples: int = 20,
    ) -> "HazardTable":
        """Build a table from durations.

        `event_times` are durations that ended in the event we care about (the
        call was answered, the agent hung up). `censored_times` are durations
        that ended some other way -- they were genuinely at risk up to that
        point and then left the population.

        The censoring is the part that matters. A call the borrower abandoned
        at 4 seconds tells us nothing about whether it would have been answered
        at 12, and counting it as a non-answer at 12 would drag every later
        bucket towards zero and make the model believe long-ringing calls never
        answer -- which is exactly when the engine would then over-dial.
        """
        if bucket_seconds <= 0:
            raise ValueError("bucket_seconds must be positive")

        longest = 0.0
        for value in list(event_times) + list(censored_times):
            longest = max(longest, value)
        n_buckets = min(MAX_BUCKETS, int(longest / bucket_seconds) + 1)
        if n_buckets <= 0:
            n_buckets = 1

        at_risk = [0] * n_buckets
        events = [0] * n_buckets

        def bucket_of(value: float) -> int:
            return min(n_buckets - 1, max(0, int(value / bucket_seconds)))

        for value in event_times:
            index = bucket_of(value)
            events[index] += 1
            # At risk in this bucket and in every earlier one it survived.
            for k in range(index + 1):
                at_risk[k] += 1
        for value in censored_times:
            index = bucket_of(value)
            for k in range(index + 1):
                at_risk[k] += 1

        return cls(
            bucket_seconds=bucket_seconds,
            at_risk=tuple(at_risk),
            events=tuple(events),
            prior_median=prior_median,
            prior_sigma=prior_sigma,
            min_samples=min_samples,
        )

    @classmethod
    def prior_only(
        cls,
        *,
        prior_median: float,
        prior_sigma: float,
        bucket_seconds: float = BUCKET_SECONDS,
    ) -> "HazardTable":
        """A table with no data at all. The honest state at campaign start."""
        return cls(
            bucket_seconds=bucket_seconds,
            prior_median=prior_median,
            prior_sigma=prior_sigma,
        )

    # -- the hazards ----------------------------------------------------

    @property
    def samples(self) -> int:
        return self.at_risk[0] if self.at_risk else 0

    def prior_bucket_hazard(self, index: int) -> float:
        """The prior's own hazard for one bucket.

        P(event in [lo, hi) | survived to lo) for the log-normal. Approaches 1
        in the far tail, which is correct: a call still ringing well past the
        distribution has essentially no chance of surviving another bucket
        without resolving one way or the other.
        """
        low = index * self.bucket_seconds
        high = low + self.bucket_seconds
        survived = 1.0 - lognormal_cdf(low, self.prior_median, self.prior_sigma)
        if survived <= 1e-9:
            return 1.0
        died = lognormal_cdf(high, self.prior_median, self.prior_sigma) - lognormal_cdf(
            low, self.prior_median, self.prior_sigma
        )
        return _clamp(died / survived)

    def bucket_hazard(self, index: int) -> float:
        """The blended hazard for one bucket.

        Shrinkage, not a hard threshold: a bucket with two observations is
        dominated by the prior, one with two hundred is dominated by the data,
        and nothing jumps discontinuously as a bucket crosses a sample count.
        """
        prior = self.prior_bucket_hazard(index)
        if index >= len(self.at_risk):
            return prior
        n = self.at_risk[index]
        if n <= 0:
            return prior
        observed = self.events[index]
        weight = float(self.min_samples)
        return _clamp((observed + prior * weight) / (n + weight))

    def hazard(self, elapsed: float, window: float) -> float:
        """P(event within `window` | it has not happened in `elapsed` so far).

        Composed across buckets as one minus the product of the per-bucket
        survivals, with the final partial bucket taken to a fractional power so
        that the answer moves smoothly with the window rather than jumping
        every two seconds. A window is rarely a whole number of buckets, and a
        discontinuity here would show up directly in the number of calls the
        engine decides to place.
        """
        if window <= 0.0:
            return 0.0
        elapsed = max(0.0, elapsed)

        survival = 1.0
        index = int(elapsed / self.bucket_seconds)
        # How much of the starting bucket is still ahead of us.
        offset = elapsed - index * self.bucket_seconds
        remaining_in_bucket = self.bucket_seconds - offset
        remaining = window

        guard = 0
        while remaining > 1e-9 and guard < MAX_BUCKETS:
            span = min(remaining, remaining_in_bucket)
            hazard = self.bucket_hazard(index)
            survival *= (1.0 - hazard) ** (span / self.bucket_seconds)
            remaining -= span
            index += 1
            remaining_in_bucket = self.bucket_seconds
            guard += 1

        return _clamp(1.0 - survival)


def _clamp(value: float) -> float:
    """Keep probabilities inside [0, 1].

    Not defensive noise. These feed a variance term p(1-p), and a value even
    slightly outside the range makes that negative, which makes a standard
    deviation imaginary and takes the whole tail bound with it.
    """
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
