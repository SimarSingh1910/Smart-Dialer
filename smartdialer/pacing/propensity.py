"""Per-borrower answer probability.

One global answer rate is a summary of a population that is not homogeneous.
Somebody 90 days past due on their third attempt at 9pm does not answer at the
same rate as somebody 15 days past due on their first attempt at 11am, and the
gap between those two is large enough to matter.

WHY THIS IS A CONTROL LEVER AND NOT JUST ACCURACY.

The obvious use is a better estimate of how many of the next N calls will be
answered. The more interesting use is choosing WHO to dial.

The number of connects is the product of dials and answer probability, but the
VARIANCE of the connect count depends on how many dials it took. Reaching a
target of six connects by dialling eight likely-answerers is a much narrower
distribution than reaching it by dialling thirty unlikely ones -- and it is the
width of that distribution, not its mean, that decides whether the answers
overshoot the free agents. So the same connect target, delivered with fewer
dials, means shorter customer wait.

That gives the safety controller a second dial to turn. Near the abandon limit,
dial the low-propensity numbers: they keep the campaign moving with very little
chance of a simultaneous burst of answers. When connects are needed and there
is headroom, dial the high-propensity ones. Neither of those is possible with a
single campaign-wide p.

THE ESTIMATOR IS DELIBERATELY BORING.

A Laplace-smoothed lookup on four coarse features. Not a model, no training,
nothing to serve or version. The brief explicitly permits a statistical
approach, and a lookup table has a property no learned model here would have:
you can read a cell and say exactly why it holds the number it does. A cell
with three observations is shrunk almost entirely to the campaign mean, so a
thin slice cannot make the system aggressive -- the same principle as the
Wilson bound in stats.py, applied to segmentation instead of to time.

Pure. No I/O; the counts are handed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

__all__ = ["PropensityKey", "PropensityTable", "BorrowerFeatures"]

# How many observations a cell needs before it is worth roughly as much as the
# campaign mean. Below this it is dominated by the mean; well above it, the
# cell speaks for itself.
DEFAULT_PRIOR_WEIGHT = 30.0


@dataclass(frozen=True, slots=True)
class BorrowerFeatures:
    """The four features, as they appear on a borrower about to be dialled."""

    hour_of_day: int
    attempt_number: int
    prior_outcome: str | None
    dpd_bucket: str | None

    def key(self) -> "PropensityKey":
        return PropensityKey(
            hour_of_day=self.hour_of_day,
            # Attempts are capped into a bucket. The difference between a first
            # and a second attempt is real; the difference between a seventh
            # and an eighth is noise on a cell that will never have enough data
            # to justify its own number.
            attempt_bucket=min(self.attempt_number, 3),
            prior_outcome=self.prior_outcome or "none",
            dpd_bucket=self.dpd_bucket or "unknown",
        )


@dataclass(frozen=True, slots=True)
class PropensityKey:
    hour_of_day: int
    attempt_bucket: int
    prior_outcome: str
    dpd_bucket: str

    def as_tuple(self) -> tuple:
        return (
            self.hour_of_day,
            self.attempt_bucket,
            self.prior_outcome,
            self.dpd_bucket,
        )


@dataclass(frozen=True, slots=True)
class PropensityTable:
    """Empirical answer rates per cell, shrunk towards the campaign mean."""

    # key tuple -> (answered, dialled)
    cells: Mapping[tuple, tuple[int, int]] = field(default_factory=dict)
    campaign_mean: float = 0.2
    prior_weight: float = DEFAULT_PRIOR_WEIGHT

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping],
        *,
        campaign_mean: float,
        prior_weight: float = DEFAULT_PRIOR_WEIGHT,
    ) -> "PropensityTable":
        """Build from aggregate rows: hour, attempt, outcome, bucket, counts."""
        cells: dict[tuple, tuple[int, int]] = {}
        for row in rows:
            key = PropensityKey(
                hour_of_day=int(row["hour_of_day"]),
                attempt_bucket=min(int(row["attempt_bucket"]), 3),
                prior_outcome=row.get("prior_outcome") or "none",
                dpd_bucket=row.get("dpd_bucket") or "unknown",
            ).as_tuple()
            answered, dialled = cells.get(key, (0, 0))
            cells[key] = (
                answered + int(row["answered"]),
                dialled + int(row["dialled"]),
            )
        return cls(
            cells=cells,
            campaign_mean=campaign_mean,
            prior_weight=prior_weight,
        )

    def probability(self, features: BorrowerFeatures) -> float:
        """This borrower's answer probability.

        Laplace smoothing towards the campaign mean with a fixed pseudo-count.
        A cell with no history returns the mean exactly, which is the correct
        answer to "we have never dialled anybody like this before" -- and it
        degrades continuously, so nothing lurches as a cell fills up.
        """
        answered, dialled = self.cells.get(features.key().as_tuple(), (0, 0))
        weight = self.prior_weight
        value = (answered + self.campaign_mean * weight) / (dialled + weight)
        return max(0.0, min(1.0, value))

    def probabilities(self, candidates: Iterable[BorrowerFeatures]) -> list[float]:
        return [self.probability(features) for features in candidates]

    @property
    def observed_cells(self) -> int:
        return len(self.cells)

    def describe(self, features: BorrowerFeatures) -> dict:
        """The cell behind one probability, for the decision log.

        "Why did you think this person would answer?" should be answerable with
        the counts, not with a number that came from somewhere.
        """
        key = features.key()
        answered, dialled = self.cells.get(key.as_tuple(), (0, 0))
        return {
            "key": key.as_tuple(),
            "answered": answered,
            "dialled": dialled,
            "campaign_mean": self.campaign_mean,
            "probability": self.probability(features),
        }
