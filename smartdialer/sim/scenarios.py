"""The scenarios, as data.

Six of them, and the last three are the failure cases the brief asks to see
demonstrated rather than described. Each is a plain dataclass so a scenario is
a row in a table rather than a function -- adding one is editing this list, and
the runner does not grow a branch for it.

What is deliberately NOT here: worker crash. That case is covered by the
reaper's tests, which kill a worker at four different points in the call
lifecycle and assert what recovery does about each. A scenario that killed a
worker mid-run would produce a less specific version of the same evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class EventKind:
    """Something the simulation does TO the system while it runs."""

    # The borrowers stop answering. Scenario D: the model's central input
    # collapses from 70% to 10% with no warning.
    ANSWER_RATE = "ANSWER_RATE"
    # The carrier stops taking calls entirely, for a fixed period.
    OUTAGE = "OUTAGE"
    # Agents disappear. Scenario F: 40 of 100 log out within a few seconds.
    AGENTS_OFFLINE = "AGENTS_OFFLINE"


@dataclass(frozen=True, slots=True)
class ScenarioEvent:
    at_seconds: float
    kind: str
    value: float = 0.0
    # ANSWER_RATE only: the talk time moves with the answer rate, because in
    # practice they move together -- a campaign reaching fewer people is
    # usually reaching different people.
    talk_median_seconds: float | None = None
    duration_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class Scenario:
    key: str
    description: str
    provider: str  # "fast" | "flaky"
    answer_rate: float
    talk_median_seconds: float
    agents: int = 50
    borrowers: int = 4000
    duration_seconds: float = 300.0
    events: tuple[ScenarioEvent, ...] = field(default_factory=tuple)


SCENARIOS: dict[str, Scenario] = {
    "A": Scenario(
        key="A",
        description="20% answer rate, 120s talk, provider A",
        provider="fast",
        answer_rate=0.20,
        talk_median_seconds=120.0,
    ),
    "B": Scenario(
        key="B",
        description="50% answer rate, 90s talk, provider A",
        provider="fast",
        answer_rate=0.50,
        talk_median_seconds=90.0,
    ),
    "C": Scenario(
        key="C",
        description="70% answer rate, 180s talk, provider A",
        provider="fast",
        answer_rate=0.70,
        talk_median_seconds=180.0,
    ),
    "D": Scenario(
        key="D",
        description="70% collapsing to 10% at t=300s, provider B (duplicates, reordering)",
        provider="flaky",
        answer_rate=0.70,
        talk_median_seconds=90.0,
        duration_seconds=420.0,
        events=(
            ScenarioEvent(
                at_seconds=300.0,
                kind=EventKind.ANSWER_RATE,
                value=0.10,
                talk_median_seconds=200.0,
            ),
        ),
    ),
    "E": Scenario(
        key="E",
        description="50% answer rate, provider outage from t=200s to t=260s",
        provider="fast",
        answer_rate=0.50,
        talk_median_seconds=90.0,
        events=(
            ScenarioEvent(
                at_seconds=200.0, kind=EventKind.OUTAGE, duration_seconds=60.0
            ),
        ),
    ),
    "F": Scenario(
        key="F",
        description="100 agents, 40 of them log out at t=180s",
        provider="fast",
        answer_rate=0.50,
        talk_median_seconds=90.0,
        agents=100,
        borrowers=6000,
        events=(
            ScenarioEvent(at_seconds=180.0, kind=EventKind.AGENTS_OFFLINE, value=40),
        ),
    ),
}

ORDER = ("A", "B", "C", "D", "E", "F")
