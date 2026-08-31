"""The telecom provider interface.

The dialer knows this file and nothing else about how a call is placed. Each
provider module translates its own API and its own webhook shapes into the
types here, so the difference between a well-behaved provider and one that
duplicates, reorders and times out is entirely contained inside that provider's
module -- there is not one conditional anywhere in the dialer that asks which
provider it is talking to.

The part of this interface that carries the most weight is not a method, it is
the exception hierarchy. When a call fails to place, the only question that
matters for correctness is:

    DO WE KNOW whether a stranger's phone is ringing?

ProviderRejected means no, definitively: nothing was placed, and the agent and
borrower can be released immediately. ProviderTimeout means we have no idea,
and releasing anything on that basis risks either dialling the same person
twice or stranding a live call with nobody to answer it. Those two get opposite
handling, so they are separate types rather than one error with a message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable, Protocol, runtime_checkable

from smartdialer.core.models import NormalisedEvent

__all__ = [
    "EventSink",
    "NormalisedEvent",
    "ProviderCallRef",
    "ProviderCallStatus",
    "ProviderError",
    "ProviderHealth",
    "ProviderRejected",
    "ProviderTimeout",
    "ProviderUnavailable",
    "TelecomProvider",
]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base for everything a provider can go wrong with."""


class ProviderRejected(ProviderError):
    """The provider answered, and the answer was no.

    A bad number, a blocked destination, an account limit. We KNOW no call was
    placed, so the agent goes straight back to the pool and the borrower is
    returned without spending an attempt. This is the cheap failure.
    """


class ProviderTimeout(ProviderError):
    """The provider did not answer in time. We do not know what happened.

    This is the expensive failure and the one worth being careful about. The
    call may have been placed and be ringing right now, or the request may
    never have arrived. Callers must NOT release the agent or re-dial the
    borrower on this: recovery looks the call up by its idempotency key and
    finds out, and until it does the agent stays reserved. Holding an agent we
    might not need costs utilisation; releasing one who might be about to be
    bridged to a live borrower costs an abandoned call.
    """


class ProviderUnavailable(ProviderError):
    """The provider is down and refused the request outright.

    Like ProviderRejected in that nothing was placed, but unlike it in what it
    says about the next call: this is the signal the circuit breaker is built
    to react to, so it is a separate type.
    """


# ---------------------------------------------------------------------------
# Values crossing the boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderCallRef:
    """The provider's acknowledgement that it has taken the call.

    Carries the idempotency key back so a caller adopting a call after a crash
    can be certain the ref it is looking at is the one it asked for.
    """

    provider: str
    provider_call_id: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class ProviderCallStatus:
    """What the provider currently believes about one call.

    Used by reconciliation, which is why `live` is a field of its own rather
    than something to infer from `state`. After a crash the only question the
    reaper needs answered is "is there a human on this line right now?", and
    that answer has to survive a provider using a vocabulary we do not model.

    `facts` uses `calls` column names, so a status poll can be applied through
    exactly the same fact-absorbing path as a webhook.
    """

    provider_call_id: str
    live: bool
    state: str
    facts: dict[str, datetime] = field(default_factory=dict)
    ended_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """A provider's recent behaviour, as we observed it.

    One of the eight signals the brief lists for the pacing engine. Rates are
    over a rolling window, so a provider that failed badly an hour ago and is
    fine now reads as fine.
    """

    name: str
    reachable: bool
    failure_rate: float
    timeout_rate: float
    samples: int
    avg_setup_seconds: float

    @property
    def is_healthy(self) -> bool:
        """Deliberately generous: this is a signal, not the circuit breaker.

        The breaker in safety/ decides what to do about a sick provider, and it
        decides on our own measurements. A provider that graded its own health
        and was believed would be a way for an external system to influence the
        safety path, which is exactly what we do not want.
        """
        return self.reachable and self.failure_rate < 0.5


# A sink is where a provider delivers events. In production it is the webhook
# endpoint's ingest function; in the simulation it is wired straight to the
# worker. Providers never write to the database themselves -- they hand over a
# NormalisedEvent and the dialer decides what it means.
EventSink = Callable[[NormalisedEvent], Awaitable[None]]


@runtime_checkable
class TelecomProvider(Protocol):
    """What the dialer requires of a carrier."""

    name: str

    async def place_call(
        self, *, to: str, from_: str, idempotency_key: str
    ) -> ProviderCallRef:
        """Start an outbound call.

        The idempotency key is generated and persisted by us BEFORE this is
        called, and passed in here, so that a crash between the two is
        recoverable via find_by_idempotency_key. A provider that cannot honour
        an idempotency key has to fake one convincingly in its own module.

        Raises ProviderRejected, ProviderTimeout or ProviderUnavailable.
        """
        ...

    async def bridge(self, provider_call_id: str, agent_leg_id: str) -> None:
        """Join the borrower's leg to an agent's leg.

        Separate from place_call because the latency of this operation is the
        engineering component of customer wait -- the dead air a borrower hears
        after saying hello. Keeping it its own call is what lets a provider
        implement it as a conference join (sub-100ms) rather than a fresh call
        setup (1-2s), and what lets the simulation measure the difference.
        """
        ...

    async def get_call_status(self, provider_call_id: str) -> ProviderCallStatus:
        """Ask what is happening on a call. The reconciliation primitive."""
        ...

    async def find_by_idempotency_key(self, key: str) -> ProviderCallRef | None:
        """Find a call we may or may not have successfully placed.

        The crash-recovery primitive, and the reason the idempotency key is
        written down before the provider is called at all.
        """
        ...

    async def hangup(self, provider_call_id: str) -> None:
        """End a call. Used when a borrower answers and there is no agent."""
        ...

    async def health(self) -> ProviderHealth:
        ...
