"""Provider B: slow, unreliable, and dishonest about ordering.

Everything the brief asks for in a badly behaved carrier:

  * setup between roughly 3 and 12 seconds, with a fat tail
  * a 12% rejection rate
  * 8% of requests never answered at all, so our own client timeout fires --
    and half of those calls were placed anyway, which is the case the
    idempotency key exists for
  * 15% of events delivered more than once
  * 20% of events held back long enough to arrive after later ones
  * an occasional event delayed by half a minute
  * bridging that takes 300-900ms, which the borrower hears as dead air

The point of this provider is not that it is annoying. It is that the dialer
contains no code that knows about any of it. Every one of these behaviours is
absorbed by mechanisms that were there anyway: the unique index on
(provider, provider_event_id) for the duplicates, rank monotonicity for the
reordering, the idempotency key for the timeouts, and the circuit breaker for
the moment it gives up entirely.
"""

from __future__ import annotations

from smartdialer.core.clock import Clock
from smartdialer.providers.base import EventSink
from smartdialer.providers.simulated import ProviderProfile, SimulatedProvider

FLAKY_PROFILE = ProviderProfile(
    name="mock_flaky",
    # Median about 5s, with a tail reaching past 12s.
    setup_median_seconds=5.0,
    setup_sigma=0.55,
    client_timeout_seconds=8.0,
    reject_rate=0.12,
    timeout_rate=0.08,
    timeout_but_placed_rate=0.5,
    # A fresh call leg rather than a conference join. This is pure customer
    # wait, and no amount of clever pacing recovers it.
    bridge_min_seconds=0.30,
    bridge_max_seconds=0.90,
    duplicate_rate=0.15,
    duplicate_max_copies=3,
    reorder_rate=0.20,
    reorder_delay_seconds=12.0,
    delayed_rate=0.05,
    delayed_seconds=30.0,
)


def make_flaky_provider(
    clock: Clock, *, seed: int = 0, event_sink: EventSink | None = None, **overrides
) -> SimulatedProvider:
    """Provider B."""
    profile = FLAKY_PROFILE.replace(**overrides) if overrides else FLAKY_PROFILE
    return SimulatedProvider(profile, clock=clock, seed=seed, event_sink=event_sink)
