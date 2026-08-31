"""Provider A: fast, reliable, well-behaved.

Setup in a second or two, a 2% rejection rate, events delivered once, in order,
immediately. Bridging is a conference join rather than a fresh call leg, so it
completes in well under a tenth of a second -- which is the difference between
a borrower hearing dead air after saying hello and not.

This is the provider the predictive path prefers for over-dial calls. Its
setup time has a short tail, and a short tail is what makes the ring-to-answer
hazard model worth anything: if post-dial delay varies by seconds, knowing how
long a call has been ringing tells you very little about when it will be
answered.
"""

from __future__ import annotations

from smartdialer.core.clock import Clock
from smartdialer.providers.base import EventSink
from smartdialer.providers.simulated import ProviderProfile, SimulatedProvider

FAST_PROFILE = ProviderProfile(
    name="mock_fast",
    setup_median_seconds=1.6,
    setup_sigma=0.30,
    client_timeout_seconds=5.0,
    reject_rate=0.02,
    timeout_rate=0.0,
    bridge_min_seconds=0.05,
    bridge_max_seconds=0.10,
    # Delivers each event once, in order, with no delay.
    duplicate_rate=0.0,
    reorder_rate=0.0,
    delayed_rate=0.0,
)


def make_fast_provider(
    clock: Clock, *, seed: int = 0, event_sink: EventSink | None = None, **overrides
) -> SimulatedProvider:
    """Provider A. `overrides` adjusts the profile for a scenario -- the answer
    rate and talk duration in particular are properties of the campaign being
    dialled, not of the carrier."""
    profile = FAST_PROFILE.replace(**overrides) if overrides else FAST_PROFILE
    return SimulatedProvider(profile, clock=clock, seed=seed, event_sink=event_sink)
