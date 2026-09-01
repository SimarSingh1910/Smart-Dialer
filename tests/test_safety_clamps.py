"""The safety clamps, tested without a database.

This file exists because of the split in safety/controller.py. `_decide` is
pure -- snapshot and proposal in, approved number and reason out -- so the
safety-critical logic can be tested exhaustively with dataclasses instead of at
whatever coverage is affordable when every case costs a campaign fixture and a
round trip.

The allocator here is a stub that records what it was asked for. `_decide`
never touches it; it is only present because SafetyController takes one.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from smartdialer.core.clock import VirtualClock
from smartdialer.core.logging import StructuredLogger
from smartdialer.core.models import Campaign, CampaignMode
from smartdialer.pacing.engine import (
    PacingProposal,
    PacingSnapshot,
    ProviderHealthSignal,
    RecentBehaviour,
)
from smartdialer.safety.controller import Reason, SafetyController

NOW = datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc)


class StubAllocator:
    """Records requests. Never used by _decide, which is the point."""

    def __init__(self) -> None:
        self.requests: list[int] = []

    async def reserve(self, *, campaign, n):
        self.requests.append(n)
        raise AssertionError("_decide must not reach the allocator")


def make_controller(max_signal_age_seconds: float = 5.0) -> SafetyController:
    clock = VirtualClock(start=NOW)
    return SafetyController(
        allocator=StubAllocator(),
        db=None,  # _decide does no I/O; execute() is tested against a real one
        clock=clock,
        logger=StructuredLogger("clamps", clock),
        max_signal_age_seconds=max_signal_age_seconds,
    )


def campaign(**overrides) -> Campaign:
    defaults = dict(
        id=uuid4(),
        name="clamp-test",
        mode=CampaignMode.PROGRESSIVE,
        max_concurrent=1000,
        abandon_budget_pct=Decimal("3.0"),
        target_shortfall_eps=Decimal("0.02"),
        max_overdial_ratio=Decimal("2.0"),
        active=True,
        wrap_up_seconds=10,
    )
    defaults.update(overrides)
    return Campaign(**defaults)


def snapshot(**overrides) -> PacingSnapshot:
    defaults = dict(
        mode=CampaignMode.PROGRESSIVE,
        snapshot_taken_at=NOW,
        now=NOW,
        agents_available=10,
        provider_health=ProviderHealthSignal(name="mock_fast"),
        recent_campaign_behaviour=RecentBehaviour(initiated=100, answered=40),
    )
    defaults.update(overrides)
    return PacingSnapshot(**defaults)


def decide(proposed: int, *, snap=None, camp=None, controller=None):
    controller = controller or make_controller()
    return controller._decide(
        PacingProposal(n=proposed, mode=CampaignMode.PROGRESSIVE, reason="test"),
        snap or snapshot(),
        camp or campaign(),
    )


# ---------------------------------------------------------------------------
# Each clamp, on its own
# ---------------------------------------------------------------------------


def test_an_unclamped_proposal_passes_through_and_says_so():
    decision = decide(10)
    assert decision.approved == 10
    assert decision.reason_code == Reason.UNCLAMPED
    assert decision.clamps == ()


def test_the_kill_switch_stops_everything():
    controller = make_controller()
    controller.kill_switch = True
    decision = decide(10, controller=controller)
    assert decision.approved == 0
    assert decision.reason_code == Reason.KILL_SWITCH


def test_an_inactive_campaign_stops_everything():
    decision = decide(10, camp=campaign(active=False))
    assert decision.approved == 0
    assert decision.reason_code == Reason.KILL_SWITCH


def test_outside_the_dialing_window_nothing_is_approved():
    """10:00 UTC against a 12:00-18:00 window."""
    decision = decide(
        10,
        camp=campaign(
            dialing_window_start=time(12, 0), dialing_window_end=time(18, 0)
        ),
    )
    assert decision.approved == 0
    assert decision.reason_code == Reason.WINDOW_CLOSED


def test_inside_the_dialing_window_dialing_proceeds():
    decision = decide(
        10,
        camp=campaign(
            dialing_window_start=time(9, 0), dialing_window_end=time(18, 0)
        ),
    )
    assert decision.approved == 10


def test_a_window_that_wraps_past_midnight_is_handled():
    """20:00-06:00 is two ranges, not an empty one. A naive start <= t <= end
    would close the campaign for the entire night it is meant to be open."""
    late = snapshot(now=NOW.replace(hour=23), snapshot_taken_at=NOW.replace(hour=23))
    decision = decide(
        10,
        snap=late,
        camp=campaign(dialing_window_start=time(20, 0), dialing_window_end=time(6, 0)),
    )
    assert decision.approved == 10


def test_stale_signals_force_the_progressive_floor():
    """A proposal built on a reading six seconds old is a bet on a world that
    has moved on. Falling back to one-agent-one-call needs no prediction."""
    stale = snapshot(agents_available=4, now=NOW + timedelta(seconds=6))
    decision = decide(20, snap=stale)
    assert decision.approved == 4
    assert Reason.STALE_SIGNALS in decision.clamps


def test_fresh_signals_do_not_trigger_the_stale_clamp():
    fresh = snapshot(agents_available=4, now=NOW + timedelta(seconds=1))
    decision = decide(6, snap=fresh)
    assert decision.approved == 6
    assert Reason.STALE_SIGNALS not in decision.clamps


def test_the_hard_ratio_binds_regardless_of_what_the_engine_wants():
    """The clamp the brief cares most about: an absolute ceiling from the
    campaign's policy row that model confidence cannot widen."""
    decision = decide(
        10_000, snap=snapshot(agents_available=10), camp=campaign(max_overdial_ratio=Decimal("2.0"))
    )
    assert decision.approved == 20
    assert Reason.HARD_RATIO in decision.clamps


def test_a_ratio_of_one_is_exactly_progressive():
    decision = decide(50, camp=campaign(max_overdial_ratio=Decimal("1.0")))
    assert decision.approved == 10


def test_campaign_concurrency_counts_everything_in_flight():
    """A backlog of ringing calls tightens the cap by itself."""
    # calls_ringing is per-call ring durations now, not a count.
    busy = snapshot(
        agents_available=50, calls_ringing=tuple(1.0 for _ in range(90)), calls_connected=5
    )
    decision = decide(50, snap=busy, camp=campaign(max_concurrent=100))
    assert decision.approved == 5
    assert Reason.CAMPAIGN_CONCURRENCY in decision.clamps


def test_concurrency_already_exhausted_approves_nothing():
    full = snapshot(agents_available=50, calls_ringing=tuple(1.0 for _ in range(100)))
    decision = decide(50, snap=full, camp=campaign(max_concurrent=100))
    assert decision.approved == 0
    assert decision.reason_code == Reason.CAMPAIGN_CONCURRENCY


def test_nothing_proposed_is_distinguishable_from_being_clamped():
    """An idle tick with agents available and nothing to dial is not the same
    event as a tick the safety system stopped, and the log must not conflate
    them."""
    decision = decide(0)
    assert decision.approved == 0
    assert decision.reason_code == Reason.NOTHING_PROPOSED
    assert decision.clamps == ()


# ---------------------------------------------------------------------------
# Clamps in combination
# ---------------------------------------------------------------------------


def test_the_reason_code_names_the_binding_clamp():
    """Several clamps can have room while one bites. The row records the one
    that actually decided the number."""
    snap = snapshot(agents_available=10, calls_ringing=tuple(1.0 for _ in range(95)))
    decision = decide(100, snap=snap, camp=campaign(max_concurrent=100))
    # hard ratio allows 20, concurrency allows 5 -> concurrency is binding
    assert decision.approved == 5
    assert decision.reason_code == Reason.CAMPAIGN_CONCURRENCY
    assert Reason.HARD_RATIO in decision.clamps


def test_a_clamp_with_room_is_not_recorded():
    decision = decide(5, camp=campaign(max_concurrent=1000))
    assert decision.clamps == ()
    # But its limit is still in the terms, so "was it even checked?" is
    # answerable from the row rather than from reading the code.
    assert "limit_campaign_concurrency" in decision.terms
    assert "limit_hard_ratio" in decision.terms


def test_clamps_never_produce_a_negative_approval():
    over = snapshot(agents_available=5, calls_ringing=tuple(1.0 for _ in range(500)))
    decision = decide(50, snap=over, camp=campaign(max_concurrent=100))
    assert decision.approved == 0


def test_no_clamp_can_ever_increase_the_proposal():
    """The controller is a set of ceilings. If any path could raise a number,
    the engine's output would stop being an upper bound on risk."""
    for proposed in range(0, 40):
        for available in (0, 1, 5, 20):
            snap = snapshot(agents_available=available)
            decision = decide(proposed, snap=snap)
            assert decision.approved <= proposed, (proposed, available)


def test_decide_is_pure_and_never_touches_the_allocator():
    controller = make_controller()
    for _ in range(10):
        decide(10, controller=controller)
    assert controller._allocator.requests == []


def test_decide_is_deterministic():
    controller = make_controller()
    snap, camp = snapshot(), campaign()
    first = decide(17, snap=snap, camp=camp, controller=controller)
    for _ in range(20):
        again = decide(17, snap=snap, camp=camp, controller=controller)
        assert (again.approved, again.reason_code, again.clamps) == (
            first.approved,
            first.reason_code,
            first.clamps,
        )
