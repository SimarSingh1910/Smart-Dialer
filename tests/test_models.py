"""Tests for the domain models. Pure -- no database needed."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from smartdialer.core.models import (
    AGENT_TRANSITIONS,
    Agent,
    AgentState,
    Call,
    CallState,
    CALL_STATE_RANK,
    CampaignCounters,
    IllegalTransition,
    TERMINAL_CALL_STATES,
    assert_call_advances,
    assert_legal_agent_transition,
)


# --- agent transitions -----------------------------------------------------


@pytest.mark.parametrize(
    "current,target",
    [
        (AgentState.OFFLINE, AgentState.AVAILABLE),
        (AgentState.AVAILABLE, AgentState.RESERVED),
        (AgentState.RESERVED, AgentState.DIALING),
        (AgentState.RESERVED, AgentState.AVAILABLE),  # lease expiry / failed dial
        (AgentState.DIALING, AgentState.CONNECTED),
        (AgentState.DIALING, AgentState.AVAILABLE),  # no answer
        (AgentState.CONNECTED, AgentState.WRAP_UP),
        (AgentState.WRAP_UP, AgentState.AVAILABLE),
        (AgentState.AVAILABLE, AgentState.PAUSED),
        (AgentState.PAUSED, AgentState.AVAILABLE),
    ],
)
def test_legal_agent_transitions_are_allowed(current, target):
    assert_legal_agent_transition(current, target)


@pytest.mark.parametrize(
    "current,target",
    [
        # Skipping the dial: an agent cannot be connected to a call that was
        # never placed.
        (AgentState.AVAILABLE, AgentState.CONNECTED),
        # Going backwards mid-call.
        (AgentState.CONNECTED, AgentState.AVAILABLE),
        # An offline agent cannot be reserved -- this is the one that matters,
        # because it is what a stale worker would try to do.
        (AgentState.OFFLINE, AgentState.RESERVED),
        (AgentState.WRAP_UP, AgentState.CONNECTED),
    ],
)
def test_illegal_agent_transitions_raise(current, target):
    """Loud, not silently corrected. A transition nobody expected means a bug,
    and quietly fixing it up destroys the evidence."""
    with pytest.raises(IllegalTransition):
        assert_legal_agent_transition(current, target)


def test_every_agent_state_can_reach_offline():
    """An agent can always be logged out or time out on heartbeat, from any
    state. If some state could not reach OFFLINE, an agent could get wedged
    there forever and the reaper would have no way to free them."""
    for state in AgentState:
        if state is AgentState.OFFLINE:
            continue
        assert AgentState.OFFLINE in AGENT_TRANSITIONS[state], state


def test_agent_transition_table_covers_every_state():
    assert set(AGENT_TRANSITIONS) == set(AgentState) - {AgentState.OFFLINE} | {
        AgentState.OFFLINE
    }


def test_no_agent_state_transitions_to_itself():
    """A self-transition would be a no-op that still bumps the version and
    rewrites state_changed_at, which would corrupt the longest-idle-first
    ordering the allocator relies on."""
    for state, targets in AGENT_TRANSITIONS.items():
        assert state not in targets, state


# --- call ranks ------------------------------------------------------------


def test_call_rank_is_monotonic_through_the_happy_path():
    happy_path = [
        CallState.QUEUED,
        CallState.RESERVED,
        CallState.INITIATED,
        CallState.RINGING,
        CallState.ANSWERED,
        CallState.CONNECTED,
        CallState.COMPLETED,
    ]
    ranks = [CALL_STATE_RANK[state] for state in happy_path]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


def test_every_call_state_has_a_rank():
    assert set(CALL_STATE_RANK) == set(CallState)


def test_advancing_a_call_backwards_raises():
    with pytest.raises(IllegalTransition):
        assert_call_advances(CallState.CONNECTED, CallState.RINGING)


def test_a_terminal_call_cannot_advance_to_another_terminal_state():
    """The COMPLETED, ANSWERED, RINGING case from the brief settles at
    COMPLETED. Equal ranks do not advance, so a late FAILED cannot overwrite a
    COMPLETED either."""
    with pytest.raises(IllegalTransition):
        assert_call_advances(CallState.COMPLETED, CallState.FAILED)


def test_abandoned_is_terminal_and_distinct_from_completed_and_failed():
    """The compliance-critical distinction. Abandoned reached a human and
    wasted their time; failed never reached one; completed did its job."""
    assert CallState.ABANDONED in TERMINAL_CALL_STATES
    assert CallState.ABANDONED is not CallState.COMPLETED
    assert CallState.ABANDONED is not CallState.FAILED


def test_ringing_advances_to_answered():
    assert_call_advances(CallState.RINGING, CallState.ANSWERED)


# --- row mapping -----------------------------------------------------------


def test_agent_from_row_maps_the_state_to_the_enum():
    agent = Agent.from_row(
        {
            "id": uuid.uuid4(),
            "campaign_id": uuid.uuid4(),
            "state": "AVAILABLE",
            "version": 3,
        }
    )
    assert agent.state is AgentState.AVAILABLE
    assert agent.version == 3
    assert agent.lease_owner is None


def test_call_from_row_and_is_terminal():
    row = {
        "id": uuid.uuid4(),
        "campaign_id": uuid.uuid4(),
        "borrower_id": uuid.uuid4(),
        "provider": "mock_fast",
        "idempotency_key": "k",
        "state": "ABANDONED",
        "state_rank": 9,
        "attempt": 1,
        "is_overdial": True,
        "version": 4,
        "answered_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "predicted_p": Decimal("0.42"),
    }
    call = Call.from_row(row)
    assert call.is_terminal
    assert call.is_overdial
    assert call.predicted_p == Decimal("0.42")


def test_from_row_ignores_extra_columns():
    """Several queries return aggregates alongside the base row; the mappers
    must not care."""
    agent = Agent.from_row(
        {
            "id": uuid.uuid4(),
            "campaign_id": uuid.uuid4(),
            "state": "RESERVED",
            "version": 1,
            "rows_behind": 17,
        }
    )
    assert agent.state is AgentState.RESERVED


# --- counters --------------------------------------------------------------


def test_abandon_rate_is_a_share_of_answered_calls():
    """Denominator is answered, not dialled. Dividing by dials would flatter
    the number whenever the answer rate is low, which is exactly when the
    dialer is most likely to be over-dialling."""
    counters = CampaignCounters(
        campaign_id=uuid.uuid4(),
        calls_initiated=1000,
        calls_answered=200,
        calls_abandoned=6,
    )
    assert counters.abandon_rate_pct == pytest.approx(3.0)


def test_abandon_rate_is_zero_before_any_call_is_answered():
    """No answers means no division, not a crash. This runs on the first tick
    of every campaign."""
    counters = CampaignCounters(campaign_id=uuid.uuid4())
    assert counters.abandon_rate_pct == 0.0
