"""Domain objects: enums, dataclasses, and the legal-transition tables.

Deliberately plain dataclasses, not an ORM and not pydantic models. These carry
rows between the SQL layer and the domain logic; they do not know how to load
or save themselves. Keeping persistence out of them is what lets the pacing
engine take a snapshot dataclass and be provably unable to reach the database.

Every dataclass mirrors a table in migrations/001_init.sql. When the schema
changes, both change together.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, time
from decimal import Decimal
from typing import Any, Mapping
from uuid import UUID


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------


class AgentState(str, enum.Enum):
    """The agent lifecycle from the brief.

    Inherits from str so the value can go straight into a query parameter and
    come back out of a row without conversion helpers on every call site.
    """

    OFFLINE = "OFFLINE"
    AVAILABLE = "AVAILABLE"
    RESERVED = "RESERVED"
    DIALING = "DIALING"
    CONNECTED = "CONNECTED"
    WRAP_UP = "WRAP_UP"
    PAUSED = "PAUSED"


# The complete set of legal agent transitions.
#
# This is a whitelist, not a guideline. An attempt to move an agent along an
# edge that is not here raises -- it is never silently corrected, because a
# transition nobody expected means a bug, and quietly fixing it up destroys the
# evidence needed to find it. The reaper reconciles; the state machine refuses.
AGENT_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    # An agent logs in.
    AgentState.OFFLINE: frozenset({AgentState.AVAILABLE}),
    # Reserved by a worker that is about to dial for them.
    AgentState.AVAILABLE: frozenset(
        {AgentState.RESERVED, AgentState.PAUSED, AgentState.OFFLINE}
    ),
    # RESERVED -> AVAILABLE covers lease expiry, a failed dial, and a cancel.
    AgentState.RESERVED: frozenset(
        {AgentState.DIALING, AgentState.AVAILABLE, AgentState.OFFLINE}
    ),
    # DIALING -> CONNECTED on a successful bridge; -> AVAILABLE on no answer.
    AgentState.DIALING: frozenset(
        {AgentState.CONNECTED, AgentState.AVAILABLE, AgentState.OFFLINE}
    ),
    # A live call ends and the agent writes up the outcome.
    AgentState.CONNECTED: frozenset({AgentState.WRAP_UP, AgentState.OFFLINE}),
    AgentState.WRAP_UP: frozenset(
        {AgentState.AVAILABLE, AgentState.PAUSED, AgentState.OFFLINE}
    ),
    AgentState.PAUSED: frozenset({AgentState.AVAILABLE, AgentState.OFFLINE}),
}

# States in which an agent counts against the dialable pool: they are already
# doing something for a call and must not be handed a second one.
AGENT_BUSY_STATES = frozenset(
    {AgentState.RESERVED, AgentState.DIALING, AgentState.CONNECTED}
)


# ---------------------------------------------------------------------------
# Call lifecycle
# ---------------------------------------------------------------------------


class CallState(str, enum.Enum):
    """The call lifecycle from the brief, plus ABANDONED.

    ABANDONED means a borrower answered and there was no agent to bridge to.
    It is terminal like COMPLETED and FAILED, but it must never be confused
    with either: COMPLETED is success, FAILED never reached a human, ABANDONED
    reached a human and wasted their time. Only the last one is a compliance
    event, and only a distinct state makes it countable.
    """

    QUEUED = "QUEUED"
    RESERVED = "RESERVED"
    INITIATED = "INITIATED"
    RINGING = "RINGING"
    ANSWERED = "ANSWERED"
    CONNECTED = "CONNECTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    ABANDONED = "ABANDONED"


# Rank ordering used to apply provider events that arrive out of order.
#
# A call's state only ever moves to a strictly higher rank, so the sequence
# COMPLETED, ANSWERED, RINGING settles at COMPLETED instead of walking
# backwards. All terminal states share rank 9: once a call is over, it is over,
# and whichever terminal state arrived first is the one that stands.
#
# This mirrors the generated state_rank column in the calls table exactly.
# The database computes it; this copy exists so Python can reason about ranks
# without a round trip. test_schema.py asserts the two agree, so they cannot
# drift apart unnoticed.
CALL_STATE_RANK: dict[CallState, int] = {
    CallState.QUEUED: 0,
    CallState.RESERVED: 1,
    CallState.INITIATED: 2,
    CallState.RINGING: 3,
    CallState.ANSWERED: 4,
    CallState.CONNECTED: 5,
    CallState.COMPLETED: 9,
    CallState.FAILED: 9,
    CallState.CANCELLED: 9,
    CallState.ABANDONED: 9,
}

TERMINAL_CALL_STATES = frozenset(
    {CallState.COMPLETED, CallState.FAILED, CallState.CANCELLED, CallState.ABANDONED}
)

# A call that is out of the queue but not yet finished. These hold an agent, a
# borrower, or a provider resource, so they are what the reaper looks for and
# what the pacing snapshot counts.
IN_FLIGHT_CALL_STATES = frozenset(
    {
        CallState.RESERVED,
        CallState.INITIATED,
        CallState.RINGING,
        CallState.ANSWERED,
        CallState.CONNECTED,
    }
)


class BorrowerState(str, enum.Enum):
    """Borrower lifecycle. Text rather than an enum type in the database,
    matching the brief, but constrained by a CHECK so it cannot drift."""

    PENDING = "PENDING"
    RESERVED = "RESERVED"
    DONE = "DONE"
    EXHAUSTED = "EXHAUSTED"


class CampaignMode(str, enum.Enum):
    PROGRESSIVE = "PROGRESSIVE"
    PREDICTIVE = "PREDICTIVE"


class IllegalTransition(Exception):
    """Raised when code tries to move an entity along an edge that does not
    exist. Loud on purpose: see the note on AGENT_TRANSITIONS."""


def assert_legal_agent_transition(current: AgentState, target: AgentState) -> None:
    if target not in AGENT_TRANSITIONS.get(current, frozenset()):
        raise IllegalTransition(f"agent cannot move {current.value} -> {target.value}")


def assert_call_advances(current: CallState, target: CallState) -> None:
    """Guard for code paths that intend to move a call forward.

    Event application does not use this -- it applies rank monotonicity in SQL,
    where a stale event simply matches no rows. This is for the dialer's own
    transitions, where going backwards would be a bug worth crashing on.
    """
    if CALL_STATE_RANK[target] <= CALL_STATE_RANK[current]:
        raise IllegalTransition(
            f"call cannot move {current.value} (rank {CALL_STATE_RANK[current]}) "
            f"-> {target.value} (rank {CALL_STATE_RANK[target]})"
        )


# ---------------------------------------------------------------------------
# Row objects
# ---------------------------------------------------------------------------
#
# Each has a from_row() that takes a dict_row mapping. They are tolerant of
# extra keys, because several queries return joined or aggregated columns
# alongside the base row.


@dataclass(frozen=True, slots=True)
class Campaign:
    id: UUID
    name: str
    mode: CampaignMode
    max_concurrent: int
    abandon_budget_pct: Decimal
    target_shortfall_eps: Decimal
    max_overdial_ratio: Decimal
    active: bool
    wrap_up_seconds: int
    dialing_window_start: time | None = None
    dialing_window_end: time | None = None
    created_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Campaign":
        return cls(
            id=row["id"],
            name=row["name"],
            mode=CampaignMode(row["mode"]),
            max_concurrent=row["max_concurrent"],
            abandon_budget_pct=row["abandon_budget_pct"],
            target_shortfall_eps=row["target_shortfall_eps"],
            max_overdial_ratio=row["max_overdial_ratio"],
            active=row["active"],
            wrap_up_seconds=row["wrap_up_seconds"],
            dialing_window_start=row.get("dialing_window_start"),
            dialing_window_end=row.get("dialing_window_end"),
            created_at=row.get("created_at"),
        )


@dataclass(frozen=True, slots=True)
class Agent:
    id: UUID
    campaign_id: UUID
    state: AgentState
    version: int
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    current_call_id: UUID | None = None
    state_changed_at: datetime | None = None
    wrap_up_ends_at: datetime | None = None
    last_heartbeat_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Agent":
        return cls(
            id=row["id"],
            campaign_id=row["campaign_id"],
            state=AgentState(row["state"]),
            version=row["version"],
            lease_owner=row.get("lease_owner"),
            lease_expires_at=row.get("lease_expires_at"),
            current_call_id=row.get("current_call_id"),
            state_changed_at=row.get("state_changed_at"),
            wrap_up_ends_at=row.get("wrap_up_ends_at"),
            last_heartbeat_at=row.get("last_heartbeat_at"),
        )


@dataclass(frozen=True, slots=True)
class Borrower:
    id: UUID
    campaign_id: UUID
    phone: str
    state: BorrowerState
    attempts: int
    max_attempts: int
    next_eligible_at: datetime
    version: int
    last_outcome: str | None = None
    dpd_bucket: str | None = None
    priority: int = 0
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Borrower":
        return cls(
            id=row["id"],
            campaign_id=row["campaign_id"],
            phone=row["phone"],
            state=BorrowerState(row["state"]),
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            next_eligible_at=row["next_eligible_at"],
            version=row["version"],
            last_outcome=row.get("last_outcome"),
            dpd_bucket=row.get("dpd_bucket"),
            priority=row.get("priority", 0),
            lease_owner=row.get("lease_owner"),
            lease_expires_at=row.get("lease_expires_at"),
        )


@dataclass(frozen=True, slots=True)
class Call:
    id: UUID
    campaign_id: UUID
    borrower_id: UUID
    provider: str
    idempotency_key: str
    state: CallState
    state_rank: int
    attempt: int
    is_overdial: bool
    version: int
    agent_id: UUID | None = None
    provider_call_id: str | None = None
    predicted_p: Decimal | None = None
    initiated_at: datetime | None = None
    ringing_at: datetime | None = None
    answered_at: datetime | None = None
    connected_at: datetime | None = None
    ended_at: datetime | None = None
    wait_ms: int | None = None
    failure_reason: str | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    created_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_CALL_STATES

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Call":
        return cls(
            id=row["id"],
            campaign_id=row["campaign_id"],
            borrower_id=row["borrower_id"],
            provider=row["provider"],
            idempotency_key=row["idempotency_key"],
            state=CallState(row["state"]),
            state_rank=row["state_rank"],
            attempt=row["attempt"],
            is_overdial=row["is_overdial"],
            version=row["version"],
            agent_id=row.get("agent_id"),
            provider_call_id=row.get("provider_call_id"),
            predicted_p=row.get("predicted_p"),
            initiated_at=row.get("initiated_at"),
            ringing_at=row.get("ringing_at"),
            answered_at=row.get("answered_at"),
            connected_at=row.get("connected_at"),
            ended_at=row.get("ended_at"),
            wait_ms=row.get("wait_ms"),
            failure_reason=row.get("failure_reason"),
            lease_owner=row.get("lease_owner"),
            lease_expires_at=row.get("lease_expires_at"),
            created_at=row.get("created_at"),
        )


@dataclass(frozen=True, slots=True)
class ProviderEvent:
    """One raw event as the provider delivered it.

    provider_event_id is the provider's own identifier for this delivery. It is
    the deduplication key, so a provider that does not supply one has to have
    a stable id synthesised for it inside its own module -- never here.
    """

    provider: str
    provider_event_id: str
    event_type: str
    payload: dict[str, Any]
    provider_call_id: str | None = None
    provider_ts: datetime | None = None
    id: int | None = None
    received_at: datetime | None = None
    applied: bool = False
    apply_result: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ProviderEvent":
        return cls(
            provider=row["provider"],
            provider_event_id=row["provider_event_id"],
            event_type=row["event_type"],
            payload=row["payload"],
            provider_call_id=row.get("provider_call_id"),
            provider_ts=row.get("provider_ts"),
            id=row.get("id"),
            received_at=row.get("received_at"),
            applied=row.get("applied", False),
            apply_result=row.get("apply_result"),
        )


@dataclass(frozen=True, slots=True)
class PacingDecision:
    """The audit record for one tick. `inputs` holds the whole snapshot plus
    every intermediate term, which is what makes "why 17 and not 10?"
    answerable after the fact rather than a matter of reconstruction."""

    campaign_id: UUID
    mode: CampaignMode
    proposed: int
    approved: int
    reason_code: str
    inputs: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    ts: datetime | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "PacingDecision":
        return cls(
            campaign_id=row["campaign_id"],
            mode=CampaignMode(row["mode"]),
            proposed=row["proposed"],
            approved=row["approved"],
            reason_code=row["reason_code"],
            inputs=row.get("inputs") or {},
            id=row.get("id"),
            ts=row.get("ts"),
        )


@dataclass(frozen=True, slots=True)
class CampaignCounters:
    """Summed across shards on read. See the note in the migration on why the
    counters are sharded rather than one row per campaign."""

    campaign_id: UUID
    calls_initiated: int = 0
    calls_answered: int = 0
    calls_connected: int = 0
    calls_abandoned: int = 0
    calls_failed: int = 0

    @property
    def abandon_rate_pct(self) -> float:
        """Abandons as a percentage of calls a human answered.

        The denominator is answered calls, not dialled calls: dropping 3 of
        100 people who picked up is the number a regulator cares about, and
        dividing by dials instead would flatter the figure whenever the answer
        rate is low.
        """
        if self.calls_answered == 0:
            return 0.0
        return 100.0 * self.calls_abandoned / self.calls_answered

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "CampaignCounters":
        return cls(
            campaign_id=row["campaign_id"],
            calls_initiated=row.get("calls_initiated") or 0,
            calls_answered=row.get("calls_answered") or 0,
            calls_connected=row.get("calls_connected") or 0,
            calls_abandoned=row.get("calls_abandoned") or 0,
            calls_failed=row.get("calls_failed") or 0,
        )
