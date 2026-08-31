"""The Safety Controller.

The pacing engine says what it would like. This decides what actually happens,
and it is the only module in the system holding a reference to the allocator.
That is the architectural claim of the whole submission: the path from a
prediction to a ringing telephone runs through here, and the engine has no
other way to reach a carrier because it imports nothing that can.

Every clamp here is a pure function of MEASURED state -- agent counts, call
counts, the campaign's own policy row, the clock. Not one of them takes an
input the engine produces. A model that becomes very confident cannot widen
the hard ratio, cannot raise the concurrency cap, and cannot talk the
controller out of the dialling window. The most an over-confident engine can do
is propose a large number and have it clamped, which is a logged event rather
than an incident.

Fail closed. The entire body is wrapped: any exception yields zero approved
calls and a CONTROLLER_ERROR row in the decision log. A safety component that
crashes into "carry on" is not a safety component, and the cost of being wrong
in this direction is idle agents, while the cost of being wrong in the other is
a compliance event.

Two clamps named in the design are not here yet, and their slots are marked
below: the AIMD abandon-budget credit and the provider circuit breaker both
arrive in step 8. Their absence is safe rather than merely unfinished -- with
no credit granted, the engine's predictive path proposes the progressive floor,
which is exactly what every clamp here degrades to anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smartdialer.allocator.allocator import CallAllocator, DialTicket
from smartdialer.core.clock import Clock
from smartdialer.core.logging import StructuredLogger
from smartdialer.core.models import Campaign
from smartdialer.domain.snapshot import is_within_dialing_window
from smartdialer.pacing.engine import PacingProposal, PacingSnapshot


class Reason:
    """Why a decision came out the way it did.

    One of these lands in pacing_decisions.reason_code on every tick, so a run
    of idle ticks explains itself. UNCLAMPED means the engine got what it asked
    for -- which is information too.
    """

    UNCLAMPED = "UNCLAMPED"
    NOTHING_PROPOSED = "NOTHING_PROPOSED"
    KILL_SWITCH = "KILL_SWITCH"
    WINDOW_CLOSED = "WINDOW_CLOSED"
    STALE_SIGNALS = "STALE_SIGNALS"
    HARD_RATIO = "HARD_RATIO"
    CAMPAIGN_CONCURRENCY = "CAMPAIGN_CONCURRENCY"
    CONTROLLER_ERROR = "CONTROLLER_ERROR"
    # Reserved for step 8, so the vocabulary does not change under the
    # simulation's reporting when they arrive.
    ABANDON_BUDGET = "ABANDON_BUDGET"
    PROVIDER_BREAKER = "PROVIDER_BREAKER"


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    """What the controller decided, and everything needed to audit it."""

    proposed: int
    approved: int
    reason_code: str
    clamps: tuple[str, ...] = ()
    dialled: int = 0
    tickets: tuple[DialTicket, ...] = ()
    terms: dict[str, Any] = field(default_factory=dict)


class SafetyController:
    def __init__(
        self,
        *,
        allocator: CallAllocator,
        clock: Clock,
        logger: StructuredLogger,
        max_signal_age_seconds: float = 5.0,
    ) -> None:
        self._allocator = allocator
        self._clock = clock
        self._log = logger
        self._max_signal_age = max_signal_age_seconds
        # The operator's stop button. Separate from campaigns.active, which is
        # campaign policy: this one stops THIS process immediately without a
        # database round trip, for when somebody needs the dialling to stop now
        # and explain later.
        self.kill_switch = False

    async def execute(
        self,
        *,
        proposal: PacingProposal,
        snapshot: PacingSnapshot,
        campaign: Campaign,
    ) -> SafetyDecision:
        """Clamp the proposal and, if anything survives, dial it."""
        try:
            return await self._execute(
                proposal=proposal, snapshot=snapshot, campaign=campaign
            )
        except Exception as exc:  # noqa: BLE001 - this is the fail-closed net
            self._log.error(
                "safety_controller_failed_closed",
                error=repr(exc),
                proposed=proposal.n,
                campaign_id=str(campaign.id),
            )
            return SafetyDecision(
                proposed=proposal.n,
                approved=0,
                reason_code=Reason.CONTROLLER_ERROR,
                clamps=(Reason.CONTROLLER_ERROR,),
                terms={"error": repr(exc)},
            )

    async def _execute(
        self,
        *,
        proposal: PacingProposal,
        snapshot: PacingSnapshot,
        campaign: Campaign,
    ) -> SafetyDecision:
        n = max(0, proposal.n)
        clamps: list[str] = []
        terms: dict[str, Any] = {"proposed": proposal.n}

        def clamp(limit: int, reason: str) -> int:
            """Apply one ceiling, recording it if it actually bound.

            Clamps are recorded only when they change the number, so the log
            distinguishes "the concurrency cap was the binding constraint" from
            "the concurrency cap was checked and had room".
            """
            nonlocal n
            terms[f"limit_{reason.lower()}"] = limit
            if limit < n:
                n = max(0, limit)
                clamps.append(reason)
            return n

        # 1. Operator kill switch, and the campaign's own active flag. Checked
        #    first because nothing below it matters if dialling is off.
        if self.kill_switch or not campaign.active:
            return self._stop(
                proposal,
                Reason.KILL_SWITCH,
                terms | {"kill_switch": self.kill_switch, "active": campaign.active},
            )

        # 2. The dialling window. A collections campaign calling outside its
        #    permitted hours is a compliance breach on its own, regardless of
        #    how the call goes.
        if not is_within_dialing_window(campaign, snapshot.now):
            return self._stop(proposal, Reason.WINDOW_CLOSED, terms)

        # 3. Stale signals force progressive.
        #
        #    This is the clamp that stops a slow tick turning into abandoned
        #    calls. Over-dialling is a bet on a reading of the world; if the
        #    reading is five seconds old, agents have been taken and calls have
        #    been answered since, and the bet is being placed on a world that
        #    no longer exists. Falling back to one-agent-one-call needs no
        #    prediction to be correct.
        age = snapshot.age_seconds
        terms["snapshot_age_seconds"] = age
        if age > self._max_signal_age:
            clamp(snapshot.agents_available, Reason.STALE_SIGNALS)

        # 4. The hard over-dial ratio. Absolute, from the campaign's policy
        #    row, and never widened by model confidence -- that is the point of
        #    it. In progressive mode the proposal is already at the floor, so
        #    this never binds; it is here so that it is impossible to reach the
        #    allocator without passing it.
        clamp(
            int(float(campaign.max_overdial_ratio) * snapshot.agents_available),
            Reason.HARD_RATIO,
        )

        # 5. Campaign concurrency. Counts everything already in flight, so a
        #    backlog of ringing calls tightens the cap by itself.
        clamp(campaign.max_concurrent - snapshot.calls_in_flight, Reason.CAMPAIGN_CONCURRENCY)

        # --- step 8 slots in here -------------------------------------
        # 6. ABANDON_BUDGET: clamp to agents_available + AIMD over-dial credit.
        # 7. PROVIDER_BREAKER: CLOSED passes, HALF_OPEN allows one probe call,
        #    OPEN approves nothing while existing calls keep reconciling.
        # Both are ceilings computed from measured outcomes, so they belong in
        # this sequence and nowhere else.

        if n <= 0:
            reason = clamps[-1] if clamps else Reason.NOTHING_PROPOSED
            return SafetyDecision(
                proposed=proposal.n,
                approved=0,
                reason_code=reason,
                clamps=tuple(clamps),
                terms=terms,
            )

        tickets = await self._allocator.dial(campaign=campaign, n=n)
        terms["dialled"] = len(tickets)

        # Fewer tickets than approved is ordinary: another worker took the
        # agents first, or the campaign ran out of borrowers. Worth recording,
        # because a persistent gap between approved and dialled is what a
        # starved campaign looks like from the outside.
        return SafetyDecision(
            proposed=proposal.n,
            approved=n,
            reason_code=clamps[-1] if clamps else Reason.UNCLAMPED,
            clamps=tuple(clamps),
            dialled=len(tickets),
            tickets=tuple(tickets),
            terms=terms,
        )

    def _stop(
        self, proposal: PacingProposal, reason: str, terms: dict[str, Any]
    ) -> SafetyDecision:
        return SafetyDecision(
            proposed=proposal.n,
            approved=0,
            reason_code=reason,
            clamps=(reason,),
            terms=terms,
        )
