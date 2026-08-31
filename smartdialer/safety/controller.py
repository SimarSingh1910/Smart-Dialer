"""The Safety Controller.

The pacing engine says what it would like. This decides what actually happens,
and it is the only module in the system holding a reference to the allocator.
That is the architectural claim of the whole submission: the path from a
prediction to a ringing telephone runs through here, and the engine has no
other way to reach a carrier because it imports nothing that can.

The module is split into two halves on purpose.

    _decide()   PURE. Snapshot and proposal in, approved number and reason
                out. No database, no carrier, no clock reads, no I/O of any
                kind. Every clamp is arithmetic over measured state.

    execute()   The wrapper. Calls _decide, reserves through the allocator,
                writes the decision row, and only then hands the batch to the
                carrier.

Splitting them is not tidiness. The clamps are the safety-critical logic and
they are now testable with a dataclass and no database at all, which means they
can be tested exhaustively -- every boundary, every combination -- instead of
at whatever coverage is affordable when each case costs a campaign fixture.

Every clamp is a pure function of MEASURED state: agent counts, call counts,
the campaign's own policy row, the snapshot's age. Not one of them takes an
input the engine produces. A model that becomes very confident cannot widen the
hard ratio, cannot raise the concurrency cap, and cannot talk the controller
out of the dialling window. The most an over-confident engine can do is propose
a large number and have it clamped, which is a logged event rather than an
incident.

Fail closed. execute() is wrapped whole: any exception yields approved = 0,
dialed = 0 and an EXCEPTION row in the decision log. A safety component that
crashes into "carry on" is not a safety component, and the cost of being wrong
in this direction is idle agents, while the cost of being wrong in the other is
a compliance event.

Two clamps named in the design are not here yet and their slots are marked
below: the AIMD abandon-budget credit and the provider circuit breaker both
arrive in step 8. Their absence is safe rather than merely unfinished -- with
no credit granted, the engine's predictive path proposes the progressive floor,
which is what every clamp here degrades to anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from smartdialer.allocator.allocator import CallAllocator, DialTicket
from smartdialer.core.clock import Clock
from smartdialer.core.db import Database
from smartdialer.core.logging import StructuredLogger
from smartdialer.core.models import Campaign
from smartdialer.domain.decisions import Shortfall, record_decision
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
    EXCEPTION = "EXCEPTION"
    # Reserved for step 8, so the vocabulary does not change under the
    # simulation's reporting when they arrive.
    ABANDON_BUDGET = "ABANDON_BUDGET"
    PROVIDER_BREAKER = "PROVIDER_BREAKER"


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    """The pure half: what the clamps concluded.

    Contains no outcome, because at this point nothing has been attempted.
    """

    proposed: int
    approved: int
    reason_code: str
    clamps: tuple[str, ...] = ()
    terms: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """The decision plus what actually came of it."""

    decision: SafetyDecision
    dialed: int = 0
    shortfall_reason: str = Shortfall.NONE
    tickets: tuple[DialTicket, ...] = ()
    decision_id: int | None = None

    @property
    def proposed(self) -> int:
        return self.decision.proposed

    @property
    def approved(self) -> int:
        return self.decision.approved

    @property
    def reason_code(self) -> str:
        return self.decision.reason_code

    @property
    def clamps(self) -> tuple[str, ...]:
        return self.decision.clamps


class SafetyController:
    def __init__(
        self,
        *,
        allocator: CallAllocator,
        db: Database,
        clock: Clock,
        logger: StructuredLogger,
        max_signal_age_seconds: float = 5.0,
    ) -> None:
        self._allocator = allocator
        self._db = db
        self._clock = clock
        self._log = logger
        self._max_signal_age = max_signal_age_seconds
        # The operator's stop button. Separate from campaigns.active, which is
        # campaign policy: this one stops THIS process immediately without a
        # database round trip, for when somebody needs the dialling to stop now
        # and explain later.
        self.kill_switch = False

    # -- the pure half --------------------------------------------------

    def _decide(
        self,
        proposal: PacingProposal,
        snapshot: PacingSnapshot,
        campaign: Campaign,
    ) -> SafetyDecision:
        """Apply every clamp, in order. No I/O, and none possible.

        Clamps are recorded only when they actually bind, so the log
        distinguishes "the concurrency cap was the binding constraint" from
        "the concurrency cap was checked and had room".
        """
        n = max(0, proposal.n)
        clamps: list[str] = []
        terms: dict[str, Any] = {"proposed": proposal.n}

        def clamp(limit: int, reason: str) -> None:
            nonlocal n
            terms[f"limit_{reason.lower()}"] = limit
            if limit < n:
                n = max(0, limit)
                clamps.append(reason)

        # 1. Operator kill switch, and the campaign's own active flag. First,
        #    because nothing below it matters if dialling is off.
        if self.kill_switch or not campaign.active:
            terms.update({"kill_switch": self.kill_switch, "active": campaign.active})
            return SafetyDecision(
                proposed=proposal.n,
                approved=0,
                reason_code=Reason.KILL_SWITCH,
                clamps=(Reason.KILL_SWITCH,),
                terms=terms,
            )

        # 2. The dialling window. A collections campaign calling outside its
        #    permitted hours is a compliance breach on its own, regardless of
        #    how the call goes.
        if not is_within_dialing_window(campaign, snapshot.now):
            return SafetyDecision(
                proposed=proposal.n,
                approved=0,
                reason_code=Reason.WINDOW_CLOSED,
                clamps=(Reason.WINDOW_CLOSED,),
                terms=terms,
            )

        # 3. Stale signals force progressive.
        #
        #    This is the clamp that stops a slow tick turning into abandoned
        #    calls. Over-dialling is a bet on a reading of the world; if the
        #    reading is five seconds old, agents have been taken and calls have
        #    been answered since, and the bet is being placed on a world that
        #    no longer exists. Falling back to one-agent-one-call needs no
        #    prediction to be correct.
        terms["snapshot_age_seconds"] = snapshot.age_seconds
        if snapshot.age_seconds > self._max_signal_age:
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
        clamp(
            campaign.max_concurrent - snapshot.calls_in_flight,
            Reason.CAMPAIGN_CONCURRENCY,
        )

        # --- step 8 slots in here -------------------------------------
        # 6. ABANDON_BUDGET: clamp to agents_available + AIMD over-dial credit.
        # 7. PROVIDER_BREAKER: CLOSED passes, HALF_OPEN allows one probe call,
        #    OPEN approves nothing while existing calls keep reconciling.
        # Both are ceilings computed from measured outcomes, so they belong in
        # this sequence and nowhere else.

        if n <= 0:
            return SafetyDecision(
                proposed=proposal.n,
                approved=0,
                reason_code=clamps[-1] if clamps else Reason.NOTHING_PROPOSED,
                clamps=tuple(clamps),
                terms=terms,
            )

        return SafetyDecision(
            proposed=proposal.n,
            approved=n,
            reason_code=clamps[-1] if clamps else Reason.UNCLAMPED,
            clamps=tuple(clamps),
            terms=terms,
        )

    # -- the wrapper ----------------------------------------------------

    async def execute(
        self,
        *,
        proposal: PacingProposal,
        snapshot: PacingSnapshot,
        campaign: Campaign,
        ts: datetime,
        log_inputs: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Decide, reserve, record, then dial.

        The order of the last three is load-bearing. Reservation commits the
        call rows; the decision row is written next; only then is the batch
        handed to the carrier. Placing before the decision row existed would
        let a carrier answer -- and a placement fail -- with no row to
        attribute the shortfall to, so the one number that explains a bad tick
        would go missing exactly when a tick went badly.
        """
        try:
            return await self._execute(
                proposal=proposal,
                snapshot=snapshot,
                campaign=campaign,
                ts=ts,
                log_inputs=log_inputs or {},
            )
        except Exception as exc:  # noqa: BLE001 - this is the fail-closed net
            self._log.error(
                "safety_controller_failed_closed",
                error=repr(exc),
                proposed=proposal.n,
                campaign_id=str(campaign.id),
            )
            decision = SafetyDecision(
                proposed=proposal.n,
                approved=0,
                reason_code=Reason.EXCEPTION,
                clamps=(Reason.EXCEPTION,),
                terms={"error": repr(exc)},
            )
            # Best effort: an exception here has already cost us the tick, and
            # failing to log it must not also cost us the loop.
            decision_id = await self._try_record(
                campaign=campaign,
                ts=ts,
                decision=decision,
                dialed=0,
                shortfall_reason=Shortfall.NONE,
                log_inputs={"error": repr(exc), **(log_inputs or {})},
            )
            return ExecutionResult(decision=decision, decision_id=decision_id)

    async def _execute(
        self,
        *,
        proposal: PacingProposal,
        snapshot: PacingSnapshot,
        campaign: Campaign,
        ts: datetime,
        log_inputs: dict[str, Any],
    ) -> ExecutionResult:
        decision = self._decide(proposal, snapshot, campaign)

        batch = None
        if decision.approved > 0:
            batch = await self._allocator.reserve(campaign=campaign, n=decision.approved)

        dialed = len(batch.tickets) if batch else 0
        shortfall = batch.shortfall_reason if batch else Shortfall.NONE

        async with self._db.transaction() as cur:
            decision_id = await record_decision(
                cur,
                campaign_id=campaign.id,
                ts=ts,
                mode=campaign.mode,
                proposed=decision.proposed,
                approved=decision.approved,
                dialed=dialed,
                reason_code=decision.reason_code,
                shortfall_reason=shortfall,
                inputs={
                    **log_inputs,
                    "safety_terms": decision.terms,
                    "clamps": list(decision.clamps),
                    "agents_reserved": batch.agents_reserved if batch else 0,
                    "borrowers_reserved": batch.borrowers_reserved if batch else 0,
                },
            )

        # The decision row now exists, so a carrier answering in a millisecond
        # still has somewhere to attribute its outcome.
        if batch is not None:
            self._allocator.place_all(batch, decision_id=decision_id)

        return ExecutionResult(
            decision=decision,
            dialed=dialed,
            shortfall_reason=shortfall,
            tickets=batch.tickets if batch else (),
            decision_id=decision_id,
        )

    async def _try_record(
        self,
        *,
        campaign: Campaign,
        ts: datetime,
        decision: SafetyDecision,
        dialed: int,
        shortfall_reason: str,
        log_inputs: dict[str, Any],
    ) -> int | None:
        try:
            async with self._db.transaction() as cur:
                return await record_decision(
                    cur,
                    campaign_id=campaign.id,
                    ts=ts,
                    mode=campaign.mode,
                    proposed=decision.proposed,
                    approved=decision.approved,
                    dialed=dialed,
                    reason_code=decision.reason_code,
                    shortfall_reason=shortfall_reason,
                    inputs=log_inputs,
                )
        except Exception as exc:  # noqa: BLE001
            self._log.error("could_not_record_decision", error=repr(exc))
            return None
