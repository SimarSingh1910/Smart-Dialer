"""The call allocator: agents, borrowers, and the moment we call the carrier.

Two things in this module carry the weight.

THE ORDERING. The call row, with the idempotency key we are about to send, is
written and COMMITTED before the provider is called. If the worker dies in the
gap, recovery finds a row saying "we may have placed this call" and asks the
provider about that exact key. Calling first and recording afterwards loses the
call on any crash in between, and a lost call here is a live one ringing a real
person with nobody responsible for it. This is the intent log, and it is the
reason the sequence below is split across a transaction and a task rather than
written as one tidy function.

WHAT WE DO WHEN THE CARRIER FAILS. The three provider exceptions get three
different responses, and the distinction is the entire point of that hierarchy:

    ProviderRejected     we know nothing was placed, and the number is at
                         fault -> fail the call, free the agent, spend one of
                         the borrower's attempts
    ProviderUnavailable  we know nothing was placed, and WE are at fault ->
                         fail the call, free the agent, return the borrower
                         without spending an attempt
    ProviderTimeout      we do not know -> change nothing at all

That last one is the one worth defending. Doing nothing looks like a bug and is
the only correct move: the call may be ringing right now. Releasing the agent
risks bridging them to a second borrower while the first is still live;
re-dialling the borrower risks calling one person twice about one debt. So the
agent stays reserved, the call stays INITIATED with its lease ticking, and the
reaper reconciles against the provider once it can. That costs utilisation
exactly when things are already going badly, and it is the trade this system
deliberately makes.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Sequence
from uuid import UUID

from smartdialer.core.clock import Clock
from smartdialer.core.config import Settings
from smartdialer.core.db import Database
from smartdialer.core.logging import StructuredLogger
from smartdialer.core.models import AgentState, Call, CallState, Campaign
from smartdialer.domain.agents import release_agent, reserve_agents, transition_agent
from smartdialer.domain.borrowers import (
    record_attempt,
    release_borrower,
    reserve_borrowers,
)
from smartdialer.domain.calls import (
    attach_provider_call_id,
    create_call,
    terminate_call,
)
from smartdialer.providers.base import (
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
    TelecomProvider,
)

# How long a borrower waits before being eligible again, by reason. A number
# the carrier rejected is probably still bad in a minute; one we could not dial
# because our own provider was down should come back quickly.
RETRY_AFTER_REJECTED = 300.0
RETRY_AFTER_UNAVAILABLE = 15.0


@dataclass(frozen=True, slots=True)
class DialTicket:
    """A call that exists in the database and is about to exist on a carrier.

    Carries the agent's version so the failure paths can compare-and-swap that
    agent back without re-reading it -- and so that if something else moved the
    agent meanwhile, the swap misses and we leave it to the reaper instead of
    fighting over it.
    """

    call_id: UUID
    campaign_id: UUID
    agent_id: UUID
    agent_version: int
    borrower_id: UUID
    borrower_version: int
    phone: str
    idempotency_key: str
    provider_name: str


class CallAllocator:
    """Turns an approved number of calls into calls.

    Only the Safety Controller holds a reference to this. The pacing engine
    cannot reach it, which is what makes the safety boundary a fact about the
    import graph rather than a promise in a docstring.
    """

    def __init__(
        self,
        *,
        db: Database,
        clock: Clock,
        providers: Sequence[TelecomProvider],
        settings: Settings,
        logger: StructuredLogger,
        from_number: str = "+911140000000",
    ) -> None:
        if not providers:
            raise ValueError("the allocator needs at least one provider")
        self._db = db
        self._clock = clock
        self._providers = list(providers)
        self._settings = settings
        self._log = logger
        self._from = from_number
        self._tasks: set[asyncio.Task] = set()

    @property
    def providers(self) -> list[TelecomProvider]:
        return list(self._providers)

    def provider_for(self, name: str) -> TelecomProvider | None:
        return next((p for p in self._providers if p.name == name), None)

    def _choose_provider(self) -> TelecomProvider:
        """Which carrier to dial through.

        The first configured one, for now. Step 8 makes this breaker-aware --
        route away from a provider whose breaker is open, and prefer the fast,
        low-variance carrier for over-dial calls, where a wide spread in setup
        time is what makes the timing model useless.
        """
        return self._providers[0]

    # -- allocation -----------------------------------------------------

    async def dial(self, *, campaign: Campaign, n: int) -> list[DialTicket]:
        """Reserve, record intent, and start placing `n` calls.

        The database work is ONE transaction and one batch, not a loop of `n`
        round trips. That is not premature optimisation: it is the fix named in
        the scale analysis for the first thing that breaks at a thousand
        agents, where every worker otherwise hammers the head of the same
        index one row at a time. The semantics are identical -- SKIP LOCKED
        hands each row to exactly one worker whether you ask for one or twenty.

        Returns the tickets whose provider calls are now in flight. Placing
        happens in background tasks: the carrier takes one to twelve seconds to
        answer, and a tick loop that waited for it would stop pacing.
        """
        if n <= 0:
            return []

        tickets: list[DialTicket] = []
        provider = self._choose_provider()

        async with self._db.transaction() as cur:
            agents = await reserve_agents(
                cur,
                campaign_id=campaign.id,
                worker_id=self._settings.worker_id,
                n=n,
                lease_seconds=self._settings.lease_seconds,
                now=self._clock.now(),
            )
            if not agents:
                return []

            borrowers = await reserve_borrowers(
                cur,
                campaign_id=campaign.id,
                worker_id=self._settings.worker_id,
                n=len(agents),
                lease_seconds=self._settings.lease_seconds,
                now=self._clock.now(),
            )

            # Fewer borrowers than agents is the normal end of a campaign, not
            # an error. The agents we cannot use go straight back so they are
            # available to another worker on the next tick rather than sitting
            # reserved until their lease expires.
            for surplus in agents[len(borrowers) :]:
                await release_agent(
                    cur,
                    agent_id=surplus.agent_id,
                    expected_version=surplus.version,
                    expected_state=AgentState.RESERVED,
                    now=self._clock.now(),
                )

            for agent, borrower in zip(agents, borrowers):
                now = self._clock.now()
                call_id = uuid.uuid4()
                # Generated here, written here, sent to the carrier later.
                idempotency_key = f"{call_id}"

                await create_call(
                    cur,
                    call_id=call_id,
                    campaign_id=campaign.id,
                    borrower_id=borrower.borrower_id,
                    provider=provider.name,
                    idempotency_key=idempotency_key,
                    now=now,
                    worker_id=self._settings.worker_id,
                    agent_id=agent.agent_id,
                    is_overdial=False,
                    attempt=borrower.attempts + 1,
                    lease_seconds=self._settings.lease_seconds,
                )

                # The agent is now dialling for this call. Same transaction as
                # the call row, so there is never a committed state in which a
                # call exists for an agent who does not know about it.
                moved = await transition_agent(
                    cur,
                    agent_id=agent.agent_id,
                    expected_version=agent.version,
                    expected_state=AgentState.RESERVED,
                    target_state=AgentState.DIALING,
                    now=now,
                    current_call_id=call_id,
                )
                if moved is None:
                    # Cannot happen while we hold the reservation, so if it
                    # ever does, something is writing to agents outside the
                    # discipline in agents.py and we want the evidence.
                    self._log.error(
                        "agent_lost_between_reserve_and_dial",
                        agent_id=str(agent.agent_id),
                        call_id=str(call_id),
                    )
                    raise RuntimeError(
                        f"agent {agent.agent_id} moved between reservation and dial"
                    )

                tickets.append(
                    DialTicket(
                        call_id=call_id,
                        campaign_id=campaign.id,
                        agent_id=agent.agent_id,
                        agent_version=moved.version,
                        borrower_id=borrower.borrower_id,
                        borrower_version=borrower.version,
                        phone=borrower.phone,
                        idempotency_key=idempotency_key,
                        provider_name=provider.name,
                    )
                )

        # The transaction has committed. Everything above is durable, so a
        # crash from here on is recoverable from the rows we just wrote.
        for ticket in tickets:
            self._spawn(self._place(ticket, provider))
        return tickets

    async def _place(self, ticket: DialTicket, provider: TelecomProvider) -> None:
        """Ask the carrier for the call, and handle the three ways it can fail."""
        log = self._log.bind(
            call_id=str(ticket.call_id),
            agent_id=str(ticket.agent_id),
            borrower_id=str(ticket.borrower_id),
            provider=provider.name,
        )
        try:
            ref = await provider.place_call(
                to=ticket.phone,
                from_=self._from,
                idempotency_key=ticket.idempotency_key,
            )
        except ProviderTimeout as exc:
            # DO NOT touch the agent, the borrower or the call. See the module
            # docstring: we do not know whether a phone is ringing, and every
            # available action is wrong if it is. The lease expires and the
            # reaper reconciles against the provider.
            log.warning(
                "place_call_timed_out_leaving_for_reconciliation",
                error=str(exc),
                idempotency_key=ticket.idempotency_key,
            )
            return
        except ProviderUnavailable as exc:
            log.warning("provider_unavailable", error=str(exc))
            await self._abort(
                ticket,
                reason="provider_unavailable",
                spend_attempt=False,
                retry_after=RETRY_AFTER_UNAVAILABLE,
            )
            return
        except ProviderRejected as exc:
            log.info("call_rejected_by_provider", error=str(exc))
            await self._abort(
                ticket,
                reason="provider_rejected",
                spend_attempt=True,
                retry_after=RETRY_AFTER_REJECTED,
            )
            return
        except Exception as exc:  # noqa: BLE001
            # An error we did not anticipate tells us nothing about whether the
            # call was placed, so it is treated exactly like a timeout: change
            # nothing, and let reconciliation find out. Failing closed here
            # means holding a resource we might not need, which is the cheap
            # mistake.
            log.error("place_call_failed_unexpectedly", error=repr(exc))
            return

        async with self._db.transaction() as cur:
            updated = await attach_provider_call_id(
                cur,
                call_id=ticket.call_id,
                provider_call_id=ref.provider_call_id,
                now=self._clock.now(),
            )
        if updated is None:
            # The call already carries a DIFFERENT provider id, which means it
            # was placed twice. Loud, because it is the one outcome the
            # idempotency key exists to prevent.
            log.error(
                "call_already_has_a_different_provider_id",
                provider_call_id=ref.provider_call_id,
            )
            return

        log.info("call_placed", provider_call_id=ref.provider_call_id)

    async def _abort(
        self,
        ticket: DialTicket,
        *,
        reason: str,
        spend_attempt: bool,
        retry_after: float,
    ) -> None:
        """Unwind a call we know was never placed.

        Whether the borrower spends an attempt is the whole decision here. A
        carrier rejecting the number is evidence about the number, so it counts
        and eventually exhausts them. Our own provider being down is evidence
        about us, and charging borrowers for our outage would slowly mark
        perfectly reachable people EXHAUSTED -- a quiet, permanent loss that
        nobody would trace back to an afternoon when a carrier was flaky.
        """
        now = self._clock.now()
        async with self._db.transaction() as cur:
            await terminate_call(
                cur,
                call_id=ticket.call_id,
                target_state=CallState.FAILED,
                now=now,
                worker_id=self._settings.worker_id,
                failure_reason=reason,
            )
            released = await release_agent(
                cur,
                agent_id=ticket.agent_id,
                expected_version=ticket.agent_version,
                expected_state=AgentState.DIALING,
                now=now,
            )
            if released is None:
                # Somebody else moved the agent -- most likely the reaper after
                # a slow provider. Leave it alone rather than forcing a write.
                self._log.warning(
                    "agent_not_released_after_failed_dial",
                    agent_id=str(ticket.agent_id),
                    call_id=str(ticket.call_id),
                )

            if spend_attempt:
                await record_attempt(
                    cur,
                    borrower_id=ticket.borrower_id,
                    now=now,
                    outcome=reason,
                    retry_after_seconds=retry_after,
                )
            else:
                await release_borrower(
                    cur,
                    borrower_id=ticket.borrower_id,
                    expected_version=ticket.borrower_version,
                    now=now,
                    retry_after_seconds=retry_after,
                )

    # -- task lifecycle -------------------------------------------------

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def drain(self) -> None:
        """Wait for in-flight place_call tasks to finish.

        For the simulation and the tests, which need a quiet point to assert
        against. The worker never calls this -- waiting for the carrier is the
        thing the tick loop must not do.
        """
        while True:
            pending = [t for t in self._tasks if not t.done()]
            if not pending:
                return
            done, _ = await asyncio.wait(pending, timeout=0)
            if not done:
                return

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)


def call_is_bound_to_agent(call: Call) -> bool:
    """A call with an agent behind it. The progressive invariant is stated in
    terms of these: over-dial calls have no agent by construction."""
    return call.agent_id is not None
