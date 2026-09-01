"""Getting an answered borrower to an agent, or admitting we cannot.

Shared by the worker and the reaper, because both reach this situation and they
must handle it identically. The worker gets here when a carrier reports a call
answered; the reaper gets here when it reconciles a call whose worker died
mid-answer and finds the borrower still on the line. If each had its own copy,
the abandon accounting would drift between them -- and an abandon the reaper
recorded as a FAILED call is a compliance event that never happened as far as
the numbers are concerned.

The decision is always the same three-way one:

    an agent is bound to the call    -> bridge them
    no agent, but one is free        -> reserve and bridge
    no agent, and none is free       -> ABANDONED, counted, hung up

Nothing here is allowed to reclassify the third case. A dialer that logs its
abandons as failures optimises the metric instead of the behaviour.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Mapping
from uuid import UUID

from smartdialer.core.clock import Clock
from smartdialer.core.config import Settings
from smartdialer.core.db import Database
from smartdialer.core.logging import StructuredLogger
from smartdialer.core.models import AgentState, Call, Campaign
from smartdialer.domain.agents import get_agent, reserve_agents, transition_agent
from smartdialer.domain.calls import abandon_call, connect_call, get_call
from smartdialer.domain.settlement import move_agent, settle_call
from smartdialer.domain.snapshot import load_campaign
from smartdialer.providers.base import ProviderError, TelecomProvider


class BridgeOutcome:
    CONNECTED = "CONNECTED"
    ABANDONED = "ABANDONED"
    # The call ended underneath us -- the borrower hung up before we got
    # there. Not an abandon we caused, and its own terminal handling applies.
    ALREADY_ENDED = "ALREADY_ENDED"
    NO_PROVIDER = "NO_PROVIDER"


class CallBridger:
    def __init__(
        self,
        *,
        db: Database,
        clock: Clock,
        providers: Mapping[str, TelecomProvider],
        settings: Settings,
        logger: StructuredLogger,
        campaign_id: UUID,
    ) -> None:
        self._db = db
        self._clock = clock
        self._providers = dict(providers)
        self._settings = settings
        self._log = logger
        self._campaign_id = campaign_id

    # -- the three-way decision -----------------------------------------

    async def handle_answered(self, call: Call) -> str:
        """A human just said hello. Everything from here is a race against them.

        The borrower's patience is a couple of seconds. That is the entire
        budget for finding an agent, asking the carrier to join the legs, and
        having audio flow -- which is why bridge latency is treated as a
        first-class property of a carrier rather than an implementation detail.
        """
        # The agent bound to this call may no longer be able to take it. After
        # a crash they can have gone OFFLINE on a stale heartbeat, or been
        # reclaimed and given to somebody else. Bridging to them anyway would
        # connect a borrower to an empty seat -- which is worse than an
        # abandon, because it looks like a success in every metric we have.
        if call.agent_id is not None and not await self._agent_can_take_call(call):
            self._log.warning(
                "bound_agent_can_no_longer_take_the_call",
                call_id=str(call.id),
                agent_id=str(call.agent_id),
            )
            call = await self._unbind_agent(call)

        if call.agent_id is None:
            call = await self._find_an_agent(call)
            if call.agent_id is None:
                await self.abandon(call, reason="no_agent_available")
                return BridgeOutcome.ABANDONED

        provider = self._providers.get(call.provider)
        if provider is None or call.provider_call_id is None:
            self._log.error("cannot_bridge_without_provider", call_id=str(call.id))
            return BridgeOutcome.NO_PROVIDER

        try:
            await provider.bridge(call.provider_call_id, str(call.agent_id))
        except ProviderError as exc:
            # The borrower hung up while we were getting to them, or the
            # carrier could not join the legs. Either way a human answered and
            # got nobody, so it is an abandon and it is counted as one. Calling
            # it a FAILED call would be more comfortable and would understate
            # the number the campaign is actually judged on.
            self._log.warning("bridge_failed", call_id=str(call.id), error=str(exc))
            await self.abandon(call, reason="bridge_failed")
            return BridgeOutcome.ABANDONED

        return await self.mark_connected(call)

    async def _agent_can_take_call(self, call: Call) -> bool:
        """Is the agent bound to this call still in a position to answer it?

        They must be in a state that expects a call and still pointing at THIS
        one. An agent who has been reclaimed and reserved for somebody else
        fails both tests, and taking them would drop that other call to rescue
        this one.
        """
        async with self._db.transaction() as cur:
            agent = await get_agent(cur, agent_id=call.agent_id)
        if agent is None:
            return False
        if agent.state not in (
            AgentState.RESERVED,
            AgentState.DIALING,
            AgentState.CONNECTED,
        ):
            return False
        return agent.current_call_id == call.id

    async def _unbind_agent(self, call: Call) -> Call:
        """Detach an unusable agent so the call can look for another.

        Guarded on the agent id we read, so a concurrent rescue that has
        already rebound the call is not undone by this one.
        """
        async with self._db.transaction() as cur:
            await cur.execute(
                "UPDATE calls SET agent_id = NULL "
                "WHERE id = %(call_id)s AND agent_id = %(agent_id)s",
                {"call_id": call.id, "agent_id": call.agent_id},
            )
            refreshed = await get_call(cur, call_id=call.id)
        return refreshed or call

    async def _find_an_agent(self, call: Call) -> Call:
        """One last look for a free agent, for a call that has none.

        Only over-dial calls and crash-recovered ones get here. Progressive
        calls always have an agent already, which is exactly what progressive
        mode buys.
        """
        now = self._clock.now()
        async with self._db.transaction() as cur:
            agents = await reserve_agents(
                cur,
                campaign_id=call.campaign_id,
                worker_id=self._settings.worker_id,
                n=1,
                lease_seconds=self._settings.lease_seconds,
                now=now,
            )
            if not agents:
                return call
            agent = agents[0]
            await cur.execute(
                "UPDATE calls SET agent_id = %(agent_id)s WHERE id = %(call_id)s "
                "AND agent_id IS NULL",
                {"agent_id": agent.agent_id, "call_id": call.id},
            )
            moved = await transition_agent(
                cur,
                agent_id=agent.agent_id,
                expected_version=agent.version,
                expected_state=AgentState.RESERVED,
                target_state=AgentState.DIALING,
                now=now,
                current_call_id=call.id,
                lease_expires_at=now
                + timedelta(seconds=self._settings.lease_seconds),
            )
            if moved is None:
                return call
            refreshed = await get_call(cur, call_id=call.id)
        return refreshed or call

    async def mark_connected(self, call: Call) -> str:
        """Put the call and its agent into CONNECTED, from either direction.

        Two things reach here: our own bridge() returning, and the carrier's
        "bridged" webhook. Both are the same fact learned twice, so this is
        idempotent -- and connect_call() returning None does NOT mean the call
        died. It means the call had already reached at least CONNECTED, which
        is the normal outcome when the other path won the race. Reading that as
        failure is what once left agents in DIALING through whole conversations.
        """
        now = self._clock.now()
        async with self._db.transaction() as cur:
            connected = await connect_call(
                cur, call_id=call.id, now=now, worker_id=self._settings.worker_id
            )
            if connected is None:
                current = await get_call(cur, call_id=call.id)
                if current is None or current.is_terminal:
                    self._log.warning(
                        "call_ended_before_connect", call_id=str(call.id)
                    )
                    return BridgeOutcome.ALREADY_ENDED
                connected = current

            if connected.agent_id is not None:
                await move_agent(
                    cur,
                    agent_id=connected.agent_id,
                    expected=(AgentState.DIALING, AgentState.RESERVED),
                    target=AgentState.CONNECTED,
                    now=now,
                    log=self._log,
                    current_call_id=connected.id,
                    lease_expires_at=now
                    + timedelta(seconds=self._settings.lease_seconds),
                )
        self._log.info(
            "call_connected",
            call_id=str(call.id),
            agent_id=str(connected.agent_id) if connected.agent_id else None,
            wait_ms=connected.wait_ms,
        )
        return BridgeOutcome.CONNECTED

    async def abandon(self, call: Call, *, reason: str) -> None:
        """Record an abandoned call, then hang up on the borrower.

        The record comes FIRST and the hangup's failure is not allowed to
        prevent it. If the carrier will not take the request we have still
        abandoned the call, and a compliance event going uncounted because a
        hangup timed out is the worst possible way to lose one.
        """
        now = self._clock.now()
        async with self._db.transaction() as cur:
            abandoned = await abandon_call(
                cur,
                call_id=call.id,
                now=now,
                worker_id=self._settings.worker_id,
                reason=reason,
            )
        self._log.warning(
            "call_abandoned",
            call_id=str(call.id),
            borrower_id=str(call.borrower_id),
            agent_id=str(call.agent_id) if call.agent_id else None,
            reason=reason,
        )

        provider = self._providers.get(call.provider)
        if provider is not None and call.provider_call_id is not None:
            try:
                await provider.hangup(call.provider_call_id)
            except ProviderError as exc:
                self._log.warning(
                    "hangup_failed_after_abandon",
                    call_id=str(call.id),
                    error=str(exc),
                )

        if abandoned is not None:
            await self.settle(abandoned)

    async def settle(self, call: Call) -> None:
        """Finish a call: release the agent, decide the borrower's future."""
        now = self._clock.now()
        async with self._db.transaction() as cur:
            campaign: Campaign | None = await load_campaign(
                cur, campaign_id=self._campaign_id
            )
            await settle_call(
                cur,
                call=call,
                now=now,
                wrap_up_seconds=campaign.wrap_up_seconds if campaign else 10,
                log=self._log,
            )
