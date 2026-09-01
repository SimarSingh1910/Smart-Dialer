"""The reaper: crash recovery and reconciliation.

Everything in this system that survives a crash survives because of leases and
this loop. A worker that dies stops renewing; the lease expires; the reaper
finds the rows and puts the world back together. There is no distributed
protocol, no leader election and no coordination service -- crash recovery is a
query, which is the strongest argument for keeping the state and the locks in
the same database.

THE ORDER OF THE SWEEP IS THE DESIGN.

Calls are reconciled BEFORE agents are reclaimed, because an agent's fate
follows their call and never the other way round. Reclaiming an agent first
would mean deciding what happened to a person on the basis of a lease timer,
when the carrier is sitting there able to tell us whether their borrower is
still on the line.

FAIL CLOSED, AND SAY WHAT THAT COSTS.

When the carrier cannot be reached, the reaper does nothing: the call stays
where it is, the agent stays reserved, and it tries again later with backoff.
That looks like a bug and is the only defensible behaviour. Releasing an agent
who might be bridged to a live borrower risks handing them a second call while
the first is still up; cancelling a call we cannot ask about risks re-dialling
somebody who is already on the phone with us. So the system holds resources it
might not need, and utilisation drops exactly when things are already going
badly. That is the trade, it is deliberate, and it is the honest answer to
"what are you least confident about?" -- this recovery path depends on the
carrier's status API being both accurate and available.

IDEMPOTENCE.

Running the sweep twice in a row must change nothing the second time, and every
query here is written to make that true rather than to be made true by a flag.
An adopted call gets its provider id, so it is no longer un-adopted. A
reconciled live call gets a fresh lease, so it leaves the worklist until that
expires. A detached borrower has no lease, so the detach query stops seeing
them. A sweep that reported work forever would make "is recovery keeping up?"
unanswerable, which is the question you actually need answered at 3am.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, fields
from datetime import timedelta
from typing import Sequence
from uuid import UUID

from smartdialer.core.clock import Clock
from smartdialer.core.config import Settings
from smartdialer.core.db import Database
from smartdialer.core.logging import StructuredLogger
from smartdialer.core.models import AgentState, Call, CallState
from smartdialer.domain.agents import (
    agents_with_expired_leases,
    expire_stale_heartbeats,
    expire_wrap_up,
    release_agent,
)
from smartdialer.domain.borrowers import (
    detach_expired_leases,
    release_expired_leases,
)
from smartdialer.domain.calls import (
    attach_provider_call_id,
    calls_over_lifetime,
    calls_with_expired_leases,
    extend_call_lease,
    get_call,
    terminate_call,
)
from smartdialer.domain.settlement import settle_call
from smartdialer.domain.snapshot import load_campaign
from smartdialer.providers.base import (
    ProviderError,
    ProviderTimeout,
    TelecomProvider,
)
from smartdialer.workers.bridging import BridgeOutcome, CallBridger


@dataclass
class ReaperReport:
    """What one sweep did. Every field is a count of an action taken.

    Returned rather than only logged so that idempotence is testable as an
    assertion -- run the sweep twice, and the second report must be empty.
    """

    wrap_ups_expired: int = 0
    heartbeats_expired: int = 0
    agents_released: int = 0
    calls_adopted: int = 0
    calls_cancelled: int = 0
    calls_settled: int = 0
    calls_bridged: int = 0
    calls_abandoned: int = 0
    calls_force_failed: int = 0
    borrowers_released: int = 0
    borrowers_detached: int = 0
    # Not a change. Counted separately so an unreachable carrier is visible in
    # the report without making the sweep look like it did work.
    calls_unreachable: int = 0

    @property
    def changes(self) -> int:
        return sum(
            getattr(self, f.name)
            for f in fields(self)
            if f.name != "calls_unreachable"
        )

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class Reaper:
    def __init__(
        self,
        *,
        db: Database,
        clock: Clock,
        campaign_id: UUID,
        providers: Sequence[TelecomProvider],
        settings: Settings,
        logger: StructuredLogger | None = None,
    ) -> None:
        self._db = db
        self._clock = clock
        self._campaign_id = campaign_id
        self._settings = settings
        self._providers = {p.name: p for p in providers}
        self._log = (logger or StructuredLogger("reaper", clock)).bind(
            campaign_id=str(campaign_id), worker_id=settings.worker_id
        )
        self._bridger = CallBridger(
            db=db,
            clock=clock,
            providers=self._providers,
            settings=settings,
            logger=self._log,
            campaign_id=campaign_id,
        )
        self._running = False

    # -- the loop -------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        self._log.info("reaper_started", every=self._settings.reaper_seconds)
        try:
            while self._running:
                try:
                    report = await self.sweep()
                    if report.changes:
                        self._log.info("reaper_swept", **report.as_dict())
                except Exception as exc:  # noqa: BLE001
                    # One bad sweep must not stop recovery. The next pass finds
                    # the same rows -- that is what idempotence is for.
                    self._log.error("reaper_sweep_failed", error=repr(exc))
                await self._clock.sleep(self._settings.reaper_seconds)
        finally:
            self._log.info("reaper_stopped")

    def stop(self) -> None:
        self._running = False

    async def sweep(self) -> ReaperReport:
        """One full pass. Idempotent."""
        report = ReaperReport()
        now = self._clock.now()

        # 1. Deterministic timers first. Wrap-up expiry is not recovery at all
        #    -- the timer was set when the call ended and nothing external can
        #    move it -- so doing it first means freed agents are available to
        #    the reconciliation below rather than a sweep later.
        async with self._db.transaction() as cur:
            report.wrap_ups_expired = len(await expire_wrap_up(cur, now=now))

        # 2. Calls, before agents. An agent's fate follows their call.
        await self._reconcile_expired_calls(report)
        await self._force_close_ancient_calls(report)

        # 3. Agents whose lease expired with no call behind them. Anything
        #    still holding a call was dealt with above, or deliberately left
        #    alone because the carrier could not be reached.
        await self._reclaim_idle_agents(report)

        # 4. Agents who stopped reporting in.
        async with self._db.transaction() as cur:
            report.heartbeats_expired = len(
                await expire_stale_heartbeats(
                    cur,
                    now=self._clock.now(),
                    timeout_seconds=self._settings.heartbeat_timeout_seconds,
                )
            )

        # 5. Borrowers. Last, because whether a borrower is free depends on
        #    whether their call is still live, and steps 2-3 are what decides
        #    that.
        async with self._db.transaction() as cur:
            report.borrowers_released = len(
                await release_expired_leases(cur, now=self._clock.now())
            )
            report.borrowers_detached = len(
                await detach_expired_leases(cur, now=self._clock.now())
            )

        return report

    # -- calls ----------------------------------------------------------

    async def _reconcile_expired_calls(self, report: ReaperReport) -> None:
        async with self._db.transaction() as cur:
            expired = await calls_with_expired_leases(cur, now=self._clock.now())

        for call in expired:
            try:
                await self._reconcile(call, report)
            except Exception as exc:  # noqa: BLE001
                # One unrecoverable call must not stop the others. It keeps its
                # expired lease and comes back on the next sweep.
                self._log.error(
                    "call_reconciliation_failed",
                    call_id=str(call.id),
                    error=repr(exc),
                )

    async def _reconcile(self, call: Call, report: ReaperReport) -> None:
        """Work out what actually happened to one call, and act on it."""
        log = self._log.bind(call_id=str(call.id), provider=call.provider)
        provider = self._providers.get(call.provider)
        if provider is None:
            log.error("no_provider_configured_for_call")
            return

        # A call with no provider id is the orphan case: we wrote the intent,
        # and then either the carrier never heard from us or it did and we
        # never heard back. Only the carrier can tell us which.
        if call.provider_call_id is None:
            adopted = await self._adopt(call, provider, report, log)
            if adopted is None:
                return
            call = adopted

        try:
            status = await provider.get_call_status(call.provider_call_id)
        except (ProviderTimeout, ProviderError) as exc:
            # FAIL CLOSED. We do not know whether a human is on this line, so
            # nothing moves: not the call, not the agent, not the borrower. The
            # lease stays expired so the next sweep tries again.
            report.calls_unreachable += 1
            log.warning("provider_unreachable_leaving_call_alone", error=str(exc))
            return

        if not status.live:
            await self._close_out(call, status, report, log)
            return

        # The call is genuinely still up.
        if call.answered_at is not None or status.state in ("answered", "connected"):
            await self._rescue(call, status, report, log)
        else:
            # Still ringing. Nothing to do but take ownership so this call is
            # not reconciled again on every sweep from here to eternity.
            await self._take_ownership(call)
            log.info("call_still_ringing_lease_extended", state=call.state.value)

    async def _adopt(
        self, call: Call, provider: TelecomProvider, report: ReaperReport, log
    ) -> Call | None:
        """Ask the carrier whether the call we intended actually exists.

        The whole reason the idempotency key is generated and committed before
        place_call. Note that the answer comes from the CARRIER: the key must
        never be turned back into a call id locally, however derivable that
        might look, because a fabricated id would make this path silently
        succeed on calls that were never placed and the orphan case would go
        untested forever.
        """
        try:
            ref = await provider.find_by_idempotency_key(call.idempotency_key)
        except ProviderError as exc:
            report.calls_unreachable += 1
            log.warning("cannot_look_up_idempotency_key", error=str(exc))
            return None

        if ref is None:
            # The carrier has never heard of it. Nothing was placed, nobody's
            # phone is ringing, and the intent row can be closed.
            now = self._clock.now()
            async with self._db.transaction() as cur:
                cancelled = await terminate_call(
                    cur,
                    call_id=call.id,
                    target_state=CallState.CANCELLED,
                    now=now,
                    worker_id=self._settings.worker_id,
                    failure_reason="never_placed_with_provider",
                )
                if cancelled is not None:
                    campaign = await load_campaign(cur, campaign_id=self._campaign_id)
                    await settle_call(
                        cur,
                        call=cancelled,
                        now=now,
                        wrap_up_seconds=campaign.wrap_up_seconds if campaign else 10,
                        log=log,
                    )
            report.calls_cancelled += 1
            log.info("orphan_call_cancelled", idempotency_key=call.idempotency_key)
            return None

        async with self._db.transaction() as cur:
            updated = await attach_provider_call_id(
                cur,
                call_id=call.id,
                provider_call_id=ref.provider_call_id,
                now=self._clock.now(),
            )
        report.calls_adopted += 1
        log.info("orphan_call_adopted", provider_call_id=ref.provider_call_id)
        return updated

    async def _close_out(self, call: Call, status, report: ReaperReport, log) -> None:
        """The carrier says the call is over. Record it and free everybody."""
        now = self._clock.now()
        # A call that reached a human completed; one that never did failed.
        # Derived from the facts rather than from the carrier's vocabulary,
        # which differs between carriers and is not ours to interpret.
        answered = call.answered_at is not None or "answered_at" in status.facts
        target = CallState.COMPLETED if answered else CallState.FAILED

        async with self._db.transaction() as cur:
            closed = await terminate_call(
                cur,
                call_id=call.id,
                target_state=target,
                now=now,
                worker_id=self._settings.worker_id,
                failure_reason=status.ended_reason or "reconciled_as_ended",
            )
            settled = closed or await get_call(cur, call_id=call.id)
            if settled is not None:
                campaign = await load_campaign(cur, campaign_id=self._campaign_id)
                await settle_call(
                    cur,
                    call=settled,
                    now=now,
                    wrap_up_seconds=campaign.wrap_up_seconds if campaign else 10,
                    log=log,
                )
        report.calls_settled += 1
        log.info("call_reconciled_as_ended", outcome=target.value)

    async def _rescue(self, call: Call, status, report: ReaperReport, log) -> None:
        """The crash-after-ANSWERED case, which is the headline one.

        A worker died with a borrower on the line. The carrier confirms they
        are still there, so there are exactly two honest outcomes: get them to
        an agent, or admit we cannot and record an abandon. There is no third
        option where this quietly becomes a FAILED call.
        """
        await self._take_ownership(call)
        fresh = await self._reload(call.id)
        if fresh is None or fresh.is_terminal:
            return

        outcome = await self._bridger.handle_answered(fresh)
        if outcome == BridgeOutcome.CONNECTED:
            report.calls_bridged += 1
            log.info("crashed_call_bridged_to_agent")
        elif outcome == BridgeOutcome.ABANDONED:
            report.calls_abandoned += 1
            log.warning("crashed_call_abandoned_no_agent_free")

    async def _force_close_ancient_calls(self, report: ReaperReport) -> None:
        """The last resort, and it is alarmed.

        A call still non-terminal past max_call_lifetime means reconciliation
        itself has failed -- the carrier is not answering, or is answering with
        something we cannot act on. Closing it is a guess, and because all
        terminal states share rank 9 that guess becomes permanent even if the
        carrier later disagrees. The disagreement is recorded as a
        TERMINAL_CONFLICT on the event, so how often this guess is wrong is a
        number rather than a feeling.
        """
        now = self._clock.now()
        async with self._db.transaction() as cur:
            ancient = await calls_over_lifetime(
                cur,
                now=now,
                max_seconds=self._settings.max_call_lifetime_seconds,
            )

        for call in ancient:
            async with self._db.transaction() as cur:
                failed = await terminate_call(
                    cur,
                    call_id=call.id,
                    target_state=CallState.FAILED,
                    now=now,
                    worker_id=self._settings.worker_id,
                    failure_reason="exceeded_max_call_lifetime",
                )
                if failed is not None:
                    campaign = await load_campaign(cur, campaign_id=self._campaign_id)
                    await settle_call(
                        cur,
                        call=failed,
                        now=now,
                        wrap_up_seconds=campaign.wrap_up_seconds if campaign else 10,
                        log=self._log,
                    )
            report.calls_force_failed += 1
            self._log.error(
                "call_force_failed_past_max_lifetime",
                call_id=str(call.id),
                state=call.state.value,
                created_at=str(call.created_at),
                alert=True,
            )

    # -- agents ---------------------------------------------------------

    async def _reclaim_idle_agents(self, report: ReaperReport) -> None:
        """Agents reserved for a call that never happened.

        The short-lease case. An agent in RESERVED has no call bound to them
        yet, so there is nothing to reconcile and nothing to be careful about:
        the worker that claimed them is gone and they should be dialling for
        somebody else within seconds. DIALING and CONNECTED agents are excluded
        here on purpose -- they have a call, and that call is what decides
        their fate.
        """
        now = self._clock.now()
        async with self._db.transaction() as cur:
            candidates = await agents_with_expired_leases(
                cur, now=now, states=(AgentState.RESERVED,)
            )
            for agent in candidates:
                if agent.current_call_id is not None:
                    call = await get_call(cur, call_id=agent.current_call_id)
                    if call is not None and not call.is_terminal:
                        # Their call is still in flight. Reconciliation owns
                        # them; a lease timer does not get to overrule it.
                        continue
                released = await release_agent(
                    cur,
                    agent_id=agent.id,
                    expected_version=agent.version,
                    expected_state=AgentState.RESERVED,
                    now=now,
                )
                if released is not None:
                    report.agents_released += 1
                    self._log.info(
                        "agent_reclaimed_after_lease_expiry",
                        agent_id=str(agent.id),
                        held_by=agent.lease_owner,
                    )

    # -- helpers --------------------------------------------------------

    async def _take_ownership(self, call: Call) -> None:
        async with self._db.transaction() as cur:
            await extend_call_lease(
                cur,
                call_id=call.id,
                seconds=self._settings.lease_seconds,
                now=self._clock.now(),
                owner=self._settings.worker_id,
            )

    async def _reload(self, call_id: UUID) -> Call | None:
        async with self._db.transaction() as cur:
            return await get_call(cur, call_id=call_id)
