"""The dialer worker: one tick loop, and one event handler.

The tick loop is the mandated pipeline and nothing else:

    campaign -> snapshot -> pacing engine -> safety controller -> allocator
                                                               -> provider

Each arrow is a plain function call and each stage is testable on its own. The
worker's job is to run that sequence every 250ms and to write down what
happened; it makes no pacing decisions of its own, which is why there is no
arithmetic in tick().

The event handler is where policy lives. domain/calls.py decides what an event
MEANS -- it moves the call, absorbs the timestamps, counts the outcome -- and
then this module decides what to DO about it: bridge the borrower to their
agent, put the agent into wrap-up, give the borrower a retry, or record that
somebody said hello and nobody was there.

Ticks and events run concurrently and neither holds a lock on the other. They
do not need to: everything they touch is guarded in the database by the
patterns in agents.py and calls.py, so the worst a race can do is make a
compare-and-swap miss, which is a logged non-event.

Why 250ms: small batches at a high tick rate place the same number of calls as
large batches at a low one, but with materially lower simultaneity. Ten calls
that ring at the same instant can overshoot the agent pool; ten spread over a
second mostly cannot. Nothing in the statistics changes, and the customer wait
does.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Sequence
from uuid import UUID

from smartdialer.allocator.allocator import CallAllocator
from smartdialer.core.clock import Clock
from smartdialer.core.config import Settings
from smartdialer.core.db import Database
from smartdialer.core.logging import StructuredLogger
from smartdialer.core.models import (
    AgentState,
    Call,
    CallState,
    Campaign,
    NormalisedEvent,
)
from smartdialer.domain.agents import (
    expire_wrap_up,
    get_agent,
    release_agent,
    reserve_agents,
    transition_agent,
)
from smartdialer.domain.borrowers import mark_done, record_attempt
from smartdialer.domain.calls import (
    abandon_call,
    apply_event,
    connect_call,
    get_call,
)
from smartdialer.domain.snapshot import (
    build_raw_snapshot,
    load_campaign,
    to_pacing_snapshot,
)
from smartdialer.pacing.engine import ProviderHealthSignal, propose
from smartdialer.providers.base import ProviderError, TelecomProvider
from smartdialer.safety.controller import ExecutionResult, SafetyController

# How long before a borrower who was not reached is dialled again.
RETRY_AFTER_NO_ANSWER = 900.0
RETRY_AFTER_FAILED = 600.0
RETRY_AFTER_ABANDONED = 1800.0


class DialerWorker:
    """One worker process's share of one campaign.

    Several of these run against the same campaign and the same database. They
    coordinate through nothing but Postgres -- no leader, no lock service, no
    shared cache. Two workers reaching for the same agent is resolved by
    SELECT ... FOR UPDATE SKIP LOCKED inside the allocator, and the question
    "which source of truth wins?" has no answer here because there is only one.
    """

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
        self._log = (logger or StructuredLogger("dialer", clock)).bind(
            campaign_id=str(campaign_id), worker_id=settings.worker_id
        )
        self._providers = {p.name: p for p in providers}

        self.allocator = CallAllocator(
            db=db,
            clock=clock,
            providers=list(providers),
            settings=settings,
            logger=self._log,
        )
        self.controller = SafetyController(
            allocator=self.allocator,
            db=db,
            clock=clock,
            logger=self._log,
            max_signal_age_seconds=settings.max_signal_age_seconds,
        )

        self._running = False
        self._event_tasks: set[asyncio.Task] = set()
        # Surfaced rather than swallowed: a handler that raised has left some
        # call in a state nobody reconciled, and the tests assert this is empty.
        self.event_errors: list[BaseException] = []

    # -- wiring ---------------------------------------------------------

    def attach_providers(self) -> None:
        """Point every carrier's event stream at this worker.

        In production the carriers post webhooks to the API and the ingester
        calls handle_event; in the simulation they call it directly. Same
        function either way, which is what makes the simulation a test of the
        real path rather than of a parallel one.
        """
        for provider in self._providers.values():
            setter = getattr(provider, "set_event_sink", None)
            if setter is not None:
                setter(self.handle_event)

    # -- the tick loop --------------------------------------------------

    async def run(self) -> None:
        self._running = True
        self.attach_providers()
        self._log.info("worker_started", tick_seconds=self._settings.tick_seconds)
        try:
            while self._running:
                try:
                    await self.tick()
                except Exception as exc:  # noqa: BLE001
                    # One bad tick must not stop the loop. The controller
                    # already fails closed inside itself, so reaching here
                    # means the snapshot or the decision log failed, and the
                    # right response is to skip this tick and try again.
                    self._log.error("tick_failed", error=repr(exc))
                await self._clock.sleep(self._settings.tick_seconds)
        finally:
            self._log.info("worker_stopped")

    def stop(self) -> None:
        self._running = False

    async def tick(self) -> ExecutionResult:
        """One pass through the pipeline. Returns what the controller decided
        and what came of it."""
        now = self._clock.now()

        async with self._db.transaction() as cur:
            campaign = await load_campaign(cur, campaign_id=self._campaign_id)
            if campaign is None:
                raise RuntimeError(f"campaign {self._campaign_id} does not exist")

            # Wrap-up expiry lives here until the reaper takes it over in step
            # 6. It is a deterministic timer rather than a recovery action, so
            # running it on the tick keeps freed agents visible to the very
            # snapshot that decides how many calls to place.
            await expire_wrap_up(cur, now=now)

            raw = await build_raw_snapshot(
                cur, campaign_id=campaign.id, now=now, window_seconds=60.0
            )

        health = await self._health_signal()
        snapshot = to_pacing_snapshot(
            raw,
            campaign=campaign,
            now=self._clock.now(),
            provider_health=health,
            # Granted by the AIMD budget in step 8. Zero means the predictive
            # path proposes the progressive floor, which is the safe default.
            overdial_credit=0,
        )

        proposal = propose(snapshot)

        # The controller writes the decision row itself, because it is the only
        # component that knows all three numbers: what was proposed, what
        # survived the clamps, and what actually started.
        return await self.controller.execute(
            proposal=proposal,
            snapshot=snapshot,
            campaign=campaign,
            ts=now,
            log_inputs={
                "engine_reason": proposal.reason,
                "engine_terms": proposal.terms,
                "snapshot": _snapshot_for_log(snapshot),
            },
        )

    async def _health_signal(self) -> ProviderHealthSignal:
        """Reduce the carriers' health to the plain numbers the engine takes.

        Worst-of across providers, and a carrier that cannot even answer a
        health question counts as unreachable rather than as unknown. The
        conversion happens here so that `pacing` never imports `providers`.
        """
        worst: ProviderHealthSignal | None = None
        for provider in self._providers.values():
            try:
                health = await provider.health()
                signal = ProviderHealthSignal(
                    name=health.name,
                    reachable=health.reachable,
                    failure_rate=health.failure_rate,
                    timeout_rate=health.timeout_rate,
                    avg_setup_seconds=health.avg_setup_seconds,
                    samples=health.samples,
                )
            except Exception:  # noqa: BLE001
                signal = ProviderHealthSignal(
                    name=provider.name, reachable=False, failure_rate=1.0
                )
            if worst is None or (signal.failure_rate, not signal.reachable) > (
                worst.failure_rate,
                not worst.reachable,
            ):
                worst = signal
        return worst or ProviderHealthSignal()

    # -- events ---------------------------------------------------------

    async def handle_event(self, event: NormalisedEvent) -> None:
        """Apply one provider event and act on what it turned out to mean.

        Applying and acting are separated on purpose. The apply step is
        idempotent and order-independent and lives in domain/calls.py; the act
        step runs ONLY when the apply step reports a real transition, so a
        provider that delivers ANSWERED three times bridges the call once.
        """
        try:
            now = self._clock.now()
            async with self._db.transaction() as cur:
                application = await apply_event(
                    cur, event=event, now=now, worker_id=self._settings.worker_id
                )

            if not application.transitioned or application.call is None:
                return

            call = application.call
            if application.new_state is CallState.ANSWERED:
                await self._on_answered(call, application.needs_agent)
            elif application.new_state is CallState.CONNECTED:
                # The carrier's own confirmation of the bridge. It races our
                # connect_call and either one may land first, so both funnel
                # into the same idempotent step rather than assuming ours won.
                await self._mark_connected(call)
            elif call.is_terminal:
                await self._settle(call)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.event_errors.append(exc)
            self._log.error(
                "event_handler_failed",
                error=repr(exc),
                provider_event_id=event.provider_event_id,
            )

    async def _on_answered(self, call: Call, needs_agent: bool) -> None:
        """A human just said hello. Everything from here is a race against them.

        Two paths. A progressive call already has an agent dialling for it, so
        this is a bridge. An over-dial call has nobody, so we look for a free
        agent and, failing that, admit the abandon.
        """
        if needs_agent:
            await self._rescue_or_abandon(call)
            return

        provider = self._providers.get(call.provider)
        if provider is None or call.provider_call_id is None:
            self._log.error("cannot_bridge_without_provider", call_id=str(call.id))
            return

        try:
            await provider.bridge(call.provider_call_id, str(call.agent_id))
        except ProviderError as exc:
            # The borrower hung up while we were getting to them, or the
            # carrier could not join the legs. Either way a human answered and
            # got nobody, so it is an abandon and it is counted as one. Calling
            # this a FAILED call would be more comfortable and would understate
            # the number the campaign is actually judged on.
            self._log.warning(
                "bridge_failed", call_id=str(call.id), error=str(exc)
            )
            await self._abandon(call, reason="bridge_failed")
            return

        await self._mark_connected(call)

    async def _mark_connected(self, call: Call) -> None:
        """Put the call and its agent into CONNECTED, from either direction.

        Two things can get here: our own bridge() returning, and the carrier's
        "bridged" webhook. Both are the same fact learned twice, so this has to
        be idempotent -- and, importantly, connect_call() returning None does
        NOT mean the call died. It means the call had already reached at least
        CONNECTED, which is the normal outcome when the other path won the
        race. Reading that as failure is what left agents in DIALING through a
        whole conversation.
        """
        now = self._clock.now()
        async with self._db.transaction() as cur:
            connected = await connect_call(
                cur, call_id=call.id, now=now, worker_id=self._settings.worker_id
            )
            if connected is None:
                current = await get_call(cur, call_id=call.id)
                if current is None or current.is_terminal:
                    # This one really did end -- the borrower hung up between
                    # the bridge and this write. The terminal handler cleans
                    # the agent up; do not fight it.
                    self._log.warning("call_ended_before_connect", call_id=str(call.id))
                    return
                connected = current

            if connected.agent_id is not None:
                await self._move_agent(
                    cur,
                    agent_id=connected.agent_id,
                    expected=(AgentState.DIALING,),
                    target=AgentState.CONNECTED,
                    now=now,
                    current_call_id=connected.id,
                )
        self._log.info(
            "call_connected", call_id=str(call.id), agent_id=str(connected.agent_id)
        )

    async def _rescue_or_abandon(self, call: Call) -> None:
        """An over-dial call was answered with no agent bound to it.

        Never happens in progressive mode -- there is no such call. In
        predictive mode it is the moment the bet either pays or costs: one last
        look for a free agent, and if there is none, the borrower is dropped
        and it goes in the abandon column.

        Step 8 improves the losing branch with a bounded couple of seconds of
        hold and an offer to call back, which converts some abandons into a
        worse-but-not-terrible experience. The accounting does not change.
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
                rescued = None
            else:
                agent = agents[0]
                await cur.execute(
                    "UPDATE calls SET agent_id = %(agent_id)s WHERE id = %(call_id)s",
                    {"agent_id": agent.agent_id, "call_id": call.id},
                )
                rescued = await transition_agent(
                    cur,
                    agent_id=agent.agent_id,
                    expected_version=agent.version,
                    expected_state=AgentState.RESERVED,
                    target_state=AgentState.DIALING,
                    now=now,
                    current_call_id=call.id,
                )

        if rescued is None:
            await self._abandon(call, reason="no_agent_available")
            return

        refreshed = await self._reload(call.id)
        if refreshed is not None:
            await self._on_answered(refreshed, needs_agent=False)

    async def _abandon(self, call: Call, *, reason: str) -> None:
        """Record an abandoned call, and hang up on the borrower.

        The hangup comes second and its failure is not allowed to prevent the
        record. If the carrier will not take the request we still abandoned the
        call, and a compliance event that goes uncounted because a hangup
        timed out is the worst possible way to lose one.
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
            await self._settle(abandoned)

    async def _settle(self, call: Call) -> None:
        """A call is over. Release the agent and decide the borrower's future.

        The agent's next state depends on where they were, not on why the call
        ended: somebody who was talking has notes to write and goes to WRAP_UP;
        somebody who was only dialling goes straight back to the pool.

        The borrower is DONE if they actually spoke to an agent, and otherwise
        spends an attempt and comes back after a backoff. `connected_at` is the
        test rather than the call's final state, because a call that connected
        and then failed still reached the person.
        """
        now = self._clock.now()
        async with self._db.transaction() as cur:
            if call.agent_id is not None:
                agent = await get_agent(cur, agent_id=call.agent_id)
                if agent is not None and agent.current_call_id == call.id:
                    if agent.state is AgentState.CONNECTED:
                        await self._move_agent(
                            cur,
                            agent_id=agent.id,
                            expected=(AgentState.CONNECTED,),
                            target=AgentState.WRAP_UP,
                            now=now,
                            wrap_up_ends_at=now
                            + timedelta(seconds=await self._wrap_up_seconds(cur)),
                        )
                    elif agent.state in (AgentState.DIALING, AgentState.RESERVED):
                        await release_agent(
                            cur,
                            agent_id=agent.id,
                            expected_version=agent.version,
                            expected_state=agent.state,
                            now=now,
                        )

            if call.connected_at is not None:
                await mark_done(
                    cur, borrower_id=call.borrower_id, outcome=call.state.value
                )
            else:
                await record_attempt(
                    cur,
                    borrower_id=call.borrower_id,
                    now=now,
                    outcome=call.failure_reason or call.state.value,
                    retry_after_seconds=_retry_after(call),
                )

    async def _wrap_up_seconds(self, cur) -> int:
        campaign: Campaign | None = await load_campaign(cur, campaign_id=self._campaign_id)
        return campaign.wrap_up_seconds if campaign else 10

    async def _move_agent(
        self,
        cur,
        *,
        agent_id: UUID,
        expected: tuple[AgentState, ...],
        target: AgentState,
        now,
        **columns,
    ) -> bool:
        """Read an agent, then compare-and-swap it once.

        Events arrive knowing about a call, not about an agent's row version,
        so there has to be a read first. If the swap misses, something else
        moved the agent between the two and we do NOT retry: forcing the write
        is how two workers end up believing they own one agent. It is logged
        and left to the reaper, which is the component whose job that is.
        """
        agent = await get_agent(cur, agent_id=agent_id)
        if agent is not None and agent.state is target:
            # Already there. The other side of a race got here first and did
            # exactly what we were about to do, which is a success, not a
            # conflict worth logging.
            return True
        if agent is None or agent.state not in expected:
            self._log.warning(
                "agent_not_in_expected_state",
                agent_id=str(agent_id),
                state=agent.state.value if agent else None,
                expected=[state.value for state in expected],
                target=target.value,
            )
            return False
        moved = await transition_agent(
            cur,
            agent_id=agent_id,
            expected_version=agent.version,
            expected_state=agent.state,
            target_state=target,
            now=now,
            **columns,
        )
        if moved is None:
            self._log.warning(
                "agent_transition_lost_a_race",
                agent_id=str(agent_id),
                target=target.value,
            )
            return False
        return True

    async def _reload(self, call_id: UUID) -> Call | None:
        async with self._db.transaction() as cur:
            return await get_call(cur, call_id=call_id)

    # -- lifecycle ------------------------------------------------------

    async def drain(self) -> None:
        """Reach a quiet point. For the simulation and the tests only."""
        await self.allocator.drain()
        for provider in self._providers.values():
            drainer = getattr(provider, "drain_deliveries", None)
            if drainer is not None:
                await drainer()

    async def close(self) -> None:
        self.stop()
        await self.allocator.close()
        for task in list(self._event_tasks):
            task.cancel()


def _retry_after(call: Call) -> float:
    if call.state is CallState.ABANDONED:
        # A borrower we dropped waits longest. Ringing somebody back promptly
        # after hanging up on them is how a compliance problem becomes a
        # complaint.
        return RETRY_AFTER_ABANDONED
    if call.answered_at is not None:
        return RETRY_AFTER_FAILED
    return RETRY_AFTER_NO_ANSWER


def _snapshot_for_log(snapshot) -> dict:
    """The snapshot, flattened for the decision log.

    Every field the engine saw, so the proposal can be recomputed from the row.
    The per-call age arrays are included: they are the inputs the step 7 hazard
    model will run on, and a log that omitted them would not be able to explain
    a predictive decision at all.
    """
    return {
        "mode": snapshot.mode.value,
        "taken_at": snapshot.taken_at,
        "age_seconds": snapshot.age_seconds,
        "agents_available": snapshot.agents_available,
        "agents_reserved": snapshot.agents_reserved,
        "agents_dialing": snapshot.agents_dialing,
        "agents_wrap_up": snapshot.agents_wrap_up,
        "agents_offline": snapshot.agents_offline,
        "calls_ringing": snapshot.calls_ringing,
        "calls_connected": snapshot.calls_connected,
        "calls_in_flight": snapshot.calls_in_flight,
        "historical_answer_rate": snapshot.historical_answer_rate,
        "call_setup_time_p95": snapshot.call_setup_time_p95,
        "avg_call_duration": snapshot.avg_call_duration,
        "provider_health": {
            "name": snapshot.provider_health.name,
            "reachable": snapshot.provider_health.reachable,
            "failure_rate": snapshot.provider_health.failure_rate,
            "timeout_rate": snapshot.provider_health.timeout_rate,
        },
        "recent_campaign_behaviour": {
            "initiated": snapshot.recent_campaign_behaviour.initiated,
            "answered": snapshot.recent_campaign_behaviour.answered,
            "connected": snapshot.recent_campaign_behaviour.connected,
            "abandoned": snapshot.recent_campaign_behaviour.abandoned,
            "failed": snapshot.recent_campaign_behaviour.failed,
        },
        "ring_seconds": list(snapshot.ring_seconds),
        "talk_seconds": list(snapshot.talk_seconds),
        "wrap_up_remaining": list(snapshot.wrap_up_remaining),
    }
