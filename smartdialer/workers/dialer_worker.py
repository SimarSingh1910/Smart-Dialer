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
from typing import Sequence
from uuid import UUID

from smartdialer.allocator.allocator import CallAllocator
from smartdialer.core.clock import Clock
from smartdialer.core.config import Settings
from smartdialer.core.db import Database
from smartdialer.core.logging import StructuredLogger
from smartdialer.core.models import Call, CallState, NormalisedEvent
from smartdialer.domain.agents import expire_wrap_up
from smartdialer.domain.calls import apply_event
from smartdialer.domain.history import CampaignHistory, build_history, empty_history
from smartdialer.domain.snapshot import (
    build_raw_snapshot,
    load_campaign,
    to_pacing_snapshot,
)
from smartdialer.pacing.engine import ProviderHealthSignal, propose
from smartdialer.providers.base import TelecomProvider
from smartdialer.workers.bridging import CallBridger
from smartdialer.safety.controller import ExecutionResult, SafetyController



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

        self._bridger = CallBridger(
            db=db,
            clock=clock,
            providers=self._providers,
            settings=settings,
            logger=self._log,
            campaign_id=campaign_id,
        )

        # The learned distributions, rebuilt on their own schedule. Hazard
        # curves and propensity cells move over minutes; agent counts move
        # every tick. Rebuilding them together would spend most of the tick
        # budget re-deriving a curve that has not changed -- and unlike the
        # snapshot, staleness here is harmless, which is why only the snapshot
        # has a freshness clamp in the safety controller.
        self._history: CampaignHistory | None = None
        self._history_refresh_seconds = 15.0

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
            history = await self._refresh_history(
                cur, campaign=campaign, raw=raw, now=now
            )

        health = await self._health_signal()
        snapshot = to_pacing_snapshot(
            raw,
            campaign=campaign,
            now=self._clock.now(),
            provider_health=health,
            history=history,
            # Granted by the AIMD budget in step 8, where it becomes a ceiling
            # the controller applies. Carried here only so it appears in the
            # decision log; the engine does not spend it.
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
                # The sentence a human reads first, and the numbers behind it.
                # "Why 17 and not 10" is answered by these fields alone.
                "engine_explanation": proposal.explain(),
                "mu_A": proposal.mu_A,
                "sigma_A": proposal.sigma_A,
                "mu_G": proposal.mu_G,
                "sigma_G": proposal.sigma_G,
                "p_hat": proposal.p_hat,
                "epsilon": proposal.epsilon,
                "window_seconds": proposal.window_seconds,
                "changepoint_detected": proposal.changepoint_detected,
                "used_exact_dp": proposal.used_exact_dp,
                "search_trace": [list(pair) for pair in proposal.search_trace],
                "snapshot": _snapshot_for_log(snapshot),
            },
        )

    async def _refresh_history(self, cur, *, campaign, raw, now) -> CampaignHistory:
        """Rebuild the learned distributions if they have gone stale.

        Guarded by age rather than rebuilt every tick. Four extra queries at
        4Hz would dominate the tick budget, and the answer would be the same
        curve each time.
        """
        if (
            self._history is not None
            and self._history.age_seconds(now) < self._history_refresh_seconds
        ):
            return self._history
        try:
            self._history = await build_history(
                cur,
                campaign_id=campaign.id,
                now=now,
                campaign_answer_rate=raw.baseline_rate or raw.recent.answer_rate or 0.2,
                talk_prior_median=max(1.0, raw.avg_call_duration or 120.0),
            )
        except Exception as exc:  # noqa: BLE001
            # A failed rebuild must not stop the tick. The previous tables, or
            # the priors, are still a defensible basis for a decision; no
            # decision at all is not.
            self._log.error("history_refresh_failed", error=repr(exc))
            self._history = self._history or empty_history(now)
        return self._history

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
                await self._bridger.mark_connected(call)
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
        """A human said hello. The bridger owns what happens next.

        Delegated rather than implemented here because the reaper reaches the
        same situation after a crash, and the two must behave identically --
        especially about when an unbridged answer is recorded as an abandon.
        Two copies of that decision would drift towards whichever path was
        exercised more.
        """
        await self._bridger.handle_answered(call)

    async def _settle(self, call: Call) -> None:
        await self._bridger.settle(call)

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




def _snapshot_for_log(snapshot) -> dict:
    """The snapshot, flattened for the decision log.

    Every field the engine saw, so the proposal can be recomputed from the row
    -- the engine being a pure function is what makes that possible, and this
    is what makes it useful. The per-call age arrays are included because they
    are the actual inputs to the hazard model: a log that recorded only "six
    calls ringing" could not explain why six calls justified seventeen new ones
    on one tick and four on the next.
    """
    return {
        "mode": snapshot.mode.value,
        "snapshot_taken_at": snapshot.snapshot_taken_at,
        "age_seconds": snapshot.age_seconds,
        "agents_available": snapshot.agents_available,
        "agents_reserved": snapshot.agents_reserved,
        "agents_dialing": snapshot.agents_dialing,
        "agents_wrap_up": snapshot.agents_wrap_up,
        "agents_offline": snapshot.agents_offline,
        "calls_ringing": list(snapshot.calls_ringing),
        "n_ringing": snapshot.n_ringing,
        "calls_connected": snapshot.calls_connected,
        "calls_in_flight": snapshot.calls_in_flight,
        "historical_answer_rate": snapshot.historical_answer_rate,
        "call_setup_time_p95": snapshot.call_setup_time_p95,
        "call_setup_time_p50": snapshot.call_setup_time_p50,
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
        "talk_seconds": list(snapshot.talk_seconds),
        "wrap_up_remaining": list(snapshot.wrap_up_remaining),
        "candidate_propensities": list(snapshot.candidate_propensities[:20]),
        "observed_answers_30s": snapshot.observed_answers_30s,
        "predicted_answers_30s": snapshot.predicted_answers_30s,
    }
