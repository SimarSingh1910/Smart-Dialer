"""A simulated carrier, and the knobs that make one behave badly.

The brief asks for at least two mock providers that behave differently: one
fast and reliable, one slow with timeouts, duplicate events and reordering.
They share this engine and differ only in a ProviderProfile, for two reasons.

First, "behaves differently" should mean different NUMBERS, not different code.
If the flaky provider were a separate implementation, a test passing against
the fast one would say nothing about the other, because they would not be the
same system under different conditions -- they would be two systems.

Second, the brief asks for these to be config knobs so the simulation can vary
them. Scenario D turns the answer rate down mid-run and Scenario E takes a
provider offline entirely, and neither should need a new class.

Determinism: every random decision about a call is drawn from a generator
seeded with (run seed, idempotency key). Two runs of the same scenario produce
identical behaviour REGARDLESS of the order in which the event loop happens to
schedule the calls -- which a single shared generator would not give us, since
interleaving would change who drew which number.

This module also simulates the borrower: whether they answer, how long the
phone rings first, how long they talk, and how long they will tolerate silence
after saying hello before hanging up. That last one is what turns a pacing
mistake into a measurable abandoned call.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque

from smartdialer.core.clock import Clock
from smartdialer.core.models import NormalisedEvent
from smartdialer.core.runtime import drain_tasks
from smartdialer.providers.base import (
    EventSink,
    ProviderCallRef,
    ProviderCallStatus,
    ProviderError,
    ProviderHealth,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
)


@dataclass(frozen=True)
class ProviderProfile:
    """Everything that distinguishes one carrier from another."""

    name: str

    # --- call setup ---------------------------------------------------
    # Post-dial delay: how long between asking for a call and the phone
    # starting to ring. Log-normal, because it is a latency: bounded below,
    # with a long right tail.
    setup_median_seconds: float = 2.0
    setup_sigma: float = 0.35
    # How long we wait for the provider's API before giving up on it. Hitting
    # this is the ProviderTimeout case, and it is the expensive one.
    client_timeout_seconds: float = 5.0

    # --- reliability --------------------------------------------------
    reject_rate: float = 0.02
    timeout_rate: float = 0.0
    # Of the requests that time out, the fraction where the call WAS actually
    # placed and is now ringing somebody with no record on our side. This is
    # the case the idempotency key exists for, and a provider whose timeouts
    # were always harmless would never exercise it.
    timeout_but_placed_rate: float = 0.5

    # --- bridging -----------------------------------------------------
    # The engineering component of customer wait: dead air after hello. A
    # provider that parks agents in a conference bridges in tens of
    # milliseconds; one that sets up a fresh leg takes closer to a second.
    bridge_min_seconds: float = 0.05
    bridge_max_seconds: float = 0.10

    # --- event delivery misbehaviour ----------------------------------
    duplicate_rate: float = 0.0
    duplicate_max_copies: int = 3
    # Reordering is modelled as what actually causes it: variable delivery
    # latency. An event held back by several seconds naturally arrives after
    # the ones that came later.
    reorder_rate: float = 0.0
    reorder_delay_seconds: float = 12.0
    delayed_rate: float = 0.0
    delayed_seconds: float = 30.0

    # --- the borrower -------------------------------------------------
    answer_rate: float = 0.5
    ring_median_seconds: float = 9.0
    ring_sigma: float = 0.45
    no_answer_ring_seconds: float = 25.0
    talk_median_seconds: float = 120.0
    talk_sigma: float = 0.6
    # How long a borrower who has said hello will hold on before hanging up.
    # The window in which a pacing mistake can still be rescued.
    answer_patience_seconds: float = 2.0

    def replace(self, **changes) -> "ProviderProfile":
        from dataclasses import replace as _replace

        return _replace(self, **changes)


@dataclass
class SimulationControls:
    """The mutable dials the simulation turns mid-run.

    Separate from the profile because the profile describes what a carrier IS
    and these describe what is happening to it right now. Scenario D drops
    answer_rate at t=300s; Scenario E sets outage_until.
    """

    answer_rate: float | None = None
    talk_median_seconds: float | None = None
    # While set and in the future, every request is refused outright.
    outage_until: datetime | None = None
    # Added on top of the profile's own reject rate, for "the provider starts
    # failing" without a full outage.
    extra_reject_rate: float = 0.0


@dataclass
class _SimCall:
    """One call as the carrier sees it. Never leaves this module."""

    provider_call_id: str
    idempotency_key: str
    to: str
    from_: str
    placed_at: datetime
    will_answer: bool
    ring_seconds: float
    talk_seconds: float
    ringing_at: datetime | None = None
    answered_at: datetime | None = None
    connected_at: datetime | None = None
    ended_at: datetime | None = None
    ended_reason: str | None = None
    bridged: bool = False

    @property
    def live(self) -> bool:
        return self.ended_at is None


class SimulatedProvider:
    """A carrier that behaves however its profile says it does."""

    def __init__(
        self,
        profile: ProviderProfile,
        *,
        clock: Clock,
        seed: int = 0,
        event_sink: EventSink | None = None,
    ) -> None:
        self.name = profile.name
        self.profile = profile
        self.controls = SimulationControls()
        self._clock = clock
        self._seed = seed
        self._sink = event_sink
        self._calls: dict[str, _SimCall] = {}
        self._by_key: dict[str, str] = {}
        self._sequence = 0
        # (timestamp, outcome) for the rolling health window.
        self._outcomes: Deque[tuple[datetime, str]] = deque()
        self._setup_times: Deque[tuple[datetime, float]] = deque()
        self._script_tasks: set[asyncio.Task] = set()
        self._delivery_tasks: set[asyncio.Task] = set()
        # Sink failures are recorded rather than raised: a carrier does not
        # crash because our webhook handler did. Tests assert this is empty,
        # so the failures are visible rather than swallowed.
        self.sink_errors: list[BaseException] = []

    def set_event_sink(self, sink: EventSink) -> None:
        self._sink = sink

    # -- randomness -----------------------------------------------------

    def _rng(self, key: str, purpose: str = "") -> random.Random:
        """A generator that depends only on the run seed and the call.

        Not on how the event loop interleaved the calls, which is what makes a
        scenario reproducible when a dozen calls are in flight at once.
        """
        return random.Random(f"{self._seed}:{self.name}:{key}:{purpose}")

    @staticmethod
    def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
        return rng.lognormvariate(math.log(max(median, 1e-6)), sigma)

    # -- health ---------------------------------------------------------

    def _record(self, outcome: str, setup_seconds: float | None = None) -> None:
        now = self._clock.now()
        self._outcomes.append((now, outcome))
        if setup_seconds is not None:
            self._setup_times.append((now, setup_seconds))
        self._trim(now)

    def _trim(self, now: datetime, window_seconds: float = 30.0) -> None:
        cutoff = now - timedelta(seconds=window_seconds)
        while self._outcomes and self._outcomes[0][0] < cutoff:
            self._outcomes.popleft()
        while self._setup_times and self._setup_times[0][0] < cutoff:
            self._setup_times.popleft()

    async def health(self) -> ProviderHealth:
        now = self._clock.now()
        self._trim(now)
        total = len(self._outcomes)
        failures = sum(1 for _, o in self._outcomes if o in ("fail", "timeout"))
        timeouts = sum(1 for _, o in self._outcomes if o == "timeout")
        setups = [s for _, s in self._setup_times]
        return ProviderHealth(
            name=self.name,
            reachable=not self._in_outage(now),
            failure_rate=(failures / total) if total else 0.0,
            timeout_rate=(timeouts / total) if total else 0.0,
            samples=total,
            avg_setup_seconds=(sum(setups) / len(setups)) if setups else 0.0,
        )

    def _in_outage(self, now: datetime) -> bool:
        until = self.controls.outage_until
        return until is not None and now < until

    # -- the interface --------------------------------------------------

    async def place_call(
        self, *, to: str, from_: str, idempotency_key: str
    ) -> ProviderCallRef:
        now = self._clock.now()

        if self._in_outage(now):
            self._record("fail")
            raise ProviderUnavailable(f"{self.name} is not accepting calls")

        # Genuine idempotency. Asking twice with the same key returns the same
        # call rather than placing a second one -- which is the whole reason
        # the key is generated and persisted before we ever get here.
        existing_id = self._by_key.get(idempotency_key)
        if existing_id is not None:
            return ProviderCallRef(
                provider=self.name,
                provider_call_id=existing_id,
                idempotency_key=idempotency_key,
            )

        rng = self._rng(idempotency_key, "place")
        roll = rng.random()
        timeout_rate = self.profile.timeout_rate
        reject_rate = self.profile.reject_rate + self.controls.extra_reject_rate

        if roll < timeout_rate:
            # No response at all. We sit here until our own client timeout
            # fires, which is what makes a slow provider expensive even when it
            # eventually works.
            await self._clock.sleep(self.profile.client_timeout_seconds)
            if rng.random() < self.profile.timeout_but_placed_rate:
                # The worst case, and the one worth building for: the request
                # DID land, and somebody's phone is ringing, but the caller
                # will never learn the call id from this response.
                self._start_call(to=to, from_=from_, idempotency_key=idempotency_key)
            self._record("timeout")
            raise ProviderTimeout(f"{self.name} did not respond in time")

        if roll < timeout_rate + reject_rate:
            await self._clock.sleep(0.05)
            self._record("fail")
            raise ProviderRejected(f"{self.name} rejected the call to {to}")

        setup = self._lognormal(
            rng, self.profile.setup_median_seconds, self.profile.setup_sigma
        )
        await self._clock.sleep(setup)
        call = self._start_call(to=to, from_=from_, idempotency_key=idempotency_key)
        self._record("ok", setup_seconds=setup)
        return ProviderCallRef(
            provider=self.name,
            provider_call_id=call.provider_call_id,
            idempotency_key=idempotency_key,
        )

    async def bridge(self, provider_call_id: str, agent_leg_id: str) -> None:
        call = self._calls.get(provider_call_id)
        if call is None:
            raise ProviderRejected(f"{self.name} has no call {provider_call_id}")
        if not call.live:
            # The borrower hung up while we were deciding. Not our bug, but the
            # caller has to hear about it rather than believe it bridged.
            raise ProviderRejected(f"call {provider_call_id} has already ended")

        rng = self._rng(call.idempotency_key, "bridge")
        await self._clock.sleep(
            rng.uniform(self.profile.bridge_min_seconds, self.profile.bridge_max_seconds)
        )
        if not call.live:
            raise ProviderRejected(f"call {provider_call_id} ended while bridging")
        call.bridged = True
        call.connected_at = self._clock.now()
        await self._emit(call, "bridged", call.connected_at)

    async def hangup(self, provider_call_id: str) -> None:
        call = self._calls.get(provider_call_id)
        if call is None or not call.live:
            return
        await self._end(call, "hangup_by_dialer")

    async def get_call_status(self, provider_call_id: str) -> ProviderCallStatus:
        if self._in_outage(self._clock.now()):
            # A provider that is down cannot tell us what is happening either.
            # This is the case that leaves the reaper holding an agent it
            # cannot safely release.
            raise ProviderTimeout(f"{self.name} is unreachable")
        call = self._calls.get(provider_call_id)
        if call is None:
            return ProviderCallStatus(
                provider_call_id=provider_call_id, live=False, state="unknown"
            )
        facts = {
            column: value
            for column, value in (
                ("ringing_at", call.ringing_at),
                ("answered_at", call.answered_at),
                ("connected_at", call.connected_at),
                ("ended_at", call.ended_at),
            )
            if value is not None
        }
        if not call.live:
            state = "ended"
        elif call.bridged:
            state = "connected"
        elif call.answered_at is not None:
            state = "answered"
        elif call.ringing_at is not None:
            state = "ringing"
        else:
            state = "initiated"
        return ProviderCallStatus(
            provider_call_id=provider_call_id,
            live=call.live,
            state=state,
            facts=facts,
            ended_reason=call.ended_reason,
        )

    async def find_by_idempotency_key(self, key: str) -> ProviderCallRef | None:
        if self._in_outage(self._clock.now()):
            raise ProviderTimeout(f"{self.name} is unreachable")
        provider_call_id = self._by_key.get(key)
        if provider_call_id is None:
            return None
        return ProviderCallRef(
            provider=self.name, provider_call_id=provider_call_id, idempotency_key=key
        )

    # -- the borrower's side --------------------------------------------

    def _call_id_for(self, idempotency_key: str) -> str:
        """A globally unique id for a call, derived from its idempotency key.

        Deliberately NOT a per-process counter. A counter restarts at one when
        the carrier restarts, so a second run reissues ids -- and every event
        id is built from the call id, which means the dialer's deduplication
        would silently discard the new run's events as copies of the old run's.
        That failure is invisible: no error, no duplicate row, just calls that
        stop progressing. Hashing the key gives ids that are stable across a
        replay of the same scenario (so runs stay reproducible) and distinct
        across different ones.
        """
        digest = hashlib.sha1(
            f"{self._seed}:{self.name}:{idempotency_key}".encode()
        ).hexdigest()
        return f"{self.name}-{digest[:16]}"

    def _start_call(self, *, to: str, from_: str, idempotency_key: str) -> _SimCall:
        self._sequence += 1
        rng = self._rng(idempotency_key, "borrower")
        answer_rate = (
            self.controls.answer_rate
            if self.controls.answer_rate is not None
            else self.profile.answer_rate
        )
        talk_median = (
            self.controls.talk_median_seconds
            if self.controls.talk_median_seconds is not None
            else self.profile.talk_median_seconds
        )
        call = _SimCall(
            provider_call_id=self._call_id_for(idempotency_key),
            idempotency_key=idempotency_key,
            to=to,
            from_=from_,
            placed_at=self._clock.now(),
            will_answer=rng.random() < answer_rate,
            ring_seconds=self._lognormal(
                rng, self.profile.ring_median_seconds, self.profile.ring_sigma
            ),
            talk_seconds=self._lognormal(rng, talk_median, self.profile.talk_sigma),
        )
        self._calls[call.provider_call_id] = call
        self._by_key[idempotency_key] = call.provider_call_id
        self._spawn(self._run_call(call), self._script_tasks)
        return call

    async def _run_call(self, call: _SimCall) -> None:
        """The life of one call from the carrier's point of view."""
        try:
            await self._clock.sleep(0.2)
            if not call.live:
                return
            call.ringing_at = self._clock.now()
            await self._emit(call, "ringing", call.ringing_at)

            if not call.will_answer:
                await self._clock.sleep(self.profile.no_answer_ring_seconds)
                if call.live:
                    await self._end(call, "no_answer", event_type="no_answer")
                return

            await self._clock.sleep(call.ring_seconds)
            if not call.live:
                return
            call.answered_at = self._clock.now()
            await self._emit(call, "answered", call.answered_at)

            # The borrower is now listening to silence. Polling rather than an
            # event is deliberate: it keeps this readable, and on a virtual
            # clock a sleeping task is just an entry in a heap.
            #
            # The step is a FRACTION of the wait rather than a fixed tenth of a
            # second. Virtual time is advanced in one jump by tests and by the
            # simulation, and every wake-up in that jump costs a full settle
            # pass -- so a fixed 0.1s poll turns a ten-minute advance into six
            # thousand of them and the run appears to hang. Twenty polls give
            # ample resolution against a patience measured in seconds.
            patience = self.profile.answer_patience_seconds
            waited = 0.0
            step = max(0.05, patience / 20.0)
            while waited < patience:
                if call.bridged or not call.live:
                    break
                await self._clock.sleep(step)
                waited += step

            if not call.live:
                return
            if not call.bridged:
                # Nobody came. This is the abandoned call, from the only point
                # of view that counts.
                await self._end(call, "borrower_hung_up_waiting")
                return

            await self._clock.sleep(call.talk_seconds)
            if call.live:
                await self._end(call, "completed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a mock must not kill the run
            self.sink_errors.append(exc)

    async def _end(
        self, call: _SimCall, reason: str, *, event_type: str = "completed"
    ) -> None:
        call.ended_at = self._clock.now()
        call.ended_reason = reason
        await self._emit(call, event_type, call.ended_at, payload={"reason": reason})

    # -- event delivery, including the bad behaviour --------------------

    async def _emit(
        self,
        call: _SimCall,
        event_type: str,
        ts: datetime,
        payload: dict | None = None,
    ) -> None:
        """Deliver one event, however badly this provider delivers events.

        The event id is derived from the call and the event type, so every copy
        of a duplicated event carries the SAME id. That is the point: a
        duplicate is the same event delivered twice, and a provider that gave
        each copy a fresh id would not be duplicating, it would be lying, and
        no deduplication scheme could save us.
        """
        event = NormalisedEvent(
            provider=self.name,
            provider_event_id=f"{call.provider_call_id}:{event_type}",
            provider_call_id=call.provider_call_id,
            event_type=event_type,
            provider_ts=ts,
            payload={"to": call.to, **(payload or {})},
        )

        rng = self._rng(call.idempotency_key, f"deliver:{event_type}")
        copies = 1
        if rng.random() < self.profile.duplicate_rate:
            copies = rng.randint(2, max(2, self.profile.duplicate_max_copies))

        for copy in range(copies):
            delay = 0.0
            if rng.random() < self.profile.reorder_rate:
                delay += self.profile.reorder_delay_seconds
            if rng.random() < self.profile.delayed_rate:
                delay += self.profile.delayed_seconds
            if copy > 0:
                # Retries of a webhook do not arrive simultaneously.
                delay += rng.uniform(0.1, 1.5)

            if delay <= 0:
                await self._deliver(event)
            else:
                self._spawn(self._deliver_later(event, delay), self._delivery_tasks)

    async def _deliver_later(self, event: NormalisedEvent, delay: float) -> None:
        await self._clock.sleep(delay)
        await self._deliver(event)

    async def _deliver(self, event: NormalisedEvent) -> None:
        if self._sink is None:
            return
        try:
            await self._sink(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.sink_errors.append(exc)

    # -- task lifecycle --------------------------------------------------

    def _spawn(self, coro, registry: set[asyncio.Task]) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        registry.add(task)
        task.add_done_callback(registry.discard)
        return task

    async def drain_deliveries(self) -> None:
        """Wait for event deliveries that are not waiting on the clock.

        Used by tests and the simulation to reach a quiet point. Call scripts
        are deliberately NOT drained: they are supposed to be sitting in the
        middle of a 120-second conversation.
        """
        await drain_tasks(list(self._delivery_tasks))

    async def close(self) -> None:
        for task in list(self._script_tasks | self._delivery_tasks):
            task.cancel()
        pending = list(self._script_tasks | self._delivery_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # -- simulation controls --------------------------------------------

    def set_answer_rate(self, rate: float) -> None:
        self.controls.answer_rate = rate

    def start_outage(self, seconds: float) -> None:
        self.controls.outage_until = self._clock.now() + timedelta(seconds=seconds)

    def end_outage(self) -> None:
        self.controls.outage_until = None

    @property
    def live_calls(self) -> int:
        return sum(1 for call in self._calls.values() if call.live)


def assert_conforms(provider: object) -> None:
    """Cheap structural check used by the conformance tests."""
    for method in (
        "place_call",
        "bridge",
        "get_call_status",
        "find_by_idempotency_key",
        "hangup",
        "health",
    ):
        if not callable(getattr(provider, method, None)):
            raise ProviderError(f"{provider!r} is missing {method}")
