"""Provider tests.

Two halves, and the split is the point.

The conformance half runs the SAME assertions against both providers. If a
behaviour is part of the interface, it has to hold for the reliable carrier and
the terrible one alike -- that is what makes it an interface rather than a
description of provider A.

The misbehaviour half proves provider B actually is terrible. A "flaky" mock
that never produces a duplicate would let the whole event-handling design go
untested while looking thoroughly covered, so these tests fail if B starts
behaving.

Everything runs on the VirtualClock. Provider B's timeout path waits 8 seconds
and its delayed events arrive 30 seconds late; on a real clock this file would
take minutes and nobody would run it.
"""

from __future__ import annotations

import asyncio

import pytest

from smartdialer.core.clock import VirtualClock
from smartdialer.core.models import NormalisedEvent
from smartdialer.providers.base import (
    ProviderError,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
    TelecomProvider,
)
from smartdialer.providers.mock_fast import make_fast_provider
from smartdialer.providers.mock_flaky import make_flaky_provider
from smartdialer.providers.simulated import assert_conforms

FROM = "+911140000000"


class Collector:
    """Stands in for the event ingest path. Records order and duplicates."""

    def __init__(self) -> None:
        self.events: list[NormalisedEvent] = []

    async def __call__(self, event: NormalisedEvent) -> None:
        self.events.append(event)

    def types(self, provider_call_id: str | None = None) -> list[str]:
        return [
            e.event_type
            for e in self.events
            if provider_call_id is None or e.provider_call_id == provider_call_id
        ]

    def ids(self) -> list[str]:
        return [e.provider_event_id for e in self.events]


def make(kind: str, clock: VirtualClock, sink, *, seed: int = 1, **overrides):
    factory = make_fast_provider if kind == "fast" else make_flaky_provider
    return factory(clock, seed=seed, event_sink=sink, **overrides)


# ---------------------------------------------------------------------------
# Conformance: identical assertions, both providers
# ---------------------------------------------------------------------------


@pytest.fixture(params=["fast", "flaky"])
def kind(request) -> str:
    return request.param


async def test_provider_interface_conformance(kind: str):
    """Both satisfy the protocol, structurally and by isinstance."""
    clock = VirtualClock()
    provider = make(kind, clock, Collector())
    assert_conforms(provider)
    assert isinstance(provider, TelecomProvider)
    assert provider.name in ("mock_fast", "mock_flaky")


async def test_placing_a_call_eventually_rings_the_borrower(kind: str):
    """Whatever the carrier, a placed call reaches the borrower and reports it.

    Seeded so neither provider draws its rejection path here; the failure
    behaviours have tests of their own below.
    """
    clock = VirtualClock()
    sink = Collector()
    provider = make(kind, clock, sink, seed=7)

    ref = await place(provider, clock, "+919000000001", "k-ring")
    assert ref is not None
    assert ref.idempotency_key.startswith("k-ring")
    assert ref.provider == provider.name

    await clock.advance(2.0)
    assert "ringing" in sink.types(ref.provider_call_id)
    await provider.close()


async def place(provider, clock: VirtualClock, to: str, key: str):
    """Place a call while driving the clock, retrying on failure.

    Two things are going on here, and both are what a real caller has to do.

    place_call sleeps for the carrier's setup time, so on a virtual clock it
    cannot complete unless somebody advances time underneath it: the request
    runs as a task and the clock moves on without it.

    And provider B rejects or times out a fair share of requests, so a test
    that wants a live call has to retry -- with a NEW idempotency key each
    time, because reusing the old one would return the same failed attempt.
    The winning key comes back on the ref, so callers use `ref.idempotency_key`
    rather than assuming the first one was the one that worked.
    """
    for attempt in range(12):
        attempt_key = key if attempt == 0 else f"{key}-r{attempt}"
        task = asyncio.ensure_future(
            provider.place_call(to=to, from_=FROM, idempotency_key=attempt_key)
        )
        for _ in range(40):
            if task.done():
                break
            await clock.advance(1.0)
        try:
            return await task
        except ProviderError:
            continue
    raise AssertionError(f"{provider.name} refused 12 attempts to place a call")


async def test_the_same_idempotency_key_never_places_two_calls(kind: str):
    """The guarantee the whole crash-recovery story rests on.

    A worker that dies after the provider took the call, restarts, and asks
    again with the same key must get the SAME call back -- not a second one
    ringing the same borrower about the same debt.
    """
    clock = VirtualClock()
    provider = make(kind, clock, Collector(), seed=7)

    first = await place(provider, clock, "+919000000002", "k-same")
    # Asking again with the key that worked. No clock driving needed: a
    # provider honouring the key answers from what it already has rather than
    # setting up a second call.
    second = await provider.place_call(
        to="+919000000002", from_=FROM, idempotency_key=first.idempotency_key
    )

    assert first.provider_call_id == second.provider_call_id
    await provider.close()


async def test_a_placed_call_can_be_found_by_its_idempotency_key(kind: str):
    clock = VirtualClock()
    provider = make(kind, clock, Collector(), seed=7)

    ref = await place(provider, clock, "+919000000003", "k-find")
    found = await provider.find_by_idempotency_key(ref.idempotency_key)
    missing = await provider.find_by_idempotency_key("k-never-used")

    assert found is not None and found.provider_call_id == ref.provider_call_id
    assert missing is None
    await provider.close()


async def test_status_reports_a_call_as_live_then_ended(kind: str):
    """The reconciliation primitive. `live` has to be trustworthy on both."""
    clock = VirtualClock()
    provider = make(kind, clock, Collector(), seed=7, answer_rate=1.0)

    ref = await place(provider, clock, "+919000000004", "k-status")
    await clock.advance(1.0)
    assert (await provider.get_call_status(ref.provider_call_id)).live is True

    await provider.hangup(ref.provider_call_id)
    status = await provider.get_call_status(ref.provider_call_id)
    assert status.live is False
    assert status.state == "ended"
    await provider.close()


async def test_status_of_an_unknown_call_is_not_an_error(kind: str):
    """Asking about a call the carrier has never heard of is a normal thing
    for recovery to do, and must not raise."""
    clock = VirtualClock()
    provider = make(kind, clock, Collector())
    status = await provider.get_call_status("no-such-call")
    assert status.live is False
    assert status.state == "unknown"
    await provider.close()


async def test_bridging_a_dead_call_is_refused_not_silently_ignored(kind: str):
    """A borrower who hung up while we were choosing an agent.

    The caller has to learn that the bridge did not happen; a silent success
    would leave an agent believing they are on a call with nobody.
    """
    clock = VirtualClock()
    provider = make(kind, clock, Collector(), seed=7, answer_rate=1.0)
    ref = await place(provider, clock, "+919000000005", "k-dead")
    await provider.hangup(ref.provider_call_id)

    with pytest.raises(ProviderRejected):
        await provider.bridge(ref.provider_call_id, "agent-leg-1")
    await provider.close()


async def test_an_answered_call_that_is_bridged_gets_connected(kind: str):
    """The happy path end to end, on both carriers: ringing, answered, bridged,
    and a conversation that ends."""
    clock = VirtualClock()
    sink = Collector()
    provider = make(
        kind, clock, sink, seed=7, answer_rate=1.0, talk_median_seconds=30.0
    )
    ref = await place(provider, clock, "+919000000006", "k-happy")

    # Wait for the answer, then bridge promptly, as a healthy dialer would.
    for _ in range(60):
        await clock.advance(1.0)
        await provider.drain_deliveries()
        if "answered" in sink.types(ref.provider_call_id):
            break
    assert "answered" in sink.types(ref.provider_call_id)

    bridge = asyncio.ensure_future(provider.bridge(ref.provider_call_id, "agent-leg-1"))
    for _ in range(10):
        if bridge.done():
            break
        await clock.advance(0.5)
    await bridge

    await clock.advance(120.0)
    await provider.drain_deliveries()
    types = sink.types(ref.provider_call_id)
    assert "bridged" in types
    assert "completed" in types
    await provider.close()


async def test_a_borrower_nobody_bridges_to_hangs_up(kind: str):
    """The abandoned call, from the borrower's side.

    They said hello, nobody came, and after their patience runs out they hang
    up. The carrier reports the call as over. This is what the whole predictive
    design exists to keep rare, and the mock has to produce it or the
    simulation would show an abandon rate of zero for the wrong reason.
    """
    clock = VirtualClock()
    sink = Collector()
    provider = make(kind, clock, sink, seed=7, answer_rate=1.0)
    ref = await place(provider, clock, "+919000000007", "k-abandon")

    for _ in range(80):
        await clock.advance(1.0)
        await provider.drain_deliveries()
        if "completed" in sink.types(ref.provider_call_id):
            break

    status = await provider.get_call_status(ref.provider_call_id)
    assert status.live is False
    assert status.ended_reason == "borrower_hung_up_waiting"
    await provider.close()


async def test_a_provider_in_outage_refuses_everything(kind: str):
    """Failure scenario 2. Both carriers can be taken down the same way, and
    the difference between "rejected" and "unreachable" is preserved."""
    clock = VirtualClock()
    provider = make(kind, clock, Collector(), seed=7)
    provider.start_outage(60.0)

    with pytest.raises(ProviderUnavailable):
        await provider.place_call(to="+919000000008", from_=FROM, idempotency_key="k-out")

    # And it cannot answer questions about existing calls either, which is the
    # situation that leaves the reaper holding an agent it dare not release.
    with pytest.raises(ProviderTimeout):
        await provider.find_by_idempotency_key("k-out")

    health = await provider.health()
    assert health.reachable is False

    provider.end_outage()
    ref = await place(provider, clock, "+919000000008", "k-out")
    assert ref is not None
    await provider.close()


async def test_runs_are_reproducible_from_the_seed(kind: str):
    """Two runs with the same seed behave identically.

    Not a nicety: the failure scenarios are only worth anything if a run that
    exposes a bug can be run again.
    """

    async def run() -> list[str]:
        clock = VirtualClock()
        sink = Collector()
        provider = make(kind, clock, sink, seed=99)
        for index in range(6):
            try:
                await place(provider, clock, f"+9190000001{index:02d}", f"k-r{index}")
            except Exception as exc:  # noqa: BLE001 - the failures are the data
                sink.events.append(
                    NormalisedEvent(
                        provider=provider.name,
                        provider_event_id=f"error:{index}",
                        provider_call_id="",
                        event_type=type(exc).__name__,
                    )
                )
        await clock.advance(60.0)
        await provider.drain_deliveries()
        await provider.close()
        return sink.ids()

    assert await run() == await run()


async def test_sink_failures_do_not_kill_the_provider(kind: str):
    """Our webhook handler raising must not take the carrier down with it."""
    clock = VirtualClock()

    async def broken(event: NormalisedEvent) -> None:
        raise RuntimeError("ingest is down")

    provider = make(kind, clock, broken, seed=7, answer_rate=1.0)
    ref = await place(provider, clock, "+919000000009", "k-sink")
    await clock.advance(30.0)

    assert provider.sink_errors, "the failure must be recorded, not swallowed"
    assert (await provider.get_call_status(ref.provider_call_id)) is not None
    await provider.close()


# ---------------------------------------------------------------------------
# Provider B must actually misbehave
# ---------------------------------------------------------------------------


async def drive(provider, clock: VirtualClock, sink: Collector, calls: int, seconds: float):
    """Place `calls` calls and run the carrier for `seconds` of virtual time."""
    tasks = [
        asyncio.ensure_future(
            provider.place_call(
                to=f"+9199000{index:05d}", from_=FROM, idempotency_key=f"drive-{index}"
            )
        )
        for index in range(calls)
    ]
    for _ in range(int(seconds)):
        await clock.advance(1.0)
        await provider.drain_deliveries()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    await clock.advance(60.0)
    await provider.drain_deliveries()
    return outcomes


async def test_flaky_provider_emits_duplicates_and_reordering():
    """The headline test for provider B.

    Over enough calls it must produce at least one duplicated event and at
    least one pair delivered out of causal order. Both are checked against what
    the carrier itself claims happened, not against a fixed expectation, so the
    test stays meaningful if the rates are tuned.
    """
    clock = VirtualClock()
    sink = Collector()
    provider = make_flaky_provider(clock, seed=5, event_sink=sink, answer_rate=0.8)

    await drive(provider, clock, sink, calls=40, seconds=90)

    ids = sink.ids()
    assert len(ids) > len(set(ids)), "provider B must deliver some event twice"

    # Causal order: for each call, the position at which each event type first
    # arrived should follow ringing -> answered -> completed. At least one call
    # must violate that, or nothing is being reordered.
    order = {"ringing": 0, "answered": 1, "bridged": 2, "completed": 3, "no_answer": 3}
    reordered = 0
    seen: dict[str, list[int]] = {}
    for event in sink.events:
        seen.setdefault(event.provider_call_id, []).append(
            order.get(event.event_type, 9)
        )
    for ranks in seen.values():
        if any(b < a for a, b in zip(ranks, ranks[1:])):
            reordered += 1
    assert reordered > 0, "provider B must deliver some events out of order"
    await provider.close()


async def test_flaky_provider_times_out_and_sometimes_places_the_call_anyway():
    """The expensive failure, and the reason for find_by_idempotency_key.

    Some requests get no response. Among those, some DID place a call -- the
    borrower's phone is ringing and the dialer has no call id. The only way
    back is the key we wrote down first.
    """
    clock = VirtualClock()
    provider = make_flaky_provider(clock, seed=3, event_sink=Collector())

    outcomes = await drive(provider, clock, Collector(), calls=60, seconds=30)
    timeouts = [
        index
        for index, outcome in enumerate(outcomes)
        if isinstance(outcome, ProviderTimeout)
    ]
    assert timeouts, "provider B must produce timeouts"

    orphans = [
        index
        for index in timeouts
        if await provider.find_by_idempotency_key(f"drive-{index}") is not None
    ]
    assert orphans, (
        "at least one timed-out request must have placed a call anyway -- "
        "otherwise crash recovery is never exercised"
    )
    await provider.close()


async def test_fast_provider_is_orderly():
    """The control. Provider A must NOT misbehave, or the comparison between
    the two proves nothing."""
    clock = VirtualClock()
    sink = Collector()
    provider = make_fast_provider(clock, seed=5, event_sink=sink, answer_rate=0.8)

    outcomes = await drive(provider, clock, sink, calls=40, seconds=90)

    ids = sink.ids()
    assert len(ids) == len(set(ids)), "provider A must never duplicate an event"
    assert not any(isinstance(o, ProviderTimeout) for o in outcomes)
    await provider.close()


async def test_the_two_providers_differ_where_it_matters():
    """Same workload, both carriers, and the numbers must actually diverge --
    setup latency and reliability are the two that change pacing."""
    results = {}
    for name, factory in (("fast", make_fast_provider), ("flaky", make_flaky_provider)):
        clock = VirtualClock()
        provider = factory(clock, seed=11, event_sink=Collector())
        tasks = [
            asyncio.ensure_future(
                provider.place_call(
                    to=f"+9199000{i:05d}", from_=FROM, idempotency_key=f"cmp-{i}"
                )
            )
            for i in range(40)
        ]
        for _ in range(20):
            await clock.advance(1.0)
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        # Read health before time moves past the rolling window, or both
        # providers report the same empty-window zero and the test proves
        # nothing.
        health = await provider.health()
        results[name] = {
            "failures": sum(1 for o in outcomes if isinstance(o, Exception)),
            "setup": health.avg_setup_seconds,
        }
        await provider.close()

    assert results["flaky"]["failures"] > results["fast"]["failures"]
    assert results["flaky"]["setup"] > results["fast"]["setup"]
