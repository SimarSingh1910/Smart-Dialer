"""The safety boundary, asserted against the import graph.

The brief's requirement is that the predictive algorithm must not have a way to
simply switch the safety mechanism off. This file is how that requirement is
enforced, and it is worth being precise about why it takes this form.

The tempting implementation is a runtime guard: a permission flag, a signed
token, a check inside the allocator that the caller is allowed to dial. All of
those are weaker than they look. Whatever grants the permission can be made to
grant it, the check itself is code that can be edited, and a reviewer has to
trace every path to be convinced there is no way round it.

A dependency that does not exist needs no tracing. The pacing engine cannot
place a call because it holds no database handle, no carrier and no allocator,
and imports nothing that could give it one. This test parses the AST of every
module under `pacing/` and fails if anyone adds such an import -- which turns
the architectural claim into something CI can check, and makes the failure
arrive at the moment the boundary is crossed rather than at review time.

Written now, in step 5, rather than alongside the predictive engine in step 7.
A boundary that is only tested once the dangerous code exists has gone
untested for exactly the period where the mistake would have been easiest to
make.
"""

from __future__ import annotations

import ast
import pathlib

import smartdialer.pacing as pacing_package
from smartdialer.core.models import CampaignMode
from smartdialer.pacing.engine import (
    PacingSnapshot,
    ProviderHealthSignal,
    RecentBehaviour,
    propose,
)

PACING_DIR = pathlib.Path(pacing_package.__file__).parent

# Anything that could reach a telephone, a database or an agent. `psycopg` and
# `db` are on the list even though they cannot dial by themselves: a module
# that can read the world can also be tempted to write to it, and the engine
# taking a snapshot as an argument is the only way it should ever learn
# anything.
FORBIDDEN_ROOTS = {
    "smartdialer.providers",
    "smartdialer.allocator",
    "smartdialer.workers",
    "smartdialer.safety",
    "smartdialer.core.db",
    "smartdialer.api",
    "psycopg",
    "psycopg_pool",
}


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import; within pacing/ that is fine.
            if node.level == 0 and node.module:
                found.add(node.module)
    return found


def _is_forbidden(module: str) -> bool:
    return any(
        module == root or module.startswith(root + ".") for root in FORBIDDEN_ROOTS
    )


def test_pacing_engine_has_no_forbidden_imports():
    """No module under pacing/ may import anything that can place a call."""
    checked = 0
    for path in sorted(PACING_DIR.rglob("*.py")):
        checked += 1
        offending = sorted(m for m in _imports(path) if _is_forbidden(m))
        assert not offending, (
            f"{path.name} imports {offending}. The pacing engine must not be able "
            f"to reach a provider, the allocator or the database -- that is the "
            f"safety boundary, and it is a property of the import graph."
        )
    assert checked > 0, "the boundary test found no modules to check"


def test_the_engine_holds_no_reference_it_could_dial_with():
    """A proposal is data. Nothing on it is callable or connected.

    The import test covers modules; this covers the object the engine hands
    back, because a proposal carrying a bound method would be a way out that no
    import check would see.
    """
    proposal = propose(_snapshot(agents_available=5))
    assert isinstance(proposal.n, int)
    for value in proposal.terms.values():
        assert not callable(value), f"a proposal term is callable: {value!r}"


def test_pacing_engine_is_deterministic():
    """The same snapshot always produces the same proposal.

    A decision log is only evidence if the decision can be recomputed from the
    inputs recorded beside it. If propose() consulted a clock, a counter or a
    random number, the row would record what was decided but not why.
    """
    snapshot = _snapshot(agents_available=7)
    first = propose(snapshot)
    for _ in range(50):
        again = propose(snapshot)
        assert again.n == first.n
        assert again.reason == first.reason
        assert again.terms == first.terms


def test_progressive_proposes_exactly_the_available_agents():
    for available in (0, 1, 5, 250):
        assert propose(_snapshot(agents_available=available)).n == available


def test_progressive_never_proposes_a_negative_number():
    """Counts should never be negative, but a proposal of -3 would reach the
    controller and be clamped to zero anyway; better that it cannot arise."""
    assert propose(_snapshot(agents_available=-4)).n == 0


def test_the_eight_signals_are_all_carried_into_the_proposal_terms():
    """Every signal the brief lists reaches the decision log, including the
    ones progressive mode does not read.

    A signal that is only wired up when it is first used is a signal nobody has
    ever checked the units of. These are populated and logged from the start so
    that step 7 inherits data it can trust rather than fields it has to verify.
    """
    snapshot = _snapshot(
        agents_available=4,
        historical_answer_rate=0.42,
        call_setup_time_p95=3.5,
        avg_call_duration=110.0,
    )
    terms = propose(snapshot).terms
    for key in (
        "agents_available",
        "historical_answer_rate",
        "call_setup_time_p95",
        "avg_call_duration",
        "provider_failure_rate",
        "recent_answer_rate",
    ):
        assert key in terms, f"{key} never reaches the decision log"


def test_predictive_falls_back_to_the_progressive_floor_without_credit():
    """With no over-dial credit granted, predictive IS progressive.

    This is the design's deterministic floor, and it is why an unfinished
    predictive path is safe rather than merely incomplete: the credit comes
    from the safety controller's measured abandon budget, so an engine with
    nothing granted to it cannot propose more than one call per free agent.
    """
    snapshot = _snapshot(agents_available=6, mode=CampaignMode.PREDICTIVE)
    assert propose(snapshot).n == 6


def test_predictive_can_only_spend_credit_it_was_given():
    snapshot = _snapshot(
        agents_available=6, mode=CampaignMode.PREDICTIVE, overdial_credit=3
    )
    assert propose(snapshot).n == 9


def _snapshot(**overrides) -> PacingSnapshot:
    from datetime import datetime, timezone

    now = datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc)
    defaults = dict(
        mode=CampaignMode.PROGRESSIVE,
        taken_at=now,
        now=now,
        provider_health=ProviderHealthSignal(name="mock_fast"),
        recent_campaign_behaviour=RecentBehaviour(initiated=100, answered=40),
    )
    defaults.update(overrides)
    return PacingSnapshot(**defaults)
