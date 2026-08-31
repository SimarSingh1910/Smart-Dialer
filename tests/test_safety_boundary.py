"""The safety boundary, asserted against the TRANSITIVE import closure.

The brief's requirement is that the predictive algorithm must not have a way to
simply switch the safety mechanism off. This file is how that requirement is
enforced, and the form it takes matters.

The tempting implementation is a runtime guard: a permission flag, a signed
token, a check inside the allocator that the caller is allowed to dial. All of
those are weaker than they look. Whatever grants the permission can be made to
grant it, the check itself is code that can be edited, and a reviewer has to
trace every path to be convinced there is no way round it. A dependency that
does not exist needs no tracing.

WHY THIS IS AN ALLOWLIST, AND WHY IT IS TRANSITIVE.

The first version of this test was a denylist over direct imports: it failed if
anything under `pacing/` imported a provider, the allocator or the database. It
protected against the three things I had thought of, and it would have stayed
green through the failure that actually matters -- `pacing` imports
`core.models`, somebody later adds `from smartdialer.core.db import ...` to
`core.models` for a shared enum, and the engine now transitively holds a
database handle with every existing test still passing.

A denylist enumerates the ways in; there are always more. An allowlist
enumerates the ways in that are ACCEPTABLE; there are few, and adding one is a
deliberate act that shows up in review. So this walks the whole closure and
requires every module in it to be named.

Note that the allowed `core` modules are listed LITERALLY rather than matched
by a `smartdialer.core.` prefix. A prefix match would readmit `core.db` the
moment somebody imported it, which is the exact hole this test exists to close.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

import smartdialer
import smartdialer.pacing as pacing_package
from smartdialer.core.models import CampaignMode
from smartdialer.pacing.engine import (
    PacingSnapshot,
    ProviderHealthSignal,
    RecentBehaviour,
    propose,
)

PACKAGE_ROOT = pathlib.Path(smartdialer.__file__).resolve().parent
PACING_DIR = pathlib.Path(pacing_package.__file__).resolve().parent

# Pure `core` modules the engine is allowed to see. Listed one by one, not by
# prefix. Every addition here is a decision somebody has to justify, which is
# the entire mechanism.
ALLOWED_PROJECT_MODULES = frozenset(
    {
        "smartdialer",
        "smartdialer.core",
        "smartdialer.core.models",
        "smartdialer.core.clock",
    }
)

# Non-stdlib third-party packages the engine may use. Empty on purpose: the
# statistics in here are a few dozen lines of arithmetic, and a dependency that
# could reach a socket has no business in the module that decides how hard to
# dial.
ALLOWED_THIRD_PARTY: frozenset[str] = frozenset()


def _module_name_for(path: pathlib.Path) -> str:
    relative = path.resolve().relative_to(PACKAGE_ROOT.parent)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _path_for(module: str) -> pathlib.Path | None:
    """Locate a project module's source without importing it."""
    if not module.startswith("smartdialer"):
        return None
    relative = pathlib.Path(*module.split("."))
    root = PACKAGE_ROOT.parent
    for candidate in (root / relative.with_suffix(".py"), root / relative / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _direct_imports(path: pathlib.Path) -> set[str]:
    """Every module named by an import statement in one file.

    Resolves `from x.y import z` to `x.y` and also to `x.y.z`, because that
    second form is how a submodule is usually pulled in and the distinction is
    invisible in the syntax. Checking both is the conservative choice.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Relative import. Resolve against this file's package.
                package = _module_name_for(path).rsplit(".", node.level)[0]
                base = f"{package}.{node.module}" if node.module else package
            else:
                base = node.module or ""
            if not base:
                continue
            found.add(base)
            if _path_for(base) is not None:
                for alias in node.names:
                    submodule = f"{base}.{alias.name}"
                    if _path_for(submodule) is not None:
                        found.add(submodule)
    return found


def _is_stdlib(module: str) -> bool:
    return module.split(".")[0] in sys.stdlib_module_names


def _closure() -> dict[str, list[str]]:
    """Every module reachable from `pacing/`, with the chain that reached it.

    Breadth-first, so the chain reported for an offending module is the
    shortest one -- which is the one somebody has to read to understand how the
    boundary was crossed.
    """
    chains: dict[str, list[str]] = {}
    queue: list[tuple[str, list[str]]] = []

    for path in sorted(PACING_DIR.rglob("*.py")):
        name = _module_name_for(path)
        chains[name] = [name]
        queue.append((name, [name]))

    while queue:
        module, chain = queue.pop(0)
        path = _path_for(module)
        if path is None:
            continue
        for imported in sorted(_direct_imports(path)):
            if imported in chains:
                continue
            chains[imported] = chain + [imported]
            queue.append((imported, chain + [imported]))
    return chains


def test_pacing_engine_import_closure_is_allowlisted():
    """Every module the engine can transitively reach must be on the list."""
    chains = _closure()
    assert chains, "the boundary test found no modules to check"

    offenders: list[str] = []
    for module, chain in sorted(chains.items()):
        if _is_stdlib(module):
            continue
        if module.startswith("smartdialer.pacing"):
            continue
        if module in ALLOWED_PROJECT_MODULES:
            continue
        if module.split(".")[0] in ALLOWED_THIRD_PARTY:
            continue
        offenders.append(" -> ".join(chain))

    assert not offenders, (
        "the pacing engine can reach modules that are not on the allowlist.\n"
        "Each line is the shortest import chain from pacing/ to the offender:\n  "
        + "\n  ".join(offenders)
        + "\n\nThe engine must be a pure function of its snapshot. If this import "
        "is genuinely harmless, add it to ALLOWED_PROJECT_MODULES deliberately -- "
        "do not widen the check."
    )


def test_the_closure_walk_actually_follows_transitive_edges():
    """A guard on the guard.

    The allowlist test passing means nothing if the walk never leaves the
    pacing package -- a broken resolver would report an empty closure and a
    clean bill of health. This asserts the walk genuinely crosses at least one
    package boundary and reaches something two hops out.
    """
    chains = _closure()
    assert "smartdialer.core.models" in chains, "the walk did not leave pacing/"
    assert any(len(chain) >= 2 for chain in chains.values())
    # core.models imports enum and uuid, so the walk must have gone further
    # still. If it had stopped at direct imports this would be empty.
    assert {"enum", "uuid", "decimal"} & set(chains), (
        "the walk did not follow edges out of core.models, so it is not "
        "transitive and would miss exactly the case it exists to catch"
    )


def test_a_deliberate_transitive_violation_is_caught():
    """Prove the test can fail, by simulating the failure it was written for.

    `core.models` is a module the engine legitimately imports. Here it is
    pretended to import `core.db` -- the two-hop violation that the original
    direct-import denylist would have waved through -- and the checking logic
    must reject it and name the full chain.

    Done against a synthetic closure rather than by editing a source file, so
    the test cannot leave the repository in a broken state if it fails
    part-way.
    """
    chains = _closure()
    assert "smartdialer.core.db" not in chains, "the real closure is clean"

    poisoned = dict(chains)
    poisoned["smartdialer.core.db"] = [
        "smartdialer.pacing.engine",
        "smartdialer.core.models",
        "smartdialer.core.db",
    ]

    offenders = [
        " -> ".join(chain)
        for module, chain in poisoned.items()
        if not _is_stdlib(module)
        and not module.startswith("smartdialer.pacing")
        and module not in ALLOWED_PROJECT_MODULES
        and module.split(".")[0] not in ALLOWED_THIRD_PARTY
    ]
    assert len(offenders) == 1
    assert offenders[0] == (
        "smartdialer.pacing.engine -> smartdialer.core.models -> smartdialer.core.db"
    ), "the failure message must name the whole chain, not just the module"


# ---------------------------------------------------------------------------
# The engine as an object
# ---------------------------------------------------------------------------


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


def test_progressive_mode_equals_agents_available():
    for available in (0, 1, 5, 250):
        assert propose(_snapshot(agents_available=available)).n == available


def test_progressive_never_proposes_a_negative_number():
    """Counts should never be negative, but a proposal of -3 would reach the
    controller and be clamped to zero anyway; better that it cannot arise."""
    assert propose(_snapshot(agents_available=-4)).n == 0


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
