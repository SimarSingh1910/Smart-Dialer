"""Writing the simulation down.

Three outputs, and they answer three different questions.

    <scenario>_<mode>.csv       what happened, tick by tick. Backing data.
    the summary table           did predictive beat progressive, and at what
                                cost. The actual deliverable.
    --explain                   why THIS tick proposed THIS number. The
                                interview question, answered from the run
                                rather than from memory.

The summary table is the one that matters. The thesis of the whole submission
is one comparison -- higher utilization at an abandon rate under budget -- and
if it is not visible in five columns it is not a thesis, it is a hope.
"""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Iterable, Sequence

from smartdialer.sim.runner import COLUMNS, RunResult

OUTPUT_DIR = pathlib.Path("sim_output")


def write_run(result: RunResult, *, directory: pathlib.Path = OUTPUT_DIR) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{result.scenario}_{result.mode.lower()}"

    path = directory / f"{stem}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(result.rows)

    # One JSON object per tick, so --explain can seek to a tick without
    # loading a run into memory and without the CSV growing a column that is
    # a paragraph.
    with (directory / f"{stem}_explain.jsonl").open("w", encoding="utf-8") as handle:
        for entry in result.narratives:
            handle.write(json.dumps(entry, default=str) + "\n")

    return path


# ---------------------------------------------------------------------------
# The summary table
# ---------------------------------------------------------------------------

HEADERS = (
    ("scenario", 8),
    ("mode", 12),
    ("utilization%", 13),
    ("connects/agent-hr", 18),
    ("abandon%", 9),
    ("wait_p50", 9),
    ("wait_p95", 9),
)


def summary_table(results: Sequence[RunResult]) -> str:
    lines = [" | ".join(name.ljust(width) for name, width in HEADERS)]
    lines.append("-+-".join("-" * width for _, width in HEADERS))

    for result in results:
        s = result.summary
        cells = (
            s["scenario"],
            s["mode"],
            f"{s['utilization_pct']:.1f}",
            f"{s['connects_per_agent_hour']:.1f}",
            f"{s['abandon_pct']:.2f}",
            f"{s['wait_p50_ms']:.0f}",
            f"{s['wait_p95_ms']:.0f}",
        )
        lines.append(
            " | ".join(str(c).ljust(width) for c, (_, width) in zip(cells, HEADERS))
        )
    return "\n".join(lines)


def verdict(results: Sequence[RunResult], *, budget_pct: float = 3.0) -> str:
    """The check the submission actually has to pass, printed as a sentence.

    Two claims, and both have to hold. Higher utilization on its own is easy --
    dial harder and abandon more people. The claim is higher utilization AND an
    abandon rate inside the budget, which is the only version of the result
    that means anything.
    """
    by_scenario: dict[str, dict[str, dict]] = {}
    for result in results:
        by_scenario.setdefault(result.scenario, {})[result.mode] = result.summary

    lines = []
    for key in sorted(by_scenario):
        modes = by_scenario[key]
        if "PROGRESSIVE" not in modes or "PREDICTIVE" not in modes:
            continue
        prog, pred = modes["PROGRESSIVE"], modes["PREDICTIVE"]
        gain = pred["utilization_pct"] - prog["utilization_pct"]
        within = pred["abandon_pct"] <= budget_pct
        mark = "ok  " if gain > 0 and within else "note"
        lines.append(
            f"  {mark} {key}: utilization {prog['utilization_pct']:.1f}% -> "
            f"{pred['utilization_pct']:.1f}% ({gain:+.1f} pts), "
            f"abandon {pred['abandon_pct']:.2f}% "
            f"({'within' if within else 'OVER'} the {budget_pct:.0f}% budget)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# --explain
# ---------------------------------------------------------------------------


def explain(
    scenario: str,
    tick: int,
    *,
    mode: str = "predictive",
    directory: pathlib.Path = OUTPUT_DIR,
) -> str:
    """The narrative for one tick: why that number and not a larger one.

    Reads the run that was already written rather than re-running the
    scenario, which is the only way this can be a fast answer to a question
    asked out loud.
    """
    path = directory / f"{scenario}_{mode.lower()}_explain.jsonl"
    if not path.exists():
        return f"no run at {path}. Run `python tasks.py sim` first."

    entry = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            candidate = json.loads(line)
            if candidate["tick"] == tick:
                entry = candidate
                break
    if entry is None:
        return f"tick {tick} is not in {path}"

    engine = entry.get("engine", {})
    terms = entry.get("terms", {})
    budget = entry.get("budget", {})
    breaker = entry.get("breaker", {})

    trace = engine.get("search_trace") or []
    trace_line = ", ".join(f"N={int(n)} -> P(shortfall)={p:.1%}" for n, p in trace[-6:])

    out = [
        f"scenario {scenario} / {mode.upper()} / tick {tick} (t = {entry['t']}s)",
        "",
        "AGENTS",
        f"  mu_G = {engine.get('mu_G', 0):.2f}  (sd {engine.get('sigma_G', 0):.2f})"
        "   free agents now, plus wrap-ups ending inside the window, plus the"
        " chance each connected agent frees",
        f"  wrap-ups freeing inside the window: {engine.get('wrap_up_freeing')}",
        "",
        "ANSWERS",
        f"  mu_A = {engine.get('mu_A', 0):.2f}  (sd {engine.get('sigma_A', 0):.2f})"
        "   hazards of the calls already ringing, plus the new calls",
        f"  p_hat = {engine.get('p_hat', 0):.3f}  (Wilson lower bound, not the raw rate)",
        f"  window = {engine.get('window_seconds', 0):.2f}s"
        f"   exact Poisson-binomial: {engine.get('used_exact_dp')}",
        "",
        "THE SEARCH",
        f"  epsilon = {engine.get('epsilon', 0):.1%}",
        f"  {trace_line}",
        f"  -> proposed {entry['proposed']}",
        "",
        "SAFETY",
        f"  hard ratio allowed      {terms.get('limit_hard_ratio')}",
        f"  concurrency allowed     {terms.get('limit_campaign_concurrency')}",
        f"  abandon budget allowed  {terms.get('limit_abandon_budget')}"
        f"   (credit {terms.get('overdial_credit')}, {budget.get('credit_reason')})",
        f"  breaker                 {terms.get('breaker_state')}"
        f"   healthy providers: {breaker.get('healthy')}",
        f"  clamps that bound       {entry['clamps'] or 'none'}",
        f"  -> approved {entry['approved']} ({entry['reason_code']}), "
        f"{entry['overdial']} of them over the progressive floor",
        "",
        f"OUTCOME  dialed {entry['dialed']} ({entry['shortfall_reason']})",
    ]
    if engine.get("changepoint_detected"):
        out.append("")
        out.append(
            "  NOTE: the changepoint detector fired on this tick -- observed"
            " answers fell below the 5th percentile of the model's own"
            " prediction, so the credit was halved immediately rather than"
            " left to drift down."
        )
    return "\n".join(out)


def iter_summaries(results: Iterable[RunResult]):
    for result in results:
        yield result.summary
