# SmartDialer

Progressive and predictive outbound dialer for a debt-collections contact centre.

The pipeline is fixed by the brief and the code follows it literally:

```
Campaign -> Pacing Engine -> Safety Controller -> Call Allocator -> Telecom Provider
```

The pacing engine proposes a number of calls to start. The Safety Controller
decides what is actually allowed: approve, reduce, reject, or fall back to
progressive. The engine is a pure function with no database handle, no provider
handle and no allocator reference, so it cannot place a call — there is nothing
in scope with which to place one. A test parses the import graph and fails if
that ever stops being true.

* `docs/architecture.md` — component diagram, the safety boundary, one tick
* `docs/state-machines.md` — both state machines, the rank table, facts vs state
* `docs/ADR.md` — what was chosen, why, and what it makes harder

## Requirements

* Python 3.11+
* PostgreSQL 15+ running locally (no Docker; the project talks to a normal
  PostgreSQL service)

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

Create the role and database once, using a superuser connection:

```bash
python tasks.py db --superuser-dsn postgresql://postgres:<password>@localhost:5432/postgres
```

Then:

```bash
cp .env.example .env      # adjust SMARTDIALER_DSN if needed
python tasks.py up        # check the connection
python tasks.py migrate   # apply migrations/*.sql
python tasks.py seed      # demo campaign, 100 agents, 5,000 borrowers
python tasks.py test      # the test suite
python tasks.py sim       # six scenarios, both modes
python tasks.py loadtest  # 1,000 agents, 20 workers
```

There is also a `Makefile` (`make up`, `make migrate`, ...) which forwards to
`tasks.py`, so the commands are identical on a machine that has `make`.

**Run `python tasks.py up` before anything else.** It prints the PostgreSQL
version and confirms the connection, which is the one thing that has to be true
for the rest to mean anything. `.env.example` already points at the role and
database that `tasks.py db` creates, so on a default setup the `cp` needs no
editing. If the connection is wrong, the database tests *skip* rather than fail
— deliberate, so `pytest` still does something useful on a clone with no
database, but it means a skipped run can look like a passing one. `up` failing
tells you immediately; `237 passed` is what a real run prints.

## What each command prints

**`python tasks.py test`** — 237 tests. The database tests skip rather than fail
if no database is reachable, so `pytest` still does something useful on a fresh
clone.

```
........................................................................ [ 30%]
........................................................................ [ 60%]
........................................................................ [ 91%]
.....................                                                    [100%]
```

**`python tasks.py sim`** — runs six scenarios in both modes (about 25 minutes;
the campaigns run on virtual time, but the database work is real) and prints the
comparison:

```
scenario | mode         | utilization%  | connects/agent-hr  | abandon%  | wait_p50  | wait_p95
---------+--------------+---------------+--------------------+-----------+-----------+----------
B        | PROGRESSIVE  | 65.1          | 26.9               | 0.00      | 212       | 317
B        | PREDICTIVE   | 68.6          | 28.6               | 5.56      | 203       | 316
```

Predictive gains 0.2–3.5 points of utilization in five of the six scenarios and
**does not hold the abandon rate under the 3% budget in any of them.** The
safety machinery all behaves as designed — the credit collapses on every
abandon, the breaker opens and recovers through its probe, the floor tracks a
shrinking agent pool — but the budget bounds how fast the over-dial allowance
grows rather than the rate itself. §12 of `docs/ADR.md` has the full table, the
diagnosis, and what I would change next.

Per-tick CSVs land in `sim_output/`. Useful flags:

```bash
python tasks.py sim --scenario C            # one scenario, both modes
python tasks.py sim --seed 12               # a different reproducible run
python tasks.py sim --explain C --tick 240  # why THAT tick proposed THAT number
```

`--explain` prints the decision for one tick of a finished run: the agent-side
forecast, the answers-side forecast, the binary search over `N`, every clamp and
what it allowed, and what was finally dialled. It reads the run that was already
written, so it answers "why 17 and not 10" in a second.

**`python tasks.py loadtest`** — reservation latency p50/p99, ticks per second,
database round trips per call, and `EXPLAIN ANALYZE` for the allocation and
snapshot queries.

**`python tasks.py run`** — starts a worker against the seeded campaign on the
real clock, dialing a mock carrier. Ctrl-C to stop.

## The scenarios

| | Answer rate | Talk | Conditions |
| --- | --- | --- | --- |
| A | 20% | 120s | Provider A |
| B | 50% | 90s | Provider A |
| C | 70% | 180s | Provider A |
| D | 70% → 10% at t=300s | 90s → 200s | Provider B: duplicates, reordering, timeouts |
| E | 50% | 90s | Provider outage, t=200–260s |
| F | 50% | 90s | 100 agents, 40 log out at t=180s |

Worker-crash recovery is not a scenario: `tests/test_reaper.py` kills a worker at
four different points in the call lifecycle and asserts what recovery does about
each, which is more specific evidence than a scenario would produce.

## Layout

| Path | Contents |
| --- | --- |
| `smartdialer/core/` | clock, config, database pool, models, structured logging |
| `smartdialer/domain/` | agent, call and borrower state machines; the snapshot query |
| `smartdialer/pacing/` | pacing engine, hazard tables, statistics (pure, no I/O) |
| `smartdialer/safety/` | safety controller, AIMD abandon budget, circuit breaker |
| `smartdialer/allocator/` | agent and borrower allocation, the carrier hand-off |
| `smartdialer/providers/` | provider interface and two mock carriers |
| `smartdialer/workers/` | dialer worker, reaper, bridging |
| `smartdialer/sim/` | scenarios, runner, report |
| `smartdialer/loadtest/` | the load test |
| `migrations/` | numbered SQL migrations |
| `docs/` | architecture, state machines, decision record |

## A note on time

No domain code calls `time.time()`, `datetime.now()` or `asyncio.sleep()`.
Everything takes an injected `Clock`: `RealClock` in production, `VirtualClock`
in tests and simulation. That is what makes a 60-second provider outage cost
nothing to test and every failure scenario reproducible from a seed.
