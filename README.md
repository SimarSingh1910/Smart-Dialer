# SmartDialer

Progressive and predictive outbound dialer for a debt-collections contact centre.

The pipeline is fixed:

```
Campaign -> Pacing Engine -> Safety Controller -> Call Allocator -> Telecom Provider
```

The pacing engine proposes a number of calls to start. The Safety Controller decides
what is actually allowed: approve, reduce, reject, or fall back to progressive.

> Full architecture notes live in `docs/`. This file covers running it.

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

Create the database once, using a superuser connection:

```bash
python tasks.py db --superuser-dsn postgresql://postgres:<password>@localhost:5432/postgres
```

That creates the `smartdialer` role and database. Then:

```bash
cp .env.example .env      # adjust SMARTDIALER_DSN if needed
python tasks.py up        # check the connection
python tasks.py migrate   # apply migrations/*.sql
python tasks.py seed      # demo campaign, agents, borrowers
python tasks.py test      # test suite
python tasks.py sim       # simulation scenarios
```

There is also a `Makefile` (`make up`, `make migrate`, ...) which simply forwards to
`tasks.py`, so the commands are identical on a machine that has `make`.

## Layout

| Path | Contents |
| --- | --- |
| `smartdialer/core/` | clock, config, database pool, structured logging |
| `smartdialer/domain/` | agent, call and borrower state machines |
| `smartdialer/pacing/` | pacing engine (pure functions, no I/O) |
| `smartdialer/safety/` | safety controller, abandon budget, circuit breaker |
| `smartdialer/allocator/` | agent and borrower allocation |
| `smartdialer/providers/` | provider interface and mock providers |
| `smartdialer/workers/` | dialer worker, reaper, event ingester |
| `smartdialer/sim/` | simulation harness |
| `migrations/` | numbered SQL migrations |
| `docs/` | architecture, state machines, decision record |

## A note on time

No domain code calls `time.time()`, `datetime.now()` or `asyncio.sleep()` directly.
Everything takes an injected `Clock`. `RealClock` in production, `VirtualClock` in
tests and simulation, which is what makes a 260-second provider outage cost
nothing to test and makes every failure scenario reproducible.
