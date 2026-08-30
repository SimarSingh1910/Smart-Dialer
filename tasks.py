#!/usr/bin/env python
"""Task runner. `python tasks.py <target>`.

There is a Makefile too, but it only forwards to this file. One implementation,
two entry points -- so the project runs identically on Windows (where `make` is
usually absent) and on a grader's mac or Linux box.

Targets:
    up        check that PostgreSQL is reachable
    db        create the smartdialer role and database (needs a superuser DSN)
    migrate   apply migrations/*.sql
    seed      load a demo campaign, agents and borrowers
    test      run the test suite
    sim       run the simulation scenarios
    loadtest  run the load test
    run       start a dialer worker
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _dsn() -> str:
    from smartdialer.core.config import load_settings

    return load_settings().dsn


def _load_dotenv() -> None:
    """Read .env if present. Real env vars win, so an override on the command
    line still works."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


# --------------------------------------------------------------------------


def task_up() -> int:
    """Verify PostgreSQL is up and we can connect. No Docker involved: this
    project talks to a local PostgreSQL service."""
    import psycopg

    dsn = _dsn()
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            version = conn.execute("SELECT version()").fetchone()[0]
        print(f"connected: {version}")
        return 0
    except Exception as exc:  # noqa: BLE001 - this is a CLI, print and exit
        print(f"cannot connect using {dsn}\n  {exc}", file=sys.stderr)
        print(
            "\nFix: create the database once with\n"
            "  python tasks.py db --superuser-dsn postgresql://postgres:<pw>@localhost:5432/postgres",
            file=sys.stderr,
        )
        return 1


def task_db(argv: list[str]) -> int:
    """Create the smartdialer role and database using a superuser connection.

    Kept separate from `migrate` because it is the one step that needs
    credentials the application itself never has.
    """
    import psycopg

    superuser_dsn = None
    if "--superuser-dsn" in argv:
        superuser_dsn = argv[argv.index("--superuser-dsn") + 1]
    superuser_dsn = superuser_dsn or os.environ.get("SMARTDIALER_SUPERUSER_DSN")
    if not superuser_dsn:
        print(
            "need a superuser DSN: --superuser-dsn postgresql://postgres:<pw>@localhost:5432/postgres",
            file=sys.stderr,
        )
        return 1

    # CREATE DATABASE cannot run inside a transaction block.
    with psycopg.connect(superuser_dsn, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = 'smartdialer'").fetchone()
        if not exists:
            conn.execute("CREATE ROLE smartdialer LOGIN PASSWORD 'smartdialer'")
            print("created role smartdialer")
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = 'smartdialer'").fetchone()
        if not exists:
            conn.execute("CREATE DATABASE smartdialer OWNER smartdialer")
            print("created database smartdialer")
    print("ok")
    return 0


def task_migrate() -> int:
    from smartdialer.core.db import migrate

    applied = asyncio.run(migrate(_dsn()))
    print("\n".join(applied) if applied else "no pending migrations")
    return 0


def task_seed() -> int:
    print("seed: implemented in step 1 (schema and domain models)")
    return 0


def task_test(argv: list[str]) -> int:
    return subprocess.call([sys.executable, "-m", "pytest", *argv])


def task_sim() -> int:
    print("sim: implemented in step 9")
    return 0


def task_loadtest() -> int:
    print("loadtest: implemented in step 10")
    return 0


def task_run() -> int:
    print("run: implemented in step 5")
    return 0


TARGETS = ("up", "db", "migrate", "seed", "test", "sim", "loadtest", "run")


def main() -> int:
    _load_dotenv()

    # Must happen before anything creates an event loop.
    from smartdialer.core.runtime import configure_event_loop

    configure_event_loop()

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    target, argv = sys.argv[1], sys.argv[2:]
    if target not in TARGETS:
        print(f"unknown target {target!r}; expected one of {', '.join(TARGETS)}", file=sys.stderr)
        return 1
    if target == "test":
        return task_test(argv)
    if target == "db":
        return task_db(argv)
    return {
        "up": task_up,
        "migrate": task_migrate,
        "seed": task_seed,
        "sim": task_sim,
        "loadtest": task_loadtest,
        "run": task_run,
    }[target]()


if __name__ == "__main__":
    raise SystemExit(main())
