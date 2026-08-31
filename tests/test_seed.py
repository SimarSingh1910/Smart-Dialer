"""Seed determinism and idempotency, against a throwaway schema.

The obvious way to test this is a throwaway database, but CREATE DATABASE needs
a privilege the application role deliberately does not have. A throwaway schema
gives the same isolation -- its own tables, its own enum types, dropped whole at
the end -- without asking for rights the dialer should never hold.

The schema is selected by putting search_path in the connection options, so the
migration runner and the seeder are exercised exactly as they are in
production, with no test-only code path inside either.
"""

from __future__ import annotations

import urllib.parse

import psycopg
import pytest

from smartdialer.core.db import migrate
from smartdialer.core.seed import DEMO_CAMPAIGN_ID, reset, seed

SCHEMA = "seed_test"


@pytest.fixture()
async def scoped_dsn(dsn: str):
    """A DSN pointing at a private schema with the full schema migrated into it."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        conn.execute(f"CREATE SCHEMA {SCHEMA}")

    options = urllib.parse.quote(f"-csearch_path={SCHEMA}")
    scoped = f"{dsn}?options={options}"
    await migrate(scoped)
    try:
        yield scoped
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")


def counts(scoped: str) -> dict[str, int]:
    with psycopg.connect(scoped) as conn:
        return {
            table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("campaigns", "agents", "borrowers", "campaign_counters")
        }


async def test_seed_creates_the_documented_fixture(scoped_dsn):
    """The definition of done for step 1: one campaign, 100 agents, 5,000
    borrowers, plus the 16 counter shards the workers hash into."""
    result = seed(scoped_dsn)
    assert result.campaign_id == DEMO_CAMPAIGN_ID
    assert counts(scoped_dsn) == {
        "campaigns": 1,
        "agents": 100,
        "borrowers": 5_000,
        "campaign_counters": 16,
    }


async def test_seed_is_idempotent(scoped_dsn):
    """Running seed twice must be a no-op, not a primary key violation.

    This is the hole an earlier version actually fell into: COPY has no ON
    CONFLICT clause, so the second run died on borrowers_pkey. Re-seeding
    during development has to be safe, so it is asserted rather than assumed.
    """
    seed(scoped_dsn)
    first = counts(scoped_dsn)
    seed(scoped_dsn)
    assert counts(scoped_dsn) == first


async def test_seed_is_deterministic_across_a_full_reset(scoped_dsn):
    """Identical data after a wipe, down to the ids.

    This is what makes a simulation result reproducible: same seed, same
    borrowers, same dpd buckets, same dialling order.
    """
    seed(scoped_dsn)
    with psycopg.connect(scoped_dsn) as conn:
        before = conn.execute(
            "SELECT id, phone, dpd_bucket, priority FROM borrowers ORDER BY phone"
        ).fetchall()

    reset(scoped_dsn)
    assert counts(scoped_dsn)["borrowers"] == 0

    seed(scoped_dsn)
    with psycopg.connect(scoped_dsn) as conn:
        after = conn.execute(
            "SELECT id, phone, dpd_bucket, priority FROM borrowers ORDER BY phone"
        ).fetchall()

    assert before == after


async def test_agents_start_offline(scoped_dsn):
    """Agents log in explicitly. A campaign that arrives with agents already
    AVAILABLE would hide the login path, and the simulation needs to drive
    logins itself so it can also drive 40 of them logging out at once."""
    seed(scoped_dsn)
    with psycopg.connect(scoped_dsn) as conn:
        states = conn.execute("SELECT DISTINCT state FROM agents").fetchall()
    assert states == [("OFFLINE",)]


async def test_borrowers_are_spread_across_dpd_buckets(scoped_dsn):
    """Answer propensity is heterogeneous, and the pacing engine treats that as
    a control lever. A seed that put everyone in one bucket would make the
    per-call probability table look useless when it is not."""
    seed(scoped_dsn)
    with psycopg.connect(scoped_dsn) as conn:
        rows = conn.execute(
            "SELECT dpd_bucket, count(*) FROM borrowers GROUP BY 1"
        ).fetchall()
    distribution = dict(rows)
    assert set(distribution) == {"0-30", "31-60", "61-90", "90+"}
    # Roughly the configured 40/30/20/10 split, with slack for sampling noise.
    assert distribution["0-30"] > distribution["90+"]
    assert sum(distribution.values()) == 5_000


async def test_migrations_are_idempotent(scoped_dsn):
    """A second migrate applies nothing. The runner tracks what it has done, so
    a redeploy is safe."""
    assert await migrate(scoped_dsn) == []
