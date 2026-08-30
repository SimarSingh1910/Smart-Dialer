"""Demo data: one campaign, 100 agents, 5,000 borrowers.

Deterministic on purpose. Every id is derived from a fixed namespace via
uuid5, and the random attributes come from a seeded generator, so two runs
produce byte-identical data. That means a simulation result can be reproduced
exactly, and it means re-running seed is idempotent -- it upserts rather than
piling up a second copy of the campaign.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass

import psycopg

# Fixed namespace so ids are stable across runs and machines.
NAMESPACE = uuid.UUID("6f2a1d4c-0000-4000-8000-000000000001")

DEMO_CAMPAIGN_ID = uuid.uuid5(NAMESPACE, "campaign:demo")

# Days-past-due buckets with their share of the book and a rough answer
# propensity. Later-stage delinquency answers less often -- that heterogeneity
# is what makes a per-call probability worth having instead of one global rate.
DPD_BUCKETS: list[tuple[str, float, int]] = [
    # (bucket, share of portfolio, dialling priority)
    ("0-30", 0.40, 0),
    ("31-60", 0.30, 1),
    ("61-90", 0.20, 2),
    ("90+", 0.10, 3),
]


@dataclass(frozen=True)
class SeedResult:
    campaign_id: uuid.UUID
    agents: int
    borrowers: int


def seed(
    dsn: str,
    *,
    agents: int = 100,
    borrowers: int = 5_000,
    rng_seed: int = 20260831,
) -> SeedResult:
    rng = random.Random(rng_seed)

    with psycopg.connect(dsn) as conn:
        with conn.transaction():
            # Upsert the campaign so re-seeding is safe.
            conn.execute(
                """
                INSERT INTO campaigns (id, name, mode, max_concurrent,
                                       abandon_budget_pct, target_shortfall_eps,
                                       max_overdial_ratio, wrap_up_seconds)
                VALUES (%(id)s, %(name)s, 'PROGRESSIVE', 1000, 3.0, 0.02, 2.0, 10)
                ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name
                """,
                {"id": DEMO_CAMPAIGN_ID, "name": "Demo collections campaign"},
            )

            # Counter shards. 16 is the same fan-out the workers hash into.
            conn.execute(
                """
                INSERT INTO campaign_counters (campaign_id, shard)
                SELECT %(id)s, generate_series(0, 15)
                ON CONFLICT DO NOTHING
                """,
                {"id": DEMO_CAMPAIGN_ID},
            )

            # Agents start OFFLINE. A campaign with agents already AVAILABLE
            # would hide the login path, and the simulation drives logins
            # explicitly so it can also drive 40 of them logging out at once.
            agent_rows = [
                (uuid.uuid5(NAMESPACE, f"agent:{index}"), DEMO_CAMPAIGN_ID)
                for index in range(agents)
            ]
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO agents (id, campaign_id, state)
                    VALUES (%s, %s, 'OFFLINE')
                    ON CONFLICT (id) DO NOTHING
                    """,
                    agent_rows,
                )

            # Borrowers. COPY rather than INSERT: 5,000 rows one statement at a
            # time is slow enough to be annoying during development, and the
            # load test seeds far more than this.
            #
            # COPY has no ON CONFLICT clause, so it goes into a temporary table
            # first and is inserted from there. That keeps COPY's speed while
            # making a second `seed` run a no-op rather than a primary key
            # violation -- re-seeding during development has to be safe.
            buckets = [b for b, _, _ in DPD_BUCKETS]
            weights = [w for _, w, _ in DPD_BUCKETS]
            priority_of = {b: p for b, _, p in DPD_BUCKETS}

            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TEMP TABLE borrowers_staging "
                    "(LIKE borrowers INCLUDING DEFAULTS) ON COMMIT DROP"
                )
                with cur.copy(
                    "COPY borrowers_staging (id, campaign_id, phone, state, attempts, "
                    "max_attempts, dpd_bucket, priority) FROM STDIN"
                ) as copy:
                    for index in range(borrowers):
                        bucket = rng.choices(buckets, weights=weights, k=1)[0]
                        copy.write_row(
                            (
                                uuid.uuid5(NAMESPACE, f"borrower:{index}"),
                                DEMO_CAMPAIGN_ID,
                                f"+9198{index:08d}",
                                "PENDING",
                                0,
                                3,
                                bucket,
                                priority_of[bucket],
                            )
                        )

                cur.execute(
                    """
                    INSERT INTO borrowers (id, campaign_id, phone, state, attempts,
                                           max_attempts, dpd_bucket, priority)
                    SELECT id, campaign_id, phone, state, attempts,
                           max_attempts, dpd_bucket, priority
                    FROM borrowers_staging
                    ON CONFLICT (id) DO NOTHING
                    """
                )

    return SeedResult(campaign_id=DEMO_CAMPAIGN_ID, agents=agents, borrowers=borrowers)


def reset(dsn: str) -> None:
    """Drop all campaign data, keeping the schema.

    Order matters because of the foreign keys. TRUNCATE ... CASCADE would be
    shorter, but naming the tables explicitly means adding a table later fails
    loudly here instead of quietly leaving its rows behind.
    """
    with psycopg.connect(dsn) as conn:
        with conn.transaction():
            conn.execute(
                "TRUNCATE provider_events, pacing_decisions, calls, "
                "campaign_counters, borrowers, agents, campaigns RESTART IDENTITY"
            )
