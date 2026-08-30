"""Test-suite wide setup.

The event loop policy has to be selected before pytest-asyncio builds its first
loop, so this happens at import time of conftest -- which is the one place
where an import side effect is the intended mechanism.
"""

from __future__ import annotations

import os
import uuid

import pytest

from smartdialer.core.runtime import configure_event_loop

configure_event_loop()


def _dsn() -> str:
    """Resolve the DSN the same way the application does, including .env.

    Tests read .env because a developer who has configured the project once
    should not have to configure it again for the test suite.
    """
    if "SMARTDIALER_DSN" not in os.environ:
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line.startswith("SMARTDIALER_DSN="):
                        os.environ.setdefault("SMARTDIALER_DSN", line.split("=", 1)[1])
    from smartdialer.core.config import load_settings

    return load_settings().dsn


@pytest.fixture(scope="session")
def dsn() -> str:
    """Skip the database tests rather than fail them when no database is
    reachable, so `pytest` still does something useful on a fresh clone."""
    import psycopg

    value = _dsn()
    try:
        with psycopg.connect(value, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no database at {value}: {exc}")
    return value


@pytest.fixture()
def conn(dsn: str):
    """A connection whose work is always rolled back.

    Every database test runs inside one transaction that is discarded at the
    end, so tests never see each other's rows and never leave any behind.
    """
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        transaction = connection.transaction()
        transaction.__enter__()
        try:
            yield connection
        finally:
            transaction.__exit__(psycopg.Rollback, psycopg.Rollback(), None)


@pytest.fixture()
def campaign_id(conn) -> uuid.UUID:
    """A throwaway campaign with counter shards, rolled back with the test."""
    new_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO campaigns (id, name) VALUES (%s, %s)", (new_id, "test campaign")
    )
    conn.execute(
        "INSERT INTO campaign_counters (campaign_id, shard) "
        "SELECT %s, generate_series(0, 15)",
        (new_id,),
    )
    return new_id
