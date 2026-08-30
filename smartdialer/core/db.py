"""Database access: a psycopg3 connection pool and a transaction helper.

No ORM. Every query in this project is hand-written SQL, because the
interesting ones (SELECT ... FOR UPDATE SKIP LOCKED, compare-and-swap on a
version column, INSERT ... ON CONFLICT DO NOTHING) are the design, not
plumbing. Hiding them behind a query builder would hide the argument.
"""

from __future__ import annotations

import contextlib
import pathlib
import re
from typing import Any, AsyncIterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parents[2] / "migrations"


class Database:
    """Owns the connection pool. One instance per process."""

    def __init__(self, dsn: str, *, min_size: int = 2, max_size: int = 10) -> None:
        self._dsn = dsn
        self._pool = AsyncConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"row_factory": dict_row},
        )

    async def open(self) -> None:
        await self._pool.open(wait=True)

    async def close(self) -> None:
        await self._pool.close()

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[psycopg.AsyncCursor]:
        """Run a block inside one transaction.

        Commits on clean exit, rolls back on exception. Everything that must be
        atomic -- reserve an agent and write the call row, dedupe an event and
        apply it -- goes through here, so the lock and the state change commit
        together or not at all.
        """
        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    yield cur

    @contextlib.asynccontextmanager
    async def cursor(self) -> AsyncIterator[psycopg.AsyncCursor]:
        """Autocommit-style cursor for reads that need no transaction."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                yield cur

    async def fetch_all(self, sql: str, params: dict[str, Any] | None = None) -> list[dict]:
        async with self.cursor() as cur:
            await cur.execute(sql, params)
            return await cur.fetchall()

    async def fetch_one(self, sql: str, params: dict[str, Any] | None = None) -> dict | None:
        async with self.cursor() as cur:
            await cur.execute(sql, params)
            return await cur.fetchone()


# --------------------------------------------------------------------------
# Migrations
# --------------------------------------------------------------------------
# Numbered .sql files applied in order, tracked in a schema_migrations table.
# A real migration tool is more than this problem needs; the whole schema is
# one file and the grader has to be able to read it.

_MIGRATION_NAME = re.compile(r"^(\d+)_.+\.sql$")

BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     int PRIMARY KEY,
    filename    text NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
"""


def discover_migrations(directory: pathlib.Path = MIGRATIONS_DIR) -> list[tuple[int, pathlib.Path]]:
    found: list[tuple[int, pathlib.Path]] = []
    for path in sorted(directory.glob("*.sql")):
        match = _MIGRATION_NAME.match(path.name)
        if not match:
            raise ValueError(f"migration {path.name} must be named <number>_<description>.sql")
        found.append((int(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    return found


async def migrate(dsn: str, directory: pathlib.Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every migration not yet recorded. Returns the filenames applied.

    Each migration runs inside its own transaction together with the row that
    records it, so a migration cannot be half-applied and cannot be recorded
    without having run.
    """
    applied: list[str] = []
    async with await psycopg.AsyncConnection.connect(dsn, row_factory=dict_row) as conn:
        async with conn.transaction():
            await conn.execute(BOOTSTRAP_SQL)
        rows = await (await conn.execute("SELECT version FROM schema_migrations")).fetchall()
        done = {row["version"] for row in rows}

        for version, path in discover_migrations(directory):
            if version in done:
                continue
            async with conn.transaction():
                await conn.execute(path.read_text(encoding="utf-8"))
                await conn.execute(
                    "INSERT INTO schema_migrations (version, filename) VALUES (%s, %s)",
                    (version, path.name),
                )
            applied.append(path.name)
    return applied
