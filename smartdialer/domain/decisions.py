"""The pacing decision log.

One row per tick, whether or not anything was dialled. The empty ticks matter
as much as the busy ones: "why did the dialer sit idle for four minutes?" is
answered by a run of rows whose reason_code says WINDOW_CLOSED or
PROVIDER_BREAKER, and by nothing else.

`inputs` holds the entire snapshot plus every intermediate term the engine
computed. That is what makes the decision reproducible after the fact rather
than merely plausible -- the engine is a pure function, so anyone holding this
row can recompute the proposal and check it.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg import AsyncCursor

from smartdialer.core.models import CampaignMode, PacingDecision


async def record_decision(
    cur: AsyncCursor,
    *,
    campaign_id: UUID,
    ts: datetime,
    mode: CampaignMode,
    proposed: int,
    approved: int,
    reason_code: str,
    inputs: dict[str, Any],
) -> int:
    """Write one tick's decision. Returns the row id.

    `ts` comes from the injected clock rather than the column default, so a
    simulation's decision log is on the same timeline as the CSV it is compared
    against.

    default=str on the dump because the inputs carry datetimes, UUIDs and
    Decimals straight off the snapshot. Losing a decision row to a
    serialisation error would be a poor trade for type fidelity in an audit
    log that is read by humans.
    """
    await cur.execute(
        """
        INSERT INTO pacing_decisions
            (campaign_id, ts, mode, proposed, approved, reason_code, inputs)
        VALUES
            (%(campaign_id)s, %(ts)s, %(mode)s, %(proposed)s, %(approved)s,
             %(reason_code)s, %(inputs)s)
        RETURNING id
        """,
        {
            "campaign_id": campaign_id,
            "ts": ts,
            "mode": mode.value,
            "proposed": proposed,
            "approved": approved,
            "reason_code": reason_code,
            "inputs": json.dumps(inputs, default=str),
        },
    )
    return (await cur.fetchone())["id"]


async def recent_decisions(
    cur: AsyncCursor, *, campaign_id: UUID, limit: int = 50
) -> list[PacingDecision]:
    """Most recent decisions first. Served by pacing_decisions_campaign_ts_idx."""
    await cur.execute(
        """
        SELECT * FROM pacing_decisions
        WHERE campaign_id = %(campaign_id)s
        ORDER BY ts DESC, id DESC
        LIMIT %(limit)s
        """,
        {"campaign_id": campaign_id, "limit": limit},
    )
    return [PacingDecision.from_row(row) for row in await cur.fetchall()]
