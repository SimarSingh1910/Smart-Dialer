"""The pacing decision log.

One row per tick, whether or not anything was dialled. The empty ticks matter
as much as the busy ones: "why did the dialer sit idle for four minutes?" is
answered by a run of rows whose reason_code says WINDOW_CLOSED or
PROVIDER_BREAKER, and by nothing else.

Each row carries three numbers, and they are three different facts:

    proposed  what the pacing engine wanted
    approved  what the safety controller allowed  -> reason_code says why
    dialed    what actually started               -> shortfall_reason says why

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


class Shortfall:
    """Why fewer calls started than the controller approved.

    NO_AGENTS and NO_BORROWERS are the two that matter and they mean opposite
    things. NO_AGENTS is a pacing signal: the dialer had authority it could not
    use because the floor was empty, which is the gap predictive mode exists to
    close. NO_BORROWERS is campaign exhaustion and has nothing to do with
    pacing at all. A single "we dialled less than we wanted" number cannot tell
    a campaign running dry from the safety system working correctly.
    """

    NONE = "NONE"
    NO_AGENTS = "NO_AGENTS"
    NO_BORROWERS = "NO_BORROWERS"
    PROVIDER_REJECTED = "PROVIDER_REJECTED"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    MIXED = "MIXED"

    @classmethod
    def combine(cls, existing: str | None, incoming: str) -> str:
        """Fold a late provider outcome into a reason already recorded.

        Provider failures arrive seconds after the tick that caused them, so
        the row is written once at decision time and amended once when a
        carrier answers. Two different causes become MIXED rather than the
        second silently overwriting the first.
        """
        if incoming == cls.NONE:
            return existing or cls.NONE
        if existing in (None, cls.NONE):
            return incoming
        if existing == incoming:
            return existing
        return cls.MIXED


async def record_decision(
    cur: AsyncCursor,
    *,
    campaign_id: UUID,
    ts: datetime,
    mode: CampaignMode,
    proposed: int,
    approved: int,
    dialed: int,
    reason_code: str,
    shortfall_reason: str,
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
            (campaign_id, ts, mode, proposed, approved, dialed,
             reason_code, shortfall_reason, inputs)
        VALUES
            (%(campaign_id)s, %(ts)s, %(mode)s, %(proposed)s, %(approved)s,
             %(dialed)s, %(reason_code)s, %(shortfall_reason)s, %(inputs)s)
        RETURNING id
        """,
        {
            "campaign_id": campaign_id,
            "ts": ts,
            "mode": mode.value,
            "proposed": proposed,
            "approved": approved,
            "dialed": dialed,
            "reason_code": reason_code,
            "shortfall_reason": shortfall_reason,
            "inputs": json.dumps(inputs, default=str),
        },
    )
    return (await cur.fetchone())["id"]


async def amend_shortfall_reason(
    cur: AsyncCursor, *, decision_id: int, incoming: str
) -> None:
    """Fold a provider outcome into a decision already written.

    Done in SQL rather than read-modify-write, because several placements from
    one tick can fail concurrently and a lost update here would misattribute
    the cause of a shortfall -- which is the one thing this column exists to
    get right.
    """
    await cur.execute(
        """
        UPDATE pacing_decisions
        SET shortfall_reason = CASE
                WHEN shortfall_reason IS NULL OR shortfall_reason = 'NONE'
                    THEN %(incoming)s
                WHEN shortfall_reason = %(incoming)s
                    THEN shortfall_reason
                ELSE 'MIXED'
            END
        WHERE id = %(decision_id)s
        """,
        {"decision_id": decision_id, "incoming": incoming},
    )


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
