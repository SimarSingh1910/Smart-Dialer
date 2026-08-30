"""Tests that the schema is what the domain code believes it is.

These are cheap and they catch the expensive class of bug: Python and
PostgreSQL disagreeing about what a state means. They run against a real
database and skip when none is configured.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from smartdialer.core.models import (
    AgentState,
    BorrowerState,
    CallState,
    CALL_STATE_RANK,
    IN_FLIGHT_CALL_STATES,
    TERMINAL_CALL_STATES,
)


def _enum_labels(conn, type_name: str) -> list[str]:
    """Read the labels in declaration order straight from the catalog.

    enum_range() would come back as a single text value that needs parsing;
    pg_enum.enumsortorder is the authoritative ordering and needs none.
    """
    rows = conn.execute(
        """
        SELECT e.enumlabel AS label
        FROM pg_enum e
        JOIN pg_type t ON t.oid = e.enumtypid
        WHERE t.typname = %s
        ORDER BY e.enumsortorder
        """,
        (type_name,),
    ).fetchall()
    return [row["label"] for row in rows]


def test_agent_enum_matches_python(conn):
    assert _enum_labels(conn, "agent_state") == [s.value for s in AgentState]


def test_call_enum_matches_python(conn):
    assert _enum_labels(conn, "call_state") == [s.value for s in CallState]


def test_borrower_check_constraint_matches_python(conn, campaign_id):
    """The borrower state is text plus a CHECK, so prove the CHECK admits every
    value the Python enum has and rejects one it does not."""
    for state in BorrowerState:
        conn.execute(
            "INSERT INTO borrowers (id, campaign_id, phone, state) VALUES (%s,%s,%s,%s)",
            (uuid.uuid4(), campaign_id, "+911", state.value),
        )
    with pytest.raises(psycopg.errors.CheckViolation):
        # Nested transaction == SAVEPOINT: the expected failure
        # rolls back to here instead of poisoning the whole test.
        with conn.transaction():
            conn.execute(
                "INSERT INTO borrowers (id, campaign_id, phone, state) VALUES (%s,%s,%s,%s)",
                (uuid.uuid4(), campaign_id, "+911", "NONSENSE"),
            )


def _make_call(conn, campaign_id, *, state="QUEUED", suffix="") -> uuid.UUID:
    borrower_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO borrowers (id, campaign_id, phone) VALUES (%s,%s,%s)",
        (borrower_id, campaign_id, f"+9199{suffix}"),
    )
    call_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO calls (id, campaign_id, borrower_id, provider, idempotency_key, state) "
        "VALUES (%s,%s,%s,'mock_fast',%s,%s)",
        (call_id, campaign_id, borrower_id, f"key-{call_id}", state),
    )
    return call_id


def test_generated_state_rank_matches_python_table(conn, campaign_id):
    """The single most important assertion in this file.

    CALL_STATE_RANK exists in Python so the engine can reason about ordering
    without a round trip, and as a generated column in SQL so out-of-order
    events can be filtered inside the UPDATE. Two copies of one fact drift.
    This makes them drift loudly, on the next test run, instead of silently in
    production."""
    call_id = _make_call(conn, campaign_id, suffix="0001")
    for state, expected_rank in CALL_STATE_RANK.items():
        conn.execute("UPDATE calls SET state = %s WHERE id = %s", (state.value, call_id))
        row = conn.execute(
            "SELECT state_rank FROM calls WHERE id = %s", (call_id,)
        ).fetchone()
        assert row["state_rank"] == expected_rank, f"rank disagreement for {state.value}"


def test_state_rank_cannot_be_written_by_hand(conn, campaign_id):
    """It is derived. If application code could set it, it could set it wrong,
    and the whole out-of-order defence rests on it being right."""
    borrower_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO borrowers (id, campaign_id, phone) VALUES (%s,%s,%s)",
        (borrower_id, campaign_id, "+9199002"),
    )
    with pytest.raises(psycopg.errors.GeneratedAlways):
        # Nested transaction == SAVEPOINT: the expected failure
        # rolls back to here instead of poisoning the whole test.
        with conn.transaction():
            conn.execute(
                "INSERT INTO calls (id, campaign_id, borrower_id, provider, "
                "idempotency_key, state, state_rank) VALUES (%s,%s,%s,'mock_fast',%s,'QUEUED',7)",
                (uuid.uuid4(), campaign_id, borrower_id, "key-manual-rank"),
            )


def test_terminal_states_all_share_the_top_rank(conn, campaign_id):
    """Once a call is over it is over: no terminal state may outrank another,
    or a late FAILED could overwrite a COMPLETED."""
    ranks = {CALL_STATE_RANK[state] for state in TERMINAL_CALL_STATES}
    assert ranks == {9}


def test_in_flight_states_rank_below_terminal():
    assert all(CALL_STATE_RANK[s] < 9 for s in IN_FLIGHT_CALL_STATES)


def test_idempotency_key_is_unique(conn, campaign_id):
    """The intent log's guarantee: one key, one call, however many times a
    crashed worker retries."""
    _make_call(conn, campaign_id, suffix="0003")
    borrower_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO borrowers (id, campaign_id, phone) VALUES (%s,%s,%s)",
        (borrower_id, campaign_id, "+9199004"),
    )
    duplicate_key = conn.execute("SELECT idempotency_key FROM calls LIMIT 1").fetchone()[
        "idempotency_key"
    ]
    with pytest.raises(psycopg.errors.UniqueViolation):
        # Nested transaction == SAVEPOINT: the expected failure
        # rolls back to here instead of poisoning the whole test.
        with conn.transaction():
            conn.execute(
                "INSERT INTO calls (id, campaign_id, borrower_id, provider, idempotency_key) "
                "VALUES (%s,%s,%s,'mock_fast',%s)",
                (uuid.uuid4(), campaign_id, borrower_id, duplicate_key),
            )


def test_provider_event_id_is_unique_per_provider(conn):
    """The deduplication mechanism. Three ANSWERED deliveries, one row."""
    for _ in range(1):
        conn.execute(
            "INSERT INTO provider_events (provider, provider_event_id, event_type, payload) "
            "VALUES ('mock_flaky','evt-1','ANSWERED','{}')"
        )
    with pytest.raises(psycopg.errors.UniqueViolation):
        # Nested transaction == SAVEPOINT: the expected failure
        # rolls back to here instead of poisoning the whole test.
        with conn.transaction():
            conn.execute(
                "INSERT INTO provider_events (provider, provider_event_id, event_type, payload) "
                "VALUES ('mock_flaky','evt-1','ANSWERED','{}')"
            )
    # The same id from a different provider is a different event.
    conn.execute(
        "INSERT INTO provider_events (provider, provider_event_id, event_type, payload) "
        "VALUES ('mock_fast','evt-1','ANSWERED','{}')"
    )


def test_a_borrower_cannot_have_two_live_calls(conn, campaign_id):
    """The database-level backstop behind borrower reservation. Reservation is
    what prevents this; the index is what proves it."""
    borrower_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO borrowers (id, campaign_id, phone) VALUES (%s,%s,%s)",
        (borrower_id, campaign_id, "+9199005"),
    )
    conn.execute(
        "INSERT INTO calls (id, campaign_id, borrower_id, provider, idempotency_key, state) "
        "VALUES (%s,%s,%s,'mock_fast','key-live-1','RINGING')",
        (uuid.uuid4(), campaign_id, borrower_id),
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        # Nested transaction == SAVEPOINT: the expected failure
        # rolls back to here instead of poisoning the whole test.
        with conn.transaction():
            conn.execute(
                "INSERT INTO calls (id, campaign_id, borrower_id, provider, idempotency_key, state) "
                "VALUES (%s,%s,%s,'mock_fast','key-live-2','INITIATED')",
                (uuid.uuid4(), campaign_id, borrower_id),
            )


def test_a_borrower_can_be_redialled_after_a_terminal_call(conn, campaign_id):
    """The same index must not block a legitimate retry: attempt 2 after
    attempt 1 failed is the normal path, not an error."""
    borrower_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO borrowers (id, campaign_id, phone) VALUES (%s,%s,%s)",
        (borrower_id, campaign_id, "+9199006"),
    )
    conn.execute(
        "INSERT INTO calls (id, campaign_id, borrower_id, provider, idempotency_key, state) "
        "VALUES (%s,%s,%s,'mock_fast','key-attempt-1','FAILED')",
        (uuid.uuid4(), campaign_id, borrower_id),
    )
    conn.execute(
        "INSERT INTO calls (id, campaign_id, borrower_id, provider, idempotency_key, "
        "state, attempt) VALUES (%s,%s,%s,'mock_fast','key-attempt-2','INITIATED',2)",
        (uuid.uuid4(), campaign_id, borrower_id),
    )


def test_campaign_check_constraints_reject_nonsense(conn):
    """The compliance knobs are constrained in the schema, so a bad UPDATE from
    an operator tool fails at the database rather than quietly making the
    dialer aggressive."""
    with pytest.raises(psycopg.errors.CheckViolation):
        # Nested transaction == SAVEPOINT: the expected failure
        # rolls back to here instead of poisoning the whole test.
        with conn.transaction():
            conn.execute(
                "INSERT INTO campaigns (id, name, max_overdial_ratio) VALUES (%s,'bad',0.5)",
                (uuid.uuid4(),),
            )
    with pytest.raises(psycopg.errors.CheckViolation):
        # Nested transaction == SAVEPOINT: the expected failure
        # rolls back to here instead of poisoning the whole test.
        with conn.transaction():
            conn.execute(
                "INSERT INTO campaigns (id, name, target_shortfall_eps) VALUES (%s,'bad',1.5)",
                (uuid.uuid4(),),
            )
    with pytest.raises(psycopg.errors.CheckViolation):
        # Nested transaction == SAVEPOINT: the expected failure
        # rolls back to here instead of poisoning the whole test.
        with conn.transaction():
            conn.execute(
                "INSERT INTO campaigns (id, name, mode) VALUES (%s,'bad','TURBO')",
                (uuid.uuid4(),),
            )


def test_agent_state_enum_rejects_an_invented_state(conn, campaign_id):
    with pytest.raises(psycopg.errors.InvalidTextRepresentation):
        # Nested transaction == SAVEPOINT: the expected failure
        # rolls back to here instead of poisoning the whole test.
        with conn.transaction():
            conn.execute(
                "INSERT INTO agents (id, campaign_id, state) VALUES (%s,%s,'SUPERVISING')",
                (uuid.uuid4(), campaign_id),
            )


def test_every_expected_index_exists(conn):
    """The partial indexes are load-bearing: allocation, the reaper sweep and
    the snapshot all depend on them, and losing one turns a fast query into a
    sequential scan that only shows up under load."""
    rows = conn.execute(
        "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
    ).fetchall()
    names = {row["indexname"] for row in rows}
    for expected in (
        "agents_available_idx",
        "agents_lease_idx",
        "agents_campaign_state_idx",
        "borrowers_dialable_idx",
        "calls_inflight_idx",
        "calls_lease_idx",
        "calls_provider_call_idx",
        "calls_one_live_per_borrower_idx",
        "provider_events_unapplied_idx",
        "pacing_decisions_campaign_ts_idx",
    ):
        assert expected in names, f"missing index {expected}"
