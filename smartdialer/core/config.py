"""Process configuration, read once from the environment.

Deliberately a plain dataclass rather than pydantic-settings: these are
deployment knobs, not user input, and the failure mode we care about is "the
DSN is wrong", which a KeyError already tells us clearly.

Campaign-level knobs (abandon budget, over-dial ratio, epsilon) do NOT live
here. They live in the campaigns table, because they are per-campaign
compliance settings that an operator changes without a redeploy.
"""

from __future__ import annotations

import os
import socket
import uuid
from dataclasses import dataclass, field

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/smartdialer"


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # --- database ---
    dsn: str = field(default_factory=lambda: os.environ.get("SMARTDIALER_DSN", DEFAULT_DSN))
    db_pool_min: int = field(default_factory=lambda: _env_int("SMARTDIALER_DB_POOL_MIN", 2))
    db_pool_max: int = field(default_factory=lambda: _env_int("SMARTDIALER_DB_POOL_MAX", 10))

    # --- worker identity ---
    # Every lease in the system is stamped with this. It has to be unique per
    # process, not per host, because we run several workers on one box.
    worker_id: str = field(
        default_factory=lambda: os.environ.get(
            "SMARTDIALER_WORKER_ID", f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        )
    )

    # --- timing ---
    # 250ms: small batches at a high tick rate produce the same call volume as
    # large batches at a low one, but with materially lower simultaneity, which
    # is what actually drives customer wait.
    tick_seconds: float = field(default_factory=lambda: _env_float("SMARTDIALER_TICK_SECONDS", 0.25))
    reaper_seconds: float = field(default_factory=lambda: _env_float("SMARTDIALER_REAPER_SECONDS", 1.0))
    # Two lease tiers, and the gap between them is deliberate.
    #
    # An agent RESERVED with no call row yet has nothing behind them to
    # reconcile, so if the worker holding them dies they should be back in the
    # pool almost immediately. The batch-reservation window is exactly where a
    # crash strands agents, and thirty seconds of idle capacity per agent is a
    # real cost at any scale.
    #
    # Once a call row exists the lease is promoted to `lease_seconds`, because
    # reclaiming that agent now means asking the carrier what happened to the
    # call first, and reconciliation needs headroom to retry a slow provider.
    reserve_lease_seconds: float = field(
        default_factory=lambda: _env_float("SMARTDIALER_RESERVE_LEASE_SECONDS", 5.0)
    )
    lease_seconds: float = field(default_factory=lambda: _env_float("SMARTDIALER_LEASE_SECONDS", 30.0))
    heartbeat_timeout_seconds: float = field(
        default_factory=lambda: _env_float("SMARTDIALER_HEARTBEAT_TIMEOUT_SECONDS", 30.0)
    )
    # Any snapshot input older than this forces the safety controller to fall
    # back to progressive. Predicting from stale state is how you abandon calls.
    max_signal_age_seconds: float = field(
        default_factory=lambda: _env_float("SMARTDIALER_MAX_SIGNAL_AGE_SECONDS", 5.0)
    )
    max_call_lifetime_seconds: float = field(
        default_factory=lambda: _env_float("SMARTDIALER_MAX_CALL_LIFETIME_SECONDS", 900.0)
    )

    # --- logging ---
    log_level: str = field(default_factory=lambda: os.environ.get("SMARTDIALER_LOG_LEVEL", "INFO"))


def load_settings() -> Settings:
    return Settings()
