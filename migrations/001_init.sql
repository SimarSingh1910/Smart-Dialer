-- SmartDialer schema.
--
-- Design note that runs through the whole file: PostgreSQL is the only source
-- of truth. Allocation, deduplication, idempotency, crash recovery and
-- concurrency control are all solved here, in the same transaction as the
-- state change they protect. That is why there is no Redis and no queue -- a
-- lock in one system and the state it protects in another is two sources of
-- truth, and the question "which one wins?" has no good answer.

-- ---------------------------------------------------------------------------
-- Enumerations
-- ---------------------------------------------------------------------------

-- The agent lifecycle from the brief, unchanged. Enums rather than text so an
-- impossible state cannot be written at all.
CREATE TYPE agent_state AS ENUM
  ('OFFLINE','AVAILABLE','RESERVED','DIALING','CONNECTED','WRAP_UP','PAUSED');

-- The call lifecycle from the brief, plus ABANDONED.
--
-- ABANDONED is a call the borrower answered when no agent was free to take it.
-- The brief does not list it, but it must be distinguishable from COMPLETED (a
-- call that did its job) and from FAILED (a call that never reached a human).
-- In collections an abandoned call is a compliance event, so it is a state of
-- its own, it is counted, and it is budgeted.
CREATE TYPE call_state AS ENUM
  ('QUEUED','RESERVED','INITIATED','RINGING','ANSWERED','CONNECTED',
   'COMPLETED','FAILED','CANCELLED','ABANDONED');

-- ---------------------------------------------------------------------------
-- Campaigns
-- ---------------------------------------------------------------------------
-- Compliance knobs live here rather than in process configuration, because an
-- operator has to be able to tighten the abandon budget on a live campaign
-- without a redeploy.

CREATE TABLE campaigns (
  id                   uuid PRIMARY KEY,
  name                 text NOT NULL,
  mode                 text NOT NULL DEFAULT 'PROGRESSIVE'
                         CHECK (mode IN ('PROGRESSIVE','PREDICTIVE')),
  max_concurrent       int  NOT NULL DEFAULT 1000 CHECK (max_concurrent > 0),
  -- Regulatory ceiling on abandoned calls, as a percentage of connects.
  abandon_budget_pct   numeric NOT NULL DEFAULT 3.0 CHECK (abandon_budget_pct >= 0),
  -- Target probability that a tick's dialling overshoots the agents that will
  -- be free. This is the epsilon in the pacing engine's tail bound.
  target_shortfall_eps numeric NOT NULL DEFAULT 0.02
                         CHECK (target_shortfall_eps > 0 AND target_shortfall_eps < 1),
  -- Absolute cap on over-dialling. Never widened by model confidence.
  max_overdial_ratio   numeric NOT NULL DEFAULT 2.0 CHECK (max_overdial_ratio >= 1.0),
  -- Operator kill switch and dialling window; both are safety clamps.
  active               boolean NOT NULL DEFAULT true,
  dialing_window_start time,
  dialing_window_end   time,
  -- Seconds an agent spends in WRAP_UP after a call. Deterministic, which is
  -- why wrap-up agents contribute to the pacing forecast without variance.
  wrap_up_seconds      int NOT NULL DEFAULT 10 CHECK (wrap_up_seconds >= 0),
  created_at           timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Agents
-- ---------------------------------------------------------------------------

CREATE TABLE agents (
  id                uuid PRIMARY KEY,
  campaign_id       uuid NOT NULL REFERENCES campaigns(id),
  state             agent_state NOT NULL DEFAULT 'OFFLINE',
  -- Optimistic concurrency. Every transition is a compare-and-swap on this
  -- column, so a worker holding a stale read cannot overwrite a newer state;
  -- it gets 0 rows affected, re-reads, and re-decides.
  version           bigint NOT NULL DEFAULT 0,
  -- Leases. A worker that crashes stops renewing, the lease expires, and the
  -- reaper recovers the agent. This is what makes crash recovery a query
  -- rather than a distributed protocol.
  lease_owner       text,
  lease_expires_at  timestamptz,
  current_call_id   uuid,
  state_changed_at  timestamptz NOT NULL DEFAULT now(),
  wrap_up_ends_at   timestamptz,
  last_heartbeat_at timestamptz
);

-- The allocation query's index. Partial, so it only holds AVAILABLE agents --
-- typically a small fraction of the table, which keeps it hot in cache even
-- when the agent count grows.
CREATE INDEX agents_available_idx ON agents (campaign_id, state_changed_at)
  WHERE state = 'AVAILABLE';
-- The reaper's index: only leased rows, so the sweep is proportional to work
-- in flight rather than to the size of the fleet.
CREATE INDEX agents_lease_idx ON agents (lease_expires_at)
  WHERE lease_expires_at IS NOT NULL;
-- Serves the snapshot's per-state counts.
CREATE INDEX agents_campaign_state_idx ON agents (campaign_id, state);
-- Finds the agent bound to a call during reconciliation.
CREATE INDEX agents_current_call_idx ON agents (current_call_id)
  WHERE current_call_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Borrowers
-- ---------------------------------------------------------------------------

CREATE TABLE borrowers (
  id               uuid PRIMARY KEY,
  campaign_id      uuid NOT NULL REFERENCES campaigns(id),
  phone            text NOT NULL,
  state            text NOT NULL DEFAULT 'PENDING'
                     CHECK (state IN ('PENDING','RESERVED','DONE','EXHAUSTED')),
  attempts         int  NOT NULL DEFAULT 0,
  max_attempts     int  NOT NULL DEFAULT 3,
  -- Retry backoff lands here, so a failed call is naturally re-queued later
  -- without a separate scheduler.
  next_eligible_at timestamptz NOT NULL DEFAULT now(),
  last_outcome     text,
  -- Days-past-due bucket. One of the keys of the per-call answer-probability
  -- lookup: propensity is heterogeneous, and that heterogeneity is a control
  -- lever, not noise.
  dpd_bucket       text,
  priority         int NOT NULL DEFAULT 0,
  lease_owner      text,
  lease_expires_at timestamptz,
  version          bigint NOT NULL DEFAULT 0
);

-- Drives borrower selection: highest priority first, then whatever became
-- eligible longest ago. Partial on PENDING for the same reason as above.
CREATE INDEX borrowers_dialable_idx
  ON borrowers (campaign_id, priority DESC, next_eligible_at)
  WHERE state = 'PENDING';
CREATE INDEX borrowers_lease_idx ON borrowers (lease_expires_at)
  WHERE lease_expires_at IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Calls
-- ---------------------------------------------------------------------------

CREATE TABLE calls (
  id               uuid PRIMARY KEY,
  campaign_id      uuid NOT NULL REFERENCES campaigns(id),
  borrower_id      uuid NOT NULL REFERENCES borrowers(id),
  -- NULL for a predictive over-dial: the call exists before an agent is bound
  -- to it. That is precisely the risk predictive dialling takes on.
  agent_id         uuid REFERENCES agents(id),
  provider         text NOT NULL,
  provider_call_id text,

  -- Generated by us BEFORE the provider is called, and passed to the provider.
  -- If the worker dies between writing this row and the provider responding,
  -- recovery asks the provider about this key instead of guessing. This is the
  -- intent-log (outbox) pattern and it is what prevents orphaned calls.
  idempotency_key  text NOT NULL UNIQUE,

  state            call_state NOT NULL DEFAULT 'QUEUED',

  -- Derived from state, never written by hand.
  --
  -- Out-of-order provider events are advanced by rank, so a late RINGING
  -- cannot drag a COMPLETED call backwards. Making this a generated column
  -- means state and rank are structurally incapable of disagreeing -- the
  -- alternative, two columns written together by application code, drifts the
  -- first time somebody forgets.
  state_rank       int GENERATED ALWAYS AS (
                     CASE state
                       WHEN 'QUEUED'    THEN 0
                       WHEN 'RESERVED'  THEN 1
                       WHEN 'INITIATED' THEN 2
                       WHEN 'RINGING'   THEN 3
                       WHEN 'ANSWERED'  THEN 4
                       WHEN 'CONNECTED' THEN 5
                       ELSE 9   -- COMPLETED, FAILED, CANCELLED, ABANDONED
                     END
                   ) STORED,

  attempt          int NOT NULL DEFAULT 1,
  is_overdial      boolean NOT NULL DEFAULT false,
  -- The answer probability the engine assigned at dial time. Kept so the
  -- model can be checked against outcomes after the fact; a predictor nobody
  -- calibrates is a guess with a decimal point.
  predicted_p      numeric,

  -- Timestamps are FACTS, absorbed unconditionally and idempotently even when
  -- the event that carried them is too late to move the state. See the note on
  -- monotonic state in docs/state-machines.md.
  initiated_at     timestamptz,
  ringing_at       timestamptz,
  answered_at      timestamptz,
  connected_at     timestamptz,
  ended_at         timestamptz,

  -- answered_at -> connected_at. How long a human waited after saying hello.
  -- The metric the whole predictive design exists to keep small.
  wait_ms          int,
  failure_reason   text,

  lease_owner      text,
  lease_expires_at timestamptz,
  version          bigint NOT NULL DEFAULT 0,
  created_at       timestamptz NOT NULL DEFAULT now()
);

-- Everything currently in flight, for the pacing snapshot.
CREATE INDEX calls_inflight_idx ON calls (campaign_id, state)
  WHERE state IN ('RESERVED','INITIATED','RINGING','ANSWERED','CONNECTED');
CREATE INDEX calls_lease_idx ON calls (lease_expires_at)
  WHERE lease_expires_at IS NOT NULL;
-- Provider events arrive keyed by the provider's own call id.
CREATE INDEX calls_provider_call_idx ON calls (provider, provider_call_id);
CREATE INDEX calls_campaign_created_idx ON calls (campaign_id, created_at DESC);

-- A borrower must never be on two live calls at once.
--
-- Borrower reservation (SELECT ... FOR UPDATE SKIP LOCKED) is the mechanism
-- that prevents this. This index is the backstop that makes the invariant a
-- property of the database rather than of the code being correct. If it ever
-- fires, there is a bug in allocation and we want to know loudly.
CREATE UNIQUE INDEX calls_one_live_per_borrower_idx ON calls (borrower_id)
  WHERE state IN ('QUEUED','RESERVED','INITIATED','RINGING','ANSWERED','CONNECTED');

-- ---------------------------------------------------------------------------
-- Provider events
-- ---------------------------------------------------------------------------
-- Raw, immutable record of everything a provider told us. Events are written
-- here first and applied second, so an event we cannot yet make sense of (one
-- naming a call we have not finished inserting) is never lost.

CREATE TABLE provider_events (
  id                bigserial PRIMARY KEY,
  provider          text NOT NULL,
  provider_event_id text NOT NULL,
  provider_call_id  text,
  event_type        text NOT NULL,
  provider_ts       timestamptz,
  received_at       timestamptz NOT NULL DEFAULT now(),
  payload           jsonb NOT NULL,
  applied           boolean NOT NULL DEFAULT false,
  apply_result      text,

  -- THE deduplication mechanism. A provider that sends ANSWERED three times
  -- gets one row here, so it causes one state transition. No application-level
  -- "have I seen this?" check, which would race with itself.
  UNIQUE (provider, provider_event_id)
);

-- The sweeper's worklist: events stored but not yet applied.
CREATE INDEX provider_events_unapplied_idx ON provider_events (id) WHERE NOT applied;
CREATE INDEX provider_events_call_idx ON provider_events (provider, provider_call_id);

-- ---------------------------------------------------------------------------
-- Pacing decisions
-- ---------------------------------------------------------------------------
-- One row per tick. This table exists to answer "why did it dial 17 and not
-- 10?" months later, which is both an engineering need and a compliance one.

CREATE TABLE pacing_decisions (
  id          bigserial PRIMARY KEY,
  campaign_id uuid NOT NULL REFERENCES campaigns(id),
  ts          timestamptz NOT NULL DEFAULT now(),
  mode        text NOT NULL,
  proposed    int NOT NULL,
  approved    int NOT NULL,
  -- Which clamp bound the decision: HARD_RATIO, ABANDON_BUDGET, STALE_SIGNALS...
  reason_code text NOT NULL,
  -- The full snapshot plus every intermediate term of the tail bound.
  inputs      jsonb NOT NULL
);

CREATE INDEX pacing_decisions_campaign_ts_idx ON pacing_decisions (campaign_id, ts DESC);

-- ---------------------------------------------------------------------------
-- Campaign counters
-- ---------------------------------------------------------------------------
-- Running totals maintained in the same transaction as the state change, so
-- the snapshot never pays for a COUNT(*) over the calls table.
--
-- Sharded from the start. A single row per campaign would be updated by every
-- worker on every call and would serialise them all on one row lock -- the
-- classic hot-row problem. Each worker updates the shard its id hashes to and
-- readers sum across shards.

CREATE TABLE campaign_counters (
  campaign_id     uuid NOT NULL REFERENCES campaigns(id),
  shard           int  NOT NULL,
  calls_initiated bigint NOT NULL DEFAULT 0,
  calls_answered  bigint NOT NULL DEFAULT 0,
  calls_connected bigint NOT NULL DEFAULT 0,
  calls_abandoned bigint NOT NULL DEFAULT 0,
  calls_failed    bigint NOT NULL DEFAULT 0,
  PRIMARY KEY (campaign_id, shard)
);
