-- Safety state that must be shared by every worker on a campaign.
--
-- Both things in here are campaign-wide budgets, not per-process opinions. If
-- each worker kept its own over-dial credit in memory, N workers would grant N
-- times the campaign's abandon budget while every one of them reported itself
-- to be within it -- and the compliance number the campaign is judged on is
-- the sum, not the maximum. Same for the breaker: five workers independently
-- discovering a dead carrier is five probe calls at a provider that asked for
-- one.
--
-- WHY NOT COLUMNS ON `campaigns`. Every call insert takes a FOR KEY SHARE lock
-- on the campaign row through its foreign key. A column updated on every tick
-- by every worker would sit under that lock, and the tick loop and the call
-- inserts would start queueing behind each other for no reason related to
-- either. A separate row, referenced but never referencing, keeps the hot
-- write off the path the FK touches.

CREATE TABLE campaign_safety_state (
  -- ON DELETE CASCADE: this row is part of the campaign, not a fact about it.
  -- A deleted campaign has no budget and no breaker, and leaving the row
  -- behind would block the delete for no benefit.
  campaign_id      uuid PRIMARY KEY REFERENCES campaigns(id) ON DELETE CASCADE,

  -- The AIMD over-dial allowance, in calls. 0 means pure progressive, which is
  -- both the starting value and the value every failure path returns it to.
  overdial_credit  int  NOT NULL DEFAULT 0,

  -- Set when the abandon rate breached the campaign's budget. Until it passes,
  -- credit is pinned at zero regardless of how clean the recent ticks look --
  -- the point of a cooldown is that recovery is not allowed to begin at the
  -- moment of the breach.
  cooldown_until   timestamptz,

  -- NULL means the breaker is CLOSED. Set means OPEN, and 20 seconds after it
  -- was set the breaker is HALF_OPEN and one probe may be claimed.
  breaker_opened_at timestamptz,

  -- Which worker claimed the half-open probe, and when. Claimed by a
  -- conditional UPDATE -- the same compare-and-swap discipline as agent
  -- reservation -- so exactly one worker across the fleet dials the probe and
  -- the rest are told zero.
  breaker_probe_owner text,
  breaker_probe_at    timestamptz,

  -- The high-water mark for abandons already charged against the credit. An
  -- abandon that ended after this stamp has not been paid for yet; the
  -- transaction that halves the credit moves the stamp forward in the same
  -- write, so one abandoned call cannot halve the credit twice.
  updated_at       timestamptz NOT NULL DEFAULT now()
);

-- The reaper and the breaker both ask "what has this carrier done for this
-- campaign in the last thirty seconds". Without this the question is a scan of
-- every call the campaign ever placed.
CREATE INDEX calls_recent_by_provider_idx
  ON calls (campaign_id, provider, created_at DESC);
