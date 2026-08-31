-- Separate what the safety controller APPROVED from what was actually DIALED.
--
-- Once the controller both clamps and dials, one number cannot carry both
-- facts. A tick that approves 17 and places 12 has two independent stories in
-- it: the clamp decision (why not more than 17?) and the outcome (why only
-- 12?). Collapsing them loses whichever one you did not record.
--
-- The distinction is not bookkeeping. The two dominant shortfall causes mean
-- opposite things:
--
--   NO_AGENTS     a pacing signal -- the dialer wanted more capacity than the
--                 floor had, which is exactly what predictive mode exists to
--                 exploit and what the abandon budget exists to bound.
--   NO_BORROWERS  campaign exhaustion -- nothing to do with pacing at all.
--
-- With one column, a campaign quietly running out of people to call looks
-- identical to the safety system doing its job, and the utilization charts in
-- the simulation report become uninterpretable. Step 8's AIMD budget needs
-- `dialed` rather than `approved` for the same reason: over-dial credit is
-- spent when a call actually starts, not when one is authorised.

ALTER TABLE pacing_decisions ADD COLUMN dialed int NOT NULL DEFAULT 0;

-- NONE | NO_AGENTS | NO_BORROWERS | PROVIDER_REJECTED | PROVIDER_TIMEOUT | MIXED
--
-- The first three are known when the tick commits. The provider reasons are
-- written later, by the placement task, because a carrier's answer arrives
-- seconds after the decision that caused the call -- so this column is updated
-- once more after the fact, and becomes MIXED when both kinds occurred.
ALTER TABLE pacing_decisions ADD COLUMN shortfall_reason text;

-- Finding the ticks where the dialer wanted to do more than it could is the
-- first query anybody runs when a campaign underperforms.
CREATE INDEX pacing_decisions_shortfall_idx
  ON pacing_decisions (campaign_id, ts DESC)
  WHERE dialed < approved;
