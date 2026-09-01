# Architecture decision record

The four questions the brief asks -- what was chosen, why, what problem it
solves, what it makes harder -- for each decision that shaped the system.

---

## 1. PostgreSQL is the only source of truth

**What was chosen.** One PostgreSQL database. No Redis, no Kafka, no queue broker,
no ORM, no microservices.

**Why.** Every problem the brief lists under distributed-systems concerns is a
consistency problem: agent allocation, borrower allocation, duplicate jobs,
retries, idempotency, concurrent updates, worker crashes, stale state. PostgreSQL
solves all of them *in the same transaction as the state change they protect*:

| Concern | Mechanism |
| --- | --- |
| Allocation | `SELECT ... FOR UPDATE SKIP LOCKED` inside an `UPDATE` |
| Concurrent updates | compare-and-swap on a `version` column |
| Idempotency | `UNIQUE` constraint + `ON CONFLICT DO NOTHING` |
| Crash recovery | lease columns + a reaper query |
| Ordering | monotonic `state_rank`, generated from `state` |
| Leader election, if needed | `pg_try_advisory_lock` |

**What problem it solves.** A lock in one system and the state it protects in
another is two sources of truth. The interview question "your database says the
agent is AVAILABLE but your cache says RESERVED — which one wins?" has no good
answer in that design. Here it has no answer because it cannot occur: there is no
cache in the allocation path.

**Why not Kafka.** Partition ordering for provider events would buy nothing,
because the provider itself reorders and duplicates events (§2.D of the brief). We
have to handle out-of-order delivery regardless. What is actually needed is
deduplication and order-independence, which are a unique constraint and a rank
comparison.

**What it makes harder.** Polling rather than push (mitigated by a 250 ms tick and
`LISTEN/NOTIFY` for urgent wakeups); a write-throughput ceiling around 10k agents
(see the scale section); PostgreSQL as a single point of failure, which in
production needs a streaming replica with automated failover and is out of scope
for this prototype.

---

## 2. Foreign keys are used, and they impose one standing rule

Foreign keys are declared on `calls.campaign_id`, `calls.borrower_id` and
`calls.agent_id`. In a system whose entire thesis is "the database is the single
source of truth", referential integrity belongs in the database.

The price is a locking interaction that has to be understood rather than
discovered:

**An FK check takes `FOR KEY SHARE` on the parent row, which conflicts with
`FOR UPDATE`.**

Two consequences:

1. **`calls.agent_id`.** Agent reservation holds `FOR UPDATE` on the agent row and
   the call insert takes `FOR KEY SHARE` on that same row. There is no contention
   only because both happen in the same transaction, in that order. A refactor
   that splits reservation and call insert across two transactions reintroduces
   the conflict. They must stay together — which they must anyway, so that a
   reserved agent and the call it is reserved for commit atomically.

2. **`calls.campaign_id`.** Every call insert in a campaign takes `FOR KEY SHARE`
   on **one** row. Shared locks do not conflict with each other, so this is free
   today. It becomes a hard stall the moment anything takes a conflicting lock on
   that row under load.

> **Standing rule: no mutable counter, status, or frequently-written column may be
> added to the `campaigns` table.** An `UPDATE campaigns SET ...` under load would
> serialise every concurrent call insert in that campaign behind one row lock.

This is why running totals live in `campaign_counters`, sharded 16 ways, rather
than on the campaign row. The `campaigns` row holds configuration an operator
edits occasionally, and nothing else.

---

## 3. Agent lifecycle: SKIP LOCKED for allocation, CAS for everything else

**Allocation.** `UPDATE agents SET state = 'RESERVED' ... WHERE id IN (SELECT ...
ORDER BY state_changed_at LIMIT n FOR UPDATE SKIP LOCKED)`. The lock and the state
write commit together, so there is no window in which two workers both believe
they own an agent. `SKIP LOCKED` means a second worker neither blocks nor errors:
it steps over the locked rows and takes the next available agents.

**What SKIP LOCKED actually buys, precisely.** It is a *throughput* property, not
a correctness one, and the distinction was established by experiment rather than
assumed. Plain `FOR UPDATE` also never double-allocates: in READ COMMITTED a
blocked waiter re-checks the row when the lock releases (EvalPlanQual), finds it
no longer `AVAILABLE`, and discards it — and because `LockRows` sits below `Limit`
in the plan, the `Limit` node pulls a replacement row. Both variants return *n*
distinct agents.

The difference is that without `SKIP LOCKED` the second worker **waits** for the
first worker's transaction, which is open across a telecom provider call. Under a
tick loop that is the difference between workers running independently and every
worker in the fleet queueing behind the slowest provider call.

`tests/test_allocation.py` reflects this honestly: the 50-worker race asserts
correctness (and passes either way, with a comment saying so), while
`test_reservation_never_blocks_on_another_workers_lock` is the test that actually
goes red — with a `TimeoutError` — when the clause is deleted. Both states were
verified by deleting the clause and running the suite.

**Everything else is compare-and-swap** on `(id, version, state)`. Zero rows
affected means somebody else moved the agent and this worker's read was stale: it
re-reads and re-decides, and never forces the write through. Illegal transitions
are rejected in Python *before* the statement runs, because a transition that
matched zero rows would be indistinguishable from a lost race — and a lost race is
normal while an impossible transition is a bug.

---

## 4. A lease and a live call release different things

This is the subtlest rule in the allocation layer.

> A **lease** releases the **worker's claim** on a borrower.
> A **terminal call** releases the **borrower for redial**.

These are governed by two independent clocks. A call can sit non-terminal
legitimately and for a long time: the worker crashed with the call in `ANSWERED`,
the provider is unreachable, reconciliation is backing off. In that window the
worker's claim has expired but the borrower is emphatically not free.

If lease expiry alone returned the borrower to `PENDING`, the dialer would reserve
them again and insert a second call — either colliding with the
one-live-call-per-borrower index (a confusing failure on entirely correct
behaviour) or, without that index, **actually ringing the same person twice for
the same debt during a provider outage**. That is a compliance problem, not an
inefficiency.

So `release_expired_leases` frees a borrower only when no non-terminal call exists
for them. Borrowers whose lease expired but whose call is still live are held, and
are deliberately observable via `borrowers_held_by_live_call` — if that set grows
and never drains, call reconciliation is stuck and the borrowers behind it are
frozen, which is the first thing to look at when a campaign stalls.

Attempts are not incremented on lease recovery. The worker crashed; the borrower
was never reached; charging them an attempt for our failure would eventually mark
a perfectly reachable person `EXHAUSTED`.

---

## 5. State is monotonic; facts are not

`state_rank` is a `GENERATED ALWAYS` column derived from `state`, so the two are
structurally incapable of disagreeing. Provider events advance a call only to a
strictly higher rank, so `COMPLETED, ANSWERED, RINGING` settles at `COMPLETED`.
Timestamps, by contrast, are absorbed unconditionally via `COALESCE`, so a late
`ANSWERED` still records `answered_at` and the measured answer rate stays honest.

**Consequence to be aware of: all terminal states share rank 9, so the first
terminal state wins.** If the reaper force-fails a call at `max_call_lifetime` and
the provider later reports `COMPLETED`, the call stays `FAILED` forever. The facts
still absorb, so the answer rate is unaffected — but the *outcome* is then locally
inferred rather than provider truth.

That disagreement is recorded in `provider_events.apply_result` and counted, not
swallowed. "How often did my reaper guess wrong about a call the provider knew
about" is a real calibration metric for the recovery timeouts. **(Implemented in
step 3.)**

---

## 6. Time is injected everywhere

No domain code calls `time.time()`, `datetime.now()` or `asyncio.sleep()`. Every
component takes a `Clock`: `RealClock` in production, `VirtualClock` in tests and
simulation. A 260-second provider outage then costs nothing to test, and every
failure scenario is reproducible rather than timing-dependent.

The `VirtualClock`'s quiescence detection is a heuristic with a documented limit —
a task blocked on real I/O looks identical to a task that has gone quiet — and
that limit is asserted by a test rather than left to be discovered.

---

## 7. Predictive pacing is a quantile decision, not a point estimate

**What was chosen.** The engine does not estimate an answer rate and divide. It
chooses the largest `N` such that

```
P(answers in the next window > agents free in the next window) <= epsilon
```

Both sides are sums of independent Bernoullis with *different* probabilities — a
Poisson-binomial — so the mean and variance are cheap:

```
mu_A  = sum over unbound ringing calls of h(t_i, W)  +  sum over new calls of p_j
mu_G  = |AVAILABLE| + sum over wrap-ups ending in W  +  sum over connected of q_k
sigma^2 = sum h(1-h) + sum p(1-p) + sum q(1-q)     (AVAILABLE and WRAP_UP add none)
P(shortfall) = 1 - Phi((0.5 - (mu_A - mu_G)) / sigma)      -- continuity corrected
```

`P(shortfall)` is monotone in `N`, so a binary search finds the largest safe `N`.
Below `sigma = 2` the normal approximation is replaced by the exact
Poisson-binomial computed by dynamic programming — that regime is the small
campaign, where the approximation is worst and the consequences of being wrong
are highest.

**Why not a point estimate.** Customer wait and abandonment are *tail* events: a
borrower waits only when simultaneous answers exceed free agents. The mean tells
you almost nothing about the tail. Dial `N` calls at probability `p` and answers
are `Binomial(N, p)`: mean `Np`, standard deviation `sqrt(Np(1-p))`, so the
coefficient of variation falls as `1/sqrt(N)`.

**The consequence that matters most: the same over-dial ratio is reckless with 20
agents and conservative with 2,000.** Any pacing constant not scaled by pool size
is wrong. That is why `max_credit_for()` scales the over-dial credit with the
agent pool rather than using a fixed number — and it is also why sharding a
campaign costs utilization, because fragmenting the pool raises the variance on
every fragment.

**Three estimators, no ML.**

* Per-call answer probability from a Laplace-smoothed table keyed by
  `(hour, attempt, prior outcome, dpd bucket)`. Heterogeneity is a control lever,
  not just accuracy: near the abandon limit, dial low-propensity numbers.
* Ring-to-answer as an empirical **hazard** over 2-second buckets. A call ringing
  3 seconds and one ringing 18 have very different chances of being answered in
  the next 2 — treating the ringing pool with a flat `p` is what produces
  overshoot spikes.
* The same hazard on the agent side for talk time, plus wrap-up timers, which are
  deterministic and therefore free information carrying no variance at all.

**Wilson, not the raw rate.** `p_hat` is the Wilson score lower bound at 95%.
Four answers out of five is a raw rate of 80% and a lower bound of about 38%, so
a thin sample cannot make the system aggressive. This is the main defence against
over-dialling at the start of a campaign. Where `p` multiplies *risk* rather than
*reward*, the upper bound is used instead: being conservative means different
arithmetic on the two sides.

**Two-layer control — the answer to "70% drops to 10%".** The fast open-loop
predictor above handles known dynamics. The slow closed-loop AIMD credit corrects
model error from observed abandons. A changepoint check runs every tick: if
observed answers fall below the 5th percentile of what the model itself
predicted, the credit is halved immediately rather than left to drift down.
Adapt fast downward, slow upward — an abandoned call cannot be undone afterwards.

**Progressive is the floor, and the controller may raise a proposal to reach it.**
This is the only place in the system where anything raises a number, and it is
deliberate. A call placed for a free agent cannot be abandoned: that agent is
reserved before the call is placed and is waiting when the phone is answered. So
one call per free agent needs no prediction to be safe, and the engine's bound
governs only the calls above that line. Without this, a cautious tick proposes
fewer calls than there are idle agents and predictive mode performs *worse* than
progressive — the model talking the system out of the one thing that never needed
a model. Every clamp still applies to the floor afterwards, including the ones
that reduce it to zero.

**One parameter was tuned against the simulation:** `target_shortfall_eps`, from
0.02 to 0.005. The bound is a per-*tick* probability and the dialer ticks four
times a second, so 2% per tick is a great many chances to be unlucky over a
campaign, while the number that has to stay under the abandon budget is the
total. It lives in the campaign row rather than in the engine precisely so that
tuning it needs no code change.

---

## 8. Failure handling: fail closed, and reconcile rather than guess

**Three provider exceptions, three different responses.** This distinction is the
entire reason the exception hierarchy exists:

| Exception | What we know | What we do |
| --- | --- | --- |
| `ProviderRejected` | nothing was placed, the number is at fault | fail the call, free the agent, spend one of the borrower's attempts |
| `ProviderUnavailable` | nothing was placed, *we* are at fault | fail the call, free the agent, return the borrower without spending an attempt |
| `ProviderTimeout` | **we do not know** | change nothing at all |

The last one looks like a bug and is the only correct move: the call may be
ringing a real person right now. Releasing the agent risks bridging them to a
second borrower while the first is still live; re-dialling risks calling one
person twice about one debt. So the agent stays reserved, the call stays
`INITIATED` with its lease ticking, and the reaper reconciles against the carrier
when it can. Charging borrowers an attempt for *our* outage would slowly mark
perfectly reachable people `EXHAUSTED` — a quiet, permanent loss that nobody
would trace back to an afternoon when a carrier was flaky.

**The intent log.** The call row, carrying the idempotency key we are about to
send, is written and committed *before* the carrier is called. If the worker dies
in the gap, recovery finds a row saying "we may have placed this call" and asks
the carrier about that exact key. Calling first and recording afterwards loses
the call on any crash in between, and a lost call here is a live one ringing a
stranger with nobody responsible for it.

**The circuit breaker is derived, not stored.** The failure rate is recomputed
from the `calls` table every tick rather than accumulated in memory. Every worker
therefore computes the same answer, and a restarted worker does not begin by
believing a dead carrier to be healthy — that removes the "stale state" failure
category rather than handling it. Only the half-open probe claim is persisted,
because twenty workers independently discovering a dead carrier is twenty probe
calls at a provider that asked for one; the claim uses the same compare-and-swap
as agent reservation.

One detail worth stating because it took a simulation run to see: a borrower not
picking up is recorded as `no_answer`, and at a 50% answer rate half of every
window is one. Counting those as carrier failures opened the breaker on a
perfectly healthy provider and held the campaign at zero for the rest of the run
— and the numbers cannot recover while nothing is being dialled. Only outcomes
that are evidence about the *carrier* count.

**While the breaker is open, existing calls are reconciled, not cancelled.**
Cancelling in-flight calls is the one action guaranteed to make a provider outage
worse, because some of those calls have a person on the line.

**Fail closed.** Any exception anywhere in the safety controller yields
`approved = 0` and an `EXCEPTION` row in the decision log. Being wrong in this
direction costs idle agents; being wrong in the other costs a compliance event.

---

## 9. Scale: what breaks first

Measured by `python tasks.py loadtest`: 1,000 agents, 20 worker coroutines, 60
seconds of virtual time, one local PostgreSQL.

```
LOADTEST_NUMBERS
```

**100 agents.** Nothing. One process, one database, a few calls per second of
state churn.

**1,000 agents.** First break: **contention at the head of the AVAILABLE index.**
Every worker runs the same `ORDER BY state_changed_at ... FOR UPDATE SKIP LOCKED`
and they all arrive at the same rows. `SKIP LOCKED` means they do not block, but
they do scan past each other's locked rows, and that scan lengthens with the
number of workers. Fixes in order: reserve in one batch per tick rather than one
row at a time (already done — it is why `reserve()` takes `n`), then add a
randomised bucket column so workers spread across the index instead of queueing
at its head.

Second: `count(*)` in the snapshot query starts to cost real time. Fix: maintain
`campaign_counters` incrementally in the same transaction as the state change.
The table is already sharded 16 ways for exactly this.

**10,000 agents.** First break: **write amplification.** Roughly 10k agents on
90-second cycles is about 110 completions/sec, times ~6 state transitions, times
(row update + event insert) — call it 1,300 writes/sec, plus 3–5k webhooks/sec.
In the order they would actually bite:

1. Connection exhaustion → pgBouncer in transaction mode.
2. `provider_events` table and index bloat → partition by day, drop old partitions.
3. Webhook ingest latency → `COPY` into a staging table in 50ms batches and apply
   in a separate loop. Ingest must stay non-blocking; application may lag.
4. The per-campaign counter row becomes a hot row → sum over the 16 shards on
   read, which the schema already supports.

**100,000 agents.** Shard by campaign across PostgreSQL instances, one pacing
leader per campaign elected with `pg_try_advisory_lock` — not a new system. The
genuinely hard part is not the plumbing: **the abandon budget is global per
campaign.** A campaign spanning shards must sub-allocate the budget per shard and
reconcile, and sub-allocation is necessarily conservative, so utilization is
lost. Worse, per §7, fragmenting the agent pool raises the variance on each
fragment, which costs utilization again. The statistics penalise sharding twice,
and no amount of infrastructure removes either penalty.

---

## 10. What I am least confident about

**Crash reconciliation depends on the provider's status API being both accurate
and available.** When a worker dies after `ANSWERED`, the reaper asks the carrier
what happened to that call. If the carrier is down or lying, the system holds the
agent reserved rather than risk bridging them to a second borrower while the
first is still live. That is the correct trade — a double-bridge is worse than an
idle agent — but it costs utilization at exactly the moment things are already
going badly, and it is a dependency on an external system inside a recovery path
that exists precisely because external systems fail.

Second: the propensity and hazard tables are rebuilt every 15 seconds from the
campaign's own recent history. Early in a campaign they are mostly prior, and the
Wilson bound keeps that safe — but a campaign whose population changes character
mid-run (a new borrower segment loaded at noon) is learning from a mixture, and
nothing here detects the mixture. The changepoint check would catch the abandon
consequences after the fact rather than the cause.

---

## 11. The final question

> How would you build a SmartDialer that gets as much of the utilization benefit
> of predictive dialing as possible, while retaining the deterministic safety
> characteristics of progressive dialing?

**Make progressive the floor, not the alternative.** The system dials 1:1 by
default, and predictive dialing spends a bounded, explicitly-accounted over-dial
credit on top of that line. Three properties make it deterministic:

1. **The credit is a measured budget, not a prediction.** It is moved by an AIMD
   controller driven by observed abandons. A wrong model cannot widen it — only
   observed good outcomes can, one call per clean tick, and a single abandon
   halves it.
2. **The model's output is advisory and clamped by rules it cannot influence.**
   The hard over-dial ratio, the concurrency cap, the dialing window and the
   circuit breaker are pure functions of measured state. The engine is a pure
   function with no I/O; it could not reach a carrier if it wanted to, and a test
   on the import graph keeps it that way.
3. **Degradation is to progressive, not to off.** Stale signals, provider
   trouble, budget exhaustion, a changepoint, or any unhandled exception all
   resolve to `n = agents_available` or below — exactly progressive behaviour,
   with exactly progressive safety characteristics.

Beyond that, most of the remaining benefit comes from shrinking the risk window
rather than predicting harder: pre-warm the agent leg so bridging is a conference
join (50–100ms on the fast mock) rather than a fresh call setup (300–900ms on the
slow one); keep setup time low so exposure is short; skip AMD, which costs 2–4
seconds of dead air that the borrower hears as "hello? hello?"; prefer the fast,
low-variance carrier for over-dial calls, because high post-dial-delay variance
is what makes the timing hazard worthless; and give a connected call with no free
agent a bounded hold with a callback offer instead of a hard drop.

**Utilization is won by reducing uncertainty, not by betting harder on a
prediction.**
