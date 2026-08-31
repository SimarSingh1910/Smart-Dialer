# Architecture decision record

Written as the build progresses. Sections marked **(pending)** are filled in at the
step that produces the evidence.

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

## 7. Predictive pacing **(pending — steps 7 and 8)**

## 8. Failure handling **(pending — steps 4 and 6)**

## 9. Scale: what breaks first **(pending — step 10 supplies the measurements)**

## 10. What I am least confident about **(pending)**
