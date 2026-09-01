"""What happens to an agent and a borrower when a call ends.

Extracted so that the worker and the reaper cannot disagree about it. Both of
them finish calls -- the worker when the carrier reports one over, the reaper
when it reconciles a call whose worker died -- and if each had its own copy of
this logic they would drift, quietly, in the direction of whichever path was
tested harder. The one that matters most is the abandon accounting, and it is
the one nobody would notice going wrong.

Two rules, both about not inferring more than we know.

The agent's next state depends on WHERE THEY WERE, not on why the call ended.
Somebody who was talking has notes to write and goes to WRAP_UP; somebody who
was only dialling goes straight back to the pool. Deriving it from the call's
final state instead would put an agent into wrap-up for a call they never
spoke on.

The borrower is DONE only if they actually reached an agent, and the test for
that is `connected_at`, not the call's final state. A call that connected and
then failed still reached the person; a call that completed without ever
connecting did not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from psycopg import AsyncCursor

from smartdialer.core.models import AgentState, Call, CallState
from smartdialer.domain.agents import get_agent, release_agent, transition_agent
from smartdialer.domain.borrowers import mark_done, record_attempt

# How long before a borrower who was not reached is dialled again.
RETRY_AFTER_NO_ANSWER = 900.0
RETRY_AFTER_FAILED = 600.0
# A borrower we hung up on waits longest. Ringing somebody back promptly after
# dropping them is how a compliance event becomes a complaint.
RETRY_AFTER_ABANDONED = 1800.0


@dataclass(frozen=True, slots=True)
class Settlement:
    """What settling one call actually did. Returned so callers can log and
    count it rather than infer it."""

    agent_moved_to: AgentState | None = None
    borrower_done: bool = False
    attempt_spent: bool = False


async def settle_call(
    cur: AsyncCursor,
    *,
    call: Call,
    now: datetime,
    wrap_up_seconds: int,
    log: Any = None,
) -> Settlement:
    """Release the agent and decide the borrower's future. Idempotent.

    Safe to call twice on the same call, which matters because the worker and
    the reaper can both reach a finished call. The agent move is guarded on the
    agent still pointing at THIS call, so a second pass finds an agent who has
    already moved on and leaves them alone.
    """
    agent_moved_to: AgentState | None = None

    if call.agent_id is not None:
        agent = await get_agent(cur, agent_id=call.agent_id)
        # The guard that makes this idempotent, and safe. An agent whose
        # current_call_id points somewhere else has already been settled for
        # this call and reserved for another; touching them here would drop a
        # live call to tidy up a dead one.
        if agent is not None and agent.current_call_id == call.id:
            if agent.state is AgentState.CONNECTED:
                moved = await transition_agent(
                    cur,
                    agent_id=agent.id,
                    expected_version=agent.version,
                    expected_state=AgentState.CONNECTED,
                    target_state=AgentState.WRAP_UP,
                    now=now,
                    wrap_up_ends_at=now + timedelta(seconds=wrap_up_seconds),
                    lease_owner=None,
                    lease_expires_at=None,
                )
                if moved is not None:
                    agent_moved_to = AgentState.WRAP_UP
            elif agent.state in (AgentState.DIALING, AgentState.RESERVED):
                moved = await release_agent(
                    cur,
                    agent_id=agent.id,
                    expected_version=agent.version,
                    expected_state=agent.state,
                    now=now,
                )
                if moved is not None:
                    agent_moved_to = AgentState.AVAILABLE

    if call.connected_at is not None:
        await mark_done(cur, borrower_id=call.borrower_id, outcome=call.state.value)
        settlement = Settlement(agent_moved_to=agent_moved_to, borrower_done=True)
    else:
        await record_attempt(
            cur,
            borrower_id=call.borrower_id,
            now=now,
            outcome=call.failure_reason or call.state.value,
            retry_after_seconds=retry_after_for(call),
        )
        settlement = Settlement(agent_moved_to=agent_moved_to, attempt_spent=True)

    if log is not None:
        log.info(
            "call_settled",
            call_id=str(call.id),
            call_state=call.state.value,
            agent_id=str(call.agent_id) if call.agent_id else None,
            agent_moved_to=agent_moved_to.value if agent_moved_to else None,
            borrower_done=settlement.borrower_done,
        )
    return settlement


def retry_after_for(call: Call) -> float:
    if call.state is CallState.ABANDONED:
        return RETRY_AFTER_ABANDONED
    if call.answered_at is not None:
        return RETRY_AFTER_FAILED
    return RETRY_AFTER_NO_ANSWER


async def move_agent(
    cur: AsyncCursor,
    *,
    agent_id: UUID,
    expected: tuple[AgentState, ...],
    target: AgentState,
    now: datetime,
    log: Any = None,
    **columns: Any,
) -> bool:
    """Read an agent, then compare-and-swap them once.

    Events arrive knowing about a call, not about an agent's row version, so
    there has to be a read first. If the swap misses, something else moved the
    agent between the two and we do NOT retry: forcing the write is how two
    workers end up believing they own one agent. It is logged and left to the
    reaper, which is the component whose job that is.

    An agent already sitting at the target counts as success. The other side of
    a race did exactly what we were about to do, and calling that a conflict
    would turn ordinary concurrency into log noise.
    """
    agent = await get_agent(cur, agent_id=agent_id)
    if agent is not None and agent.state is target:
        return True
    if agent is None or agent.state not in expected:
        if log is not None:
            log.warning(
                "agent_not_in_expected_state",
                agent_id=str(agent_id),
                state=agent.state.value if agent else None,
                expected=[state.value for state in expected],
                target=target.value,
            )
        return False
    moved = await transition_agent(
        cur,
        agent_id=agent_id,
        expected_version=agent.version,
        expected_state=agent.state,
        target_state=target,
        now=now,
        **columns,
    )
    if moved is None:
        if log is not None:
            log.warning(
                "agent_transition_lost_a_race",
                agent_id=str(agent_id),
                target=target.value,
            )
        return False
    return True
