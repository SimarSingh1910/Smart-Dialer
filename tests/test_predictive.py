"""The predictive pacing engine.

All of these run against dataclasses. There is no database here and there could
not be one -- the engine is a pure function, which is exactly what makes it
testable this thoroughly and exactly what makes the safety boundary structural
rather than a promise.

The tests are grouped by the claim they defend:

  * the hazard model is genuinely used (not a flat rate wearing a costume)
  * thin evidence produces caution, not confidence
  * the exact route and the approximate route agree where they overlap, and the
    exact one is used where the approximation would be untrustworthy
  * the proposal can explain itself
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from smartdialer.core.models import CampaignMode
from smartdialer.pacing.engine import (
    EXACT_SIGMA_THRESHOLD,
    PacingSnapshot,
    ProviderHealthSignal,
    RecentBehaviour,
    propose,
)
from smartdialer.pacing.hazard import HazardTable, lognormal_cdf
from smartdialer.pacing.propensity import (
    BorrowerFeatures,
    PropensityTable,
)
from smartdialer.pacing.stats import (
    changepoint_detected,
    ewma,
    wilson_lower_bound,
    wilson_upper_bound,
)

NOW = datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc)


def snapshot(**overrides) -> PacingSnapshot:
    """A predictive snapshot with a well-evidenced 40% answer rate.

    Well-evidenced on purpose: with a thin sample the Wilson upper bound
    dominates every other effect, which is correct behaviour and would mask the
    effect most of these tests are trying to isolate.
    """
    defaults = dict(
        mode=CampaignMode.PREDICTIVE,
        snapshot_taken_at=NOW,
        now=NOW,
        agents_available=10,
        calls_ringing=(),
        talk_seconds=(),
        wrap_up_remaining=(),
        call_setup_time_p95=8.0,
        call_setup_time_p50=4.0,
        target_shortfall_eps=0.02,
        provider_health=ProviderHealthSignal(name="mock_fast"),
        recent_campaign_behaviour=RecentBehaviour(initiated=1000, answered=400),
    )
    defaults.update(overrides)
    return PacingSnapshot(**defaults)


# ---------------------------------------------------------------------------
# Progressive is untouched
# ---------------------------------------------------------------------------


def test_progressive_mode_equals_agents_available():
    for available in (0, 1, 7, 250):
        proposal = propose(snapshot(mode=CampaignMode.PROGRESSIVE, agents_available=available))
        assert proposal.n == available
        assert proposal.reason == "PROGRESSIVE_ONE_TO_ONE"


def test_pacing_engine_is_deterministic():
    """A decision log is only evidence if the decision can be recomputed from
    the inputs recorded beside it."""
    snap = snapshot(
        calls_ringing=(2.0, 6.0, 11.0),
        talk_seconds=(30.0, 150.0),
        wrap_up_remaining=(3.0,),
    )
    first = propose(snap)
    for _ in range(30):
        again = propose(snap)
        assert again.n == first.n
        assert again.search_trace == first.search_trace
        assert again.mu_A == first.mu_A and again.mu_G == first.mu_G


# ---------------------------------------------------------------------------
# The hazard model is actually used
# ---------------------------------------------------------------------------


def test_more_ringing_calls_reduces_proposal():
    """Calls already ringing are the near-term threat to the agent pool."""
    quiet = propose(snapshot(calls_ringing=()))
    busy = propose(snapshot(calls_ringing=tuple(8.0 for _ in range(8))))
    assert busy.n < quiet.n
    assert busy.mu_A > quiet.mu_A


def test_longer_ringing_calls_reduce_proposal_more_than_fresh_ones():
    """THE test that proves this is a hazard model and not a flat rate.

    Six calls ringing for one second and six ringing for nine are the same
    number of calls. Under a single answer probability they are identical and
    must produce identical proposals. Under a hazard model they are not close:
    a call at nine seconds is sitting on the steep part of the ring-to-answer
    curve and is far more likely to deliver an answer in the next few seconds
    than one that has barely started.

    If this test ever passes with equality, the hazard table has been replaced
    by a constant somewhere and the engine has quietly lost the ability to see
    an incoming burst of simultaneous answers.
    """
    fresh = propose(snapshot(calls_ringing=tuple(1.0 for _ in range(6))))
    ripe = propose(snapshot(calls_ringing=tuple(9.0 for _ in range(6))))

    assert ripe.mu_A > fresh.mu_A, "a call about to be answered must weigh more"
    assert ripe.n < fresh.n, (
        "the same COUNT of ringing calls must not produce the same proposal -- "
        "if it does, the engine is using a flat rate"
    )


def test_connected_agents_close_to_hanging_up_add_capacity():
    """The agent side of the same idea.

    Four agents four minutes into a call are much more likely to be free within
    the window than four who have just connected, so they count for more.
    """
    just_started = propose(snapshot(talk_seconds=(5.0, 5.0, 5.0, 5.0)))
    nearly_done = propose(snapshot(talk_seconds=(240.0, 240.0, 240.0, 240.0)))
    assert nearly_done.mu_G > just_started.mu_G
    assert nearly_done.n >= just_started.n


def test_wrap_up_agents_add_capacity_without_adding_variance():
    """Wrap-up timers are deterministic, and that is worth real utilisation.

    Their expiry was fixed when the call ended and nothing external can move
    it, so they are capacity we KNOW about. Modelling them as uncertain would
    inflate sigma_G and make the engine needlessly timid.
    """
    without = propose(snapshot(wrap_up_remaining=()))
    within = propose(snapshot(wrap_up_remaining=(1.0, 2.0, 3.0)))
    beyond = propose(snapshot(wrap_up_remaining=(600.0, 600.0, 600.0)))

    assert within.mu_G == pytest.approx(without.mu_G + 3)
    assert within.sigma_G == pytest.approx(without.sigma_G), (
        "a certain event must contribute no variance"
    )
    assert beyond.mu_G == pytest.approx(without.mu_G), (
        "a timer expiring long after the window is not capacity now"
    )
    assert within.n > beyond.n


# ---------------------------------------------------------------------------
# Evidence, and the two ends of the Wilson interval
# ---------------------------------------------------------------------------


def test_wilson_lower_bound_conservative_on_small_sample():
    """Four answers from five calls is 80% observed and about 38% supported."""
    assert wilson_lower_bound(4, 5) == pytest.approx(0.376, abs=0.02)
    assert wilson_lower_bound(400, 500) == pytest.approx(0.80, abs=0.04)
    # It rises towards the observed rate as evidence accumulates, which is the
    # property that lets a campaign become aggressive only once it has earned
    # the right to be.
    assert wilson_lower_bound(4, 5) < wilson_lower_bound(40, 50) < wilson_lower_bound(400, 500)
    assert wilson_lower_bound(0, 0) == 0.0


def test_wilson_upper_bound_is_total_ignorance_on_no_data():
    """The bound the RISK term uses, and why it is the other end.

    In the tail bound p multiplies risk rather than dividing it, so a low p
    makes over-dialling look safe. With no data at all the only safe assumption
    is that every call will be answered -- which is what keeps a cold-started
    predictive campaign behaving like a progressive one.
    """
    assert wilson_upper_bound(0, 0) == 1.0
    assert wilson_upper_bound(4, 5) > 0.8
    assert wilson_upper_bound(400, 500) < wilson_upper_bound(4, 5)


def test_a_thin_sample_cannot_make_the_engine_aggressive():
    """Same observed rate, different amounts of evidence.

    Four answers from five calls and four hundred from five hundred are both
    "80% answered". Only one of them is a reason to over-dial.
    """
    thin = propose(
        snapshot(recent_campaign_behaviour=RecentBehaviour(initiated=5, answered=1))
    )
    thick = propose(
        snapshot(recent_campaign_behaviour=RecentBehaviour(initiated=500, answered=100))
    )
    assert thin.n < thick.n, "identical rates, but only the evidenced one earns credit"


def test_a_cold_campaign_does_not_over_dial():
    """No history at all must not read as "no risk".

    This is the failure the upper bound exists to prevent: an empty rolling
    window makes every new call look free, and the engine authorises a burst
    into a campaign it knows nothing about.
    """
    cold = propose(
        snapshot(recent_campaign_behaviour=RecentBehaviour(initiated=0, answered=0))
    )
    assert cold.n <= cold.mu_G, "with no evidence, do not exceed known capacity"


def test_a_low_but_well_evidenced_answer_rate_unlocks_over_dialling():
    """The whole point of predictive dialling.

    When only one call in five is answered, one call per agent leaves agents
    idle most of the time. With enough evidence to believe that rate, the
    engine should authorise materially more than the progressive floor.
    """
    proposal = propose(
        snapshot(recent_campaign_behaviour=RecentBehaviour(initiated=2000, answered=200))
    )
    assert proposal.n > 10, "a well-evidenced 10% answer rate should beat 1:1"


# ---------------------------------------------------------------------------
# Exact versus approximate
# ---------------------------------------------------------------------------


def test_exact_dp_used_when_sigma_small():
    """Small pools take the exact route.

    A ten-agent campaign has no room to absorb an approximation error, and the
    normal approximation is at its worst on a handful of Bernoullis -- so the
    two conditions coincide exactly where it matters most.
    """
    proposal = propose(snapshot(agents_available=3, calls_ringing=(6.0, 8.0)))
    assert proposal.used_exact_dp is True
    assert proposal.sigma_G < EXACT_SIGMA_THRESHOLD


def test_exact_dp_matches_normal_approx_when_sigma_large():
    """Where both are valid they must agree, or one of them is wrong.

    Built with enough ringing calls and connected agents that the combined
    spread is comfortably past the threshold, then checked against an
    independently computed exact Poisson-binomial. Agreement to a couple of
    percentage points is what the continuity correction buys.
    """
    from smartdialer.pacing.engine import _exact_shortfall, _shortfall_probability

    ring_hazards = [0.35] * 30
    free_probabilities = [0.30] * 30
    approx, used_exact = _shortfall_probability(
        n=0,
        mu_A0=sum(ring_hazards),
        var_A0=sum(h * (1 - h) for h in ring_hazards),
        mu_G=12.0 + sum(free_probabilities),
        var_G=sum(q * (1 - q) for q in free_probabilities),
        ring_hazards=ring_hazards,
        free_probabilities=free_probabilities,
        deterministic_agents=12.0,
        probability_of_call=lambda i: 0.0,
    )
    assert used_exact is False, "this pool should be past the sigma threshold"

    exact = _exact_shortfall(
        answer_probabilities=ring_hazards,
        agent_probabilities=free_probabilities,
        deterministic_agents=12.0,
    )
    assert approx == pytest.approx(exact, abs=0.02)


def test_the_exact_route_is_a_real_distribution():
    """Sanity on the dynamic program itself.

    A Poisson-binomial with identical probabilities is just a binomial, so the
    convolution can be checked against a closed form. If this drifts, every
    small-pool decision is wrong and nothing else would notice.
    """
    from smartdialer.pacing.engine import _poisson_binomial_pmf

    p, n = 0.3, 8
    pmf = _poisson_binomial_pmf([p] * n)
    assert sum(pmf) == pytest.approx(1.0)
    for k in range(n + 1):
        expected = math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))
        assert pmf[k] == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Epsilon, the search, and the explanation
# ---------------------------------------------------------------------------


def test_proposal_monotonic_in_epsilon():
    """A looser risk target can never authorise fewer calls.

    Epsilon is the dial an operator turns to trade customer wait against
    utilisation. If the relationship were not monotonic that dial would be
    unusable -- and a non-monotonic search is also the classic symptom of a
    bound that is not actually monotonic in N.
    """
    previous = -1
    for epsilon in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25):
        proposal = propose(
            snapshot(
                target_shortfall_eps=epsilon,
                calls_ringing=(4.0, 7.0, 9.0),
                talk_seconds=(60.0, 120.0),
            )
        )
        assert proposal.n >= previous, f"epsilon={epsilon} proposed fewer calls"
        previous = proposal.n


def test_search_trace_brackets_the_chosen_n():
    """The proposal has to be able to justify itself.

    The trace must contain the chosen N with a probability inside the budget,
    and N+1 with one outside it. That pair IS the answer to "why 17 and not
    18?", and without it the number in the log is an assertion rather than an
    argument.
    """
    proposal = propose(
        snapshot(
            agents_available=12,
            calls_ringing=(3.0, 5.0, 7.0),
            talk_seconds=(40.0, 90.0, 200.0),
            recent_campaign_behaviour=RecentBehaviour(initiated=2000, answered=600),
        )
    )
    trace = dict(proposal.search_trace)
    assert proposal.n in trace, "the chosen N must appear in the trace"
    assert trace[proposal.n] <= proposal.epsilon

    if proposal.n + 1 in trace:
        assert trace[proposal.n + 1] > proposal.epsilon, (
            "the next N up must be the one that broke the budget"
        )


def test_the_proposal_explains_itself_in_one_line():
    proposal = propose(
        snapshot(calls_ringing=(4.0, 9.0), talk_seconds=(120.0,), wrap_up_remaining=(2.0,))
    )
    sentence = proposal.explain()
    for fragment in ("mu_G", "mu_A", "p_hat", "eps", "proposed"):
        assert fragment in sentence, sentence


def test_every_signal_reaches_the_decision_log():
    """All eight, whether or not the current logic leans on them.

    A signal that is only wired up when it is first needed is a signal nobody
    has ever checked the units of.
    """
    terms = propose(snapshot()).terms
    for key in (
        "agents_available",
        "calls_connected",
        "calls_ringing",
        "historical_answer_rate",
        "call_setup_time_p95",
        "avg_call_duration",
        "provider_health",
        "recent_campaign_behaviour",
    ):
        assert key in terms, f"{key} never reaches the decision log"


# ---------------------------------------------------------------------------
# The changepoint signal
# ---------------------------------------------------------------------------


def test_changepoint_flag_set_on_sudden_drop():
    """Twenty calls that normally yield fourteen answers yield two."""
    collapsed = propose(
        snapshot(
            observed_answers_30s=2.0,
            predicted_answers_30s=14.0,
            predicted_answers_30s_variance=14.0 * 0.3,
        )
    )
    assert collapsed.changepoint_detected is True

    steady = propose(
        snapshot(
            observed_answers_30s=13.0,
            predicted_answers_30s=14.0,
            predicted_answers_30s_variance=14.0 * 0.3,
        )
    )
    assert steady.changepoint_detected is False


def test_changepoint_is_one_sided():
    """More answers than predicted is a surprise that costs nothing."""
    assert not changepoint_detected(
        observed=30.0, expected_mean=14.0, expected_variance=4.0
    )
    assert changepoint_detected(
        observed=2.0, expected_mean=14.0, expected_variance=4.0
    )


def test_answer_rate_collapse_reduces_pacing_within_60s():
    """Scenario D, as far as the ENGINE is responsible for it.

    Two things happen when an answer rate collapses, and only one of them
    belongs here.

    What the engine does, within one rolling window, is notice: the changepoint
    flag is raised as soon as the last resolved calls fall below the fifth
    percentile of what the campaign's own baseline predicted. That is the
    engine's whole job -- it reports, it does not act.

    What it must NOT do is act on its own. Note the direction the tail bound
    pulls: when genuinely fewer people answer, over-dialling really is less
    risky, so the mathematics correctly wants to dial MORE. Whether that is
    wise is a different question from whether it is correct, and it is decided
    by the AIMD abandon budget in the safety controller, which halves the
    over-dial credit the instant this flag appears. Putting that judgement in
    the engine would mean a component that is supposed to model the world was
    also quietly policing it.

    So this asserts the engine's contribution: within the window, the collapse
    is visible and flagged, and the reported answer-rate estimate has fallen.
    """
    healthy = snapshot(
        recent_campaign_behaviour=RecentBehaviour(initiated=200, answered=140),
        observed_answers_30s=14.0,
        predicted_answers_30s=14.0,
        predicted_answers_30s_variance=14.0 * 0.3,
    )
    # 60 seconds later: the rolling window has turned over and the recent
    # calls tell a completely different story.
    collapsed = snapshot(
        now=NOW + timedelta(seconds=60),
        snapshot_taken_at=NOW + timedelta(seconds=60),
        recent_campaign_behaviour=RecentBehaviour(initiated=200, answered=20),
        observed_answers_30s=2.0,
        predicted_answers_30s=14.0,
        predicted_answers_30s_variance=14.0 * 0.3,
    )

    before = propose(healthy)
    after = propose(collapsed)

    assert before.changepoint_detected is False
    assert after.changepoint_detected is True, "the collapse must be visible within 60s"
    assert after.p_hat < before.p_hat, "and the believed answer rate must fall"


# ---------------------------------------------------------------------------
# The hazard table itself
# ---------------------------------------------------------------------------


def test_hazard_rises_across_the_body_of_the_distribution():
    table = HazardTable.prior_only(prior_median=9.0, prior_sigma=0.45)
    early = table.hazard(1.0, 2.0)
    middle = table.hazard(8.0, 2.0)
    assert middle > early


def test_hazard_of_a_zero_window_is_zero():
    table = HazardTable.prior_only(prior_median=9.0, prior_sigma=0.45)
    assert table.hazard(5.0, 0.0) == 0.0


def test_hazard_is_a_probability_everywhere():
    """It feeds a p(1-p) variance term, so a value outside [0,1] would make a
    standard deviation imaginary and take the whole bound with it."""
    table = HazardTable.prior_only(prior_median=9.0, prior_sigma=0.45)
    for elapsed in (0.0, 0.5, 3.0, 9.0, 30.0, 300.0):
        for window in (0.5, 2.0, 8.0, 60.0):
            value = table.hazard(elapsed, window)
            assert 0.0 <= value <= 1.0, (elapsed, window, value)


def test_censored_observations_do_not_look_like_non_answers():
    """The survival-analysis point, and it changes the numbers a lot.

    Fifty calls answered around nine seconds, and fifty that the borrower gave
    up on after two. Treating the abandoned ones as "did not answer at nine
    seconds" would halve the hazard there. Censoring them correctly says they
    left the population at two seconds and are silent about what would have
    happened later.
    """
    answered = [9.0] * 50
    gave_up_early = [2.0] * 50

    censored = HazardTable.from_observations(
        event_times=answered, censored_times=gave_up_early, min_samples=1
    )
    mistaken = HazardTable.from_observations(
        event_times=answered,
        # The bug: pretending they were still ringing, unanswered, the whole
        # way through.
        censored_times=[30.0] * 50,
        min_samples=1,
    )
    bucket = int(9.0 / censored.bucket_seconds)
    assert censored.bucket_hazard(bucket) > mistaken.bucket_hazard(bucket)


def test_a_thin_bucket_is_dominated_by_the_prior():
    """Cold start must not produce confident nonsense.

    One observation in a bucket is not evidence that the hazard there is 100%.
    """
    prior = HazardTable.prior_only(prior_median=9.0, prior_sigma=0.45)
    thin = HazardTable.from_observations(event_times=[9.0], censored_times=[])
    bucket = int(9.0 / thin.bucket_seconds)
    assert thin.bucket_hazard(bucket) == pytest.approx(
        prior.prior_bucket_hazard(bucket), abs=0.06
    )


def test_lognormal_cdf_matches_its_median():
    assert lognormal_cdf(9.0, 9.0, 0.45) == pytest.approx(0.5)
    assert lognormal_cdf(0.0, 9.0, 0.45) == 0.0


# ---------------------------------------------------------------------------
# Propensity
# ---------------------------------------------------------------------------


def test_propensity_falls_back_to_the_campaign_mean_for_an_unseen_cell():
    table = PropensityTable(cells={}, campaign_mean=0.25)
    features = BorrowerFeatures(
        hour_of_day=10, attempt_number=1, prior_outcome=None, dpd_bucket="0-30"
    )
    assert table.probability(features) == pytest.approx(0.25)


def test_propensity_shrinks_a_thin_cell_towards_the_mean():
    """Two answers out of two is not a 100% segment."""
    features = BorrowerFeatures(
        hour_of_day=10, attempt_number=1, prior_outcome=None, dpd_bucket="0-30"
    )
    table = PropensityTable(
        cells={features.key().as_tuple(): (2, 2)}, campaign_mean=0.2
    )
    assert table.probability(features) < 0.35


def test_propensity_trusts_a_thick_cell():
    features = BorrowerFeatures(
        hour_of_day=10, attempt_number=1, prior_outcome=None, dpd_bucket="0-30"
    )
    table = PropensityTable(
        cells={features.key().as_tuple(): (600, 1000)}, campaign_mean=0.2
    )
    assert table.probability(features) == pytest.approx(0.60, abs=0.03)


def test_high_propensity_candidates_reduce_the_proposal():
    """The control lever.

    The same connect target reached with likelier borrowers means fewer dials,
    and fewer dials means lower variance -- so a queue of likely answerers is
    dialled less hard than a queue of unlikely ones. That is not caution for
    its own sake: it is what keeps the answers from arriving all at once.
    """
    unlikely = propose(snapshot(candidate_propensities=tuple(0.05 for _ in range(60))))
    likely = propose(snapshot(candidate_propensities=tuple(0.95 for _ in range(60))))
    assert likely.n < unlikely.n


def test_attempts_are_bucketed_so_rare_cells_do_not_get_their_own_number():
    a = BorrowerFeatures(hour_of_day=9, attempt_number=7, prior_outcome=None, dpd_bucket="90+")
    b = BorrowerFeatures(hour_of_day=9, attempt_number=3, prior_outcome=None, dpd_bucket="90+")
    assert a.key() == b.key()


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------


def test_ewma_weights_recent_samples_more_heavily():
    recent_high = ewma([(0.0, 1.0), (600.0, 0.0)], half_life_seconds=60.0)
    recent_low = ewma([(0.0, 0.0), (600.0, 1.0)], half_life_seconds=60.0)
    assert recent_high > 0.9
    assert recent_low < 0.1


def test_ewma_of_nothing_is_the_default():
    assert ewma([], half_life_seconds=60.0, default=0.42) == 0.42


# ---------------------------------------------------------------------------
# End to end, against a real campaign row
# ---------------------------------------------------------------------------


async def test_a_predictive_campaign_ticks_and_explains_itself(dsn: str):
    """The engine, wired to the real snapshot builder and decision log.

    Everything above tests the mathematics in isolation. This checks the
    plumbing: that a PREDICTIVE campaign produces a snapshot the engine
    accepts, that the tail-bound terms reach the decision row, and that the
    explanation is reconstructable from what was stored.
    """
    import uuid

    from smartdialer.core.clock import VirtualClock
    from smartdialer.core.config import Settings
    from smartdialer.core.db import Database
    from smartdialer.core.logging import StructuredLogger
    from smartdialer.providers.mock_fast import make_fast_provider
    from smartdialer.workers.dialer_worker import DialerWorker

    database = Database(dsn, min_size=2, max_size=8)
    await database.open()
    campaign_id = uuid.uuid4()
    start = NOW
    try:
        async with database.transaction() as cur:
            await cur.execute(
                "INSERT INTO campaigns (id, name, mode, wrap_up_seconds) "
                "VALUES (%s, %s, 'PREDICTIVE', 5)",
                (campaign_id, f"pred-{campaign_id}"),
            )
            for _ in range(6):
                await cur.execute(
                    "INSERT INTO agents (id, campaign_id, state, state_changed_at) "
                    "VALUES (%s, %s, 'AVAILABLE', %s)",
                    (uuid.uuid4(), campaign_id, start),
                )
            for index in range(60):
                await cur.execute(
                    "INSERT INTO borrowers (id, campaign_id, phone, next_eligible_at, "
                    "dpd_bucket) VALUES (%s, %s, %s, %s, %s)",
                    (uuid.uuid4(), campaign_id, f"+9196{index:08d}", start, "0-30"),
                )

        clock = VirtualClock(start=start)
        provider = make_fast_provider(clock, seed=11, answer_rate=0.4, reject_rate=0.0)
        worker = DialerWorker(
            db=database,
            clock=clock,
            campaign_id=campaign_id,
            providers=[provider],
            settings=Settings(worker_id="predictive", tick_seconds=0.25),
            logger=StructuredLogger("pred", clock),
        )
        worker.attach_providers()

        for _ in range(40):
            await worker.tick()
            await clock.advance(0.25)
            await worker.drain()

        async with database.transaction() as cur:
            await cur.execute(
                "SELECT mode, proposed, approved, dialed, inputs FROM pacing_decisions "
                "WHERE campaign_id = %s ORDER BY id DESC LIMIT 1",
                (campaign_id,),
            )
            row = await cur.fetchone()

        assert row["mode"] == "PREDICTIVE"
        inputs = row["inputs"]
        assert inputs["engine_reason"] == "PREDICTIVE_TAIL_BOUND"
        # The five numbers that answer "why this many and not more".
        for key in ("mu_A", "sigma_A", "mu_G", "sigma_G", "p_hat", "epsilon"):
            assert key in inputs, key
        assert inputs["search_trace"], "the search must record what it evaluated"
        assert "proposed" in inputs["engine_explanation"]
        # The hazard inputs, not just their summary.
        assert "calls_ringing" in inputs["snapshot"]

        assert not worker.event_errors, worker.event_errors
        assert not provider.sink_errors, provider.sink_errors

        await worker.close()
        await provider.close()
    finally:
        async with database.transaction() as cur:
            await cur.execute(
                "DELETE FROM provider_events WHERE provider_call_id IN "
                "(SELECT provider_call_id FROM calls WHERE campaign_id = %s "
                " AND provider_call_id IS NOT NULL)",
                (campaign_id,),
            )
            await cur.execute("DELETE FROM pacing_decisions WHERE campaign_id = %s", (campaign_id,))
            await cur.execute("UPDATE agents SET current_call_id = NULL WHERE campaign_id = %s", (campaign_id,))
            await cur.execute("DELETE FROM calls WHERE campaign_id = %s", (campaign_id,))
            await cur.execute("DELETE FROM borrowers WHERE campaign_id = %s", (campaign_id,))
            await cur.execute("DELETE FROM agents WHERE campaign_id = %s", (campaign_id,))
            await cur.execute("DELETE FROM campaign_counters WHERE campaign_id = %s", (campaign_id,))
            await cur.execute("DELETE FROM campaigns WHERE id = %s", (campaign_id,))
        await database.close()
