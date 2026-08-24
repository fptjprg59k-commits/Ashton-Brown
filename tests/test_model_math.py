"""Distributions, pricing, game script, and usage - the pure functions."""

from __future__ import annotations

import math
import statistics

import pytest

from nflprops.distributions import Sampler, fg_make_prob, logistic, yardline_to_fg_distance
from nflprops.gamescript import pass_rate, seconds_per_play, urgency_z
from nflprops.models import (
    Alignment, DefenseProfile, InjuryStatus, Player, Position, TeamState,
)
from nflprops.pricing import (
    american_to_decimal, american_to_prob, devig_multiplicative, devig_shin,
    prob_to_american, price,
)
from nflprops.priors import Priors
from nflprops.usage import build_pool, rest_hazard, shrink


# --------------------------------------------------------------------------
# Yardage distributions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("explosiveness", [0.0, 0.35, 0.7, 1.0])
def test_reception_yardage_is_mean_preserving(explosiveness):
    """Explosiveness must reshape the distribution without moving its mean.

    This is the whole point of the parameterisation: if raising explosiveness
    also raised the projection, every tuning decision would silently smuggle in
    a change to expected production.
    """
    s = Sampler(Priors(), seed=1)
    target = 13.0
    draws = [s.reception_yards(target, explosiveness) for _ in range(60000)]
    assert statistics.fmean(draws) == pytest.approx(target, abs=0.45)


def test_explosiveness_increases_variance_and_lowers_median():
    """A boom-or-bust profile should miss low more often at the same mean."""
    s = Sampler(Priors(), seed=2)
    possession = [s.reception_yards(13.0, 0.05) for _ in range(40000)]
    deep = [s.reception_yards(13.0, 0.95) for _ in range(40000)]

    assert statistics.pstdev(deep) > statistics.pstdev(possession) * 1.5
    assert statistics.median(deep) < statistics.median(possession)
    # The deep profile pays off through a fat right tail.
    assert sum(y > 35 for y in deep) > sum(y > 35 for y in possession) * 2


def test_rush_yardage_dispersion_matches_nfl():
    """Per-carry SD should land near the observed 5-6 yards at a 4.35 mean."""
    s = Sampler(Priors(), seed=3)
    draws = [s.rush_yards(4.35, 0.35) for _ in range(60000)]
    assert statistics.fmean(draws) == pytest.approx(4.35, abs=0.2)
    assert 4.0 < statistics.pstdev(draws) < 7.0
    # Negative plays must be possible but not common.
    assert 0.04 < sum(y < 0 for y in draws) / len(draws) < 0.22


# --------------------------------------------------------------------------
# Kicking
# --------------------------------------------------------------------------


def test_fg_curve_matches_league_make_rates():
    p = Priors()
    for distance, expected in [(20, 0.99), (40, 0.92), (50, 0.75), (55, 0.61), (60, 0.45)]:
        assert fg_make_prob(distance, p) == pytest.approx(expected, abs=0.04)


def test_fg_probability_is_monotone_in_distance():
    p = Priors()
    probs = [fg_make_prob(d, p) for d in range(20, 66, 5)]
    assert all(a > b for a, b in zip(probs, probs[1:]))


def test_yardline_to_fg_distance():
    # From the opponent's 20 (yardline 80): 20 to the goal + 17 for snap/holder.
    assert yardline_to_fg_distance(80) == 37


# --------------------------------------------------------------------------
# Game script
# --------------------------------------------------------------------------


def test_trailing_teams_pass_more_and_leading_teams_pass_less():
    p = Priors()
    team = TeamState(abbr="X", base_pass_rate=0.575)
    neutral = pass_rate(team, 0, 1800, p)
    trailing = pass_rate(team, -14, 1800, p)
    leading = pass_rate(team, +14, 1800, p)

    assert trailing > neutral > leading
    assert neutral == pytest.approx(0.575, abs=0.01)


def test_script_effect_grows_as_the_clock_shrinks():
    """Down seven early is a normal game; down seven late is a two-minute drill."""
    p = Priors()
    team = TeamState(abbr="X", base_pass_rate=0.575)
    early = pass_rate(team, -7, 1700, p)
    late = pass_rate(team, -7, 240, p)
    assert late > early


def test_pace_speeds_up_when_trailing_and_slows_when_leading():
    p = Priors()
    team = TeamState(abbr="X")
    assert seconds_per_play(team, -14, 900, p) < seconds_per_play(team, 0, 900, p)
    assert seconds_per_play(team, +14, 900, p) > seconds_per_play(team, 0, 900, p)


def test_urgency_saturates():
    """tanh keeps a 40-point deficit from producing an absurd play-call."""
    p = Priors()
    z_scale = p.get("script.z_scale")
    assert abs(urgency_z(-40, 300, z_scale)) <= 1.0
    assert abs(urgency_z(-400, 60, z_scale)) <= 1.0


def test_pass_rate_stays_inside_bounds():
    p = Priors()
    team = TeamState(abbr="X", base_pass_rate=0.575)
    for diff in (-45, -21, 0, 21, 45):
        r = pass_rate(team, diff, 120, p)
        assert p.get("script.pass_rate_min") <= r <= p.get("script.pass_rate_max")


# --------------------------------------------------------------------------
# Usage
# --------------------------------------------------------------------------


def _team_with_two_receivers() -> TeamState:
    team = TeamState(abbr="X")
    team.players = [
        Player("qb", "QB", Position.QB, "X", is_starter_qb=True),
        Player("wr1", "WR1", Position.WR, "X", target_share=0.30,
               alignment=Alignment.PERIMETER),
        Player("wr2", "WR2", Position.WR, "X", target_share=0.20,
               alignment=Alignment.SLOT),
        Player("rb", "RB", Position.RB, "X", rush_share=0.70, target_share=0.10,
               alignment=Alignment.BACKFIELD),
    ]
    return team


def test_injured_out_player_share_flows_to_teammates():
    """A co-star's absence should raise the survivors' absolute share."""
    team = _team_with_two_receivers()
    defense = DefenseProfile()

    before = build_pool(team, defense)
    i = before.index_of("wr2")
    share_before = before.target_w[i] / before.target_total

    team.player("wr1").injury = InjuryStatus.OUT
    after = build_pool(team, defense)

    assert after.index_of("wr1") == -1
    j = after.index_of("wr2")
    share_after = after.target_w[j] / after.target_total
    assert share_after > share_before


def test_scheme_funnel_shifts_targets_by_alignment():
    """A defense conceding underneath should push volume to the slot."""
    team = _team_with_two_receivers()
    neutral = build_pool(team, DefenseProfile())
    funnel = build_pool(team, DefenseProfile(slot_funnel=1.4, perimeter_funnel=0.75))

    def slot_share(pool):
        return pool.target_w[pool.index_of("wr2")] / pool.target_total

    assert slot_share(funnel) > slot_share(neutral)


def test_shadow_coverage_suppresses_both_volume_and_efficiency():
    team = _team_with_two_receivers()
    plain = build_pool(team, DefenseProfile())
    shadowed = build_pool(team, DefenseProfile(shadow={"wr1": 0.75}))

    i = plain.index_of("wr1")
    j = shadowed.index_of("wr1")
    assert shadowed.target_w[j] / shadowed.target_total < plain.target_w[i] / plain.target_total
    assert shadowed.eff_mult[j] < plain.eff_mult[i]


def test_rest_hazard_only_applies_to_leading_stars_late():
    p = Priors()
    star = Player("rb", "RB", Position.RB, "X", star=True)
    role = Player("rb2", "RB2", Position.RB, "X", star=False)

    assert rest_hazard(star, margin=28, seconds_remaining=400, priors=p) > 0
    assert rest_hazard(role, margin=28, seconds_remaining=400, priors=p) == 0
    # Trailing teams never rest anyone.
    assert rest_hazard(star, margin=-28, seconds_remaining=400, priors=p) == 0
    # Nor does anyone rest in the third quarter.
    assert rest_hazard(star, margin=28, seconds_remaining=1500, priors=p) == 0


def test_rest_hazard_rises_with_margin():
    p = Priors()
    star = Player("rb", "RB", Position.RB, "X", star=True)
    assert (
        rest_hazard(star, 35, 400, p)
        > rest_hazard(star, 21, 400, p)
        > rest_hazard(star, 10, 400, p)
    )


def test_shrinkage_weights_prior_heavily_on_small_samples():
    # 12 observations against a prior worth 45: the prior should dominate.
    blended = shrink(observed=1.0, prior=0.0, n=12, prior_strength=45)
    assert blended < 0.25
    # With a large sample the observation takes over.
    assert shrink(observed=1.0, prior=0.0, n=500, prior_strength=45) > 0.9
    # No data at all returns the prior untouched.
    assert shrink(observed=1.0, prior=0.4, n=0, prior_strength=45) == 0.4


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("odds", [-500, -180, -110, 100, 145, 900])
def test_american_probability_round_trip(odds):
    p = american_to_prob(odds)
    assert prob_to_american(p) == pytest.approx(odds, abs=1)


def test_decimal_conversion():
    assert american_to_decimal(100) == pytest.approx(2.0)
    assert american_to_decimal(-110) == pytest.approx(1.909, abs=0.001)


def test_implied_probabilities_of_a_two_way_market_exceed_one():
    """That excess is the hold, and is exactly what de-vigging removes."""
    assert american_to_prob(-110) + american_to_prob(-110) > 1.0


@pytest.mark.parametrize("method", [devig_multiplicative, devig_shin])
def test_devig_returns_a_proper_distribution(method):
    a, b = method(-110, -110)
    assert a + b == pytest.approx(1.0)
    assert a == pytest.approx(0.5, abs=0.01)


def test_devig_preserves_ordering_on_a_lopsided_market():
    a, b = devig_shin(-400, +320)
    assert a + b == pytest.approx(1.0)
    assert a > b
    assert a < american_to_prob(-400)  # vig removed, so the favourite comes down


def test_positive_edge_produces_positive_expected_value():
    p = price(model_prob=0.60, odds=-110, opposing_odds=-110)
    assert p.edge > 0
    assert p.ev_per_unit > 0
    assert p.kelly_fraction > 0


def test_negative_edge_never_recommends_a_stake():
    p = price(model_prob=0.40, odds=-110, opposing_odds=-110)
    assert p.edge < 0
    assert p.ev_per_unit < 0
    assert p.kelly_fraction == 0.0


def test_push_probability_pulls_expected_value_toward_zero():
    """A push refunds the stake, so it dilutes the bet in both directions.

    A profitable bet loses expected value to pushes (some of the edge gets
    refunded instead of paid), and a losing bet recovers some. Whole-number
    lines are therefore not a free option in either direction.
    """
    good_flat = price(model_prob=0.58, odds=-110, opposing_odds=-110, p_push=0.0)
    good_push = price(model_prob=0.58, odds=-110, opposing_odds=-110, p_push=0.20)
    assert 0 < good_push.ev_per_unit < good_flat.ev_per_unit

    bad_flat = price(model_prob=0.40, odds=-110, opposing_odds=-110, p_push=0.0)
    bad_push = price(model_prob=0.40, odds=-110, opposing_odds=-110, p_push=0.20)
    assert bad_flat.ev_per_unit < bad_push.ev_per_unit < 0


def test_kelly_is_capped():
    p = price(model_prob=0.99, odds=+500, opposing_odds=-900, kelly_cap=0.05)
    assert p.kelly_fraction <= 0.05


def test_one_sided_devig_is_flagged_as_inexact():
    exact = price(0.5, -110, opposing_odds=-110)
    approx = price(0.5, -110, opposing_odds=None)
    assert exact.devig_exact is True
    assert approx.devig_exact is False


def test_logistic_is_numerically_stable_at_extremes():
    assert logistic(-800) == pytest.approx(0.0, abs=1e-12)
    assert logistic(800) == pytest.approx(1.0, abs=1e-12)
    assert logistic(0) == pytest.approx(0.5)
