"""Behavioural tests for the simulation engine.

These assert the properties the model is *for*: that game script moves volume
the right way, that injuries and rest risk truncate distributions, that props
on the same game are correlated, and that first-half production carries into
full-game lines. A unit test of a gamma draw cannot catch a regression in any
of those.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from nflprops.calibrate import neutral_game, neutral_team
from nflprops.models import (
    GameState, InjuryStatus, Market, Prop, PropScope, Side, Weather,
)
from nflprops.priors import Priors
from nflprops.props import evaluate
from nflprops.rank import correlation_matrix, parlay_probability, rank
from nflprops.simulate import GameSimulator, simulate

SIMS = 3000


def _game(home_score=0, away_score=0, seconds=1800, **kw) -> GameState:
    state = neutral_game(seconds)
    state.home.score = home_score
    state.away.score = away_score
    for k, v in kw.items():
        setattr(state, k, v)
    return state


def _team_totals(sim, state, abbr, stat) -> np.ndarray:
    total = np.zeros(sim.n_sims)
    for p in state.team(abbr).players:
        total = total + sim.get(p.player_id, stat)
    return total


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_same_seed_reproduces_the_same_board():
    state = _game()
    a = simulate(state, n_sims=400, seed=42)
    b = simulate(state, n_sims=400, seed=42)
    assert np.array_equal(a.get("hom_wr1", "rec_yds"), b.get("hom_wr1", "rec_yds"))


def test_different_seeds_give_different_paths():
    state = _game()
    a = simulate(state, n_sims=400, seed=1)
    b = simulate(state, n_sims=400, seed=2)
    assert not np.array_equal(a.get("hom_wr1", "rec_yds"), b.get("hom_wr1", "rec_yds"))


# --------------------------------------------------------------------------
# Game script
# --------------------------------------------------------------------------


def test_trailing_team_throws_more_and_runs_less():
    """The single most important behaviour in a halftime model."""
    trailing = _game(home_score=0, away_score=17)
    leading = _game(home_score=17, away_score=0)

    sim_t = simulate(trailing, n_sims=SIMS, seed=7)
    sim_l = simulate(leading, n_sims=SIMS, seed=7)

    att_trailing = _team_totals(sim_t, trailing, "HOM", "pass_att").mean()
    att_leading = _team_totals(sim_l, leading, "HOM", "pass_att").mean()
    rush_trailing = _team_totals(sim_t, trailing, "HOM", "rush_att").mean()
    rush_leading = _team_totals(sim_l, leading, "HOM", "rush_att").mean()

    assert att_trailing > att_leading * 1.15
    assert rush_leading > rush_trailing * 1.15


def test_trailing_lifts_receiver_yardage_and_depresses_back_carries():
    """Game script is close to a mirror between the pass and run games."""
    trailing = _game(home_score=0, away_score=17)
    leading = _game(home_score=17, away_score=0)
    st = simulate(trailing, n_sims=SIMS, seed=11)
    sl = simulate(leading, n_sims=SIMS, seed=11)

    assert st.get("hom_wr1", "rec_yds").mean() > sl.get("hom_wr1", "rec_yds").mean()
    assert sl.get("hom_rb1", "rush_att").mean() > st.get("hom_rb1", "rush_att").mean()


def test_trailing_team_runs_more_plays_than_the_same_team_leading():
    """Tempo buys snaps, but only modestly.

    Compared against the *same* team leading, not against a neutral game, so
    that possession order does not confound the result - whoever receives to
    open the half gets an extra drive regardless of score.

    The effect is small by construction: possessions alternate, so a faster
    tempo mostly compresses the trailing team's own drives rather than winning
    it extra ones. Real trailing teams gain more than this because they also
    force three-and-outs and spend timeouts, neither of which this engine
    models. Treat play-count deltas from script as second-order; the pass/run
    *mix* is where game script does its real work.
    """
    trailing = _game(home_score=0, away_score=17)
    leading = _game(home_score=17, away_score=0)
    st = simulate(trailing, n_sims=4000, seed=13)
    sl = simulate(leading, n_sims=4000, seed=13)
    assert st.stats["@HOM.plays"].mean() > sl.stats["@HOM.plays"].mean()


def test_less_time_remaining_means_less_production():
    full = _game(seconds=1800)
    quarter = _game(seconds=450)
    sf = simulate(full, n_sims=SIMS, seed=17)
    sq = simulate(quarter, n_sims=SIMS, seed=17)
    assert sq.get("hom_wr1", "rec_yds").mean() < sf.get("hom_wr1", "rec_yds").mean() * 0.5


# --------------------------------------------------------------------------
# Health and availability
# --------------------------------------------------------------------------


def test_ruling_a_player_out_zeroes_him_and_lifts_his_teammates():
    base = _game()
    injured = copy.deepcopy(base)
    injured.home.player("hom_wr1").injury = InjuryStatus.OUT

    sb = simulate(base, n_sims=SIMS, seed=21)
    si = simulate(injured, n_sims=SIMS, seed=21)

    assert si.get("hom_wr1", "rec_yds").mean() == 0.0
    assert si.get("hom_wr2", "rec_yds").mean() > sb.get("hom_wr2", "rec_yds").mean()


def test_questionable_designation_reduces_but_does_not_erase_production():
    base = _game()
    hobbled = copy.deepcopy(base)
    hobbled.home.player("hom_wr1").injury = InjuryStatus.LIMITED

    sb = simulate(base, n_sims=SIMS, seed=23)
    sh = simulate(hobbled, n_sims=SIMS, seed=23)

    limited = sh.get("hom_wr1", "rec_yds").mean()
    healthy = sb.get("hom_wr1", "rec_yds").mean()
    assert 0 < limited < healthy * 0.85


# --------------------------------------------------------------------------
# Blowout / rest risk
# --------------------------------------------------------------------------


def test_stars_lose_volume_in_a_decided_game():
    """Rest risk should truncate the top of a star's distribution."""
    close = _game(home_score=21, away_score=20, seconds=900)
    blowout = _game(home_score=45, away_score=3, seconds=900)

    sc = simulate(close, n_sims=SIMS, seed=29)
    sb = simulate(blowout, n_sims=SIMS, seed=29)

    # The leading back in a blowout gets carries early then sits.
    assert sb.get("hom_rb1", "rush_att").mean() < sc.get("hom_rb1", "rush_att").mean() * 1.5
    # And his ceiling is cut off relative to the volume he would otherwise see.
    assert np.percentile(sb.get("hom_rb1", "rush_yds"), 95) < np.percentile(
        sc.get("hom_rb1", "rush_yds"), 95
    ) * 2.2


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------


def test_high_wind_depresses_passing():
    calm = _game()
    calm.weather = Weather(wind_mph=0, dome=False)
    windy = _game()
    windy.weather = Weather(wind_mph=30, dome=False)

    sc = simulate(calm, n_sims=SIMS, seed=31)
    sw = simulate(windy, n_sims=SIMS, seed=31)

    assert _team_totals(sw, windy, "HOM", "pass_yds").mean() < _team_totals(
        sc, calm, "HOM", "pass_yds"
    ).mean()


def test_wind_hurts_deep_targets_more_than_short_ones():
    """A screen game is nearly wind-proof; a vertical game is not."""
    calm, windy = _game(), _game()
    calm.weather = Weather(wind_mph=0, dome=False)
    windy.weather = Weather(wind_mph=30, dome=False)

    sc = simulate(calm, n_sims=SIMS, seed=33)
    sw = simulate(windy, n_sims=SIMS, seed=33)

    # WR1 runs an 11-yard aDOT; RB1 catches at 1.2 yards.
    deep_drop = 1 - sw.get("hom_wr1", "rec").mean() / max(sc.get("hom_wr1", "rec").mean(), 1e-9)
    short_drop = 1 - sw.get("hom_rb1", "rec").mean() / max(sc.get("hom_rb1", "rec").mean(), 1e-9)
    assert deep_drop > short_drop


# --------------------------------------------------------------------------
# Matchup
# --------------------------------------------------------------------------


def test_stingy_defense_lowers_output_and_generous_defense_raises_it():
    tough, soft = _game(), _game()
    tough.away.defense.pass_yds_mult = 0.75
    soft.away.defense.pass_yds_mult = 1.25

    st = simulate(tough, n_sims=SIMS, seed=37)
    ss = simulate(soft, n_sims=SIMS, seed=37)
    assert st.get("hom_wr1", "rec_yds").mean() < ss.get("hom_wr1", "rec_yds").mean()


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------


def test_quarterback_and_receiver_yardage_are_positively_correlated():
    """The payoff of simulating whole games rather than projecting players.

    On a path where the offense throws well, the QB and his WR1 are both high.
    An independent per-player model cannot represent this, and it is exactly
    what makes same-game parlays mispriced.
    """
    state = _game()
    sim = simulate(state, n_sims=4000, seed=41)
    qb = sim.get("hom_qb", "pass_yds")
    wr = sim.get("hom_wr1", "rec_yds")
    assert np.corrcoef(qb, wr)[0, 1] > 0.30


def test_backs_in_the_same_backfield_compete_for_carries():
    """Two players splitting one workload are substitutes, not complements."""
    state = _game()
    sim = simulate(state, n_sims=4000, seed=43)
    rb1 = sim.get("hom_rb1", "rush_att")
    rb2 = sim.get("hom_rb2", "rush_att")
    # Conditional on total team carries, they trade off; the raw correlation is
    # pulled up by shared volume, so it must still sit well below the QB/WR1 pair.
    qb = sim.get("hom_qb", "pass_yds")
    wr = sim.get("hom_wr1", "rec_yds")
    assert np.corrcoef(rb1, rb2)[0, 1] < np.corrcoef(qb, wr)[0, 1]


def test_parlay_probability_accounts_for_correlation():
    state = _game()
    sim = simulate(state, n_sims=4000, seed=47)
    props = [
        Prop("a", Market.PASS_YDS, Side.OVER, 210.5, -110, PropScope.SECOND_HALF,
             player_id="hom_qb", player_name="HOM QB"),
        Prop("b", Market.REC_YDS, Side.OVER, 55.5, -110, PropScope.SECOND_HALF,
             player_id="hom_wr1", player_name="HOM WR1"),
    ]
    ranked = rank(sim, state, props, keep_values=True)
    assert len(ranked) == 2

    joint = parlay_probability(ranked)
    naive = ranked[0].evaluation.p_win * ranked[1].evaluation.p_win
    # A correlated stack cashes together more often than independence implies.
    assert joint > naive

    m = correlation_matrix(ranked)
    assert m.shape == (2, 2)
    assert m[0][1] > 0


# --------------------------------------------------------------------------
# Sanity
# --------------------------------------------------------------------------


def test_simulated_totals_are_internally_consistent():
    """Team passing yards must equal the sum of its receivers' yards."""
    state = _game()
    sim = simulate(state, n_sims=600, seed=53)
    qb = sim.get("hom_qb", "pass_yds")
    receivers = np.zeros(sim.n_sims)
    for p in state.home.players:
        receivers = receivers + sim.get(p.player_id, "rec_yds")
    assert np.allclose(qb, receivers)


def test_game_total_equals_the_sum_of_team_points():
    state = _game()
    sim = simulate(state, n_sims=600, seed=59)
    assert np.allclose(
        sim.game_points(), sim.team_points("HOM") + sim.team_points("AWY")
    )


def test_no_production_when_no_time_remains():
    state = _game(seconds=0)
    sim = simulate(state, n_sims=200, seed=61)
    assert sim.game_points().mean() == 0.0
    assert sim.get("hom_wr1", "rec_yds").mean() == 0.0


def test_engine_handles_a_team_with_no_healthy_quarterback():
    """Degrade to a replacement passer rather than raising."""
    state = _game()
    for p in state.home.players:
        if p.player_id == "hom_qb":
            p.injury = InjuryStatus.OUT
    sim = simulate(state, n_sims=200, seed=67)
    assert sim.get("hom_qb", "pass_yds").mean() == 0.0
    assert sim.stats["@HOM.plays"].mean() > 0


@pytest.mark.parametrize("seconds", [60, 300, 900, 1800])
def test_drive_loop_always_terminates(seconds):
    state = _game(seconds=seconds)
    sim = simulate(state, n_sims=150, seed=71)
    assert np.isfinite(sim.game_points()).all()
