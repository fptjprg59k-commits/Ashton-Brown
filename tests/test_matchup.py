"""Tests for the pre-game matchup engine.

The interesting assertions here are the directional ones. A weighting model
will happily produce plausible-looking numbers with a sign error in it - that
is exactly the bug this suite exists to catch, because a matchup report that
tells you to attack the *strong* side of a defence is worse than no report.
"""

from __future__ import annotations

import json
import os

import pytest

from nflprops.matchup import (
    LEAGUE_MID,
    Matchup,
    blend_rank,
    build_matchup,
    load_matchup,
    strength,
)
from nflprops.matchup_report import render_matchup_html, render_matchup_terminal

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures",
    "matchup_ind_kc_2026w2.json",
)


# --------------------------------------------------------------------------
# Rank arithmetic
# --------------------------------------------------------------------------


def test_strength_is_inverted_rank():
    assert strength(1) == pytest.approx(1.0)
    assert strength(32) == pytest.approx(0.0)
    assert strength(16.5) == pytest.approx(0.5)


def test_blend_leans_on_last_season_when_this_season_is_one_game():
    blended, w = blend_rank(prior=4, current=32, games=1, k=3.0, carryover=1.0)
    assert w["current"] == pytest.approx(0.25)
    assert blended < LEAGUE_MID, "one bad game should not erase a top-five season"


def test_blend_converges_on_this_season_as_games_accumulate():
    early, _ = blend_rank(4, 32, games=1, k=3.0)
    late, _ = blend_rank(4, 32, games=16, k=3.0)
    assert late > early
    # By the back half of a season the current year owns most of the weight.
    assert abs(late - 32) < abs(late - 4)
    _, w = blend_rank(4, 32, games=16, k=3.0)
    assert w["current"] > 0.8


def test_low_carryover_regresses_toward_average_not_toward_last_year():
    """Lost continuity is missing information, not evidence for either side."""
    kept, _ = blend_rank(28, 4, games=1, k=3.0, carryover=1.0)
    lost, _ = blend_rank(28, 4, games=1, k=3.0, carryover=0.0)
    assert kept > lost, "a fully returning unit keeps more of its bad prior"
    # With no carryover the prior's weight goes to league average.
    expected = 0.25 * 4 + 0.75 * LEAGUE_MID
    assert lost == pytest.approx(expected)


def test_blend_weights_always_sum_to_one():
    for carry in (0.0, 0.4, 1.0):
        for games in (0, 1, 5, 17):
            _, w = blend_rank(10, 20, games=games, k=4.0, carryover=carry)
            assert sum(w.values()) == pytest.approx(1.0)


def test_missing_current_season_still_produces_a_rank():
    blended, w = blend_rank(prior=6, current=None, games=0, carryover=1.0)
    assert blended == pytest.approx(6)
    assert w["current"] == 0.0


# --------------------------------------------------------------------------
# Script direction - the sign errors that matter
# --------------------------------------------------------------------------


def _synthetic(*, opp_rush_rank, opp_pass_rank, spread=0.0, favorite=None):
    """Two plain teams so one variable at a time can be moved."""

    def unit(**over):
        base = {k: {"prior": 16, "current": 16} for k in
                ("scoring", "pass", "rush", "red_zone", "third_down",
                 "explosive", "ball_security", "pass_protect")}
        base.update(over)
        return base

    def dunit(**over):
        base = {k: {"prior": 16, "current": 16} for k in
                ("scoring", "pass", "rush", "red_zone", "third_down",
                 "pressure", "takeaways", "tackling")}
        base.update(over)
        return base

    tend = {"neutral_pass_rate": {"prior": 0.57, "current": 0.57},
            "seconds_per_play": {"prior": 28.0, "current": 28.0}}

    return {
        "meta": {"games_played": 4,
                 "market": {"total": 44.0, "spread": spread, "favorite": favorite}},
        "weather": {},
        "teams": {
            "AAA": {"name": "A", "home": True, "rest": {"days": 7},
                    "offense": unit(), "defense": dunit(), "tendencies": tend},
            "BBB": {"name": "B", "home": False, "rest": {"days": 7},
                    "offense": unit(),
                    "defense": dunit(rush={"prior": opp_rush_rank, "current": opp_rush_rank},
                                     **{"pass": {"prior": opp_pass_rank, "current": opp_pass_rank}}),
                    "tendencies": tend},
        },
    }


def test_a_weak_opposing_run_defense_makes_you_run_more():
    """The bug this catches: attacking the unit that is actually good."""
    m = build_matchup(_synthetic(opp_rush_rank=32, opp_pass_rank=1))
    s = m.scripts["AAA"]
    assert s.projected_pass_rate < s.neutral_pass_rate
    soft = [a for a in s.adjustments if a[0] == "Opponent's soft spot"][0]
    assert soft[1] < 0
    assert "run defense" in soft[2]


def test_a_weak_opposing_pass_defense_makes_you_throw_more():
    m = build_matchup(_synthetic(opp_rush_rank=1, opp_pass_rank=32))
    s = m.scripts["AAA"]
    assert s.projected_pass_rate > s.neutral_pass_rate
    soft = [a for a in s.adjustments if a[0] == "Opponent's soft spot"][0]
    assert soft[1] > 0
    assert "pass defense" in soft[2]


def test_the_underdog_throws_and_the_favourite_runs():
    m = build_matchup(_synthetic(opp_rush_rank=16, opp_pass_rank=16,
                                 spread=10.0, favorite="AAA"))
    assert m.scripts["BBB"].projected_pass_rate > m.scripts["AAA"].projected_pass_rate


def test_wind_and_rain_push_both_teams_toward_the_run():
    calm = build_matchup(_synthetic(opp_rush_rank=16, opp_pass_rank=16))
    raw = _synthetic(opp_rush_rank=16, opp_pass_rank=16)
    raw["weather"] = {"wind_mph": 25, "precip_chance": 0.9, "temp_f": 40}
    storm = build_matchup(raw)
    for abbr in ("AAA", "BBB"):
        assert storm.scripts[abbr].projected_pass_rate < calm.scripts[abbr].projected_pass_rate


def test_pass_rate_stays_inside_believable_bounds():
    raw = _synthetic(opp_rush_rank=32, opp_pass_rank=1, spread=21.0, favorite="AAA")
    raw["weather"] = {"wind_mph": 40, "precip_chance": 1.0, "temp_f": 5}
    m = build_matchup(raw)
    for s in m.scripts.values():
        assert 0.30 <= s.projected_pass_rate <= 0.78


# --------------------------------------------------------------------------
# Injuries
# --------------------------------------------------------------------------


def test_injury_weights_land_on_the_named_axis_only():
    raw = _synthetic(opp_rush_rank=16, opp_pass_rank=16)
    raw["teams"]["AAA"]["injuries"] = [{
        "player": "Left Tackle", "pos": "LT", "status": "out",
        "units": ["offense"], "axes": {"pass_protect": 1.0, "pass": 0.35},
        "impact": 1.0,
    }]
    m = build_matchup(raw)
    off = m.team("AAA").offense
    assert off["pass_protect"].injury_shift > off["pass"].injury_shift > 0
    assert off["rush"].injury_shift == 0, "a tackle being out must not move the run game"
    assert off["rush"].adjusted == pytest.approx(off["rush"].blended)


def test_a_questionable_player_costs_less_than_a_ruled_out_one():
    def shift(status):
        raw = _synthetic(opp_rush_rank=16, opp_pass_rank=16)
        raw["teams"]["AAA"]["injuries"] = [{
            "player": "X", "pos": "DT", "status": status,
            "units": ["defense"], "axes": ["pressure"], "impact": 1.0,
        }]
        return build_matchup(raw).team("AAA").defense["pressure"].injury_shift

    assert shift("out") > shift("doubtful") > shift("questionable") > 0
    assert shift("healthy") == 0


def test_a_bare_axis_list_is_read_as_full_weight():
    raw = _synthetic(opp_rush_rank=16, opp_pass_rank=16)
    raw["teams"]["AAA"]["injuries"] = [{
        "player": "X", "pos": "CB", "status": "out",
        "units": ["defense"], "axes": ["pass"], "impact": 0.5,
    }]
    m = build_matchup(raw)
    assert m.team("AAA").defense["pass"].injury_shift == pytest.approx(0.5 * 8.0)


def test_injury_adjustment_cannot_push_a_rank_off_the_scale():
    raw = _synthetic(opp_rush_rank=16, opp_pass_rank=16)
    raw["teams"]["AAA"]["defense"]["pass"] = {"prior": 31, "current": 31}
    raw["teams"]["AAA"]["injuries"] = [
        {"player": f"CB{i}", "pos": "CB", "status": "out", "units": ["defense"],
         "axes": ["pass"], "impact": 1.0} for i in range(5)
    ]
    assert build_matchup(raw).team("AAA").defense["pass"].adjusted <= 32.0


# --------------------------------------------------------------------------
# Leverage
# --------------------------------------------------------------------------


def test_leverage_is_positive_when_the_offense_outranks_the_defense():
    raw = _synthetic(opp_rush_rank=32, opp_pass_rank=32)
    raw["teams"]["AAA"]["offense"]["rush"] = {"prior": 1, "current": 1}
    m = build_matchup(raw)
    row = [l for l in m.boards["AAA"] if l.axis.key == "rush"][0]
    assert row.edge > 0 and row.score > 0
    assert row.grade == "decisive"


def test_exposure_follows_the_projected_script():
    """A team that will throw gets more of its grade from passing phases."""
    passing = build_matchup(_synthetic(opp_rush_rank=1, opp_pass_rank=32))
    running = build_matchup(_synthetic(opp_rush_rank=32, opp_pass_rank=1))
    pass_exp = [l for l in passing.boards["AAA"] if l.axis.key == "pass"][0].exposure
    run_exp = [l for l in running.boards["AAA"] if l.axis.key == "pass"][0].exposure
    assert pass_exp > run_exp


def test_disagreement_reports_the_gap_between_seasons():
    raw = _synthetic(opp_rush_rank=16, opp_pass_rank=16)
    raw["teams"]["AAA"]["defense"]["rush"] = {"prior": 1, "current": 32}
    m = build_matchup(raw)
    metric = m.team("AAA").defense["rush"]
    assert metric.disagreement == pytest.approx(1.0)
    assert any(mm is metric for _, _, mm in m.open_questions())


def test_agreeing_seasons_are_not_an_open_question():
    raw = _synthetic(opp_rush_rank=16, opp_pass_rank=16)
    m = build_matchup(raw)
    assert m.open_questions() == []


def test_two_teams_are_required():
    with pytest.raises(ValueError):
        build_matchup({"meta": {}, "teams": {"AAA": {}}})


# --------------------------------------------------------------------------
# The shipped game
# --------------------------------------------------------------------------


def test_the_bundled_matchup_loads_and_both_surfaces_render():
    m = load_matchup(FIXTURE)
    assert {m.home.abbr, m.away.abbr} == {"KC", "IND"}
    assert m.home.abbr == "KC", "the Chiefs are at home"

    text = render_matchup_terminal(m)
    assert "LEVERAGE BOARD" in text and "OPEN QUESTIONS" in text

    page = render_matchup_html(m)
    assert page.count("<title>") == 1
    assert "Matchup Lab" in page
    # every section renders
    for heading in ("The read", "Leverage board", "Open questions", "Unit ranks",
                    "Play calling", "Personnel and usage", "Injuries", "Situation"):
        assert heading in page
    # No Python template leftovers escaped into the output. CSS has braces of
    # its own, so look for the names that would only appear unrendered.
    for leak in ("{FONT_", "{_e(", "{TOKENS_CSS", "{BASE_CSS", "None%"):
        assert leak not in page


def test_the_bundled_matchup_flags_the_colts_run_defense_contradiction():
    """2025 said 4th, Week 1 said 32nd. That has to survive to the report."""
    m = load_matchup(FIXTURE)
    top = m.open_questions(1)[0]
    team, unit, metric = top
    assert team.abbr == "IND" and unit == "defense"
    assert metric.prior == 4 and metric.current == 32


def test_every_leverage_row_points_at_a_real_pair_of_metrics():
    m = load_matchup(FIXTURE)
    for board in m.boards.values():
        for l in board:
            assert 1.0 <= l.off_metric.adjusted <= 32.0
            assert 1.0 <= l.def_metric.adjusted <= 32.0
            assert -1.0 <= l.edge <= 1.0
            assert l.exposure > 0


def test_the_fixture_is_valid_json_with_the_fields_the_engine_reads():
    with open(FIXTURE, encoding="utf-8") as fh:
        raw = json.load(fh)
    assert set(raw["teams"]) == {"KC", "IND"}
    for t in raw["teams"].values():
        assert t["offense"] and t["defense"] and t["tendencies"]
        for m in list(t["offense"].values()) + list(t["defense"].values()):
            assert "prior" in m and "current" in m
