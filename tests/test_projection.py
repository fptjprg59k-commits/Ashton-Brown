"""Tests for the stat-line projection.

The assertions that matter here are the accounting ones. A projection model is
easy to make plausible and hard to make *consistent* - the classic failure is a
player table whose touchdown probabilities quietly sum to more scores than the
projected score can pay for. These tests keep the books balanced.
"""

from __future__ import annotations

import math
import os

import pytest

from nflprops.matchup import build_matchup, load_matchup
from nflprops.projection import (
    MAX_SCORE_TILT,
    POINTS_PER_TD_DRIVE,
    _catch_rate,
    game_script,
    ordinal,
    project,
)

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures", "matchup_ind_kc_2026w2.json",
)


def _game(**over):
    """A plain, symmetric matchup so one thing at a time can be moved."""
    def unit():
        return {k: {"prior": 16, "current": 16} for k in
                ("scoring", "pass", "rush", "red_zone", "third_down",
                 "explosive", "ball_security", "pass_protect")}

    def dunit():
        return {k: {"prior": 16, "current": 16} for k in
                ("scoring", "pass", "rush", "red_zone", "third_down",
                 "pressure", "takeaways", "tackling")}

    tend = {"neutral_pass_rate": {"prior": 0.57, "current": 0.57},
            "seconds_per_play": {"prior": 28.0, "current": 28.0}}

    players = [
        {"name": "Back", "pos": "RB", "rush_share": 0.8, "ypc": 4.4,
         "target_share": 0.10, "ypr": 8.0, "gl_factor": 1.3},
        {"name": "Wideout", "pos": "WR", "target_share": 0.28, "adot": 11.0},
        {"name": "Tight End", "pos": "TE", "target_share": 0.20, "adot": 7.0,
         "rz_factor": 1.2},
        {"name": "Back Two", "pos": "RB", "rush_share": 0.2, "ypc": 4.0,
         "target_share": 0.05, "ypr": 7.0},
    ]
    qb = {"name": "Passer", "ypa": 7.2, "completion_pct": 0.65, "int_rate": 0.023,
          "sack_rate": 0.068, "rush_attempts": 4.0, "rush_ypc": 4.5}

    raw = {
        "meta": {"games_played": 6,
                 "market": {"total": 44.0, "spread": 0.0, "favorite": None}},
        "weather": {},
        "teams": {
            "AAA": {"name": "A Team", "abbr": "AAA", "home": True, "qb": "Passer",
                    "rest": {"days": 7}, "offense": unit(), "defense": dunit(),
                    "tendencies": tend, "players": players, "qb_profile": qb},
            "BBB": {"name": "B Team", "abbr": "BBB", "home": False, "qb": "Passer",
                    "rest": {"days": 7}, "offense": unit(), "defense": dunit(),
                    "tendencies": tend, "players": players, "qb_profile": qb},
        },
    }
    for path, value in over.items():
        node = raw
        parts = path.split(".")
        for k in parts[:-1]:
            node = node[k]
        node[parts[-1]] = value
    return build_matchup(raw)


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------


def test_team_yards_are_the_sum_of_their_parts():
    g = project(_game())
    for tp in g.teams:
        assert tp.total_yards == pytest.approx(tp.pass_yards + tp.rush_yards)
        player_rush = sum(s.rush_yards for s in tp.skill)
        assert tp.rush_yards == pytest.approx(player_rush + tp.qb.rush_yards)


def test_dropbacks_split_into_attempts_sacks_and_scrambles():
    g = project(_game())
    for tp in g.teams:
        assert tp.dropbacks == pytest.approx(
            tp.attempts + tp.sacks + tp.qb.scrambles, abs=0.01
        )


def test_designed_quarterback_runs_cost_carries_not_pass_attempts():
    """A called keeper is a run. Charging it to dropbacks would quietly take
    throws away from exactly the quarterbacks a team runs on purpose."""
    pocket = project(_game(**{"teams.AAA.qb_profile": {
        "name": "Pocket", "ypa": 7.2, "completion_pct": 0.65, "int_rate": 0.023,
        "sack_rate": 0.068, "rush_attempts": 6.0, "rush_ypc": 4.5,
        "designed_run_share": 0.0,
    }}))
    runner = project(_game(**{"teams.AAA.qb_profile": {
        "name": "Runner", "ypa": 7.2, "completion_pct": 0.65, "int_rate": 0.023,
        "sack_rate": 0.068, "rush_attempts": 6.0, "rush_ypc": 4.5,
        "designed_run_share": 1.0,
    }}))
    a, b = pocket.for_team("AAA"), runner.for_team("AAA")
    # Same carries for the quarterback either way...
    assert a.qb.rush_attempts == pytest.approx(b.qb.rush_attempts)
    # ...but the designed runner keeps his pass attempts.
    assert b.attempts > a.attempts + 4
    # and takes those carries off his running backs instead.
    assert sum(s.carries for s in b.skill) < sum(s.carries for s in a.skill)


def test_touchdown_budget_is_fully_allocated_and_not_exceeded():
    """The single most important invariant: nobody scores on credit."""
    g = project(_game())
    for tp in g.teams:
        allocated = sum(s.expected_tds for s in tp.skill) + tp.qb.rush_tds
        assert allocated == pytest.approx(tp.expected_tds, abs=1e-6)
        assert tp.pass_tds + tp.rush_tds == pytest.approx(tp.expected_tds)


def test_expected_touchdowns_follow_the_projected_score():
    g = project(_game())
    for tp in g.teams:
        assert tp.expected_tds == pytest.approx(tp.points / POINTS_PER_TD_DRIVE)


def test_td_probability_is_the_poisson_tail_of_expected_scores():
    g = project(_game())
    for tp in g.teams:
        for s in tp.skill:
            assert s.td_probability == pytest.approx(1 - math.exp(-s.expected_tds))
            # A player with under one expected score is never a favourite to score.
            if s.expected_tds < 1.0:
                assert s.td_probability < 0.64


def test_player_receptions_never_exceed_their_targets():
    g = project(_game())
    for tp in g.teams:
        for s in tp.skill:
            assert 0 <= s.receptions <= s.targets
        assert sum(s.targets for s in tp.skill) <= tp.attempts + 0.01


def test_a_symmetric_matchup_produces_a_symmetric_game():
    g = project(_game())
    assert g.home.points == pytest.approx(g.away.points)
    assert g.home.total_yards == pytest.approx(g.away.total_yards, rel=0.02)


# --------------------------------------------------------------------------
# Direction
# --------------------------------------------------------------------------


def test_the_score_stays_anchored_to_the_market():
    g = project(_game())
    for tp in g.teams:
        assert abs(tp.points - tp.market_points) <= MAX_SCORE_TILT + 1e-9


def test_a_favourite_is_projected_to_outscore_the_underdog():
    g = project(_game(**{"meta.market": {"total": 44.0, "spread": 9.0, "favorite": "AAA"}}))
    assert g.for_team("AAA").points > g.for_team("BBB").points


def test_facing_a_worse_pass_defense_produces_more_passing_yards():
    tough = project(_game())
    # Rebuild with a weak secondary opposite AAA.
    soft = project(_game(**{"teams.BBB.defense": {
        **{k: {"prior": 16, "current": 16} for k in
           ("scoring", "rush", "red_zone", "third_down", "pressure", "takeaways", "tackling")},
        "pass": {"prior": 32, "current": 32},
    }}))
    assert soft.for_team("AAA").pass_yards > tough.for_team("AAA").pass_yards


def test_a_dominant_pass_rush_produces_more_sacks():
    calm = project(_game())
    heat = project(_game(**{"teams.BBB.defense": {
        **{k: {"prior": 16, "current": 16} for k in
           ("scoring", "pass", "rush", "red_zone", "third_down", "takeaways", "tackling")},
        "pressure": {"prior": 1, "current": 1},
    }}))
    assert heat.for_team("AAA").sacks > calm.for_team("AAA").sacks


def test_weather_takes_yards_off_the_passing_game():
    clear = project(_game())
    storm = project(_game(**{"weather": {"wind_mph": 26, "precip_chance": 0.95, "temp_f": 45}}))
    assert storm.for_team("AAA").pass_yards < clear.for_team("AAA").pass_yards


def test_deeper_targets_are_caught_less_often():
    assert _catch_rate(3) > _catch_rate(11) > _catch_rate(20)
    assert 0.38 <= _catch_rate(40) <= 0.82


def test_ordinal_handles_the_teens_and_the_tens():
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 32)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "32nd"
    ]


# --------------------------------------------------------------------------
# The shipped game
# --------------------------------------------------------------------------


def test_the_bundled_game_projects_a_believable_stat_line():
    g = project(load_matchup(FIXTURE))
    assert 30 <= g.total <= 65
    for tp in g.teams:
        assert 10 <= tp.points <= 40
        assert 220 <= tp.total_yards <= 480
        assert 45 <= tp.plays <= 78
        assert 0.5 <= tp.sacks <= 6
        assert 0.1 <= tp.interceptions <= 3
        q = tp.qb
        assert 0.52 <= q.completion_pct <= 0.75
        assert 4.5 <= q.yards_per_attempt <= 10.5
        assert q.attempts >= q.completions


def test_the_bundled_game_names_its_lead_backs():
    g = project(load_matchup(FIXTURE))
    kc = g.for_team("KC")
    ind = g.for_team("IND")
    assert kc.skill[0].name == "Kenneth Walker III"
    assert ind.skill[0].name == "Jonathan Taylor"
    # Both are every-down backs; nobody else on their team sees half their carries.
    for tp in (kc, ind):
        lead = max(tp.skill, key=lambda s: s.carries)
        others = [s.carries for s in tp.skill if s is not lead]
        assert lead.carries > 2 * max(others)


def test_the_game_script_reads_as_a_sentence_and_names_both_teams():
    m = load_matchup(FIXTURE)
    text = game_script(project(m))
    assert m.home.name in text and m.away.name in text
    assert text.endswith(".")
    assert "{" not in text and "None" not in text
