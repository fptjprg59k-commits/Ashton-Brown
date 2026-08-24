"""Prop parsing, name resolution, scope handling, and the end-to-end pipeline."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from nflprops.agent import analyze
from nflprops.matching import PlayerResolver, normalize, similarity
from nflprops.models import (
    GameState, Market, Player, Position, Prop, PropScope, Side, TeamState,
)
from nflprops.props import evaluate, settled_status
from nflprops.providers.base import resolve_props
from nflprops.providers.stake import StakePropsProvider
from nflprops.serde import game_from_dict, game_to_dict, load_game
from nflprops.simulate import simulate

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")
GAME = os.path.join(FIXTURES, "game_bal_cin_halftime.json")
PROPS = os.path.join(FIXTURES, "props_paste.txt")


# --------------------------------------------------------------------------
# Name matching
# --------------------------------------------------------------------------


def test_normalize_strips_punctuation_accents_and_suffixes():
    assert normalize("Ja'Marr Chase") == "jamarr chase"
    assert normalize("Odell Beckham Jr.") == "odell beckham"
    assert normalize("Chase, Ja'Marr") == "jamarr chase"


@pytest.mark.parametrize(
    "a,b",
    [
        ("Ja'Marr Chase", "JaMarr Chase"),
        ("J. Chase", "Ja'Marr Chase"),
        ("Chase, Ja'Marr", "Ja'Marr Chase"),
        ("Marvin Harrison Jr", "Marvin Harrison Jr."),
    ],
)
def test_similar_names_match(a, b):
    assert similarity(a, b) >= 0.86


@pytest.mark.parametrize(
    "a,b",
    [
        ("Ja'Marr Chase", "Chase Brown"),
        ("Justin Jefferson", "Jauan Jennings"),
        ("Mike Williams", "Mike Evans"),
    ],
)
def test_different_players_do_not_match(a, b):
    assert similarity(a, b) < 0.86


def test_resolver_returns_none_rather_than_guessing():
    """An unmatched prop must be reported, never attached to the nearest name."""
    roster = [Player("p1", "Ja'Marr Chase", Position.WR, "CIN")]
    r = PlayerResolver(roster)
    assert r.resolve("Ja'Marr Chase") == "p1"
    assert r.resolve("JaMarr Chase") == "p1"
    assert r.resolve("Somebody Entirely Different") is None


# --------------------------------------------------------------------------
# Paste parsing
# --------------------------------------------------------------------------


def test_parses_the_standard_paste_format():
    props = StakePropsProvider.parse_text(
        "Ja'Marr Chase Over 65.5 Receiving Yards -115"
    )
    assert len(props) == 1
    p = props[0]
    assert p.player_name == "Ja'Marr Chase"
    assert p.market is Market.REC_YDS
    assert p.side is Side.OVER
    assert p.line == 65.5
    assert p.odds == -115


@pytest.mark.parametrize(
    "line",
    [
        "Ja'Marr Chase - Receiving Yards - Over 65.5 (-115)",
        "Receiving Yards | Ja'Marr Chase | Over 65.5 | -115",
        "Ja'Marr Chase Over 65.5 Rec Yards -115",
    ],
)
def test_parser_tolerates_ordering_and_separators(line):
    props = StakePropsProvider.parse_text(line)
    assert len(props) == 1
    assert props[0].market is Market.REC_YDS
    assert props[0].line == 65.5
    assert props[0].odds == -115
    assert "Chase" in props[0].player_name


def test_signed_odds_are_never_confused_with_an_unsigned_line():
    """The disambiguation rule the whole parser rests on."""
    props = StakePropsProvider.parse_text("Joe Burrow Over 249.5 Passing Yards +120")
    assert props[0].line == 249.5
    assert props[0].odds == 120


def test_two_sided_prices_are_captured_for_exact_devigging():
    props = StakePropsProvider.parse_text(
        "Tee Higgins Over 62.5 Receiving Yards -110 -110"
    )
    assert props[0].odds == -110
    assert props[0].opposing_odds == -110


def test_anytime_touchdown_is_normalised_to_a_yes_market():
    props = StakePropsProvider.parse_text("Chase Brown Anytime Touchdown +145")
    p = props[0]
    assert p.market is Market.ANYTIME_TD
    assert p.side is Side.YES
    assert p.line == 0.5


def test_longest_alias_wins_over_a_shorter_substring():
    """'passing + rushing yards' must not be read as 'passing yards'."""
    props = StakePropsProvider.parse_text(
        "Lamar Jackson Over 260.5 Passing + Rushing Yards -110"
    )
    assert props[0].market is Market.PASS_RUSH_YDS


def test_second_half_scope_is_detected():
    props = StakePropsProvider.parse_text(
        "Zay Flowers Over 30.5 Receiving Yards 2nd Half -110"
    )
    assert props[0].scope is PropScope.SECOND_HALF
    assert "Flowers" in props[0].player_name


def test_already_settled_first_half_markets_are_discarded():
    """A first-half line has nothing left to predict at the break."""
    props = StakePropsProvider.parse_text(
        "Zay Flowers Over 30.5 Receiving Yards 1st Half -110"
    )
    assert props == []


def test_comments_blank_lines_and_junk_are_handled():
    rejected = []
    props = StakePropsProvider.parse_text(
        "# a comment\n\nJa'Marr Chase Over 65.5 Receiving Yards -115\n"
        "total nonsense with no odds\n",
        rejected=rejected,
    )
    assert len(props) == 1
    assert len(rejected) == 1


def test_json_input_round_trips():
    payload = {
        "props": [
            {
                "player": "Ja'Marr Chase", "market": "rec_yds", "side": "over",
                "line": 65.5, "odds": -115, "opposing_odds": -105, "team": "CIN",
            }
        ]
    }
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        path = fh.name
    try:
        props = StakePropsProvider(path=path).fetch()
        assert len(props) == 1
        assert props[0].market is Market.REC_YDS
        assert props[0].opposing_odds == -105
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def test_game_state_round_trips_through_json():
    state = load_game(GAME)
    again = game_from_dict(json.loads(json.dumps(game_to_dict(state))))
    assert again.home.abbr == state.home.abbr
    assert again.home.score == state.home.score
    assert len(again.away.players) == len(state.away.players)
    assert again.away.player("cin_chase").target_share == pytest.approx(
        state.away.player("cin_chase").target_share
    )


def test_minimal_state_uses_defaults():
    state = game_from_dict({
        "home": {"abbr": "AAA", "score": 7},
        "away": {"abbr": "BBB", "score": 3},
    })
    assert state.seconds_remaining == 1800
    assert state.home.base_pass_rate == pytest.approx(0.575)


# --------------------------------------------------------------------------
# Scope, pushes, settled lines
# --------------------------------------------------------------------------


def _small_state() -> GameState:
    home = TeamState(abbr="HH")
    away = TeamState(abbr="AA")
    wr = Player("wr", "Test Receiver", Position.WR, "AA", target_share=0.3)
    wr.h1.rec_yds = 44.0
    wr.h1.rec = 4
    wr.h1.rec_td = 1
    away.players = [Player("qb", "Test QB", Position.QB, "AA", is_starter_qb=True), wr]
    return GameState(game_id="t", home=home, away=away, possession="AA")


def test_full_game_scope_adds_first_half_production():
    """The most expensive mistake available at halftime, guarded directly."""
    state = _small_state()
    sim = simulate(state, n_sims=400, seed=5)

    full = Prop("f", Market.REC_YDS, Side.OVER, 0.5, -110,
                PropScope.FULL_GAME, player_id="wr")
    half = Prop("h", Market.REC_YDS, Side.OVER, 0.5, -110,
                PropScope.SECOND_HALF, player_id="wr")

    ef = evaluate(sim, state, full)
    eh = evaluate(sim, state, half)
    assert ef.mean == pytest.approx(eh.mean + 44.0, abs=0.6)


def test_longest_reception_combines_by_max_not_sum():
    state = _small_state()
    state.away.player("wr").h1.longest_rec = 30.0
    sim = simulate(state, n_sims=400, seed=5)

    prop = Prop("l", Market.LONGEST_REC, Side.OVER, 25.5, -110,
                PropScope.FULL_GAME, player_id="wr")
    ev = evaluate(sim, state, prop)
    # A 30-yard catch already in the book means this cannot come back under.
    assert ev.p_win == 1.0
    assert ev.settled == "won"


def test_whole_number_lines_can_push():
    state = _small_state()
    sim = simulate(state, n_sims=2000, seed=9)
    # The line has to sit where the distribution actually lives for a push to
    # be reachable at all, so take the simulated median.
    probe = Prop("probe", Market.REC, Side.OVER, 0.5, -110,
                 PropScope.FULL_GAME, player_id="wr")
    median = evaluate(sim, state, probe).median

    prop = Prop("p", Market.REC, Side.OVER, float(median), -110,
                PropScope.FULL_GAME, player_id="wr")
    ev = evaluate(sim, state, prop)
    assert ev.p_push > 0
    assert ev.p_win + ev.p_push + ev.p_loss == pytest.approx(1.0)
    assert ev.p_win_excluding_push > ev.p_win


def test_half_point_lines_never_push():
    state = _small_state()
    sim = simulate(state, n_sims=1000, seed=9)
    prop = Prop("p", Market.REC, Side.OVER, 6.5, -110,
                PropScope.FULL_GAME, player_id="wr")
    assert evaluate(sim, state, prop).p_push == 0.0


def test_settled_detection_for_each_side():
    state = _small_state()  # 44 receiving yards, 1 receiving TD already
    over_won = Prop("a", Market.REC_YDS, Side.OVER, 30.5, -110,
                    PropScope.FULL_GAME, player_id="wr")
    under_lost = Prop("b", Market.REC_YDS, Side.UNDER, 30.5, -110,
                      PropScope.FULL_GAME, player_id="wr")
    undecided = Prop("c", Market.REC_YDS, Side.OVER, 90.5, -110,
                     PropScope.FULL_GAME, player_id="wr")
    td_won = Prop("d", Market.ANYTIME_TD, Side.YES, 0.5, -110,
                  PropScope.FULL_GAME, player_id="wr")
    second_half = Prop("e", Market.REC_YDS, Side.OVER, 30.5, -110,
                       PropScope.SECOND_HALF, player_id="wr")

    assert settled_status(state, over_won) == "won"
    assert settled_status(state, under_lost) == "lost"
    assert settled_status(state, undecided) is None
    assert settled_status(state, td_won) == "won"
    # A second-half line is never settled by first-half production.
    assert settled_status(state, second_half) is None


def test_raising_the_line_lowers_the_probability():
    state = load_game(GAME)
    sim = simulate(state, n_sims=1500, seed=15)
    probs = []
    for line in (40.5, 70.5, 100.5, 140.5, 200.5):
        prop = Prop("x", Market.REC_YDS, Side.OVER, line, -110,
                    PropScope.FULL_GAME, player_id="cin_chase")
        probs.append(evaluate(sim, state, prop).p_win)
    assert all(a >= b for a, b in zip(probs, probs[1:]))


def test_over_and_under_on_the_same_line_are_complementary():
    state = load_game(GAME)
    sim = simulate(state, n_sims=1500, seed=19)
    over = Prop("o", Market.REC_YDS, Side.OVER, 99.5, -110,
                PropScope.FULL_GAME, player_id="cin_chase")
    under = Prop("u", Market.REC_YDS, Side.UNDER, 99.5, -110,
                 PropScope.FULL_GAME, player_id="cin_chase")
    a = evaluate(sim, state, over)
    b = evaluate(sim, state, under)
    assert a.p_win + b.p_win == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# Resolution and pipeline
# --------------------------------------------------------------------------


def test_unmatched_props_are_separated_not_dropped_silently():
    state = load_game(GAME)
    props = [
        Prop("good", Market.REC_YDS, Side.OVER, 65.5, -110, player_name="Ja'Marr Chase"),
        Prop("bad", Market.REC_YDS, Side.OVER, 65.5, -110, player_name="Nonexistent Person"),
    ]
    resolved, unresolved = resolve_props(props, state)
    assert [p.prop_id for p in resolved] == ["good"]
    assert [p.prop_id for p in unresolved] == ["bad"]
    assert resolved[0].player_id == "cin_chase"
    assert resolved[0].team == "CIN"


def test_end_to_end_pipeline_ranks_the_fixture_board():
    state = load_game(GAME)
    props = StakePropsProvider(path=PROPS).fetch()
    assert len(props) > 10

    result = analyze(state, props, n_sims=1500, seed=101)

    assert result.ranked, "expected a ranked board"
    assert not result.unresolved, "fixture names should all resolve"

    # Sorted highest-probability first, as asked for.
    probs = [r.evaluation.p_win for r in result.ranked]
    assert probs == sorted(probs, reverse=True)

    # Bateman already scored, so his anytime touchdown is held out.
    assert any(r.evaluation.settled == "won" for r in result.settled)
    assert all(r.evaluation.settled is None for r in result.ranked)

    for r in result.ranked:
        assert 0.0 <= r.evaluation.p_win <= 1.0
        assert r.drivers, f"no driver attribution for {r.prop.label}"


def test_sorting_by_edge_reorders_the_board():
    state = load_game(GAME)
    props = StakePropsProvider(path=PROPS).fetch()
    by_prob = analyze(state, props, n_sims=1200, seed=103, sort="probability")
    by_edge = analyze(state, props, n_sims=1200, seed=103, sort="edge")

    edges = [r.pricing.edge for r in by_edge.ranked]
    assert edges == sorted(edges, reverse=True)
    assert [r.prop.prop_id for r in by_prob.ranked] != [
        r.prop.prop_id for r in by_edge.ranked
    ]


def test_deliberately_shaded_lines_surface_as_edges():
    """The fixture hides three mispriced lines; the model should find them."""
    state = load_game(GAME)
    props = StakePropsProvider(path=PROPS).fetch()
    result = analyze(state, props, n_sims=6000, seed=107, sort="edge")

    labels = {r.prop.player_name: r.pricing.edge for r in result.ranked}
    # Higgins was shaded in the bettor's favour, Brown and Henry against it.
    assert labels.get("Tee Higgins", 0) > 0.02
    assert min(labels.get("Chase Brown", 0), labels.get("Derrick Henry", 0)) < -0.02
