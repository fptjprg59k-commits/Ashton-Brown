"""The semantic status system and the dashboard built on it.

The rule under test throughout: a card never decides how to look. It asks the
status for a tone, and the tone maps to tokens. So these tests assert on
*status and tone*, not on colours - if the palette changes, none of this
should need touching.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from nflprops.dashboard import (
    Dashboard, FILTERS, GameCard, TeamLine, _filter_slugs,
    render_dashboard, render_dashboard_terminal,
)
from nflprops.status import (
    GameStatus, SeasonType, Tone, derive_status, describe, ordinal_period,
)
from nflprops.theme import TONE_TOKENS

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "fixtures", "scoreboard_preseason.json"
)


# --------------------------------------------------------------------------
# Halftime detection - the state the whole application exists for
# --------------------------------------------------------------------------


def test_explicit_halftime_status():
    assert derive_status("STATUS_HALFTIME", period=2, clock="0:00") is GameStatus.HALFTIME


def test_end_of_second_period_is_halftime():
    """Some feeds describe the break positionally rather than by name."""
    assert derive_status("STATUS_END_PERIOD", period=2, clock="0:00") is GameStatus.HALFTIME


def test_in_progress_at_zero_on_the_second_period_is_halftime():
    """The seconds before a feed switches to a dedicated halftime status."""
    assert derive_status("STATUS_IN_PROGRESS", period=2, clock="0:00") is GameStatus.HALFTIME


def test_end_of_first_and_third_periods_are_not_halftime():
    """The regression this guards: any quarter break turning the card green."""
    for period in (1, 3):
        status = derive_status("STATUS_END_PERIOD", period=period, clock="0:00")
        assert status is GameStatus.END_PERIOD
        assert not status.is_actionable
        assert status.tone is Tone.LIVE


def test_second_period_with_time_left_is_ordinary_live():
    assert derive_status("STATUS_IN_PROGRESS", period=2, clock="4:31") is GameStatus.LIVE


@pytest.mark.parametrize("clock", ["0:00", "00:00", "0"])
def test_zero_clock_spellings_all_count(clock):
    assert derive_status("STATUS_IN_PROGRESS", period=2, clock=clock) is GameStatus.HALFTIME


# --------------------------------------------------------------------------
# Everything else
# --------------------------------------------------------------------------


def test_completed_flag_wins_over_a_stale_period():
    """A finished game can still carry period 2 and a zeroed clock."""
    status = derive_status("STATUS_IN_PROGRESS", period=2, clock="0:00", completed=True)
    assert status is GameStatus.FINAL


def test_abandoned_games_beat_stale_live_fields():
    assert derive_status("STATUS_POSTPONED", period=2, clock="0:00") is GameStatus.POSTPONED
    assert derive_status("STATUS_CANCELED", period=3) is GameStatus.CANCELED
    assert derive_status("STATUS_SUSPENDED", period=2) is GameStatus.POSTPONED


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("STATUS_SCHEDULED", GameStatus.SCHEDULED),
        ("STATUS_IN_PROGRESS", GameStatus.LIVE),
        ("STATUS_FINAL", GameStatus.FINAL),
        ("STATUS_DELAYED", GameStatus.DELAYED),
        ("", GameStatus.SCHEDULED),
        (None, GameStatus.SCHEDULED),
    ],
)
def test_common_statuses(raw, expected):
    assert derive_status(raw, period=None) is expected


def test_status_prefix_is_optional():
    """Feeds vary on whether the STATUS_ prefix is present."""
    assert derive_status("halftime", period=2) is GameStatus.HALFTIME
    assert derive_status("final", completed=False) is GameStatus.FINAL


def test_unknown_status_with_a_live_period_does_not_claim_the_game_is_upcoming():
    assert derive_status("STATUS_SOMETHING_NEW", period=3, clock="7:00") is GameStatus.LIVE
    assert derive_status("STATUS_SOMETHING_NEW", period=None) is GameStatus.SCHEDULED


# --------------------------------------------------------------------------
# The semantic contract
# --------------------------------------------------------------------------


def test_every_status_has_a_tone_and_every_tone_has_tokens():
    """No status may invent a look that the theme cannot express."""
    for status in GameStatus:
        assert isinstance(status.tone, Tone)
        assert status.tone.value in TONE_TOKENS


def test_only_halftime_is_actionable():
    """"Actionable" means a prop board can be built - that is halftime, alone."""
    actionable = [s for s in GameStatus if s.is_actionable]
    assert actionable == [GameStatus.HALFTIME]


def test_halftime_is_the_only_green_state():
    green = [s for s in GameStatus if s.tone is Tone.HALFTIME]
    assert green == [GameStatus.HALFTIME]


def test_scheduled_games_carry_no_score():
    assert not GameStatus.SCHEDULED.has_score
    assert GameStatus.HALFTIME.has_score
    assert GameStatus.FINAL.has_score


def test_live_covers_the_states_where_the_ball_is_still_to_be_played():
    assert GameStatus.LIVE.is_live
    assert GameStatus.HALFTIME.is_live
    assert GameStatus.END_PERIOD.is_live
    assert not GameStatus.FINAL.is_live
    assert not GameStatus.SCHEDULED.is_live


def test_sort_order_puts_actionable_games_first():
    order = sorted(GameStatus, key=lambda s: s.sort_key)
    assert order[0] is GameStatus.HALFTIME
    assert order.index(GameStatus.LIVE) < order.index(GameStatus.SCHEDULED)
    assert order.index(GameStatus.SCHEDULED) < order.index(GameStatus.FINAL)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def test_halftime_label_is_unambiguous():
    d = describe(GameStatus.HALFTIME, period=2, clock="0:00")
    assert d.label == "HALFTIME"
    assert d.tone is Tone.HALFTIME


def test_live_label_reads_period_then_clock():
    assert describe(GameStatus.LIVE, period=2, clock="08:42").label == "2nd · 08:42"


def test_final_distinguishes_overtime():
    assert describe(GameStatus.FINAL).label == "FINAL"
    assert describe(GameStatus.FINAL, went_to_overtime=True).label == "FINAL/OT"


def test_end_period_names_the_quarter():
    assert describe(GameStatus.END_PERIOD, period=3).label == "END 3rd"


@pytest.mark.parametrize(
    "period,expected",
    [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (5, "OT"), (6, "2OT"), (None, "")],
)
def test_period_ordinals(period, expected):
    assert ordinal_period(period) == expected


def test_scheduled_shows_kickoff_and_a_countdown():
    start = datetime(2026, 8, 24, 19, 30, tzinfo=timezone.utc)
    now = start - timedelta(minutes=42)
    d = describe(GameStatus.SCHEDULED, start_time=start, now=now)
    assert d.label == "7:30 PM"
    assert d.detail == "Starts in 42 min"


def test_countdown_scales_with_distance():
    start = datetime(2026, 8, 24, 19, 0, tzinfo=timezone.utc)

    def at(delta):
        return describe(GameStatus.SCHEDULED, start_time=start, now=start - delta).detail

    assert at(timedelta(minutes=5)) == "Starts in 5 min"
    assert at(timedelta(minutes=150)) == "Starts in 2.5 hours"
    assert at(timedelta(hours=30)) == "In 1 day"
    assert at(timedelta(seconds=-10)) == "Starting now"


def test_countdown_is_omitted_without_a_reference_time():
    """A rendered page must be reproducible, so 'now' is never implicit."""
    start = datetime(2026, 8, 24, 19, 0, tzinfo=timezone.utc)
    assert describe(GameStatus.SCHEDULED, start_time=start).detail is None


def test_naive_datetimes_do_not_explode_mid_render():
    start = datetime(2026, 8, 24, 19, 0)
    now = datetime(2026, 8, 24, 18, 0)
    assert describe(GameStatus.SCHEDULED, start_time=start, now=now).detail is not None


# --------------------------------------------------------------------------
# Season types
# --------------------------------------------------------------------------


def test_espn_season_type_mapping():
    assert SeasonType.from_espn(1) is SeasonType.PRESEASON
    assert SeasonType.from_espn(2) is SeasonType.REGULAR
    assert SeasonType.from_espn(3) is SeasonType.POSTSEASON
    assert SeasonType.from_espn("1") is SeasonType.PRESEASON
    assert SeasonType.from_espn(None) is SeasonType.REGULAR


def test_only_non_regular_seasons_get_a_badge():
    assert SeasonType.PRESEASON.badge == "PRE"
    assert SeasonType.REGULAR.badge is None


def test_preseason_uses_the_identical_status_logic():
    """Preseason is a calendar fact, not a different kind of football."""
    assert derive_status("STATUS_HALFTIME", period=2).is_actionable


# --------------------------------------------------------------------------
# Dashboard behaviour
# --------------------------------------------------------------------------


def _card(status, gid="g", home_score=None, away_score=None, **kw) -> GameCard:
    return GameCard(
        game_id=gid,
        home=TeamLine("HOM", "Home", home_score),
        away=TeamLine("AWY", "Away", away_score),
        status=status,
        **kw,
    )


def test_dashboard_sorts_halftime_above_everything():
    dash = Dashboard(games=[
        _card(GameStatus.FINAL, "a"),
        _card(GameStatus.SCHEDULED, "b"),
        _card(GameStatus.LIVE, "c"),
        _card(GameStatus.HALFTIME, "d"),
    ])
    assert [g.game_id for g in dash.sorted_games()] == ["d", "c", "b", "a"]


def test_dashboard_counts():
    dash = Dashboard(games=[
        _card(GameStatus.HALFTIME, "a"),
        _card(GameStatus.LIVE, "b"),
        _card(GameStatus.END_PERIOD, "c"),
        _card(GameStatus.SCHEDULED, "d"),
        _card(GameStatus.FINAL, "e"),
    ])
    assert dash.halftime_count == 1
    assert dash.live_count == 3      # live, end-of-period, and halftime
    assert dash.upcoming_count == 1
    assert dash.final_count == 1


def test_leader_emphasis():
    assert _card(GameStatus.LIVE, home_score=24, away_score=21).leader == "home"
    assert _card(GameStatus.LIVE, home_score=21, away_score=24).leader == "away"
    assert _card(GameStatus.LIVE, home_score=21, away_score=21).leader is None
    assert _card(GameStatus.SCHEDULED).leader is None


def test_filter_membership():
    assert "halftime" in _filter_slugs(GameStatus.HALFTIME)
    # Halftime is a live state, so the Live filter must include it.
    assert "live" in _filter_slugs(GameStatus.HALFTIME)
    assert "upcoming" in _filter_slugs(GameStatus.SCHEDULED)
    assert "finished" in _filter_slugs(GameStatus.FINAL)
    assert "halftime" not in _filter_slugs(GameStatus.END_PERIOD)
    # Everything is in "all".
    for status in GameStatus:
        assert "all" in _filter_slugs(status)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fixture_dashboard():
    from nflprops.providers.espn import load_dashboard

    return load_dashboard(FIXTURE)


def test_fixture_covers_every_status():
    dash = _fixture_dashboard()
    seen = {g.status for g in dash.games}
    for required in (
        GameStatus.HALFTIME, GameStatus.LIVE, GameStatus.END_PERIOD,
        GameStatus.SCHEDULED, GameStatus.FINAL, GameStatus.DELAYED,
        GameStatus.POSTPONED,
    ):
        assert required in seen, f"fixture is missing {required}"


def test_fixture_detects_both_halftime_encodings():
    """One game reports halftime by name, another by end-of-period-2."""
    dash = _fixture_dashboard()
    halftime = [g for g in dash.games if g.status is GameStatus.HALFTIME]
    assert len(halftime) == 2


def test_scheduled_games_have_their_scores_cleared():
    dash = _fixture_dashboard()
    for g in dash.games:
        if g.status is GameStatus.SCHEDULED:
            assert g.home.score is None and g.away.score is None


def test_preseason_fixture_is_labelled():
    dash = _fixture_dashboard()
    assert all(g.season_type is SeasonType.PRESEASON for g in dash.games)
    assert dash.subtitle and "Preseason" in dash.subtitle


def test_html_renders_tone_attributes_not_hardcoded_colour():
    dash = _fixture_dashboard()
    out = render_dashboard(dash, now=datetime(2026, 8, 24, 23, 38, tzinfo=timezone.utc))
    assert 'data-tone="halftime"' in out
    assert 'data-tone="live"' in out
    assert 'data-tone="final"' in out
    assert "HALFTIME" in out
    # The page must declare its own ground, or it borrows the host's theme.
    assert "background:var(--ground)" in out


def test_html_marks_only_running_clocks_for_animation():
    dash = Dashboard(games=[
        _card(GameStatus.LIVE, "a", 7, 3, period=3, clock="8:42"),
        _card(GameStatus.HALFTIME, "b", 7, 3, period=2, clock="0:00"),
    ])
    # Count on the cards only - the stylesheet also contains the selector.
    markup = render_dashboard(dash).split("</style>", 1)[1]
    assert markup.count('data-running="true"') == 1
    assert markup.count('data-running="false"') == 1


def test_a_card_is_only_clickable_when_a_board_exists_behind_it():
    """The affordance must never lie.

    A halftime card with a board becomes a link; a halftime card without one
    says so plainly rather than inviting a click that goes nowhere.
    """
    from nflprops.dashboard import render_slate_fragment

    dash = Dashboard(games=[_card(GameStatus.HALFTIME, "a", 7, 3)])

    linked = render_slate_fragment(dash, link_for=lambda gid: f"#game-{gid}")
    assert '<a class="card" href="#game-a"' in linked
    assert "View prop board" in linked

    bare = render_slate_fragment(dash, link_for=lambda gid: None)
    assert "<a class=" not in bare
    assert "No props loaded" in bare
    assert "View prop board" not in bare


def test_non_halftime_games_never_offer_a_board():
    from nflprops.dashboard import render_slate_fragment

    for status in (GameStatus.LIVE, GameStatus.SCHEDULED, GameStatus.FINAL):
        out = render_slate_fragment(
            Dashboard(games=[_card(status, "a", 7, 3)]),
            link_for=lambda gid: f"#game-{gid}",
        )
        assert "View prop board" not in out
        assert "No props loaded" not in out


def test_terminal_dashboard_surfaces_the_next_command():
    dash = _fixture_dashboard()
    out = render_dashboard_terminal(dash)
    assert "HALFTIME" in out
    assert "nflprops run --game espn:" in out


def test_every_filter_button_has_matching_cards_or_a_zero_count():
    dash = _fixture_dashboard()
    out = render_dashboard(dash)
    for slug, label, _ in FILTERS:
        assert f'data-filter="{slug}"' in out


def test_empty_dashboard_renders_without_raising():
    assert "No games in this view" in render_dashboard(Dashboard(games=[]))


# --------------------------------------------------------------------------
# Combined app
# --------------------------------------------------------------------------


def _app_page():
    """Build the combined page from the bundled fixtures."""
    from nflprops.agent import analyze
    from nflprops.app import GameBoard, render_app
    from nflprops.models import PropScope
    from nflprops.providers.stake import StakePropsProvider
    from nflprops.serde import load_game

    dash = _fixture_dashboard()
    root = os.path.dirname(os.path.dirname(__file__))

    def board(gid, state_file, props_file):
        state = load_game(os.path.join(root, "fixtures", state_file))
        props = StakePropsProvider(
            path=os.path.join(root, "fixtures", props_file),
            default_scope=PropScope.FULL_GAME,
        ).fetch()
        res = analyze(state, props, n_sims=1200, seed=5)
        return GameBoard(gid, res.ranked, res.state, res.diagnostics, 1200,
                         settled=res.settled)

    boards = [
        board("401780001", "game_bal_cin_halftime.json", "props_paste.txt"),
        board("401780002", "game_gb_sea_halftime.json", "props_gb_sea.txt"),
    ]
    return render_app(dash, boards), dash, boards


def test_component_stylesheets_do_not_collide():
    """The slate and the board share one page, so their classes must be disjoint.

    This guards a bug that is invisible in either surface alone: the board's
    scoreboard styling once overrode the dashboard's card score, drawing a
    stray divider through every card. Whichever sheet is emitted second wins,
    so the only safe invariant is that they never define the same bare class.
    """
    import re as _re

    from nflprops.dashboard import _CSS as DASH
    from nflprops.report import _COMPONENT_CSS as BOARD
    from nflprops.theme import BASE_CSS as BASE, TOKENS_CSS as TOK

    def bare_classes(css):
        css = _re.sub(r"/\*.*?\*/", "", css, flags=_re.S)
        found = set()
        for _, sels in _re.findall(r"(^|\})([^{}@]+)\{", css, _re.M):
            for sel in sels.split(","):
                m = _re.match(r"\s*\.([a-zA-Z][\w-]*)\s*$", sel)
                if m:
                    found.add(m.group(1))
        return found

    dash_only = DASH.replace(TOK, "").replace(BASE, "")
    assert not (bare_classes(BOARD) & bare_classes(dash_only))


def test_app_links_only_halftime_cards_to_boards():
    page, dash, boards = _app_page()
    assert page.count('<a class="card"') == 2
    for b in boards:
        assert f'href="#{b.anchor}"' in page
        assert f'id="{b.anchor}"' in page


def test_app_embeds_a_full_board_per_game():
    page, _, boards = _app_page()
    for b in boards:
        assert len(b.ranked) > 5
    # Both boards' props are present in the one page. (Names are HTML-escaped,
    # so match on a fragment without an apostrophe.)
    assert "Chase" in page and "Derrick Henry" in page
    assert "Jaxon Smith-Njigba" in page and "Josh Jacobs" in page


def test_app_board_views_start_hidden_so_the_slate_is_the_landing_view():
    page, _, _ = _app_page()
    for chunk in page.split('<section class="view gameview"')[1:]:
        assert "hidden>" in chunk.split(">", 1)[0] + ">"


def test_app_survives_a_manifest_game_that_is_not_at_halftime():
    """A board for a game that has restarted must not be presented as live."""
    from nflprops.app import GameBoard, render_app

    page, dash, boards = _app_page()
    stale = boards[0]
    # Point the board at a game that is FINAL on this slate.
    stale = GameBoard("401780008", stale.ranked, stale.state, stale.diag, 1200)
    out = render_app(dash, [stale])
    assert '<a class="card"' not in out
    assert "No props loaded" in out


def test_app_has_no_board_when_nothing_is_at_halftime():
    from nflprops.app import render_app

    dash = Dashboard(games=[_card(GameStatus.FINAL, "a", 7, 3)])
    out = render_app(dash, [])
    assert "No game is at halftime right now" in out
    # No board *element* - the class still appears in the stylesheet.
    assert '<section class="view gameview"' not in out


# --------------------------------------------------------------------------
# Live mode wiring
# --------------------------------------------------------------------------


def test_live_mode_fetches_state_only_for_games_at_the_break(monkeypatch, tmp_path):
    """--live should pull the slate, then each halftime game's box score.

    The HTTP itself cannot be exercised here, so the provider is stubbed and
    this asserts the wiring: which games get fetched, and that a board is built
    from what comes back.
    """
    import nflprops.cli as cli
    from nflprops.providers import espn as espn_mod
    from nflprops.serde import load_game

    root = os.path.dirname(os.path.dirname(__file__))
    fetched = []

    class StubProvider:
        def dashboard(self, **kw):
            return _fixture_dashboard()

        def fetch(self, game_id):
            fetched.append(game_id)
            return load_game(
                os.path.join(root, "fixtures", "game_bal_cin_halftime.json")
            )

    monkeypatch.setattr(espn_mod, "ESPNProvider", StubProvider)

    out = tmp_path / "live.html"
    args = cli.build_parser().parse_args([
        "app", "--live",
        "--props-for", f"401780001={os.path.join(root, 'fixtures', 'props_paste.txt')}",
        "--sims", "600", "--out", str(out),
    ])
    assert args.func(args) == 0

    # Only the game that was named got fetched - not the whole slate.
    assert fetched == ["401780001"]
    page = out.read_text()
    assert '<a class="card" href="#game-401780001"' in page


def test_props_shorthand_is_refused_when_it_would_be_ambiguous(monkeypatch, capsys):
    """Two games at the break and a bare --props must not guess."""
    import nflprops.cli as cli
    from nflprops.providers import espn as espn_mod

    class StubProvider:
        def dashboard(self, **kw):
            return _fixture_dashboard()

        def fetch(self, game_id):  # pragma: no cover - must not be reached
            raise AssertionError("should not fetch when the mapping is ambiguous")

    monkeypatch.setattr(espn_mod, "ESPNProvider", StubProvider)

    args = cli.build_parser().parse_args(["app", "--live", "--props", "x.txt"])
    assert args.func(args) == 2
    err = capsys.readouterr().err
    assert "ambiguous" in err
    # It names the candidates so the next command is obvious.
    assert "401780001" in err and "401780002" in err


def test_props_shorthand_says_so_when_nothing_is_at_the_break(monkeypatch, capsys):
    import nflprops.cli as cli
    from nflprops.dashboard import Dashboard as D
    from nflprops.providers import espn as espn_mod

    class StubProvider:
        def dashboard(self, **kw):
            return D(games=[_card(GameStatus.LIVE, "a", 7, 3)])

    monkeypatch.setattr(espn_mod, "ESPNProvider", StubProvider)

    args = cli.build_parser().parse_args(["app", "--live", "--props", "x.txt"])
    assert args.func(args) == 4
    assert "Nothing is at halftime" in capsys.readouterr().err


def test_one_unreachable_box_score_does_not_lose_the_other_game(monkeypatch, tmp_path):
    """A single failed fetch costs that game only, not the whole page."""
    import nflprops.cli as cli
    from nflprops.providers import espn as espn_mod
    from nflprops.providers.base import ProviderError
    from nflprops.serde import load_game

    root = os.path.dirname(os.path.dirname(__file__))

    class StubProvider:
        def dashboard(self, **kw):
            return _fixture_dashboard()

        def fetch(self, game_id):
            if game_id == "401780001":
                raise ProviderError("boom")
            return load_game(
                os.path.join(root, "fixtures", "game_gb_sea_halftime.json")
            )

    monkeypatch.setattr(espn_mod, "ESPNProvider", StubProvider)

    out = tmp_path / "live.html"
    fx = lambda n: os.path.join(root, "fixtures", n)
    args = cli.build_parser().parse_args([
        "app", "--live",
        "--props-for", f"401780001={fx('props_paste.txt')}",
        "--props-for", f"401780002={fx('props_gb_sea.txt')}",
        "--sims", "600", "--out", str(out),
    ])
    assert args.func(args) == 0
    page = out.read_text()
    assert "#game-401780002" in page
    assert 'href="#game-401780001"' not in page
