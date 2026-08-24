"""The game dashboard: a vertical feed of status-driven game cards.

Every visual decision here is derived from ``status.py``. No card is styled
individually; a card asks its status for a tone and renders the tokens that
tone maps to. Adding a state, or changing what halftime looks like, is a
one-line change that propagates everywhere.

The layout takes the oversized rounded shapes and unmissable status separation
of a scribbled scoreboard and holds onto both, while pulling the execution
back to something you would leave open on a second monitor: restrained colour,
real typographic hierarchy, and a status rail doing the work that a highlighter
does on paper.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence

from .status import GameStatus, SeasonType, StatusDisplay, describe
from .theme import BASE_CSS, FONT_DISPLAY, FONT_MONO, FONT_LINKS, TOKENS_CSS


@dataclass
class TeamLine:
    abbr: str
    name: str = ""
    score: Optional[int] = None
    record: Optional[str] = None
    logo_url: Optional[str] = None

    @property
    def display(self) -> str:
        return self.name or self.abbr


@dataclass
class GameCard:
    """One game, already resolved into everything the view needs."""

    game_id: str
    home: TeamLine
    away: TeamLine
    status: GameStatus
    period: Optional[int] = None
    clock: Optional[str] = None
    start_time: Optional[datetime] = None
    season_type: SeasonType = SeasonType.REGULAR
    week: Optional[int] = None
    broadcast: Optional[str] = None
    venue: Optional[str] = None
    situation: Optional[str] = None      # "3rd & 7 at CIN 42"
    odds: Optional[str] = None           # "BAL -3.5 · O/U 47.5"
    went_to_overtime: bool = False

    def display(self, now: Optional[datetime] = None) -> StatusDisplay:
        return describe(
            self.status,
            period=self.period,
            clock=self.clock,
            start_time=self.start_time,
            now=now,
            went_to_overtime=self.went_to_overtime,
        )

    @property
    def leader(self) -> Optional[str]:
        """Which side is ahead, for emphasis. None while level or scoreless."""
        if self.home.score is None or self.away.score is None:
            return None
        if self.home.score > self.away.score:
            return "home"
        if self.away.score > self.home.score:
            return "away"
        return None


@dataclass
class Dashboard:
    games: List[GameCard] = field(default_factory=list)
    title: str = "Today's Games"
    subtitle: Optional[str] = None
    generated_at: Optional[datetime] = None

    def sorted_games(self) -> List[GameCard]:
        """Actionable first: halftime, then live, then upcoming, then done."""
        return sorted(
            self.games,
            key=lambda g: (
                g.status.sort_key,
                g.start_time.timestamp() if g.start_time else 0,
                g.game_id,
            ),
        )

    def count(self, predicate) -> int:
        return sum(1 for g in self.games if predicate(g))

    @property
    def live_count(self) -> int:
        return self.count(lambda g: g.status.is_live)

    @property
    def halftime_count(self) -> int:
        return self.count(lambda g: g.status.is_actionable)

    @property
    def upcoming_count(self) -> int:
        return self.count(lambda g: g.status is GameStatus.SCHEDULED)

    @property
    def final_count(self) -> int:
        return self.count(lambda g: g.status is GameStatus.FINAL)


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------

#: (id, label, statuses it admits). "all" admits everything.
FILTERS = [
    ("all", "All", None),
    ("halftime", "Halftime", {GameStatus.HALFTIME}),
    ("live", "Live", {GameStatus.LIVE, GameStatus.END_PERIOD, GameStatus.HALFTIME}),
    ("upcoming", "Upcoming", {GameStatus.SCHEDULED}),
    ("finished", "Finished", {GameStatus.FINAL}),
]


def _filter_slugs(status: GameStatus) -> str:
    """Space-separated filter ids a card belongs to, read by the client JS."""
    slugs = ["all"]
    for slug, _, allowed in FILTERS:
        if allowed and status in allowed:
            slugs.append(slug)
    if status in (GameStatus.POSTPONED, GameStatus.CANCELED, GameStatus.DELAYED):
        slugs.append("other")
    return " ".join(slugs)


# --------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------

_CSS = f"""
{TOKENS_CSS}
{BASE_CSS}

/* ---- header ---------------------------------------------------------- */
.head{{display:flex; flex-wrap:wrap; align-items:flex-end; gap:16px;
  justify-content:space-between; margin:32px 0 4px}}
h1{{font-family:{FONT_DISPLAY}; font-weight:700; font-size:34px;
  letter-spacing:.005em; margin:0; text-wrap:balance}}
.tally{{display:flex; gap:18px; flex-wrap:wrap; padding-bottom:5px}}
.tally .t{{display:flex; align-items:baseline; gap:6px; font-size:13.5px;
  color:var(--muted)}}
.tally .t b{{font-family:{FONT_MONO}; font-size:17px; font-weight:600;
  color:var(--ink); font-variant-numeric:tabular-nums}}
.tally .t.hot b{{color:var(--st-half-fg)}}

/* ---- filters --------------------------------------------------------- */
.filters{{display:flex; gap:7px; flex-wrap:wrap; margin:20px 0 18px}}
.filters button{{
  font-family:{FONT_DISPLAY}; font-weight:600; font-size:12px;
  letter-spacing:.11em; text-transform:uppercase; cursor:pointer;
  color:var(--muted); background:var(--card); border:1px solid var(--rule);
  padding:8px 15px; border-radius:999px; transition:background .15s,
  color .15s, border-color .15s;
}}
.filters button:hover{{color:var(--ink); border-color:var(--st-up-edge)}}
.filters button[aria-pressed="true"]{{
  background:var(--ink); color:var(--ground); border-color:var(--ink);
}}
.filters .cnt{{font-family:{FONT_MONO}; font-size:11px; opacity:.65;
  margin-left:5px}}

/* ---- feed ------------------------------------------------------------ */
.feed{{display:flex; flex-direction:column; gap:12px}}
.card{{
  position:relative; display:grid; align-items:center; gap:18px;
  grid-template-columns:1fr auto 1fr auto;
  background:var(--card); border:1px solid var(--rule);
  border-radius:22px; padding:20px 26px 20px 30px;
  transition:border-color .18s, transform .18s, box-shadow .18s;
}}
.card:hover{{transform:translateY(-1px);
  box-shadow:0 4px 18px -8px rgba(15,25,45,.28)}}
.card:before{{
  content:""; position:absolute; left:0; top:16px; bottom:16px; width:4px;
  border-radius:0 4px 4px 0; background:var(--edge);
}}
.card[data-tone="halftime"]{{background:var(--st-half-bg);
  border-color:var(--st-half-edge)}}
.card[data-tone="live"]{{border-color:var(--st-live-edge)}}
.card[data-tone="final"]{{background:var(--st-final-bg); opacity:.82}}
.card[data-tone="final"]:hover{{opacity:1}}
.card[data-tone="warning"]{{background:var(--st-warn-bg)}}

/* tone drives the rail colour; nothing styles a card directly */
.card[data-tone="upcoming"]{{--edge:var(--st-up-edge)}}
.card[data-tone="live"]{{--edge:var(--st-live-edge)}}
.card[data-tone="halftime"]{{--edge:var(--st-half-edge)}}
.card[data-tone="final"]{{--edge:var(--st-final-edge)}}
.card[data-tone="warning"]{{--edge:var(--st-warn-edge)}}

/* ---- teams ----------------------------------------------------------- */
.team{{display:flex; flex-direction:column; gap:3px; min-width:0}}
.team.away{{align-items:flex-end; text-align:right}}
.team .nm{{font-family:{FONT_DISPLAY}; font-weight:600; font-size:21px;
  letter-spacing:.01em; line-height:1.15; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap; max-width:100%}}
.team .side{{font-family:{FONT_DISPLAY}; font-weight:600; font-size:9.5px;
  letter-spacing:.15em; text-transform:uppercase; color:var(--faint);
  line-height:1}}
.team .meta{{font-size:12px; color:var(--faint);
  font-family:{FONT_MONO}}}
.team.trail .nm{{color:var(--muted); font-weight:500}}

/* ---- score ----------------------------------------------------------- */
.score{{display:flex; align-items:center; gap:12px;
  font-family:{FONT_MONO}; font-variant-numeric:tabular-nums;
  font-size:30px; font-weight:600; line-height:1}}
.score .dash{{color:var(--faint); font-weight:400; font-size:20px}}
.score .s.trail{{color:var(--muted); font-weight:500}}
.card[data-tone="halftime"] .score{{font-size:33px}}
.kick{{font-family:{FONT_MONO}; font-size:20px; font-weight:500;
  color:var(--muted); white-space:nowrap}}

/* ---- status block ---------------------------------------------------- */
.stat{{display:flex; flex-direction:column; align-items:flex-end; gap:4px;
  min-width:118px}}
.pill{{
  display:inline-flex; align-items:center; gap:7px;
  font-family:{FONT_DISPLAY}; font-weight:600; font-size:12px;
  letter-spacing:.12em; text-transform:uppercase; white-space:nowrap;
  padding:6px 12px; border-radius:999px;
  color:var(--fg); background:var(--bg); border:1px solid transparent;
}}
.card[data-tone="upcoming"] .pill{{--fg:var(--st-up-fg); --bg:var(--st-up-bg)}}
.card[data-tone="live"] .pill{{--fg:var(--st-live-fg); --bg:var(--st-live-bg)}}
.card[data-tone="halftime"] .pill{{--fg:#fff; --bg:var(--st-half-edge)}}
.card[data-tone="final"] .pill{{--fg:var(--st-final-fg); --bg:transparent;
  border-color:var(--st-final-edge)}}
.card[data-tone="warning"] .pill{{--fg:var(--st-warn-fg); --bg:var(--st-warn-bg)}}
.substat{{font-size:12px; color:var(--faint); text-align:right}}

.dot{{width:7px; height:7px; border-radius:50%; background:currentColor;
  flex:0 0 auto}}
/* only a running clock pulses; a stopped one (end of quarter, halftime)
   shows a steady dot, because a blinking light should mean "happening now" */
.card[data-running="true"] .dot{{animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1; transform:scale(1)}}
  50%{{opacity:.35; transform:scale(.82)}}}}

/* ---- footer strip ---------------------------------------------------- */
.under{{grid-column:1/-1; display:flex; flex-wrap:wrap; gap:8px 16px;
  align-items:center; margin-top:14px; padding-top:13px;
  border-top:1px solid var(--rule-soft); font-size:12.5px; color:var(--muted)}}
.card[data-tone="halftime"] .under{{border-top-color:var(--st-half-edge)}}
.badge{{font-family:{FONT_DISPLAY}; font-weight:600; font-size:10.5px;
  letter-spacing:.11em; text-transform:uppercase; padding:2px 7px;
  border-radius:3px; background:var(--rule-soft); color:var(--muted)}}
.card[data-tone="halftime"] .badge{{background:rgba(255,255,255,.5);
  color:var(--st-half-fg)}}
.under .sep{{color:var(--faint)}}
.ready{{margin-left:auto; display:inline-flex; align-items:center; gap:6px;
  font-family:{FONT_DISPLAY}; font-weight:600; font-size:11.5px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--st-half-fg)}}

/* ---- empty ----------------------------------------------------------- */
.empty{{display:none; text-align:center; padding:56px 20px; color:var(--muted);
  border:1px dashed var(--rule); border-radius:22px; font-size:14.5px}}
.feed[data-empty="true"] .empty{{display:block}}

footer{{margin-top:34px; padding-top:15px; border-top:1px solid var(--rule);
  font-size:12.5px; color:var(--faint); display:flex; flex-wrap:wrap;
  gap:6px 14px}}

/* ---- responsive ------------------------------------------------------ */
@media (max-width:760px){{
  .card{{grid-template-columns:1fr auto; gap:10px 14px;
    padding:18px 20px 18px 24px; border-radius:20px}}
  .team{{flex-direction:row; align-items:baseline; gap:8px}}
  .team .side{{order:-1; min-width:38px}}
  .team.away{{grid-row:2; text-align:left; align-items:baseline;
    justify-content:flex-start}}
  .team .nm{{font-size:18px}}
  .score{{display:contents}}
  .score .dash{{display:none}}
  .score .s{{font-family:{FONT_MONO}; font-size:25px; font-weight:600;
    text-align:right; font-variant-numeric:tabular-nums}}
  .score .s.home{{grid-row:1; grid-column:2}}
  .score .s.away{{grid-row:2; grid-column:2}}
  .stat{{grid-column:1/-1; flex-direction:row; align-items:center;
    justify-content:space-between; min-width:0; margin-top:4px}}
  .substat{{text-align:left}}
  .kick{{grid-column:2; grid-row:1/3; align-self:center}}
  h1{{font-size:28px}}
}}
"""

_JS = """
(function(){
  var feed = document.getElementById('feed');
  var buttons = Array.prototype.slice.call(
    document.querySelectorAll('.filters button'));

  function apply(slug){
    var shown = 0;
    Array.prototype.forEach.call(feed.querySelectorAll('.card'), function(card){
      var match = (' ' + card.dataset.filters + ' ').indexOf(' ' + slug + ' ') > -1;
      card.hidden = !match;
      if (match) shown++;
    });
    feed.dataset.empty = shown === 0 ? 'true' : 'false';
    buttons.forEach(function(b){
      b.setAttribute('aria-pressed', String(b.dataset.filter === slug));
    });
    try { localStorage.setItem('nflprops.filter', slug); } catch (e) {}
  }

  buttons.forEach(function(b){
    b.addEventListener('click', function(){ apply(b.dataset.filter); });
  });

  var initial = 'all';
  try {
    var saved = localStorage.getItem('nflprops.filter');
    if (saved && document.querySelector('[data-filter="' + saved + '"]')) {
      initial = saved;
    }
  } catch (e) {}
  apply(initial);
})();
"""


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _team_html(team: TeamLine, side: str, leader: Optional[str]) -> str:
    """One side of the card.

    The home/away micro-label is not decoration: with the home team on the
    left, a reader glancing at an unfamiliar matchup has no other cue for
    which venue the game is at, and every other sports surface they use writes
    it "away @ home". Naming the sides costs one quiet line and removes the
    ambiguity entirely.
    """
    e = html.escape
    trail = leader is not None and leader != side
    meta = e(team.record) if team.record else ""
    return (
        f'<div class="team {side}{" trail" if trail else ""}">'
        f'<span class="side">{side}</span>'
        f'<span class="nm">{e(team.display)}</span>'
        f'{f"<span class=meta>{meta}</span>" if meta else ""}'
        f"</div>"
    )


def _score_html(card: GameCard) -> str:
    if not card.status.has_score or card.home.score is None or card.away.score is None:
        disp = card.display()
        return f'<div class="kick">{html.escape(disp.label)}</div>'

    leader = card.leader
    h_cls = "s home" + (" trail" if leader == "away" else "")
    a_cls = "s away" + (" trail" if leader == "home" else "")
    return (
        f'<div class="score"><span class="{h_cls}">{card.home.score}</span>'
        f'<span class="dash">&ndash;</span>'
        f'<span class="{a_cls}">{card.away.score}</span></div>'
    )


def _under_html(card: GameCard) -> str:
    e = html.escape
    bits: List[str] = []

    badge = card.season_type.badge
    if badge:
        week = f" Wk {card.week}" if card.week else ""
        bits.append(f'<span class="badge">{e(badge)}{e(week)}</span>')

    for value in (card.situation, card.broadcast, card.venue, card.odds):
        if value:
            bits.append(e(value))

    ready = ""
    if card.status.is_actionable:
        ready = (
            '<span class="ready"><span class="dot"></span>Prop board ready</span>'
        )

    if not bits and not ready:
        return ""

    joined = '<span class="sep">·</span>'.join(
        b if b.startswith("<") else f"<span>{b}</span>" for b in bits
    )
    return f'<div class="under">{joined}{ready}</div>'


def _card_html(card: GameCard, now: Optional[datetime]) -> str:
    e = html.escape
    disp = card.display(now)
    tone = disp.tone.value

    dot = '<span class="dot"></span>' if tone in ("live", "halftime") else ""

    # An upcoming card already prints the kickoff time large and centred, so
    # repeating it in the pill wastes the loudest element on the card. Promote
    # the countdown into the pill instead: the time answers "when", the
    # countdown answers "how long have I got", and neither repeats the other.
    pill_text = disp.label
    sub_text = disp.detail
    if card.status is GameStatus.SCHEDULED and disp.detail:
        pill_text = disp.detail.replace("Starts in ", "in ").replace("Starts ", "")
        sub_text = None

    sub = f'<span class="substat">{e(sub_text)}</span>' if sub_text else ""

    return (
        f'<article class="card" data-tone="{tone}" '
        f'data-running="{"true" if card.status is GameStatus.LIVE else "false"}" '
        f'data-filters="{_filter_slugs(card.status)}" '
        f'aria-label="{e(card.away.display)} at {e(card.home.display)}, {e(disp.label)}">'
        f"{_team_html(card.home, 'home', card.leader)}"
        f"{_score_html(card)}"
        f"{_team_html(card.away, 'away', card.leader)}"
        f'<div class="stat"><span class="pill">{dot}{e(pill_text)}</span>{sub}</div>'
        f"{_under_html(card)}"
        f"</article>"
    )


def render_dashboard(
    dash: Dashboard,
    now: Optional[datetime] = None,
    title: Optional[str] = None,
    refresh_seconds: Optional[int] = None,
) -> str:
    """Render the full dashboard page.

    ``refresh_seconds`` adds a meta refresh, which is what turns a rendered
    file into a board you can actually leave open during a slate. It is off by
    default because a published or emailed copy should stay put.
    """
    e = html.escape
    games = dash.sorted_games()
    page_title = title or "NFL Game Dashboard"

    counts = {
        "all": len(games),
        "halftime": dash.halftime_count,
        "live": dash.live_count,
        "upcoming": dash.upcoming_count,
        "finished": dash.final_count,
    }

    filters = "".join(
        f'<button type="button" data-filter="{slug}" aria-pressed="false">'
        f'{e(label)}<span class="cnt">{counts.get(slug, 0)}</span></button>'
        for slug, label, _ in FILTERS
    )

    tallies = [("Games", len(games), False)]
    if dash.halftime_count:
        tallies.append(("At halftime", dash.halftime_count, True))
    tallies.append(("Live", dash.live_count, False))
    tallies.append(("Upcoming", dash.upcoming_count, False))
    tally_html = "".join(
        f'<span class="t{" hot" if hot else ""}"><b>{n}</b>{e(label)}</span>'
        for label, n, hot in tallies
    )

    cards = "".join(_card_html(g, now) for g in games)
    stamp = ""
    if dash.generated_at:
        stamp = f"As of {_format_stamp(dash.generated_at)}"

    subtitle = dash.subtitle or ""

    refresh = (
        f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">'
        if refresh_seconds
        else ""
    )

    return f"""<title>{e(page_title)}</title>
{refresh}
{FONT_LINKS}
<style>{_CSS}</style>
<div class="wrap">
  <div class="head">
    <div>
      <h1>{e(dash.title)}</h1>
      {f'<p class="eyebrow" style="margin:6px 0 0">{e(subtitle)}</p>' if subtitle else ''}
    </div>
    <div class="tally">{tally_html}</div>
  </div>

  <div class="filters" role="group" aria-label="Filter games by status">{filters}</div>

  <div class="feed" id="feed" data-empty="false">
    {cards}
    <p class="empty">No games in this view.</p>
  </div>

  <footer>
    <span>{e(stamp)}</span>
    <span>Green means <strong>halftime</strong> &mdash; the point where a prop
    board can be built for that game.</span>
  </footer>
</div>
<script>{_JS}</script>"""


def _format_stamp(value: datetime) -> str:
    hour = value.hour % 12 or 12
    meridiem = "AM" if value.hour < 12 else "PM"
    return f"{value.strftime('%b %-d')}, {hour}:{value.minute:02d} {meridiem}"


# --------------------------------------------------------------------------
# Terminal rendering
# --------------------------------------------------------------------------

_TONE_MARK = {
    "halftime": "HALF",
    "live": "LIVE",
    "upcoming": "----",
    "final": "DONE",
    "warning": "!!!!",
}


def render_dashboard_terminal(
    dash: Dashboard, now: Optional[datetime] = None
) -> str:
    """Plain-text board, for when a browser is not in the loop."""
    games = dash.sorted_games()
    lines = [
        "=" * 84,
        f"  {dash.title.upper()}",
        f"  {len(games)} games · {dash.halftime_count} at halftime · "
        f"{dash.live_count} live · {dash.upcoming_count} upcoming",
        "=" * 84,
    ]

    for g in games:
        disp = g.display(now)
        mark = _TONE_MARK.get(disp.tone.value, "    ")
        if g.status.has_score and g.home.score is not None:
            score = f"{g.home.score:>3} - {g.away.score:<3}"
        else:
            score = "    ·    "  # same width as a score, so columns stay true
        badge = g.season_type.badge or ""
        lines.append(
            f"  [{mark}]  {g.home.abbr:>4} {score} {g.away.abbr:<4}  "
            f"{disp.label:<14} {badge:<4} {disp.detail or ''}"
        )
        if g.status.is_actionable:
            lines.append(
                f"          -> prop board ready: "
                f"nflprops run --game espn:{g.game_id} --props <file>"
            )

    lines.append("=" * 84)
    return "\n".join(lines)
