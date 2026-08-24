"""Rendering: terminal board, HTML board, and JSON.

The board is sorted highest-probability-first, which is what was asked for.
One caveat is printed with it rather than buried: sorting by hit rate ranks the
*most likely* props, which are usually short-priced favourites, not the most
*profitable* ones. The edge column is what says whether the number is wrong.
"""

from __future__ import annotations

import html
import json
from typing import List, Optional, Sequence

from .models import BINARY_MARKETS, GameState, PropScope
from .rank import GameDiagnostics, RankedProp

BAR_CHARS = "▏▎▍▌▋▊▉█"


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _odds_str(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


def _bar(p: float, width: int = 12) -> str:
    filled = p * width
    whole = int(filled)
    frac = filled - whole
    out = "█" * whole
    if whole < width:
        out += BAR_CHARS[min(int(frac * 8), 7)] if frac > 0.05 else " "
    return out.ljust(width)[:width]


def context_lines(state: GameState, diag: GameDiagnostics) -> List[str]:
    home, away = state.home, state.away
    mins = diag.seconds_remaining // 60
    secs = diag.seconds_remaining % 60
    lines = [
        f"{away.abbr} {away.score}  @  {home.abbr} {home.score}"
        f"   |   Q{state.quarter}, {mins}:{secs:02d} remaining"
        f"   |   {state.possession} ball",
        f"Projected final:  {away.abbr} {diag.proj_points[away.abbr]:.1f}"
        f"  -  {home.abbr} {diag.proj_points[home.abbr]:.1f}"
        f"   ({away.abbr} win {diag.win_prob[away.abbr]:.0%},"
        f" {home.abbr} win {diag.win_prob[home.abbr]:.0%})",
        f"Projected H2 pass rate:  {away.abbr} {diag.pass_rate[away.abbr]:.0%}"
        f" (neutral {away.base_pass_rate:.0%})"
        f"   |   {home.abbr} {diag.pass_rate[home.abbr]:.0%}"
        f" (neutral {home.base_pass_rate:.0%})",
        f"Projected H2 plays:  {away.abbr} {diag.away_plays:.0f}"
        f"   |   {home.abbr} {diag.home_plays:.0f}"
        f"   |   blowout risk (17+): {diag.blowout_prob:.0%}",
    ]
    w = state.weather
    if not w.dome:
        lines.append(
            f"Conditions:  {w.temp_f:.0f}F, wind {w.wind_mph:.0f} mph,"
            f" {w.precip.value.replace('_', ' ')}"
        )
    else:
        lines.append("Conditions:  indoors")
    return lines


# --------------------------------------------------------------------------
# Terminal
# --------------------------------------------------------------------------


def render_terminal(
    ranked: Sequence[RankedProp],
    state: GameState,
    diag: GameDiagnostics,
    n_sims: int,
    verbose: bool = False,
    limit: Optional[int] = None,
) -> str:
    out: List[str] = []
    out.append("=" * 104)
    out.append(f"  HALFTIME PROP BOARD  -  {state.game_id}")
    out.append("=" * 104)
    for line in context_lines(state, diag):
        out.append("  " + line)
    out.append("-" * 104)
    out.append(
        f"  {'#':>2}  {'PROP':<44} {'HIT%':>6} {'':12} {'ODDS':>6}"
        f" {'FAIR':>6} {'EDGE':>7} {'EV':>7}  CONF"
    )
    out.append("-" * 104)

    rows = list(ranked)[: limit or len(ranked)]
    for i, r in enumerate(rows, 1):
        ev = r.evaluation
        pr = r.pricing
        label = ev.prop.label
        if ev.prop.scope is PropScope.SECOND_HALF:
            label += " [H2]"
        if len(label) > 44:
            label = label[:41] + "..."

        out.append(
            f"  {i:>2}  {label:<44} {ev.p_win:>5.1%} {_bar(ev.p_win)}"
            f" {_odds_str(pr.odds):>6} {pr.fair_prob:>5.1%}"
            f" {pr.edge:>+6.1%} {pr.ev_per_unit:>+6.2f}  {r.confidence}"
        )
        if ev.p_push > 0.005:
            out.append(f"      push {ev.p_push:.1%} (whole-number line: stake returned)")
        if verbose:
            for d in r.drivers:
                out.append(f"        - {d}")
            out.append("")

    out.append("-" * 104)
    out.append(
        f"  {len(rows)} props  |  {n_sims:,} simulated second halves  |  "
        f"MC error at 50%: +/-{50 * (1 / n_sims) ** 0.5:.2f}pp"
    )
    out.append(
        "  Sorted by hit probability. Highest-probability props are usually "
        "short-priced favourites,"
    )
    out.append(
        "  not the best bets - read the EDGE column for where the model "
        "disagrees with the book."
    )
    out.append("=" * 104)
    return "\n".join(out)


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def to_json(
    ranked: Sequence[RankedProp],
    state: GameState,
    diag: GameDiagnostics,
    n_sims: int,
) -> str:
    payload = {
        "game_id": state.game_id,
        "generated_from": {
            "away": {"abbr": state.away.abbr, "score": state.away.score},
            "home": {"abbr": state.home.abbr, "score": state.home.score},
            "quarter": state.quarter,
            "seconds_remaining": state.seconds_remaining,
            "possession": state.possession,
        },
        "n_sims": n_sims,
        "projection": {
            "points": diag.proj_points,
            "win_prob": diag.win_prob,
            "pass_rate": diag.pass_rate,
            "blowout_prob": diag.blowout_prob,
        },
        "props": [
            {
                "rank": i,
                "prop_id": r.prop.prop_id,
                "label": r.prop.label,
                "player_id": r.prop.player_id,
                "team": r.prop.team,
                "market": r.prop.market.value,
                "side": r.prop.side.value,
                "line": r.prop.line,
                "scope": r.prop.scope.value,
                "odds": r.pricing.odds,
                "probability": round(r.evaluation.p_win, 4),
                "push_probability": round(r.evaluation.p_push, 4),
                "fair_probability": round(r.pricing.fair_prob, 4),
                "edge": round(r.pricing.edge, 4),
                "ev_per_unit": round(r.pricing.ev_per_unit, 4),
                "kelly_fraction": round(r.pricing.kelly_fraction, 4),
                "devig_exact": r.pricing.devig_exact,
                "projection": {
                    "mean": round(r.evaluation.mean, 2),
                    "median": round(r.evaluation.median, 2),
                    "p10": round(r.evaluation.p10, 2),
                    "p90": round(r.evaluation.p90, 2),
                },
                "stderr": round(r.stderr, 4),
                "confidence": r.confidence,
                "drivers": r.drivers,
            }
            for i, r in enumerate(ranked, 1)
        ],
    }
    return json.dumps(payload, indent=2)




# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Saira+Condensed:wght@500;600;700&"
    "family=Source+Sans+3:ital,wght@0,400;0,600;1,400&"
    'family=IBM+Plex+Mono:wght@400;500;600&display=swap">'
)

_CSS = """
:root{
  --ground:#f5f7fa; --card:#ffffff; --ink:#151a22; --muted:#68718400;
  --muted:#687184; --faint:#8b93a3; --rule:#e2e6ed; --rule-soft:#eef1f6;
  --accent:#2d4b8e; --accent-soft:#dbe3f4; --accent-ink:#20356a;
  --pos:#12734a; --pos-soft:#d9efe3; --neg:#a82f27; --neg-soft:#f7e0de;
  --track:#e8ebf1; --tick:#151a22;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0f131b; --card:#171d28; --ink:#e6e9ef; --muted:#98a1b3;
  --faint:#798296; --rule:#28303e; --rule-soft:#1f2632;
  --accent:#7fa3e8; --accent-soft:#22304b; --accent-ink:#a9c3f2;
  --pos:#4fc98a; --pos-soft:#16352a; --neg:#f0817a; --neg-soft:#3a1f1d;
  --track:#232b38; --tick:#e6e9ef;
}}
:root[data-theme="dark"]{
  --ground:#0f131b; --card:#171d28; --ink:#e6e9ef; --muted:#98a1b3;
  --faint:#798296; --rule:#28303e; --rule-soft:#1f2632;
  --accent:#7fa3e8; --accent-soft:#22304b; --accent-ink:#a9c3f2;
  --pos:#4fc98a; --pos-soft:#16352a; --neg:#f0817a; --neg-soft:#3a1f1d;
  --track:#232b38; --tick:#e6e9ef;
}

*{box-sizing:border-box}
body{
  margin:0; padding:0 0 72px; background:var(--ground); color:var(--ink);
  font-family:"Source Sans 3",ui-sans-serif,system-ui,-apple-system,sans-serif;
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px; margin:0 auto; padding:0 20px}

/* ---- scoreboard bug ------------------------------------------------- */
.board{
  background:var(--card); border:1px solid var(--rule); border-radius:4px;
  margin:26px 0 18px; overflow:hidden;
}
.board-top{
  display:flex; align-items:stretch; flex-wrap:wrap;
  border-bottom:1px solid var(--rule);
}
.eyebrow{
  font-family:"Saira Condensed",ui-sans-serif,sans-serif; font-weight:600;
  font-size:11px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--faint);
}
.score{
  display:flex; align-items:baseline; gap:14px; padding:16px 22px;
  border-right:1px solid var(--rule); flex:0 0 auto;
}
.score .abbr{
  font-family:"Saira Condensed",ui-sans-serif,sans-serif; font-weight:700;
  font-size:26px; letter-spacing:.04em; line-height:1;
}
.score .pts{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-weight:600;
  font-size:26px; line-height:1; font-variant-numeric:tabular-nums;
}
.score .at{color:var(--faint); font-size:15px; padding:0 2px}
.score.has-ball .abbr{color:var(--accent)}
.ballmark{
  font-family:"Saira Condensed",sans-serif; font-size:10px; font-weight:600;
  letter-spacing:.1em; color:var(--accent); background:var(--accent-soft);
  padding:2px 6px; border-radius:2px; align-self:center;
}
.clock{display:flex; flex-direction:column; justify-content:center;
  padding:14px 22px; gap:2px}
.clock .t{
  font-family:"IBM Plex Mono",monospace; font-size:19px; font-weight:600;
  font-variant-numeric:tabular-nums; line-height:1.15;
}

/* ---- context tiles --------------------------------------------------- */
.tiles{display:grid; grid-template-columns:repeat(auto-fit,minmax(184px,1fr))}
.tile{padding:13px 22px; border-right:1px solid var(--rule-soft);
  border-top:1px solid var(--rule-soft); display:flex; flex-direction:column; gap:3px}
.tile:last-child{border-right:none}
.tile .v{
  font-family:"IBM Plex Mono",monospace; font-size:16px; font-weight:500;
  font-variant-numeric:tabular-nums; line-height:1.25;
}
.tile .note{font-size:12.5px; color:var(--muted); line-height:1.35}

/* ---- table ----------------------------------------------------------- */
.scroller{overflow-x:auto; background:var(--card);
  border:1px solid var(--rule); border-radius:4px}
table{border-collapse:collapse; width:100%; min-width:940px}
thead th{
  font-family:"Saira Condensed",ui-sans-serif,sans-serif; font-weight:600;
  font-size:11px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--faint); text-align:left; padding:11px 12px;
  border-bottom:1px solid var(--rule); white-space:nowrap; vertical-align:bottom;
}
tbody td{padding:13px 12px; border-bottom:1px solid var(--rule-soft);
  vertical-align:top}
tbody tr:last-child td{border-bottom:none}
.num{text-align:right; font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums; white-space:nowrap}
.rk{width:38px; text-align:right; color:var(--faint);
  font-family:"IBM Plex Mono",monospace; font-size:13px; padding-top:15px}
.name{font-weight:600; letter-spacing:-.005em}
.tag{font-family:"Saira Condensed",sans-serif; font-size:10.5px;
  letter-spacing:.09em; text-transform:uppercase; color:var(--accent);
  background:var(--accent-soft); padding:1px 5px; border-radius:2px;
  margin-left:6px; vertical-align:1px}

/* probability */
.pcell{width:112px}
.pval{font-size:16px; font-weight:600}
.ptrack{height:5px; background:var(--track); border-radius:3px;
  overflow:hidden; margin-top:6px}
.pfill{height:100%; background:var(--accent); border-radius:3px}

/* distribution strip: the model's actual claim, drawn */
.dist{width:170px; padding-top:4px}
.strip{position:relative; height:22px; margin-top:3px}
.strip .axis{position:absolute; left:0; right:0; top:10px; height:2px;
  background:var(--track); border-radius:2px}
.strip .band{position:absolute; top:7px; height:8px; background:var(--accent-soft);
  border-radius:2px}
.strip .med{position:absolute; top:5px; width:2px; height:12px;
  background:var(--accent); border-radius:1px}
.strip .line{position:absolute; top:0; width:2px; height:22px;
  background:var(--tick)}
.strip .line:after{content:""; position:absolute; left:-2px; top:0;
  border-left:3px solid transparent; border-right:3px solid transparent;
  border-top:4px solid var(--tick)}
.range{font-family:"IBM Plex Mono",monospace; font-size:11.5px;
  color:var(--muted); font-variant-numeric:tabular-nums; margin-top:1px}

/* chips */
.chip{display:inline-block; font-family:"IBM Plex Mono",monospace;
  font-size:12.5px; font-weight:500; padding:2px 7px; border-radius:3px;
  font-variant-numeric:tabular-nums}
.chip.up{color:var(--pos); background:var(--pos-soft)}
.chip.down{color:var(--neg); background:var(--neg-soft)}
.conf{font-family:"Saira Condensed",sans-serif; font-size:10.5px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
  border:1px solid var(--rule); padding:2px 7px; border-radius:2px;
  white-space:nowrap}

/* drivers */
.why{margin:8px 0 0; padding:0; list-style:none; display:flex;
  flex-direction:column; gap:2px; max-width:62ch}
.why li{font-size:12.8px; color:var(--muted); line-height:1.45;
  padding-left:13px; position:relative}
.why li:before{content:""; position:absolute; left:0; top:8px; width:5px;
  height:1px; background:var(--faint)}
.why b{color:var(--ink); font-weight:600}

/* prose */
h1{font-family:"Saira Condensed",ui-sans-serif,sans-serif; font-weight:700;
  font-size:31px; letter-spacing:.005em; margin:30px 0 2px; text-wrap:balance}
.deck{color:var(--muted); font-size:14.5px; margin:0 0 4px; max-width:70ch}
.section-label{margin:30px 0 10px}
.legend{display:flex; flex-wrap:wrap; gap:18px; margin:12px 2px 0;
  font-size:12.5px; color:var(--muted)}
.legend span{display:flex; align-items:center; gap:6px}
.sw{width:22px; height:8px; border-radius:2px; display:inline-block}
.sw.band{background:var(--accent-soft)}
.sw.med{background:var(--accent); width:2px; height:12px}
.sw.line{background:var(--tick); width:2px; height:12px}
.notes{margin-top:24px; display:grid; gap:14px;
  grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.note-card{border-left:2px solid var(--rule); padding:2px 0 2px 14px;
  font-size:13.2px; color:var(--muted); line-height:1.55; max-width:65ch}
.note-card b{color:var(--ink); font-weight:600}
.settled{margin-top:14px; font-size:13px; color:var(--muted)}
.settled code{font-family:"IBM Plex Mono",monospace; font-size:12.2px;
  background:var(--rule-soft); padding:1px 5px; border-radius:2px}
footer{margin-top:30px; padding-top:14px; border-top:1px solid var(--rule);
  font-size:12.5px; color:var(--faint)}
"""


def _strip_html(ev, line: float) -> str:
    """Draw the 80% range against the book's line.

    This is the model's actual claim in one glance: not "he projects for 118"
    but "the middle of his distribution sits here, the line sits there, and
    the spread is this wide". A prop that clears only via one explosive play
    has a visibly wider band than one that gets there on volume, even when
    both project the same number.
    """
    lo, hi = float(ev.p10), float(ev.p90)
    med = float(ev.median)
    left = min(lo, line, med)
    right = max(hi, line, med)
    span = right - left
    if span <= 0:
        span = 1.0
    pad = span * 0.12
    left -= pad
    right += pad
    span = right - left

    def pos(v: float) -> float:
        return max(0.0, min(100.0, (v - left) / span * 100.0))

    band_l, band_r = pos(lo), pos(hi)
    return (
        f'<div class="strip"><div class="axis"></div>'
        f'<div class="band" style="left:{band_l:.1f}%;width:{max(band_r - band_l, 1.2):.1f}%"></div>'
        f'<div class="med" style="left:{pos(med):.1f}%"></div>'
        f'<div class="line" style="left:{pos(line):.1f}%"></div></div>'
        f'<div class="range">{lo:.0f} – {hi:.0f}</div>'
    )


#: Drivers the HTML board already encodes elsewhere. "Line" is exactly what
#: the range strip draws, and "Weather" is a game-level fact sitting in the
#: conditions tile - repeating either on all twenty rows is noise that pushes
#: the rows that differ off the screen. The terminal board, which has no strip
#: and no tiles, keeps both.
_REDUNDANT_IN_HTML = ("Line:", "Weather:")


def _driver_html(drivers, escape) -> str:
    """Bold the driver's category so the column scans vertically."""
    out = []
    for d in drivers:
        if d.startswith(_REDUNDANT_IN_HTML):
            continue
        label, sep, rest = d.partition(": ")
        if sep:
            out.append(f"<li><b>{escape(label)}</b> {escape(rest)}</li>")
        else:
            out.append(f"<li>{escape(d)}</li>")
    return "".join(out)


def render_html(
    ranked: Sequence[RankedProp],
    state: GameState,
    diag: GameDiagnostics,
    n_sims: int,
    title: Optional[str] = None,
    settled: Optional[Sequence[RankedProp]] = None,
) -> str:
    e = html.escape
    away, home = state.away, state.home
    name = title or f"{away.abbr}–{home.abbr} Halftime Board"

    mins, secs = divmod(diag.seconds_remaining, 60)

    def score_block(team, ball: bool) -> str:
        cls = "score has-ball" if ball else "score"
        mark = '<span class="ballmark">BALL</span>' if ball else ""
        return (
            f'<div class="{cls}"><span class="abbr">{e(team.abbr)}</span>'
            f'<span class="pts">{team.score}</span>{mark}</div>'
        )

    w = state.weather
    if w.dome:
        conditions = "Indoors"
    else:
        bits = [f"{w.temp_f:.0f}&deg;F", f"wind {w.wind_mph:.0f} mph"]
        if w.precip.value != "none":
            bits.append(w.precip.value.replace("_", " "))
        conditions = ", ".join(bits)

    def lean(team) -> str:
        got = diag.pass_rate[team.abbr]
        delta = got - team.base_pass_rate
        word = "pass-leaning" if delta > 0 else "run-leaning"
        return (
            f'<div class="tile"><span class="eyebrow">{e(team.abbr)} script</span>'
            f'<span class="v">{got:.0%} pass</span>'
            f'<span class="note">{word} vs {team.base_pass_rate:.0%} neutral</span></div>'
        )

    tiles = (
        f'<div class="tile"><span class="eyebrow">Projected final</span>'
        f'<span class="v">{diag.proj_points[away.abbr]:.1f} – {diag.proj_points[home.abbr]:.1f}</span>'
        f'<span class="note">{e(home.abbr)} win {diag.win_prob[home.abbr]:.0%}</span></div>'
        + lean(away) + lean(home)
        + f'<div class="tile"><span class="eyebrow">Blowout risk</span>'
        f'<span class="v">{diag.blowout_prob:.0%}</span>'
        f'<span class="note">17+ margin; drives star rest risk</span></div>'
        f'<div class="tile"><span class="eyebrow">Conditions</span>'
        f'<span class="v">{conditions}</span>'
        f'<span class="note">~{diag.away_plays:.0f} / {diag.home_plays:.0f} H2 plays</span></div>'
    )

    rows = []
    for i, r in enumerate(ranked, 1):
        ev, pr = r.evaluation, r.pricing
        binary = ev.prop.market in BINARY_MARKETS
        tag = '<span class="tag">2H</span>' if ev.prop.scope is PropScope.SECOND_HALF else ""
        edge_cls = "up" if pr.edge > 0 else "down"
        ev_cls = "up" if pr.ev_per_unit > 0 else "down"
        dist = "" if binary else _strip_html(ev, ev.prop.line)
        push = (
            f'<div class="range">push {ev.p_push:.0%}</div>' if ev.p_push > 0.005 else ""
        )
        rows.append(
            f"<tr><td class='rk'>{i}</td>"
            f"<td><div class='name'>{e(ev.prop.label)}{tag}</div>"
            f"<ul class='why'>{_driver_html(r.drivers, e)}</ul></td>"
            f"<td class='num pcell'><div class='pval'>{ev.p_win:.1%}</div>"
            f"<div class='ptrack'><div class='pfill' style='width:{ev.p_win * 100:.1f}%'></div></div>"
            f"{push}</td>"
            f"<td class='dist'>{dist}</td>"
            f"<td class='num'>{e(_odds_str(pr.odds))}</td>"
            f"<td class='num'>{pr.fair_prob:.1%}</td>"
            f"<td class='num'><span class='chip {edge_cls}'>{pr.edge:+.1%}</span></td>"
            f"<td class='num'><span class='chip {ev_cls}'>{pr.ev_per_unit:+.2f}</span></td>"
            f"<td><span class='conf'>{e(r.confidence)}</span></td></tr>"
        )

    settled_html = ""
    if settled:
        items = ", ".join(
            f"{e(s.prop.label)} <code>already {e(s.evaluation.settled or '')}</code>"
            for s in settled
        )
        settled_html = (
            f'<p class="settled"><b>Held out as already decided:</b> {items}. '
            f"First-half production alone settles these, so ranking them at "
            f"100% against a stale price would be noise.</p>"
        )

    mc = 50 * (1 / n_sims) ** 0.5

    return f"""<title>{e(name)}</title>
{_FONTS}
<style>{_CSS}</style>
<div class="wrap">
<h1>{e(away.abbr)} at {e(home.abbr)} — halftime prop board</h1>
<p class="deck">Every offered line, ranked by how often it cashed across
{n_sims:,} simulated second halves.</p>

<div class="board">
  <div class="board-top">
    {score_block(away, state.possession == away.abbr)}
    {score_block(home, state.possession == home.abbr)}
    <div class="clock">
      <span class="eyebrow">Q{state.quarter} &middot; remaining</span>
      <span class="t">{mins}:{secs:02d}</span>
    </div>
  </div>
  <div class="tiles">{tiles}</div>
</div>

<div class="scroller"><table>
<thead><tr>
  <th></th><th>Prop &amp; what moved it</th><th class="num">Hit</th>
  <th>Range vs line</th><th class="num">Odds</th><th class="num">Fair</th>
  <th class="num">Edge</th><th class="num">EV</th><th>Conf</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table></div>

<div class="legend">
  <span><i class="sw band"></i> 80% of outcomes (10th–90th)</span>
  <span><i class="sw med"></i> median</span>
  <span><i class="sw line"></i> the book's line</span>
</div>

{settled_html}

<div class="notes">
  <p class="note-card"><b>Likely is not the same as profitable.</b> This is
  sorted by hit probability, so the top rows are short-priced favourites. The
  <b>Edge</b> column &mdash; model probability minus the book's vig-free
  probability &mdash; is where the model claims the number is actually wrong.</p>
  <p class="note-card"><b>Read the range, not just the median.</b> A wide band
  means the prop depends on one explosive play; a narrow one means it gets
  there on accumulated volume. Two props with the same median and different
  bands are different bets.</p>
  <p class="note-card"><b>How much to trust it.</b> Simulation error is about
  &plusmn;{mc:.2f} points at a 50% probability. Model priors are league-average
  defaults rather than parameters fitted to a proprietary database, so treat
  edges under roughly 3 points as noise.</p>
</div>

<footer>Generated by nflprops &middot; analysis only, not advice &middot;
if gambling stops being fun, call or text 1-800-GAMBLER</footer>
</div>"""
