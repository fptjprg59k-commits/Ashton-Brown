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

from .models import GameState, PropScope
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


_CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1a1a18;--muted:#6b6b66;--line:#e2e2dd;
--pos:#1a7f4b;--neg:#b4342a;--accent:#2f5d8f;--bar:#c9d6e6;--barfill:#2f5d8f;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#16161a;--panel:#1e1e23;--ink:#ececea;--muted:#9a9a94;--line:#33333a;
--pos:#5fc48a;--neg:#e8776c;--accent:#8fb4dd;--bar:#33333a;--barfill:#8fb4dd;}}
:root[data-theme=dark]{--bg:#16161a;--panel:#1e1e23;--ink:#ececea;--muted:#9a9a94;
--line:#33333a;--pos:#5fc48a;--neg:#e8776c;--accent:#8fb4dd;--bar:#33333a;--barfill:#8fb4dd;}
*{box-sizing:border-box}
body{margin:0;padding:28px 20px 60px;background:var(--bg);color:var(--ink);
font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1120px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px;margin-bottom:20px}
.ctx{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:14px 16px;margin-bottom:20px;font-size:13.5px}
.ctx div{padding:3px 0}
.tablewrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);
border-radius:10px}
table{border-collapse:collapse;width:100%;min-width:900px;font-size:13.5px}
th{text-align:left;font-weight:600;color:var(--muted);font-size:11px;
text-transform:uppercase;letter-spacing:.06em;padding:12px 10px;
border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:none}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.rank{color:var(--muted);text-align:right;width:34px}
.prop{font-weight:600}
.scope{color:var(--muted);font-weight:400;font-size:11.5px}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.barwrap{background:var(--bar);border-radius:3px;height:7px;width:90px;
overflow:hidden;margin-top:5px}
.barfill{background:var(--barfill);height:100%}
.drivers{margin:7px 0 0;padding:0;list-style:none;color:var(--muted);font-size:12.5px}
.drivers li{padding:1.5px 0 1.5px 12px;text-indent:-12px}
.drivers li:before{content:"– "}
.conf{font-size:11px;padding:2px 7px;border-radius:99px;border:1px solid var(--line);
color:var(--muted);white-space:nowrap}
.note{margin-top:18px;color:var(--muted);font-size:12.5px;line-height:1.6}
"""


def render_html(
    ranked: Sequence[RankedProp],
    state: GameState,
    diag: GameDiagnostics,
    n_sims: int,
    title: Optional[str] = None,
) -> str:
    e = html.escape
    name = title or f"{state.away.abbr} @ {state.home.abbr} Halftime Board"

    ctx = "".join(f"<div>{e(line)}</div>" for line in context_lines(state, diag))

    rows = []
    for i, r in enumerate(ranked, 1):
        ev, pr = r.evaluation, r.pricing
        scope = " <span class='scope'>[2nd half]</span>" if ev.prop.scope is PropScope.SECOND_HALF else ""
        push = (
            f"<div class='scope'>push {ev.p_push:.1%}</div>" if ev.p_push > 0.005 else ""
        )
        drivers = "".join(f"<li>{e(d)}</li>" for d in r.drivers)
        edge_cls = "pos" if pr.edge > 0 else "neg"
        ev_cls = "pos" if pr.ev_per_unit > 0 else "neg"
        rows.append(
            f"<tr><td class='rank'>{i}</td>"
            f"<td><div class='prop'>{e(ev.prop.label)}{scope}</div>"
            f"<ul class='drivers'>{drivers}</ul></td>"
            f"<td class='num'><strong>{ev.p_win:.1%}</strong>{push}"
            f"<div class='barwrap'><div class='barfill' style='width:{ev.p_win * 100:.1f}%'></div></div></td>"
            f"<td class='num'>{e(_odds_str(pr.odds))}</td>"
            f"<td class='num'>{pr.fair_prob:.1%}</td>"
            f"<td class='num {edge_cls}'>{pr.edge:+.1%}</td>"
            f"<td class='num {ev_cls}'>{pr.ev_per_unit:+.2f}</td>"
            f"<td class='num'>{ev.median:.0f}<div class='scope'>{ev.p10:.0f}–{ev.p90:.0f}</div></td>"
            f"<td><span class='conf'>{e(r.confidence)}</span></td></tr>"
        )

    return f"""<title>{e(name)}</title>
<style>{_CSS}</style>
<div class="wrap">
<h1>{e(name)}</h1>
<div class="sub">{n_sims:,} simulated second halves &middot; sorted by hit probability</div>
<div class="ctx">{ctx}</div>
<div class="tablewrap"><table>
<thead><tr><th></th><th>Prop</th><th class="num">Hit %</th><th class="num">Odds</th>
<th class="num">Fair</th><th class="num">Edge</th><th class="num">EV</th>
<th class="num">Median<br>80% range</th><th>Conf</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table></div>
<p class="note"><strong>Reading this board.</strong> It is sorted by hit
probability, so the top rows are the most <em>likely</em> props &mdash; which
are usually short-priced favourites, not the best bets. <strong>Edge</strong>
is model probability minus the book's vig-free probability; that column, not
the hit rate, is where the model claims the number is wrong. Median and the
80% range show the shape of the projection: a wide range means the prop
depends on one explosive play rather than accumulated volume.</p>
<p class="note">Probabilities carry Monte Carlo error of about
&plusmn;{50 * (1 / n_sims) ** 0.5:.2f} percentage points at 50%. Model priors
are league-average defaults, not fitted to a proprietary database &mdash; treat
edges under about 3 points as noise.</p>
</div>"""
