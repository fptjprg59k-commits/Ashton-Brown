"""Rendering for the matchup projection - terminal and standalone HTML.

The page leads with the stat line and keeps the reasoning behind it, in that
order. Projected score, then how the game gets played, then team totals, then
the quarterbacks, then every listed player, and only then the leverage that
produced all of it. Nothing here computes anything: every figure comes from
:mod:`nflprops.projection`, so the two surfaces can never disagree.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

from .matchup import Matchup, Metric, Team
from .projection import GameProjection, SkillLine, TeamProjection, game_script, ordinal, project
from .theme import BASE_CSS, FONT_BODY, FONT_DISPLAY, FONT_LINKS, FONT_MONO, TOKENS_CSS


def _e(x: Any) -> str:
    return html.escape(str(x if x is not None else ""))


def _pos(rank: float) -> float:
    """Rank 1..32 -> 0..100% along the leverage rail."""
    return max(0.0, min(100.0, (float(rank) - 1.0) / 31.0 * 100.0))


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

CSS = f"""
{TOKENS_CSS}
{BASE_CSS}

body{{padding-block:0 64px}}
.wrap{{max-width:1000px}}

h1,h2{{font-family:{FONT_DISPLAY}; font-weight:700; margin:0; text-wrap:balance}}
h1{{font-size:clamp(26px,4.4vw,38px); line-height:1.04; letter-spacing:-.012em}}
h2{{font-size:17px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted)}}
p{{margin:0}}
a{{color:var(--accent)}}

section{{margin-top:38px}}
.sec-head{{display:flex; flex-wrap:wrap; align-items:baseline; gap:8px 14px;
  padding-bottom:9px; border-bottom:1px solid var(--rule); margin-bottom:18px}}
.sec-head .q{{font-size:13px; color:var(--faint)}}

/* ---- masthead ---- */
.mast{{padding-block:26px 0}}
.kick{{font-family:{FONT_DISPLAY}; font-size:11px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--faint); display:flex; flex-wrap:wrap;
  gap:6px 14px; margin-bottom:10px}}
.kick b{{color:var(--ink); font-weight:600}}
.chips{{margin-top:14px; display:flex; flex-wrap:wrap; gap:7px}}
.chip{{font-family:{FONT_MONO}; font-size:11.5px; padding:4px 9px; border-radius:2px;
  background:var(--raised); border:1px solid var(--rule); color:var(--ink)}}
.chip b{{color:var(--faint); font-weight:500; letter-spacing:.04em}}

/* ---- scoreboard ---- */
.score{{margin-top:24px; background:var(--card); border:1px solid var(--rule);
  border-radius:4px; overflow:hidden}}
.score .row{{display:grid; grid-template-columns:1fr auto; align-items:center;
  gap:12px; padding:16px 20px; border-left:6px solid var(--tc)}}
.score .row + .row{{border-top:1px solid var(--rule-soft)}}
.score .nm{{font-family:{FONT_DISPLAY}; font-size:clamp(19px,3vw,26px); font-weight:700;
  line-height:1.08}}
.score .meta{{font-family:{FONT_MONO}; font-size:11.5px; color:var(--muted); margin-top:3px}}
.score .pts{{font-family:{FONT_DISPLAY}; font-size:clamp(38px,7vw,58px); font-weight:700;
  line-height:.9; font-variant-numeric:tabular-nums; letter-spacing:-.02em}}
.score .foot{{display:flex; flex-wrap:wrap; gap:6px 20px; padding:11px 20px;
  background:var(--raised); border-top:1px solid var(--rule);
  font-family:{FONT_MONO}; font-size:11.5px; color:var(--muted)}}
.score .foot b{{color:var(--ink); font-weight:600}}

/* ---- game script ---- */
.script{{background:var(--card); border:1px solid var(--rule);
  border-top:3px solid var(--accent); border-radius:3px; padding:20px 22px;
  font-size:16px; line-height:1.62; max-width:74ch}}

/* ---- split bars ---- */
.splits{{display:flex; flex-wrap:wrap; gap:14px; margin-top:18px}}
.sp{{flex:1 1 300px; min-width:0}}
.sp .hd{{display:flex; justify-content:space-between; align-items:baseline;
  font-family:{FONT_DISPLAY}; font-size:12px; letter-spacing:.09em;
  text-transform:uppercase; color:var(--muted); margin-bottom:6px}}
.sp .hd b{{color:var(--ink); font-size:13px}}
.bar2{{height:26px; display:flex; border-radius:2px; overflow:hidden;
  border:1px solid var(--rule)}}
.bar2 span{{display:flex; align-items:center; justify-content:center;
  font-family:{FONT_MONO}; font-size:11.5px; font-weight:600; min-width:0;
  white-space:nowrap; overflow:hidden}}
.bar2 .a{{background:var(--tc); color:#fff}}
.bar2 .b{{background:var(--track); color:var(--ink)}}

/* ---- tables ---- */
.tw{{overflow-x:auto; -webkit-overflow-scrolling:touch}}
table{{border-collapse:collapse; width:100%; font-size:13.5px; min-width:560px}}
th,td{{text-align:left; padding:8px 9px; border-bottom:1px solid var(--rule-soft)}}
th{{font-family:{FONT_DISPLAY}; font-size:10px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--faint); border-bottom:1px solid var(--rule);
  white-space:nowrap}}
td.n,th.n{{text-align:right; font-family:{FONT_MONO}; font-variant-numeric:tabular-nums;
  white-space:nowrap}}
td.n b{{font-weight:600}}
tbody tr:hover{{background:var(--raised)}}
.pl{{font-weight:600; white-space:nowrap}}
.pp{{font-family:{FONT_MONO}; font-size:11px; color:var(--faint); margin-left:5px}}
.tdcell{{min-width:92px}}
.meter{{display:flex; align-items:center; gap:7px; justify-content:flex-end}}
.meter .t{{position:relative; width:46px; height:6px; background:var(--track);
  border-radius:3px; overflow:hidden; flex:none}}
.meter .f{{position:absolute; inset:0 auto 0 0; background:var(--tc); border-radius:3px}}
.meter .v{{font-family:{FONT_MONO}; font-variant-numeric:tabular-nums; font-size:12.5px;
  min-width:34px; text-align:right}}

.qb{{display:flex; flex-wrap:wrap; gap:14px}}
.qbc{{flex:1 1 400px; min-width:0; background:var(--card); border:1px solid var(--rule);
  border-radius:3px; border-top:3px solid var(--tc); padding:16px 18px}}
.qbc .nm{{font-family:{FONT_DISPLAY}; font-size:20px; font-weight:700; line-height:1.1}}
.qbc .ln{{font-family:{FONT_MONO}; font-size:15px; margin-top:6px; color:var(--ink)}}
.qbc dl{{margin:14px 0 0; display:grid; grid-template-columns:repeat(auto-fit,minmax(84px,1fr));
  gap:11px 8px}}
.qbc dt{{font-family:{FONT_DISPLAY}; font-size:9.5px; letter-spacing:.11em;
  text-transform:uppercase; color:var(--faint)}}
.qbc dd{{margin:2px 0 0; font-family:{FONT_MONO}; font-size:16px; font-weight:600;
  font-variant-numeric:tabular-nums}}
.qbc .src{{margin-top:13px; padding-top:10px; border-top:1px solid var(--rule-soft);
  font-size:12px; color:var(--muted); line-height:1.5}}

.stackcols{{display:flex; flex-direction:column; gap:26px}}
.stackcols table{{min-width:640px}}
.stackcols th:first-child{{color:var(--tc); font-size:12px; letter-spacing:.08em}}
.stackcols thead tr{{border-bottom:2px solid var(--tc)}}
.cols{{display:flex; flex-wrap:wrap; gap:24px}}
.col{{flex:1 1 430px; min-width:0}}
.col > h3{{font-family:{FONT_DISPLAY}; font-size:13px; letter-spacing:.1em;
  text-transform:uppercase; margin:0 0 8px; padding-bottom:6px;
  border-bottom:2px solid var(--tc)}}

/* ---- leverage rails ---- */
.lrow{{display:grid; grid-template-columns:minmax(0,150px) 1fr 62px; gap:6px 16px;
  align-items:center; padding:12px 0; border-top:1px solid var(--rule-soft)}}
.lrow:first-child{{border-top:none}}
.lrow .ax{{font-family:{FONT_DISPLAY}; font-size:14.5px; font-weight:600; line-height:1.15}}
.lrow .at{{font-family:{FONT_MONO}; font-size:10.5px; color:var(--tc); font-weight:600;
  margin-top:2px}}
.rail{{position:relative; height:30px; min-width:0}}
.rail .track{{position:absolute; left:0; right:0; top:13px; height:4px;
  background:var(--track); border-radius:2px}}
.rail .span{{position:absolute; top:11px; height:8px; border-radius:2px; background:var(--tc)}}
.rail .mk{{position:absolute; top:5px; width:2px; height:20px; background:var(--ink)}}
.rail .mk.off{{background:var(--tc); width:3px}}
.rail .lab{{position:absolute; font-family:{FONT_MONO}; font-size:10px;
  color:var(--muted); white-space:nowrap}}
.rail .lab.hi{{top:-2px; color:var(--ink); font-weight:600}}
.rail .lab.lo{{top:21px}}
.lrow .sc{{text-align:right; font-family:{FONT_MONO}; font-size:16px; font-weight:600}}

.caveat{{margin-top:18px; background:var(--st-warn-bg); border:1px solid var(--st-warn-edge);
  border-radius:3px; padding:13px 16px; font-size:13.5px; line-height:1.55;
  color:var(--ink); max-width:80ch}}
.caveat b{{color:var(--st-warn-fg)}}

.inj{{display:flex; flex-wrap:wrap; gap:8px}}
.ic{{flex:1 1 230px; min-width:0; background:var(--card); border:1px solid var(--rule);
  border-left:4px solid var(--sc); border-radius:3px; padding:10px 12px}}
.ic .p{{font-family:{FONT_DISPLAY}; font-size:14.5px; font-weight:700}}
.ic .m{{font-family:{FONT_MONO}; font-size:11px; color:var(--muted); margin-top:2px}}
.ic .st{{font-family:{FONT_DISPLAY}; font-size:9px; letter-spacing:.12em;
  text-transform:uppercase; font-weight:600; color:var(--sc)}}

footer{{margin-top:50px; padding-top:16px; border-top:1px solid var(--rule);
  color:var(--faint); font-size:12px; line-height:1.6; max-width:84ch}}
footer b{{color:var(--muted)}}

@media (max-width:700px){{
  .lrow{{grid-template-columns:1fr 56px}}
  .rail{{grid-column:1/-1}}
  table{{min-width:520px}}
}}
"""


# --------------------------------------------------------------------------
# Fragments
# --------------------------------------------------------------------------


def _masthead(g: GameProjection) -> str:
    m = g.matchup
    meta, mk = m.meta, (m.meta.get("market") or {})
    venue = meta.get("venue") or {}
    w = m.weather or {}
    chips = []
    if mk.get("favorite"):
        chips.append(f'<span class="chip"><b>LINE</b> {_e(mk["favorite"])} -{_e(mk.get("spread"))}</span>')
    if mk.get("total"):
        chips.append(f'<span class="chip"><b>TOTAL</b> {_e(mk.get("total"))}</span>')
    if w:
        chips.append(
            f'<span class="chip"><b>WX</b> {_e(w.get("temp_f"))}F, '
            f'{_e(w.get("wind_mph"))} mph, {float(w.get("precip_chance") or 0) * 100:.0f}% rain</span>'
        )
    for t in m.teams:
        rest = t.rest or {}
        tag = f'{rest.get("days")}d rest'
        if rest.get("short_week"):
            tag += ", short week"
        chips.append(f'<span class="chip"><b>{_e(t.abbr)}</b> {_e(tag)}</span>')

    return f"""<header class="mast">
<div class="kick"><span>{_e(meta.get('label'))}</span>
  <span><b>Week {_e(meta.get('week'))}</b>, {_e(meta.get('season'))}</span>
  <span>{_e(meta.get('kickoff_local'))}</span><span>{_e(meta.get('network'))}</span></div>
<h1>{_e(m.away.name)} at {_e(m.home.name)}</h1>
<div class="chips">{''.join(chips)}</div>
</header>"""


def _scoreboard(g: GameProjection) -> str:
    m = g.matchup
    rows = []
    for tp in g.teams:
        t = m.team(tp.team)
        rows.append(f"""<div class="row" style="--tc:{_e(t.color)}">
  <div><div class="nm">{_e(t.name)}</div>
    <div class="meta">{_e(t.record)} &middot; {tp.total_yards:.0f} yds &middot;
      {tp.pass_yards:.0f} pass / {tp.rush_yards:.0f} rush &middot;
      {tp.pass_rate:.0%} pass rate</div></div>
  <div class="pts">{tp.points:.0f}</div>
</div>""")
    mk = m.meta.get("market") or {}
    return f"""<div class="score">{''.join(rows)}
<div class="foot">
  <span>Projected total <b>{g.total:.1f}</b> (market {_e(mk.get('total'))})</span>
  <span>Projected margin <b>{abs(g.margin):.1f}</b> {_e(m.home.abbr if g.margin > 0 else m.away.abbr)}
    (line {_e(mk.get('favorite'))} -{_e(mk.get('spread'))})</span>
</div></div>"""


def _team_totals(g: GameProjection) -> str:
    m = g.matchup
    rows = []
    for tp in g.teams:
        t = m.team(tp.team)
        rows.append(f"""<tr>
  <td class="pl" style="border-left:4px solid {_e(t.color)}; padding-left:9px">{_e(t.name)}</td>
  <td class="n">{tp.plays:.0f}</td>
  <td class="n">{tp.pass_rate:.0%}</td>
  <td class="n">{tp.attempts:.0f}</td>
  <td class="n">{tp.completions:.0f}</td>
  <td class="n"><b>{tp.pass_yards:.0f}</b></td>
  <td class="n">{tp.carries:.0f}</td>
  <td class="n"><b>{tp.rush_yards:.0f}</b></td>
  <td class="n"><b>{tp.total_yards:.0f}</b></td>
  <td class="n">{tp.yards_per_play:.1f}</td>
  <td class="n">{tp.sacks:.1f}</td>
  <td class="n">{tp.interceptions:.1f}</td>
  <td class="n">{tp.expected_tds:.1f}</td>
</tr>""")

    splits = []
    for tp in g.teams:
        t = m.team(tp.team)
        py = tp.pass_yards / max(1.0, tp.total_yards) * 100
        splits.append(f"""<div class="sp" style="--tc:{_e(t.color)}">
  <div class="hd"><span>{_e(t.abbr)} yardage split</span><b>{tp.total_yards:.0f} total</b></div>
  <div class="bar2"><span class="a" style="width:{py:.1f}%">{tp.pass_yards:.0f} pass</span>
    <span class="b" style="width:{100 - py:.1f}%">{tp.rush_yards:.0f} rush</span></div>
</div>""")

    return f"""<div class="tw"><table>
<thead><tr><th>Team</th><th class="n">Plays</th><th class="n">Pass%</th>
<th class="n">Att</th><th class="n">Comp</th><th class="n">Pass yds</th>
<th class="n">Carries</th><th class="n">Rush yds</th><th class="n">Total</th>
<th class="n">Y/P</th><th class="n">Sacks</th><th class="n">INT</th><th class="n">xTD</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<div class="splits">{''.join(splits)}</div>"""


def _qb_cards(g: GameProjection) -> str:
    m = g.matchup
    cards = []
    for tp in g.teams:
        t = m.team(tp.team)
        q = tp.qb
        src = (t.qb_profile or {}).get("src", "")
        cards.append(f"""<div class="qbc" style="--tc:{_e(t.color)}">
  <div class="nm">{_e(q.name)}</div>
  <div class="ln">{q.completions:.0f}/{q.attempts:.0f} &middot; {q.pass_yards:.0f} yds &middot;
    {q.pass_tds:.1f} TD &middot; {q.interceptions:.1f} INT</div>
  <dl>
    <div><dt>Comp %</dt><dd>{q.completion_pct:.1%}</dd></div>
    <div><dt>Yds / att</dt><dd>{q.yards_per_attempt:.1f}</dd></div>
    <div><dt>Sacks taken</dt><dd>{q.sacks:.1f}</dd></div>
    <div><dt>Rush att</dt><dd>{q.rush_attempts:.0f}</dd></div>
    <div><dt>Rush yds</dt><dd>{q.rush_yards:.0f}</dd></div>
    <div><dt>Rush TD odds</dt><dd>{q.td_probability:.0%}</dd></div>
  </dl>
  {f'<div class="src">{_e(src)}</div>' if src else ''}
</div>""")
    return f'<div class="qb">{"".join(cards)}</div>'


def _player_table(g: GameProjection, tp: TeamProjection) -> str:
    t = g.matchup.team(tp.team)
    rows = []
    for s in tp.skill:
        pct = s.td_probability * 100
        rows.append(f"""<tr>
  <td><span class="pl">{_e(s.name)}</span><span class="pp">{_e(s.pos)}</span></td>
  <td class="n">{s.carries:.0f}</td>
  <td class="n">{s.rush_yards:.0f}</td>
  <td class="n">{s.targets:.1f}</td>
  <td class="n">{s.receptions:.1f}</td>
  <td class="n">{s.rec_yards:.0f}</td>
  <td class="n"><b>{s.total_yards:.0f}</b></td>
  <td class="n tdcell"><span class="meter"><span class="t">
    <span class="f" style="width:{min(100.0, pct):.0f}%"></span></span>
    <span class="v">{pct:.0f}%</span></span></td>
</tr>""")
    return f"""<div class="tw"><table>
<thead><tr><th>{_e(t.name)}</th><th class="n">Car</th><th class="n">Rush yds</th>
<th class="n">Tgt</th><th class="n">Rec</th><th class="n">Rec yds</th>
<th class="n">Total yds</th><th class="n">Any TD</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


def _rail(m: Matchup, l) -> str:
    att, dfn = m.team(l.attacker), m.team(l.defender)
    o, d = _pos(l.off_metric.adjusted), _pos(l.def_metric.adjusted)
    lo, hi = min(o, d), max(o, d)
    owner = att if l.edge >= 0 else dfn

    def anchor(x: float, nudge: float) -> str:
        x2 = max(0.0, min(100.0, x + nudge))
        if x2 < 12:
            return f"left:{x2:.1f}%; transform:translateX(0)"
        if x2 > 88:
            return f"left:{x2:.1f}%; transform:translateX(-100%)"
        return f"left:{x2:.1f}%; transform:translateX(-50%)"

    close = abs(o - d) < 9
    return f"""<div class="rail">
  <div class="track"></div>
  <div class="span" style="left:{lo:.1f}%; width:{max(0.6, hi - lo):.1f}%; background:{_e(owner.color)}"></div>
  <div class="mk off" style="left:{o:.1f}%; background:{_e(att.color)}"></div>
  <div class="mk" style="left:{d:.1f}%"></div>
  <div class="lab hi" style="{anchor(o, -4.0 if close else 0.0)}">{_e(l.attacker)} {ordinal(l.off_metric.adjusted)}</div>
  <div class="lab lo" style="{anchor(d, 4.0 if close else 0.0)}">{_e(l.defender)} {ordinal(l.def_metric.adjusted)}</div>
</div>"""


def _leverage(g: GameProjection, n: int = 5) -> str:
    m = g.matchup
    board = m.top_exploits(99)
    picked = board[:n - 1] + [board[-1]]
    rows = []
    for l in picked:
        owner = m.team(l.attacker) if l.edge >= 0 else m.team(l.defender)
        cap = (f"{l.attacker} offense" if l.edge >= 0 else f"{l.defender} defense")
        rows.append(f"""<div class="lrow" style="--tc:{_e(owner.color)}">
  <div><div class="ax">{_e(l.axis.label)}</div><div class="at">{_e(cap)}</div></div>
  {_rail(m, l)}
  <div class="sc">{l.score:+.1f}</div>
</div>""")

    caveat = ""
    q = m.open_questions(1)
    if q:
        t, unit, mm = q[0]
        caveat = (f'<div class="caveat"><b>Least settled number in the game.</b> '
                  f'{_e(t.abbr)}&rsquo;s {_e(mm.label).lower()} ranked {ordinal(mm.prior)} '
                  f'last season and {ordinal(mm.current)} this one. It is blended to '
                  f'{ordinal(mm.adjusted)}, and every projection above leans on that number. '
                  f'{_e(mm.detail)}</div>')
    return "".join(rows) + caveat


_STATUS_COLOR = {"out": "var(--neg)", "doubtful": "var(--neg)",
                 "questionable": "var(--st-warn-fg)", "limited": "var(--st-warn-fg)"}


def _injuries(m: Matchup) -> str:
    cards = []
    for t in m.teams:
        for i in t.injuries:
            cards.append(f"""<div class="ic" style="--sc:{_STATUS_COLOR.get(i.status, 'var(--muted)')}">
  <div class="p">{_e(i.player)}</div>
  <div class="m">{_e(t.abbr)} &middot; {_e(i.pos)} &middot; {_e(i.injury)}</div>
  <div class="st">{_e(i.status)}</div>
</div>""")
    return f'<div class="inj">{"".join(cards)}</div>'


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


def _default_title(m: Matchup) -> str:
    return f"{m.away.name.rsplit(' ', 1)[-1]} {m.home.name.rsplit(' ', 1)[-1]} Projection"


def render_matchup_html(m: Matchup, title: Optional[str] = None) -> str:
    g = project(m)
    name = title or _default_title(m)

    src_list = m.meta.get("sources") or []
    sources = ""
    if src_list:
        links = " &middot; ".join(
            f'<a href="{_e(x.get("url"))}" rel="noreferrer">{_e(x.get("name"))}</a>'
            for x in src_list
        )
        sources = f"<b>Sources.</b> {links}<br><br>"

    return f"""<title>{_e(name)}</title>
{FONT_LINKS}
<style>{CSS}</style>
<div class="wrap">
{_masthead(g)}
{_scoreboard(g)}

<section>
  <div class="sec-head"><h2>Game script</h2></div>
  <p class="script">{_e(game_script(g))}</p>
</section>

<section>
  <div class="sec-head"><h2>Team projections</h2>
    <span class="q">xTD is expected touchdowns, the budget every player probability below is drawn from.</span></div>
  {_team_totals(g)}
</section>

<section>
  <div class="sec-head"><h2>Quarterbacks</h2></div>
  {_qb_cards(g)}
</section>

<section>
  <div class="sec-head"><h2>Players</h2>
    <span class="q">Any TD is the chance of scoring at least once, rushing or receiving.</span></div>
  <div class="stackcols">
    <div style="--tc:{_e(m.away.color)}">{_player_table(g, g.away)}</div>
    <div style="--tc:{_e(m.home.color)}">{_player_table(g, g.home)}</div>
  </div>
</section>

<section>
  <div class="sec-head"><h2>Why these numbers</h2>
    <span class="q">Each phase on a shared 1&ndash;32 scale. Further left is better at your job; the shaded span is the mismatch.</span></div>
  {_leverage(g)}
</section>

<section>
  <div class="sec-head"><h2>Injuries</h2></div>
  {_injuries(m)}
</section>

<footer>
<b>Method.</b> Ranks blend last season and this one by how much evidence this season has
produced and how much of last season's roster remains. The score starts at the market's
implied total and spread and moves by a capped tilt, because the line already prices
quarterback play and home field. Volume comes from projected pace and pass rate; yards come
from matchup-adjusted efficiency; touchdown chances are a Poisson tail on red-zone-weighted
usage, so a player with 0.8 expected scores reads near 55%, not 80%.
<br><br>
<b>Limits.</b> These are central estimates, not forecasts of a single game. Real outcomes
scatter widely around them, and a handful of the inputs are inferred rather than published.
<br><br>
{sources}
Built with <b>nflprops matchup</b> &middot; analysis only, not advice &middot;
if gambling stops being fun, call or text 1-800-GAMBLER
</footer>
</div>"""


# --------------------------------------------------------------------------
# Terminal
# --------------------------------------------------------------------------


def render_matchup_terminal(m: Matchup, width: int = 92) -> str:
    g = project(m)
    L: List[str] = []
    bar = "=" * width
    meta = m.meta
    mk = meta.get("market") or {}

    L.append(bar)
    L.append(f"  PROJECTION  -  {meta.get('label', '')}  (Week {meta.get('week')}, {meta.get('season')})")
    L.append(bar)
    for tp in g.teams:
        t = m.team(tp.team)
        L.append(f"  {t.name:<24} {tp.points:>5.1f}   {tp.total_yards:>4.0f} yds "
                 f"({tp.pass_yards:>3.0f} pass / {tp.rush_yards:>3.0f} rush)   "
                 f"{tp.pass_rate:.0%} pass rate")
    L.append(f"  Projected total {g.total:.1f} (market {mk.get('total')})   "
             f"margin {abs(g.margin):.1f} {m.home.abbr if g.margin > 0 else m.away.abbr} "
             f"(line {mk.get('favorite')} -{mk.get('spread')})")
    L.append("-" * width)
    L.append("  GAME SCRIPT")
    words, line = game_script(g).split(), "   "
    for w in words:
        if len(line) + len(w) + 1 > width - 2:
            L.append(line)
            line = "   "
        line += w + " "
    L.append(line.rstrip())
    L.append("-" * width)

    for tp in g.teams:
        q = tp.qb
        L.append(f"  {tp.team}  {q.name}")
        L.append(f"      {q.completions:.0f}/{q.attempts:.0f} ({q.completion_pct:.1%})  "
                 f"{q.pass_yards:.0f} yds  {q.yards_per_attempt:.1f} Y/A  "
                 f"{q.pass_tds:.1f} TD  {q.interceptions:.1f} INT  {q.sacks:.1f} sacks  |  "
                 f"rush {q.rush_attempts:.0f}-{q.rush_yards:.0f}, TD {q.td_probability:.0%}")
        L.append(f"      {'PLAYER':<22}{'CAR':>5}{'RUYD':>6}{'TGT':>6}{'REC':>6}"
                 f"{'REYD':>6}{'TOTAL':>7}{'TD%':>6}")
        for s in tp.skill:
            L.append(f"      {s.name:<22}{s.carries:>5.0f}{s.rush_yards:>6.0f}"
                     f"{s.targets:>6.1f}{s.receptions:>6.1f}{s.rec_yards:>6.0f}"
                     f"{s.total_yards:>7.0f}{s.td_probability:>5.0%}")
        L.append("")

    L.append("-" * width)
    L.append("  WHY  (offense rank vs the defense facing it, weighted by exposure)")
    for l in m.top_exploits(5):
        L.append(f"      {l.attacker:<4} {l.axis.label:<26} "
                 f"{l.off_metric.adjusted:>5.1f} vs {l.def_metric.adjusted:>5.1f}"
                 f"   {l.score:>+6.2f}")
    q = m.open_questions(1)
    if q:
        t, unit, mm = q[0]
        L.append(f"      least settled: {t.abbr} {mm.label} - {ordinal(mm.prior)} last year, "
                 f"{ordinal(mm.current)} now, blended {ordinal(mm.adjusted)}")
    L.append("-" * width)
    L.append("  INJURIES")
    for t in m.teams:
        for i in t.injuries:
            L.append(f"      {t.abbr:<4} {i.player:<22} {i.pos:<4} {i.status.upper():<13} {i.injury}")
    L.append(bar)
    L.append("  central estimates, not forecasts  |  analysis only, not advice  |  1-800-GAMBLER")
    return "\n".join(L)
