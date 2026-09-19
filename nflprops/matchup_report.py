"""Rendering for :mod:`nflprops.matchup` - terminal and standalone HTML.

The page is organised the way you would actually brief someone before a game:
the conclusion first, then the single picture that supports it, then the
evidence, then the caveats. Nothing here computes anything; every number on
the page comes from the engine so the two surfaces can never disagree.

The one piece of information design worth naming is the **leverage rail**. Each
matchup axis is drawn on a shared 1-32 scale where further left is better at
your job - so an offense's marker and the marker of the defense trying to stop
it sit on the same ruler, and the distance between them *is* the mismatch. The
span is filled in the colour of whichever team owns it. A reader can scan the
column of rails and see who is winning what without reading a single number,
which is the whole point of putting them on one scale.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Sequence

from .matchup import AXES, Leverage, Matchup, Metric, Script, Team
from .theme import BASE_CSS, FONT_BODY, FONT_DISPLAY, FONT_LINKS, FONT_MONO, TOKENS_CSS


def _e(x: Any) -> str:
    return html.escape(str(x if x is not None else ""))


def _pos(rank: float) -> float:
    """Rank 1..32 -> 0..100% along the leverage rail."""
    return max(0.0, min(100.0, (float(rank) - 1.0) / 31.0 * 100.0))


def _ordinal(n: float) -> str:
    i = int(round(n))
    if 10 <= i % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suf}"


def _src_mark(m: Metric) -> str:
    """A small honest badge: is this rank measured, or is it my estimate?"""
    if m.prior_src == "measured" and m.current_src == "measured":
        return '<span class="src src-m" title="Both seasons from published figures">measured</span>'
    if m.measured:
        return '<span class="src src-p" title="One season from published figures, one estimated">part</span>'
    return '<span class="src src-e" title="Estimated from surrounding reporting, not a published rank">est</span>'


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

MATCHUP_CSS = f"""
{TOKENS_CSS}
{BASE_CSS}

body{{padding-block:0 64px}}
.wrap{{max-width:1120px}}

h1,h2,h3{{font-family:{FONT_DISPLAY}; font-weight:700; margin:0;
  text-wrap:balance; letter-spacing:.005em}}
h1{{font-size:clamp(30px,5.4vw,46px); line-height:1.02; letter-spacing:-.012em}}
h2{{font-size:clamp(19px,2.6vw,24px); line-height:1.12}}
h3{{font-size:15px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted)}}
p{{margin:0}}
a{{color:var(--accent)}}

section{{margin-top:44px}}
.sec-head{{display:flex; flex-wrap:wrap; align-items:baseline; gap:10px 16px;
  padding-bottom:10px; border-bottom:2px solid var(--ink); margin-bottom:20px}}
.sec-head .q{{font-size:13.5px; color:var(--muted); font-style:italic}}
.stack{{display:flex; flex-direction:column; gap:14px}}

/* ---- masthead ---- */
.mast{{padding-block:28px 0}}
.kick{{font-family:{FONT_DISPLAY}; font-size:12px; letter-spacing:.17em;
  text-transform:uppercase; color:var(--muted); display:flex; flex-wrap:wrap;
  gap:6px 14px; margin-bottom:12px}}
.kick b{{color:var(--ink); font-weight:600}}
.mast .sub{{margin-top:12px; color:var(--muted); font-size:15.5px; max-width:66ch}}

.teamline{{display:flex; flex-wrap:wrap; align-items:stretch; gap:14px;
  margin-top:22px}}
.tcard{{flex:1 1 260px; min-width:0; background:var(--card); border:1px solid var(--rule);
  border-radius:3px; border-left:5px solid var(--tc); padding:14px 16px}}
.tcard .nm{{font-family:{FONT_DISPLAY}; font-size:22px; font-weight:700; line-height:1.05}}
.tcard .rec{{font-family:{FONT_MONO}; font-size:12.5px; color:var(--muted); margin-top:3px}}
.tcard dl{{margin:12px 0 0; display:grid; grid-template-columns:auto 1fr;
  gap:4px 12px; font-size:13px}}
.tcard dt{{color:var(--faint); font-family:{FONT_DISPLAY}; letter-spacing:.09em;
  text-transform:uppercase; font-size:10.5px; align-self:center}}
.tcard dd{{margin:0; color:var(--ink)}}
.vs{{flex:0 0 auto; align-self:center; font-family:{FONT_DISPLAY}; font-size:13px;
  letter-spacing:.2em; color:var(--faint); text-transform:uppercase}}

.market{{margin-top:14px; display:flex; flex-wrap:wrap; gap:8px}}
.chip{{font-family:{FONT_MONO}; font-size:12px; padding:5px 10px; border-radius:2px;
  background:var(--raised); border:1px solid var(--rule); color:var(--ink)}}
.chip b{{color:var(--muted); font-weight:500; letter-spacing:.04em}}

/* ---- the read ---- */
.read{{background:var(--card); border:1px solid var(--rule); border-top:3px solid var(--accent);
  border-radius:3px; padding:22px 24px}}
.read p + p{{margin-top:13px}}
.read p{{font-size:16px; line-height:1.62; max-width:70ch}}
.read .lede{{font-size:18.5px; line-height:1.48; font-weight:600}}

.scripts{{display:flex; flex-wrap:wrap; gap:14px; margin-top:22px}}
.script{{flex:1 1 320px; min-width:0; background:var(--card); border:1px solid var(--rule);
  border-radius:3px; padding:16px 18px; border-top:3px solid var(--tc)}}
.script .hd{{display:flex; align-items:baseline; justify-content:space-between; gap:12px}}
.script .who{{font-family:{FONT_DISPLAY}; font-size:17px; font-weight:700}}
.script .pts{{font-family:{FONT_MONO}; font-size:13px; color:var(--muted)}}
.split{{margin:14px 0 8px; height:30px; display:flex; border-radius:2px; overflow:hidden;
  border:1px solid var(--rule)}}
.split .sp{{display:flex; align-items:center; justify-content:center;
  font-family:{FONT_MONO}; font-size:12px; font-weight:600; min-width:0;
  white-space:nowrap; overflow:hidden}}
.split .pass{{background:var(--tc); color:#fff}}
.split .run{{background:var(--track); color:var(--ink)}}
.splitkey{{display:flex; justify-content:space-between; font-family:{FONT_DISPLAY};
  font-size:10.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--faint)}}
.adj{{margin-top:14px; border-top:1px solid var(--rule-soft); padding-top:10px}}
.adj .row{{display:grid; grid-template-columns:118px 54px 1fr; gap:4px 10px;
  font-size:12.5px; padding:5px 0; align-items:start}}
.adj .row + .row{{border-top:1px dotted var(--rule-soft)}}
.adj .nm{{font-family:{FONT_DISPLAY}; letter-spacing:.06em; text-transform:uppercase;
  font-size:10.5px; color:var(--muted); padding-top:2px}}
.adj .v{{font-family:{FONT_MONO}; font-size:12px; text-align:right; font-weight:600}}
.adj .v.up{{color:var(--accent)}} .adj .v.dn{{color:var(--muted)}}
.adj .w{{color:var(--muted); line-height:1.4}}
.adj .base{{font-size:12px; color:var(--faint); font-family:{FONT_MONO};
  padding-bottom:8px; border-bottom:1px solid var(--rule-soft); margin-bottom:4px}}

/* ---- leverage rails ---- */
.lev{{display:flex; flex-direction:column; gap:0}}
.lrow{{display:grid; grid-template-columns:minmax(0,190px) 1fr minmax(0,92px);
  gap:6px 18px; align-items:center; padding:14px 0;
  border-top:1px solid var(--rule-soft)}}
.lrow:first-child{{border-top:none}}
.lrow .who{{min-width:0}}
.lrow .ax{{font-family:{FONT_DISPLAY}; font-size:15.5px; font-weight:600; line-height:1.15}}
.lrow .at{{font-family:{FONT_MONO}; font-size:11px; color:var(--tc); font-weight:600;
  letter-spacing:.05em; margin-top:2px}}
.rail{{position:relative; height:38px; min-width:0}}
.rail .track{{position:absolute; left:0; right:0; top:17px; height:4px;
  background:var(--track); border-radius:2px}}
.rail .span{{position:absolute; top:15px; height:8px; border-radius:2px;
  background:var(--tc); opacity:.85}}
.rail .mk{{position:absolute; top:6px; width:2px; height:26px; background:var(--ink)}}
.rail .mk.off{{background:var(--tc); width:3px}}
.rail .lab{{position:absolute; top:26px; font-family:{FONT_MONO}; font-size:10.5px;
  color:var(--muted); white-space:nowrap}}
.rail .lab.hi{{top:-3px; color:var(--ink); font-weight:600}}
.rail .ends{{position:absolute; top:17px; width:100%; display:flex;
  justify-content:space-between; font-family:{FONT_DISPLAY}; font-size:9.5px;
  letter-spacing:.1em; color:var(--faint); pointer-events:none}}
.rail .ends span{{transform:translateY(-14px)}}
.lrow .sc{{text-align:right; font-family:{FONT_MONO}}}
.lrow .sc .n{{font-size:19px; font-weight:600; letter-spacing:-.02em}}
.lrow .sc .g{{font-family:{FONT_DISPLAY}; font-size:10px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); display:block}}
.lrow .why{{grid-column:1/-1; font-size:13.5px; color:var(--muted); line-height:1.5;
  max-width:78ch}}
.lrow.decisive .ax{{font-size:17px}}
.lrow.even{{opacity:.72}}

/* ---- tables ---- */
.tw{{overflow-x:auto; -webkit-overflow-scrolling:touch}}
table{{border-collapse:collapse; width:100%; font-size:13.5px; min-width:520px}}
th,td{{text-align:left; padding:8px 10px; border-bottom:1px solid var(--rule-soft);
  vertical-align:top}}
th{{font-family:{FONT_DISPLAY}; font-size:10.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--faint); border-bottom:1px solid var(--rule);
  white-space:nowrap}}
td.n,th.n{{text-align:right; font-family:{FONT_MONO}; font-variant-numeric:tabular-nums;
  white-space:nowrap}}
tbody tr:hover{{background:var(--raised)}}
td .d{{display:block; color:var(--muted); font-size:12.5px; line-height:1.45;
  margin-top:3px; max-width:62ch}}
.rk{{font-family:{FONT_MONO}; font-weight:600}}
.rk.good{{color:var(--pos)}} .rk.bad{{color:var(--neg)}}
.tr{{font-family:{FONT_MONO}; font-size:11px}}
.tr.up{{color:var(--pos)}} .tr.dn{{color:var(--neg)}} .tr.flat{{color:var(--faint)}}
.src{{font-family:{FONT_DISPLAY}; font-size:9px; letter-spacing:.09em;
  text-transform:uppercase; padding:1px 5px; border-radius:2px; margin-left:6px;
  vertical-align:1px; white-space:nowrap}}
.src-m{{background:var(--pos-soft); color:var(--pos)}}
.src-p{{background:var(--accent-soft); color:var(--accent-ink)}}
.src-e{{background:var(--track); color:var(--muted)}}

.bar{{display:block; height:5px; background:var(--tc); border-radius:2px; margin-top:4px}}

/* ---- two column ---- */
.cols{{display:flex; flex-wrap:wrap; gap:26px}}
.col{{flex:1 1 440px; min-width:0}}
.col > h3{{margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid var(--rule)}}

/* ---- cards ---- */
.notes{{display:flex; flex-direction:column; gap:12px}}
.note{{background:var(--card); border:1px solid var(--rule); border-left:4px solid var(--tc);
  border-radius:3px; padding:13px 16px}}
.note .h{{font-family:{FONT_DISPLAY}; font-size:15.5px; font-weight:700; line-height:1.2}}
.note .k{{font-family:{FONT_DISPLAY}; font-size:9.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--tc); font-weight:600}}
.note p{{margin-top:6px; color:var(--muted); font-size:13.5px; line-height:1.55;
  max-width:76ch}}

.inj{{display:flex; flex-wrap:wrap; gap:12px}}
.icard{{flex:1 1 300px; min-width:0; background:var(--card); border:1px solid var(--rule);
  border-radius:3px; padding:12px 14px; border-top:3px solid var(--sc)}}
.icard .top{{display:flex; align-items:baseline; justify-content:space-between; gap:10px}}
.icard .p{{font-family:{FONT_DISPLAY}; font-size:16px; font-weight:700}}
.icard .st{{font-family:{FONT_DISPLAY}; font-size:9.5px; letter-spacing:.13em;
  text-transform:uppercase; font-weight:600; color:var(--sc); white-space:nowrap}}
.icard .meta{{font-family:{FONT_MONO}; font-size:11.5px; color:var(--muted); margin-top:2px}}
.icard .nt{{font-size:13px; color:var(--muted); line-height:1.5; margin-top:8px}}
.icard .hit{{margin-top:8px; font-family:{FONT_MONO}; font-size:11px; color:var(--faint)}}

.flags{{display:flex; flex-direction:column; gap:10px}}
.flag{{display:grid; grid-template-columns:auto auto 1fr; gap:4px 12px;
  align-items:baseline; padding:11px 0; border-top:1px solid var(--rule-soft)}}
.flag:first-child{{border-top:none}}
.flag .l{{font-family:{FONT_DISPLAY}; font-size:14px; font-weight:700; white-space:nowrap}}
.flag .w{{font-family:{FONT_DISPLAY}; font-size:9.5px; letter-spacing:.13em;
  text-transform:uppercase; padding:2px 7px; border-radius:2px; white-space:nowrap}}
.w-high{{background:var(--neg-soft); color:var(--neg)}}
.w-medium{{background:var(--st-warn-bg); color:var(--st-warn-fg)}}
.w-low{{background:var(--track); color:var(--muted)}}
.flag .b{{color:var(--muted); font-size:13.5px; line-height:1.55; grid-column:3;
  max-width:80ch}}

footer{{margin-top:56px; padding-top:18px; border-top:1px solid var(--rule);
  color:var(--faint); font-size:12.5px; line-height:1.6; max-width:82ch}}
footer b{{color:var(--muted)}}

@media (max-width:720px){{
  .lrow{{grid-template-columns:1fr auto; gap:4px 12px}}
  .rail{{grid-column:1/-1}}
  .lrow .sc{{text-align:right}}
  table{{min-width:460px}}
  .adj .row{{grid-template-columns:1fr 54px; }}
  .adj .w{{grid-column:1/-1}}
}}
"""


# --------------------------------------------------------------------------
# Fragments
# --------------------------------------------------------------------------


def _masthead(m: Matchup) -> str:
    meta, mk = m.meta, (m.meta.get("market") or {})
    venue = meta.get("venue") or {}
    chips = []
    fav = mk.get("favorite")
    if fav:
        chips.append(f'<span class="chip"><b>LINE</b> {_e(fav)} -{_e(mk.get("spread"))}</span>')
    if mk.get("total"):
        opened = mk.get("total_open")
        moved = f' (from {_e(opened)})' if opened and opened != mk.get("total") else ""
        chips.append(f'<span class="chip"><b>TOTAL</b> {_e(mk.get("total"))}{moved}</span>')
    for k, v in (mk.get("moneyline") or {}).items():
        chips.append(f'<span class="chip"><b>{_e(k)} ML</b> {"+" if float(v) > 0 else ""}{_e(v)}</span>')

    cards = []
    for t in m.teams:
        loc = "Home" if t.home else "Away"
        cards.append(f"""<div class="tcard" style="--tc:{_e(t.color)}">
  <div class="nm">{_e(t.name)}</div>
  <div class="rec">{_e(t.record)} &middot; {loc} &middot; {_e(t.last_game)}</div>
  <dl>
    <dt>QB</dt><dd>{_e(t.qb)}</dd>
    <dt>Calls</dt><dd>{_e(t.play_caller)}</dd>
    <dt>Def</dt><dd>{_e(t.dc)}</dd>
    <dt>Rest</dt><dd>{_e((t.rest or {}).get('days'))} days{' &middot; short week' if (t.rest or {}).get('short_week') else ''}</dd>
  </dl>
</div>""")

    return f"""<header class="mast">
<div class="kick">
  <span>{_e(meta.get('label'))}</span>
  <span><b>Week {_e(meta.get('week'))}</b>, {_e(meta.get('season'))}</span>
  <span>{_e(meta.get('kickoff_local'))}</span>
  <span>{_e(meta.get('network'))}</span>
</div>
<h1>{_e(m.away.abbr)} at {_e(m.home.abbr)}: who can take what?</h1>
<p class="sub">{_e((meta.get('venue') or {}).get('name'))}, {_e(venue.get('city'))}.
{_e(meta.get('series'))}</p>
<div class="market">{''.join(chips)}</div>
<div class="teamline">{cards[0]}<div class="vs">vs</div>{cards[1]}</div>
</header>"""


def _script_card(m: Matchup, t: Team) -> str:
    s = m.scripts[t.abbr]
    p = s.projected_pass_rate * 100
    rows = [
        f'<div class="base">Neutral pass rate {s.neutral_pass_rate:.0%} &rarr; '
        f'projected {s.projected_pass_rate:.0%}</div>'
    ]
    for name, val, why in s.adjustments:
        cls = "up" if val > 0 else "dn"
        rows.append(
            f'<div class="row"><span class="nm">{_e(name)}</span>'
            f'<span class="v {cls}">{val * 100:+.1f}</span>'
            f'<span class="w">{_e(why)}</span></div>'
        )
    return f"""<div class="script" style="--tc:{_e(t.color)}">
  <div class="hd"><span class="who">{_e(t.name)}</span>
    <span class="pts">{s.projected_points:.1f} projected pts</span></div>
  <div class="split">
    <div class="sp pass" style="width:{p:.1f}%">{s.projected_pass_rate:.0%} pass</div>
    <div class="sp run" style="width:{100 - p:.1f}%">{s.projected_run_rate:.0%} run</div>
  </div>
  <div class="splitkey"><span>~{s.pass_attempts:.0f} dropbacks</span>
    <span>~{s.projected_plays:.0f} plays</span><span>~{s.rush_attempts:.0f} carries</span></div>
  <div class="adj">{''.join(rows)}</div>
</div>"""


def _rail(m: Matchup, l: Leverage) -> str:
    """The shared 1-32 ruler. Further left is better at your job."""
    att = m.team(l.attacker)
    dfn = m.team(l.defender)
    o, d = _pos(l.off_metric.adjusted), _pos(l.def_metric.adjusted)
    lo, hi = min(o, d), max(o, d)
    owner = att if l.edge >= 0 else dfn

    # When the two markers sit almost on top of each other the labels collide.
    # Push them apart along the rail rather than letting them overprint, and
    # clamp both inside the track so neither runs off the end.
    def anchor(x: float, nudge: float) -> str:
        x2 = max(0.0, min(100.0, x + nudge))
        if x2 < 12:
            return f"left:{x2:.1f}%; transform:translateX(0)"
        if x2 > 88:
            return f"left:{x2:.1f}%; transform:translateX(-100%)"
        return f"left:{x2:.1f}%; transform:translateX(-50%)"

    close = abs(o - d) < 9
    off_style = anchor(o, -4.0 if (close and o <= d) else (4.0 if close else 0.0))
    def_style = anchor(d, 4.0 if (close and o <= d) else (-4.0 if close else 0.0))

    return f"""<div class="rail">
  <div class="track"></div>
  <div class="ends"><span>1ST</span><span>32ND</span></div>
  <div class="span" style="left:{lo:.1f}%; width:{max(0.6, hi - lo):.1f}%; background:{_e(owner.color)}"></div>
  <div class="mk off" style="left:{o:.1f}%; background:{_e(att.color)}"></div>
  <div class="mk" style="left:{d:.1f}%"></div>
  <div class="lab hi" style="{off_style}">{_e(l.attacker)} off {_ordinal(l.off_metric.adjusted)}</div>
  <div class="lab" style="{def_style}">{_e(l.defender)} def {_ordinal(l.def_metric.adjusted)}</div>
</div>"""


def _leverage_row(m: Matchup, l: Leverage) -> str:
    att = m.team(l.attacker)
    owner = att if l.edge >= 0 else m.team(l.defender)
    caption = (f"{l.attacker} attacking {l.defender}" if l.edge >= 0
               else f"{l.defender} defense beating {l.attacker}")
    why = _why(m, l)
    return f"""<div class="lrow {l.grade}" style="--tc:{_e(owner.color)}">
  <div class="who">
    <div class="ax">{_e(l.axis.label)}</div>
    <div class="at">{_e(caption)}</div>
  </div>
  {_rail(m, l)}
  <div class="sc"><span class="n">{l.score:+.1f}</span><span class="g">{_e(l.grade)}</span></div>
  <div class="why">{why}</div>
</div>"""


def _why(m: Matchup, l: Leverage) -> str:
    """Build the sentence from the data rather than from a stored string."""
    att, dfn = l.attacker, l.defender
    om, dm = l.off_metric, l.def_metric
    verb = "has the edge over" if l.edge >= 0 else "is outmatched by"
    head = (
        f"<b>{_e(att)}'s {_e(om.label).lower()} ({_ordinal(om.adjusted)}) "
        f"{verb} {_e(dfn)}'s {_e(dm.label).lower()} ({_ordinal(dm.adjusted)}).</b> "
        f"This phase is projected to cover {l.exposure * 100:.0f}% of what decides the game."
    )
    bits = [head]
    if om.detail:
        bits.append(_e(om.detail))
    if dm.detail:
        bits.append(_e(dm.detail))
    if om.injury_causes:
        bits.append("Adjusted for " + _e(", ".join(sorted(set(om.injury_causes)))) + ".")
    if dm.injury_causes:
        bits.append("Adjusted for " + _e(", ".join(sorted(set(dm.injury_causes)))) + ".")
    return " ".join(bits)


def _rank_table(m: Matchup, t: Team, unit: str) -> str:
    metrics = t.offense if unit == "offense" else t.defense
    rows = []
    for key in metrics:
        mm = metrics[key]
        good = "good" if mm.adjusted <= 10 else ("bad" if mm.adjusted >= 23 else "")
        tr = {1: ('<span class="tr dn">&darr; worse</span>'),
              -1: ('<span class="tr up">&uarr; better</span>'),
              0: '<span class="tr flat">&mdash;</span>'}[mm.trend]
        inj = ""
        if mm.injury_shift >= 0.4:
            inj = (f'<span class="d">Injury adjustment +{mm.injury_shift:.1f} spots: '
                   f'{_e(", ".join(sorted(set(mm.injury_causes))))}.</span>')
        carry = ""
        if mm.carryover_note:
            carry = f'<span class="d">{_e(mm.carryover_note)}</span>'
        det = f'<span class="d">{_e(mm.detail)}</span>' if mm.detail else ""
        rows.append(f"""<tr>
  <td>{_e(mm.label)}{_src_mark(mm)}{det}{carry}{inj}</td>
  <td class="n">{_e(mm.prior) if mm.prior is not None else '&mdash;'}</td>
  <td class="n">{_e(mm.current) if mm.current is not None else '&mdash;'}</td>
  <td class="n"><span class="rk {good}">{mm.adjusted:.1f}</span></td>
  <td class="n">{tr}</td>
</tr>""")
    return f"""<div class="tw"><table>
<thead><tr><th>{_e(t.abbr)} {unit}</th><th class="n">2025</th><th class="n">2026</th>
<th class="n">Blend</th><th class="n">Trend</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


def _tendency_table(m: Matchup) -> str:
    keys: List[str] = []
    for t in m.teams:
        for k in t.tendencies:
            if k not in keys:
                keys.append(k)
    rows = []
    for k in keys:
        cells = []
        label, detail = k.replace("_", " ").title(), ""
        for t in m.teams:
            td = t.tendencies.get(k)
            if not td:
                cells.append('<td class="n">&mdash;</td><td class="n">&mdash;</td><td class="n">&mdash;</td>')
                continue
            label = td.get("label", label)
            if td.get("detail") and not detail:
                detail = f"<b>{_e(t.abbr)}:</b> {_e(td['detail'])}"
            elif td.get("detail"):
                detail += f" <b>{_e(t.abbr)}:</b> {_e(td['detail'])}"
            blended = t.tendency(k, m.games)
            fmt = (lambda v: f"{v:.0f}") if abs(float(td.get("prior") or 0)) > 3 else (lambda v: f"{v * 100:.0f}%")
            cells.append(
                f'<td class="n">{fmt(float(td.get("prior") or 0))}</td>'
                f'<td class="n">{fmt(float(td.get("current") or 0))}</td>'
                f'<td class="n"><b>{fmt(blended)}</b></td>'
            )
        det = f'<span class="d">{detail}</span>' if detail else ""
        rows.append(f"<tr><td>{_e(label)}{det}</td>" + "".join(cells) + "</tr>")
    heads = "".join(
        f'<th class="n" colspan="3">{_e(t.abbr)}</th>' for t in m.teams
    )
    sub = "".join('<th class="n">25</th><th class="n">26</th><th class="n">Blend</th>' for _ in m.teams)
    return f"""<div class="tw"><table>
<thead><tr><th rowspan="2">Play calling</th>{heads}</tr><tr>{sub}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


def _usage_table(m: Matchup, t: Team) -> str:
    rows = []
    for p in t.players:
        ts = p.get("target_share")
        bar = ""
        if ts:
            bar = f'<span class="bar" style="width:{float(ts) * 260:.0f}%; max-width:100%"></span>'
        nums = [
            f'{float(ts) * 100:.1f}%' if ts else "&mdash;",
            f'{float(p["adot"]):.1f}' if p.get("adot") else "&mdash;",
            f'{int(p["routes_per_game"])}' if p.get("routes_per_game") else "&mdash;",
            f'{float(p["tprr"]) * 100:.0f}%' if p.get("tprr") else "&mdash;",
            f'{float(p["snap_share"]) * 100:.0f}%' if p.get("snap_share") else "&mdash;",
        ]
        detail = p.get("detail") or p.get("note") or ""
        src = f'<span class="d">Source: {_e(p["src"])}.</span>' if p.get("src") else ""
        det = f'<span class="d">{_e(detail)}</span>' if detail else ""
        name_cell = (f'<b>{_e(p.get("name"))}</b> '
                     f'<span class="rk">{_e(p.get("pos"))}</span>{det}{src}{bar}')
        cells = "".join(f'<td class="n">{v}</td>' for v in nums)
        rows.append(f"<tr><td>{name_cell}</td>{cells}</tr>")
    return f"""<div class="tw"><table>
<thead><tr><th>{_e(t.abbr)} usage</th><th class="n">Tgt share</th><th class="n">aDOT</th>
<th class="n">Routes/g</th><th class="n">Tgt/route</th><th class="n">Snaps</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


_STATUS_COLOR = {
    "out": "var(--neg)", "doubtful": "var(--neg)",
    "questionable": "var(--st-warn-fg)", "limited": "var(--st-warn-fg)",
    "probable": "var(--muted)", "healthy": "var(--muted)",
}


def _injury_cards(m: Matchup, t: Team) -> str:
    if not t.injuries:
        return '<p style="color:var(--muted)">No designations on the final report.</p>'
    cards = []
    for i in t.injuries:
        hit = ""
        if i.axes and i.cost > 0:
            parts = [f"{k.replace('_', ' ')} +{i.cost * 8.0 * float(w):.1f}" for k, w in i.axes.items()]
            hit = f'<div class="hit">Model cost: {_e(", ".join(parts))} rank spots</div>'
        cards.append(f"""<div class="icard" style="--sc:{_STATUS_COLOR.get(i.status, 'var(--muted)')}">
  <div class="top"><span class="p">{_e(i.player)}</span><span class="st">{_e(i.status)}</span></div>
  <div class="meta">{_e(i.pos)} &middot; {_e(i.injury)}</div>
  <div class="nt">{_e(i.note)}</div>{hit}
</div>""")
    return f'<div class="inj">{"".join(cards)}</div>'


# --------------------------------------------------------------------------
# Narrative
# --------------------------------------------------------------------------


def summarise(m: Matchup) -> List[str]:
    """The written read, assembled from what the engine actually concluded."""
    home, away = m.home, m.away
    sh, sa = m.scripts[home.abbr], m.scripts[away.abbr]
    top = m.top_exploits(3)
    paras: List[str] = []

    lead = top[0]
    lead_team = m.team(lead.attacker)
    paras.append(
        f"<b>{_e(lead_team.name)} attacking {_e(lead.defender)} on "
        f"{_e(lead.axis.label).lower()} is the largest single mismatch on the board</b>, "
        f"at {_ordinal(lead.off_metric.adjusted)} against {_ordinal(lead.def_metric.adjusted)}. "
        f"Everything else in this game is closer than the {_e((m.meta.get('market') or {}).get('spread'))}-point "
        f"spread suggests."
    )

    def describe(t: Team, s: Script) -> str:
        delta = s.projected_pass_rate - s.neutral_pass_rate
        if delta <= -0.03:
            move = f"run more than they normally do ({s.neutral_pass_rate:.0%} neutral pass rate down to {s.projected_pass_rate:.0%})"
        elif delta >= 0.03:
            move = f"throw more than they normally do ({s.neutral_pass_rate:.0%} neutral pass rate up to {s.projected_pass_rate:.0%})"
        else:
            move = f"play close to their normal split ({s.projected_pass_rate:.0%} pass)"
        drivers = sorted(s.adjustments, key=lambda a: -abs(a[1]))[:2]
        why = "; ".join(f"{d[0].lower()} {d[1] * 100:+.1f}" for d in drivers)
        return (f"<b>{_e(t.abbr)} should {move}</b>, roughly {s.pass_attempts:.0f} dropbacks "
                f"to {s.rush_attempts:.0f} carries. Biggest movers: {why}.")

    paras.append(describe(home, sh) + " " + describe(away, sa))

    qs = m.open_questions(2)
    if qs:
        t, unit, mm = qs[0]
        paras.append(
            f"<b>The honest caveat:</b> {_e(t.abbr)}'s {_e(mm.label).lower()} is the least "
            f"settled number in this game. It ranked {_ordinal(mm.prior)} last season and "
            f"{_ordinal(mm.current)} so far this one. The model splits the difference at "
            f"{_ordinal(mm.adjusted)}, and nobody should pretend that is knowledge. "
            f"If one of those two seasons is the real team, this game looks very different "
            f"than the board below says."
        )

    # The most useful thing a matchup model can find: a team whose single best
    # path is the one the game script is most likely to take away from them.
    for t in (away, home):
        s = m.scripts[t.abbr]
        own = [l for l in m.top_exploits(99) if l.attacker == t.abbr]
        if not own:
            continue
        best = own[0]
        if best.score < 2.0:
            continue
        starved = (best.axis.family == "rush" and s.projected_pass_rate > s.neutral_pass_rate) or \
                  (best.axis.family == "pass" and s.projected_pass_rate < s.neutral_pass_rate)
        trailing = s.projected_points < m.scripts[m.opponent(t.abbr).abbr].projected_points
        if best.axis.family == "rush" and trailing:
            paras.append(
                f"<b>The tension in this game is that {_e(t.abbr)}'s best matchup is the one "
                f"they are least likely to get to use.</b> {_e(best.axis.label)} is where they "
                f"hold a {best.score:+.1f} edge, and it is also the first thing a two-score "
                f"deficit takes away. Watch their run-pass split on the opening two drives and "
                f"in the third quarter: if they are still handing it off while down a score, "
                f"they believe what this board believes. If they abandon it, the projected "
                f"{s.projected_pass_rate:.0%} pass rate is the floor, not the ceiling, and they "
                f"are throwing into their worst matchup instead."
            )
        elif starved:
            paras.append(
                f"<b>{_e(t.abbr)}'s best phase and their likely plan point in different "
                f"directions.</b> {_e(best.axis.label)} grades as their edge, but the script "
                f"pushes them the other way."
            )
        break

    nl_h, nl_a = m.net_leverage(home.abbr), m.net_leverage(away.abbr)
    ahead = home if nl_h > nl_a else away
    paras.append(
        f"Summing every phase, <b>{_e(ahead.abbr)} holds the net phase advantage</b> "
        f"({nl_h:+.1f} {_e(home.abbr)} vs {nl_a:+.1f} {_e(away.abbr)}). "
        f"That figure prices units against units. It does not price quarterback play, "
        f"home field, or coaching in a two-minute drill, which is most of the gap between "
        f"this board and the betting market."
    )
    return paras


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


def _default_title(m: Matchup) -> str:
    """Name the page after the game, the way a broadcast would."""
    def short(t: Team) -> str:
        return t.name.rsplit(" ", 1)[-1]
    return f"{short(m.away)} {short(m.home)} Matchup Lab"


def render_matchup_html(m: Matchup, title: Optional[str] = None) -> str:
    meta = m.meta
    name = title or _default_title(m)

    board = m.top_exploits(len(AXES) * 2)
    decisive = [l for l in board if l.grade in ("decisive", "clear")]
    rest = [l for l in board if l not in decisive]

    open_q = "".join(
        f"""<div class="note" style="--tc:{_e(t.color)}">
  <div class="k">{_e(t.abbr)} {_e(unit)} &middot; disagreement {mm.disagreement * 100:.0f}%</div>
  <div class="h">{_e(mm.label)}: {_ordinal(mm.prior)} last year, {_ordinal(mm.current)} this year</div>
  <p>Blended to {_ordinal(mm.adjusted)}. {_e(mm.detail)}</p>
</div>"""
        for t, unit, mm in m.open_questions(5)
    )

    narrative = "".join(
        f"""<div class="note" style="--tc:{_e(m.team(n.get('team', m.home.abbr)).color if n.get('team') in (m.home.abbr, m.away.abbr) else m.home.color)}">
  <div class="k">{_e(n.get('kind'))} &middot; {_e(n.get('team'))}</div>
  <div class="h">{_e(n.get('headline'))}</div><p>{_e(n.get('body'))}</p>
</div>"""
        for n in m.narrative
    )

    flags = "".join(
        f"""<div class="flag"><span class="l">{_e(f.label)}</span>
  <span class="w w-{_e(f.weight)}">{_e(f.weight)}</span>
  <span class="b"><b>{_e(f.team)}</b> &mdash; {_e(f.body)}</span></div>"""
        for f in m.flags
    )

    return f"""<title>{_e(name)}</title>
{FONT_LINKS}
<style>{MATCHUP_CSS}</style>
<div class="wrap">
{_masthead(m)}

<section>
  <div class="sec-head"><h2>The read</h2>
    <span class="q">What is each team going to try to do, and why?</span></div>
  <div class="read">{''.join(f'<p class="{"lede" if i == 0 else ""}">{p}</p>' for i, p in enumerate(summarise(m)))}</div>
  <div class="scripts">{_script_card(m, m.away)}{_script_card(m, m.home)}</div>
</section>

<section>
  <div class="sec-head"><h2>Leverage board</h2>
    <span class="q">Every offensive phase against the defense that has to stop it. Further left is better at your job; the shaded span is the mismatch.</span></div>
  <div class="lev">{''.join(_leverage_row(m, l) for l in decisive)}</div>
  <h3 style="margin:26px 0 4px">Closer to even</h3>
  <div class="lev">{''.join(_leverage_row(m, l) for l in rest)}</div>
</section>

<section>
  <div class="sec-head"><h2>Open questions</h2>
    <span class="q">Where last season and this season flatly contradict each other. These decide the game.</span></div>
  <div class="notes">{open_q}</div>
</section>

{f'''<section>
  <div class="sec-head"><h2>What the tape says</h2>
    <span class="q">Context the ranks cannot carry.</span></div>
  <div class="notes">{narrative}</div>
</section>''' if narrative else ''}

<section>
  <div class="sec-head"><h2>Unit ranks</h2>
    <span class="q">2025 full season, 2026 to date, and the credibility-weighted blend the model actually uses.</span></div>
  <div class="cols">
    <div class="col"><h3>{_e(m.away.abbr)} offense</h3>{_rank_table(m, m.away, 'offense')}</div>
    <div class="col"><h3>{_e(m.home.abbr)} defense</h3>{_rank_table(m, m.home, 'defense')}</div>
  </div>
  <div class="cols" style="margin-top:26px">
    <div class="col"><h3>{_e(m.home.abbr)} offense</h3>{_rank_table(m, m.home, 'offense')}</div>
    <div class="col"><h3>{_e(m.away.abbr)} defense</h3>{_rank_table(m, m.away, 'defense')}</div>
  </div>
</section>

<section>
  <div class="sec-head"><h2>Play calling</h2>
    <span class="q">Rates, not results. How each staff chooses to play before the score forces them.</span></div>
  {_tendency_table(m)}
</section>

<section>
  <div class="sec-head"><h2>Personnel and usage</h2>
    <span class="q">Who gets the ball, how far downfield, and how often per route.</span></div>
  <div class="cols">
    <div class="col">{_usage_table(m, m.away)}</div>
    <div class="col">{_usage_table(m, m.home)}</div>
  </div>
</section>

<section>
  <div class="sec-head"><h2>Injuries</h2>
    <span class="q">Final report, and what each designation costs its unit in the model.</span></div>
  <div class="cols">
    <div class="col"><h3>{_e(m.away.name)}</h3>{_injury_cards(m, m.away)}</div>
    <div class="col"><h3>{_e(m.home.name)}</h3>{_injury_cards(m, m.home)}</div>
  </div>
</section>

<section>
  <div class="sec-head"><h2>Situation</h2>
    <span class="q">Weather, rest and travel - the things that move a play sheet before anyone lines up.</span></div>
  <div class="flags">{flags}</div>
</section>

<footer>
<b>How to read this.</b> Every rank is a blend of last season and this one, weighted by
how much evidence this season has produced and by how much of last season's team still
exists. Badges mark which figures come from published numbers
(<span class="src src-m">measured</span>), which mix a published figure with an estimate
(<span class="src src-p">part</span>), and which are inferred from surrounding reporting
rather than a published league table (<span class="src src-e">est</span>). Estimated ranks
are the author's inference and should be treated as such.
<br><br>
<b>Limits.</b> Leverage scores weight units against units. They do not price quarterback
play, home field, coaching in the fourth quarter, or anything that happens on special
teams. Do not read the net figure as a point spread.
<br><br>
Built with <b>nflprops matchup</b> &middot; analysis only, not advice &middot;
if gambling stops being fun, call or text 1-800-GAMBLER
</footer>
</div>"""


# --------------------------------------------------------------------------
# Terminal
# --------------------------------------------------------------------------


def render_matchup_terminal(m: Matchup, width: int = 96) -> str:
    L: List[str] = []
    bar = "=" * width
    meta = m.meta
    L.append(bar)
    L.append(f"  MATCHUP LAB  -  {meta.get('label', '')}  (Week {meta.get('week')}, {meta.get('season')})")
    L.append(bar)
    mk = meta.get("market") or {}
    L.append(f"  {m.away.name} ({m.away.record})  at  {m.home.name} ({m.home.record})")
    L.append(f"  {meta.get('kickoff_local', '')}  |  {(meta.get('venue') or {}).get('name', '')}")
    if mk:
        L.append(f"  Line: {mk.get('favorite')} -{mk.get('spread')}   Total: {mk.get('total')}")
    w = m.weather or {}
    if w:
        L.append(f"  Weather: {w.get('temp_f')}F, wind {w.get('wind_mph')} mph, "
                 f"{float(w.get('precip_chance') or 0) * 100:.0f}% precip - {w.get('forecast', '')}")
    L.append("-" * width)

    for t in m.teams:
        s = m.scripts[t.abbr]
        L.append(f"  {t.abbr} script: {s.projected_pass_rate:.0%} pass / {s.projected_run_rate:.0%} run "
                 f"(neutral {s.neutral_pass_rate:.0%})  ~{s.pass_attempts:.0f} dropbacks, "
                 f"~{s.rush_attempts:.0f} carries, {s.projected_points:.1f} pts")
        for n, v, why in s.adjustments:
            L.append(f"      {n:<20s} {v * 100:+5.1f}  {why}")
    L.append("-" * width)
    L.append("  LEVERAGE BOARD")
    L.append(f"  {'':<4} {'PHASE':<26} {'OFF':>6} {'DEF':>6} {'EXP':>6} {'SCORE':>7}  GRADE")
    for l in m.top_exploits(len(AXES) * 2):
        L.append(f"  {l.attacker:<4} {l.axis.label:<26} "
                 f"{l.off_metric.adjusted:>6.1f} {l.def_metric.adjusted:>6.1f} "
                 f"{l.exposure:>6.3f} {l.score:>+7.2f}  {l.grade}")
    L.append("-" * width)
    L.append("  OPEN QUESTIONS (last season vs this season disagree most)")
    for t, unit, mm in m.open_questions(5):
        L.append(f"  {t.abbr:<4} {unit:<8} {mm.label:<22} {int(mm.prior):>3} -> {int(mm.current):>3}"
                 f"   blended {mm.adjusted:>5.1f}")
    L.append("-" * width)
    L.append("  INJURIES")
    for t in m.teams:
        for i in t.injuries:
            L.append(f"  {t.abbr:<4} {i.player:<22} {i.pos:<4} {i.status.upper():<13} {i.injury}")
    L.append(bar)
    L.append("  analysis only, not advice  |  1-800-GAMBLER")
    return "\n".join(L)
