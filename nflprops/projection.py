"""Turn a matchup into a stat line: score, team totals, and every player.

:mod:`nflprops.matchup` decides *who has leverage and how each team will play*.
This module spends that conclusion, converting it into the numbers you actually
want before kickoff - a projected score, passing and rushing yards for each
side, a full quarterback line, and carries, targets, receptions, yards and a
touchdown probability for every listed player.

The chain is deliberately short enough to check by hand:

    points   <- market implied total and spread, tilted by net phase leverage
    plays    <- pace
    dropbacks / carries  <- the projected pass rate from the script model
    sacks    <- dropbacks x a sack rate set by pass rush against protection
    attempts <- dropbacks - sacks - scrambles
    yards    <- attempts x yards per attempt, matchup-adjusted
    players  <- team totals split by usage share, then priced by efficiency

Two choices are worth defending.

**The market is the anchor for the score, not the model.** A units-against-units
score is a worse predictor than the closing line, because the line already
prices quarterback play, home field and everything else the leverage board
deliberately excludes. So the projection starts at the implied score and moves
it by a bounded tilt. A model that swings a projected total by ten points off
its own rank table is not being bold, it is being wrong loudly.

**Touchdowns are a rate, not a yes/no.** Expected touchdowns are allocated by
red-zone-weighted usage share, then converted with a Poisson tail,
``P(>=1) = 1 - exp(-expected)``. That is why a back projected for 0.8 expected
scores reads as a 55% chance and not an 80% one: scoring twice is a real
outcome and it has to come out of the same budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .matchup import Matchup, Team

# --------------------------------------------------------------------------
# League baselines. Everything is expressed as a move away from these.
# --------------------------------------------------------------------------

BASE_SACK_RATE = 0.068          # of dropbacks
BASE_YPA = 7.1
BASE_COMP = 0.655
BASE_INT_RATE = 0.023           # of attempts
BASE_YPC = 4.3
BASE_CATCH_INTERCEPT = 0.80     # catch rate at a 0-yard aDOT
BASE_CATCH_SLOPE = 0.0165       # lost per yard of aDOT
BASE_YAC = 4.5
POINTS_PER_TD_DRIVE = 9.3       # points per touchdown once field goals are priced in

#: How hard a matchup edge moves each rate. Small on purpose: these are
#: season-long ranks applied to one game.
K_YPA = 0.22
K_COMP = 0.10
K_YPC = 0.26
K_INT = 0.55
K_SACK = 0.50

#: Bound on how far net leverage may move a team off its implied score.
MAX_SCORE_TILT = 3.0
POINTS_PER_LEVERAGE = 0.12


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ordinal(n: float) -> str:
    """4 -> '4th', 32 -> '32nd'. Shared by both report surfaces."""
    i = int(round(n))
    suffix = "th" if 10 <= i % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suffix}"


# --------------------------------------------------------------------------
# Output shapes
# --------------------------------------------------------------------------


@dataclass
class QBLine:
    name: str
    dropbacks: float
    #: Quarterback carries that came out of a called pass, as opposed to
    #: designed runs. ``dropbacks == attempts + sacks + scrambles``.
    scrambles: float
    attempts: float
    completions: float
    pass_yards: float
    pass_tds: float
    interceptions: float
    sacks: float
    sack_yards: float
    rush_attempts: float
    rush_yards: float
    rush_tds: float
    td_probability: float          # anytime rushing touchdown

    @property
    def completion_pct(self) -> float:
        return self.completions / self.attempts if self.attempts else 0.0

    @property
    def yards_per_attempt(self) -> float:
        return self.pass_yards / self.attempts if self.attempts else 0.0

    @property
    def total_yards(self) -> float:
        return self.pass_yards + self.rush_yards


@dataclass
class SkillLine:
    name: str
    pos: str
    carries: float
    rush_yards: float
    targets: float
    receptions: float
    rec_yards: float
    expected_tds: float
    td_probability: float
    note: str = ""

    @property
    def total_yards(self) -> float:
        return self.rush_yards + self.rec_yards

    @property
    def touches(self) -> float:
        return self.carries + self.receptions


@dataclass
class TeamProjection:
    team: str
    points: float
    market_points: float
    tilt: float
    plays: float
    pass_rate: float
    dropbacks: float
    attempts: float
    completions: float
    pass_yards: float
    sacks: float
    carries: float
    rush_yards: float
    interceptions: float
    expected_tds: float
    pass_tds: float
    rush_tds: float
    qb: QBLine
    skill: List[SkillLine]

    @property
    def total_yards(self) -> float:
        return self.pass_yards + self.rush_yards

    @property
    def yards_per_play(self) -> float:
        return self.total_yards / self.plays if self.plays else 0.0


@dataclass
class GameProjection:
    home: TeamProjection
    away: TeamProjection
    matchup: Matchup

    @property
    def teams(self) -> List[TeamProjection]:
        return [self.away, self.home]

    def for_team(self, abbr: str) -> TeamProjection:
        return self.home if self.home.team == abbr else self.away

    @property
    def total(self) -> float:
        return self.home.points + self.away.points

    @property
    def margin(self) -> float:
        return self.home.points - self.away.points


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _catch_rate(adot: float) -> float:
    """Deeper targets are caught less often. Linear is close enough here."""
    return _clamp(BASE_CATCH_INTERCEPT - BASE_CATCH_SLOPE * max(0.0, adot), 0.38, 0.82)


def _yards_per_reception(player: Dict[str, Any]) -> float:
    if player.get("ypr"):
        return float(player["ypr"])
    adot = float(player.get("adot") or 8.0)
    yac = float(player.get("yac") or BASE_YAC)
    return 0.75 * adot + yac


def _weather_factor(weather: Dict[str, Any]) -> Tuple[float, float]:
    """Return (passing multiplier, rushing multiplier)."""
    if not weather:
        return 1.0, 1.0
    wind = float(weather.get("wind_mph") or 0)
    precip = float(weather.get("precip_chance") or 0)
    pen = 0.0
    if wind > 12:
        pen += 0.006 * (wind - 12)
    if precip >= 0.5:
        pen += 0.035 * precip
    return _clamp(1.0 - pen, 0.85, 1.0), _clamp(1.0 + pen * 0.35, 1.0, 1.06)


def _poisson_at_least_one(expected: float) -> float:
    return 1.0 - math.exp(-max(0.0, expected))


def _normalised(weights: List[float]) -> List[float]:
    total = sum(weights)
    if total <= 0:
        return [0.0] * len(weights)
    return [w / total for w in weights]


# --------------------------------------------------------------------------
# The projection
# --------------------------------------------------------------------------


def _project_team(m: Matchup, team: Team, opp: Team) -> TeamProjection:
    script = m.scripts[team.abbr]
    pass_mult, rush_mult = _weather_factor(m.weather)

    # --- score ------------------------------------------------------------
    market = script.projected_points
    tilt = _clamp(
        (m.net_leverage(team.abbr) - m.net_leverage(opp.abbr)) * POINTS_PER_LEVERAGE,
        -MAX_SCORE_TILT, MAX_SCORE_TILT,
    )
    points = max(3.0, market + tilt)

    # --- volume -----------------------------------------------------------
    plays = script.projected_plays
    dropbacks = plays * script.projected_pass_rate
    carries_total = plays - dropbacks

    qb_raw = team.qb_profile or {}
    qb_carries = float(qb_raw.get("rush_attempts") or 3.5)
    # A designed quarterback run is a called run, not a dropback that broke
    # down. Only scrambles come out of the passing game; designed keepers come
    # out of the carry pool, the same place a handoff would. Charging both to
    # dropbacks costs a mobile quarterback several attempts he would actually
    # throw, which is exactly wrong for the offenses that run him on purpose.
    designed_share = _clamp(float(qb_raw.get("designed_run_share", 0.35)), 0.0, 1.0)
    qb_designed = qb_carries * designed_share
    qb_scrambles = qb_carries - qb_designed

    # --- sacks ------------------------------------------------------------
    protect = team.offense["pass_protect"].strength
    rush_def = opp.defense["pressure"].strength
    base_sack = float(qb_raw.get("sack_rate") or BASE_SACK_RATE)
    sack_rate = _clamp(base_sack * (1 + K_SACK * (rush_def - protect)), 0.025, 0.14)
    sacks = dropbacks * sack_rate

    attempts = max(10.0, dropbacks - sacks - qb_scrambles)
    carries_total = max(0.0, carries_total - qb_designed)

    # --- passing efficiency ----------------------------------------------
    off_pass = team.offense["pass"].strength
    def_pass = opp.defense["pass"].strength
    edge_pass = off_pass - def_pass

    ypa = float(qb_raw.get("ypa") or BASE_YPA) * (1 + K_YPA * edge_pass) * pass_mult
    ypa = _clamp(ypa, 4.8, 10.5)
    comp = float(qb_raw.get("completion_pct") or BASE_COMP) * (1 + K_COMP * edge_pass)
    comp = _clamp(comp * (pass_mult ** 0.5), 0.50, 0.75)

    pass_yards = attempts * ypa
    completions = attempts * comp

    # --- rushing efficiency ----------------------------------------------
    off_rush = team.offense["rush"].strength
    def_rush = opp.defense["rush"].strength
    team_ypc = BASE_YPC * (1 + K_YPC * (off_rush - def_rush)) * rush_mult
    team_ypc = _clamp(team_ypc, 3.1, 5.8)

    # --- turnovers --------------------------------------------------------
    ball_sec = team.offense["ball_security"].strength
    takeaway = opp.defense["takeaways"].strength
    int_rate = float(qb_raw.get("int_rate") or BASE_INT_RATE)
    int_rate = _clamp(int_rate * (1 + K_INT * (takeaway - ball_sec)), 0.008, 0.055)
    interceptions = attempts * int_rate

    # --- touchdown budget -------------------------------------------------
    expected_tds = points / POINTS_PER_TD_DRIVE
    # Teams score through the air roughly in proportion to how they play, but
    # the goal line is always more run-heavy than the field between the 20s.
    pass_td_share = _clamp(0.30 + 0.55 * script.projected_pass_rate, 0.42, 0.72)
    pass_tds = expected_tds * pass_td_share
    rush_tds = expected_tds - pass_tds

    # --- allocate to players ---------------------------------------------
    players = [p for p in team.players if (p.get("pos") or "").upper() != "QB"]

    rush_weights = _normalised([float(p.get("rush_share") or 0.0) for p in players])
    tgt_weights = _normalised([float(p.get("target_share") or 0.0) for p in players])

    # Backs take what is left of the carry pool once designed keepers are out.
    rb_carries = max(0.0, carries_total)
    team_attempts = attempts

    rz_weights = _normalised([
        float(p.get("target_share") or 0.0) * float(p.get("rz_factor") or 1.0)
        for p in players
    ])
    gl_weights = _normalised([
        float(p.get("rush_share") or 0.0) * float(p.get("gl_factor") or 1.0)
        for p in players
    ])

    # The quarterback keeps a slice of the rushing scores he runs in himself.
    qb_rush_td_share = float(qb_raw.get("rush_td_share") or 0.16)
    qb_rush_tds = rush_tds * qb_rush_td_share
    skill_rush_tds = rush_tds - qb_rush_tds

    skill: List[SkillLine] = []
    for p, rw, tw, rzw, glw in zip(players, rush_weights, tgt_weights, rz_weights, gl_weights):
        carries = rb_carries * rw
        ypc = float(p.get("ypc") or team_ypc)
        # A back's own efficiency is pulled toward what this matchup supports.
        ypc = 0.55 * ypc + 0.45 * team_ypc
        rush_yards = carries * ypc

        targets = team_attempts * tw
        receptions = targets * _catch_rate(float(p.get("adot") or 8.0))
        rec_yards = receptions * _yards_per_reception(p)

        exp_td = pass_tds * rzw + skill_rush_tds * glw
        skill.append(SkillLine(
            name=p.get("name", "?"),
            pos=p.get("pos", ""),
            carries=carries,
            rush_yards=rush_yards,
            targets=targets,
            receptions=receptions,
            rec_yards=rec_yards,
            expected_tds=exp_td,
            td_probability=_poisson_at_least_one(exp_td),
            note=p.get("detail") or p.get("note") or "",
        ))

    # Rushing yards reconcile: backs plus the quarterback.
    qb_rush_yards = qb_carries * float(qb_raw.get("rush_ypc") or 4.6)
    rush_yards_total = sum(s.rush_yards for s in skill) + qb_rush_yards

    qb = QBLine(
        name=team.qb or (qb_raw.get("name") or "Quarterback"),
        dropbacks=dropbacks,
        scrambles=qb_scrambles,
        attempts=attempts,
        completions=completions,
        pass_yards=pass_yards,
        pass_tds=pass_tds,
        interceptions=interceptions,
        sacks=sacks,
        sack_yards=sacks * 6.8,
        rush_attempts=qb_carries,
        rush_yards=qb_rush_yards,
        rush_tds=qb_rush_tds,
        td_probability=_poisson_at_least_one(qb_rush_tds),
    )

    return TeamProjection(
        team=team.abbr,
        points=points,
        market_points=market,
        tilt=tilt,
        plays=plays,
        pass_rate=script.projected_pass_rate,
        dropbacks=dropbacks,
        attempts=attempts,
        completions=completions,
        pass_yards=pass_yards,
        sacks=sacks,
        carries=rb_carries + qb_carries,
        rush_yards=rush_yards_total,
        interceptions=interceptions,
        expected_tds=expected_tds,
        pass_tds=pass_tds,
        rush_tds=rush_tds,
        qb=qb,
        skill=sorted(skill, key=lambda s: -s.total_yards),
    )


def project(m: Matchup) -> GameProjection:
    return GameProjection(
        home=_project_team(m, m.home, m.away),
        away=_project_team(m, m.away, m.home),
        matchup=m,
    )


# --------------------------------------------------------------------------
# Game script, in words
# --------------------------------------------------------------------------


def game_script(g: GameProjection) -> str:
    """One paragraph: how this game is most likely to be played."""
    m = g.matchup
    home, away = g.home, g.away
    lead, trail = (home, away) if home.points >= away.points else (away, home)
    lead_t, trail_t = m.team(lead.team), m.team(trail.team)

    top = m.top_exploits(1)[0]
    q = m.open_questions(1)
    caveat = ""
    if q:
        t, unit, mm = q[0]
        caveat = (f" The number to distrust is {t.abbr}'s {mm.label.lower()}: "
                  f"{ordinal(mm.prior)} last season, {ordinal(mm.current)} this one, "
                  f"blended to {ordinal(mm.adjusted)}. If the current season is the real "
                  f"team, this projection is wrong in that direction.")

    weather = ""
    w = m.weather or {}
    if float(w.get("precip_chance") or 0) >= 0.5 or float(w.get("wind_mph") or 0) >= 14:
        weather = (f" Conditions take a little off both passing games: "
                   f"{w.get('wind_mph')} mph wind and a "
                   f"{float(w.get('precip_chance') or 0) * 100:.0f}% chance of rain.")

    return (
        f"{lead_t.name} {lead.points:.0f}, {trail_t.name} {trail.points:.0f}. "
        f"{lead_t.abbr} throws on {lead.pass_rate:.0%} of snaps and {trail_t.abbr} on "
        f"{trail.pass_rate:.0%}, which is the whole story: the favourite does not need "
        f"volume and the underdog cannot avoid it. "
        f"The clearest edge on the field is {top.attacker} attacking {top.defender} on "
        f"{top.axis.label.lower()}. "
        f"Expect {lead.total_yards:.0f} total yards from {lead_t.abbr} and "
        f"{trail.total_yards:.0f} from {trail_t.abbr}, with "
        f"{(home.interceptions + away.interceptions):.1f} interceptions and "
        f"{(home.sacks + away.sacks):.1f} sacks between them."
        f"{weather}{caveat}"
    )
