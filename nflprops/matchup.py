"""Pre-game matchup analysis: where two teams' strengths meet the other's holes.

The prop board answers "what is likely to happen in the second half of a game
already in progress." This module answers the question that comes before it:
*given two teams and nothing but a kickoff time, who has leverage over whom,
and what will each team therefore try to do?*

The whole model is three ideas.

**One. A rank is a blend, not a number.**
In Week 2 a current-season rank is one game of evidence and last year's rank is
seventeen. Neither is the answer on its own. Every metric carries a ``prior``
(last completed season) and a ``current`` (this season to date), and
:func:`blend_rank` combines them by credibility:

    w_current = g / (g + k)

with ``k`` set per metric family - how many games that statistic needs before
it means anything. Turnover rate needs a lot (k=8); yards per carry needs few
(k=3). The weight the current season does not claim does **not** all go to last
year: it is split between last year and league average by the unit's
``carryover``, which is how much of last season's team still exists. A team
that changed coordinator and lead back should not be described by last year's
film, and shrinking the difference toward rank 16.5 says "we don't know yet"
instead of inventing certainty in either direction.

**Two. Leverage is an offense's strength minus the defensive strength in front
of it, weighted by how often that phase will actually occur.**
A huge edge in the red zone is worth less than a modest edge on early downs
because there are twelve red zone snaps and a hundred and twenty others. So
each axis carries an ``exposure``, and exposure is *not* static - it is scaled
by the projected script. If a team is going to throw on 63% of snaps, its
receivers' edge matters more and its runners' edge matters less, in that game.

**Three. The script is predictable from the matchup itself.**
Teams do not call plays in a vacuum. :func:`project_script` starts at a team's
neutral pass rate and moves it for the four things that actually move it: the
expected margin (trailing teams throw), the specific weakness across the ball
(you run at a bad run defense), the weather, and the rest differential. That
projected pass rate then feeds back into exposure - which is the point. The
question "will they throw more or run more" is not a separate section of the
report; it is the input that decides which mismatches matter.

Nothing here simulates anything. It is a transparent weighting of stated
inputs, and every number it prints can be traced back to a field in the game
file by hand. That is deliberate: a matchup report that cannot be argued with
is not useful to someone who watches the games.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Middle of a 32-team league. Unknown regresses here, never to zero.
LEAGUE_MID = 16.5

#: Games of current-season evidence before a metric family is worth half its
#: own weight. Bigger k == noisier statistic == trust last season longer.
STABILIZE_K = {
    "scoring": 5.0,
    "pass": 4.0,
    "rush": 3.0,
    "red_zone": 7.0,
    "third_down": 6.0,
    "explosive": 6.0,
    "ball_security": 8.0,
    "takeaways": 8.0,
    "pass_protect": 4.0,
    "pressure": 4.0,
    "tackling": 5.0,
}
DEFAULT_K = 5.0

#: How much a designation costs the unit it belongs to, as a fraction of the
#: injury's stated ``impact``.
STATUS_SEVERITY = {
    "out": 1.0,
    "doubtful": 0.65,
    "questionable": 0.30,
    "limited": 0.50,
    "probable": 0.08,
    "healthy": 0.0,
}

#: An injury of impact 1.0 and axis weight 1.0 that is ruled OUT moves that
#: axis this many rank spots. A starter lost outright is worth roughly a
#: quarter of the league on the one thing he most directly controls - and
#: much less on everything else, which is what the per-axis weights are for.
INJURY_RANK_SWING = 8.0

CONFIDENCE = {"measured": 1.0, "forecast": 0.85, "estimate": 0.72}

#: League-average neutral pass rate, the anchor exposure is scaled against.
NEUTRAL_PASS_RATE = 0.57


# --------------------------------------------------------------------------
# Axes - the pairings that actually decide football games
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    """One offensive phase and the defensive phase that answers it."""

    key: str
    label: str
    off_key: str
    def_key: str
    exposure: float
    #: "pass", "rush" or "neutral" - which way projected pass rate moves it.
    family: str
    question: str


AXES: Tuple[Axis, ...] = (
    Axis("pass", "Dropback passing", "pass", "pass", 0.22, "pass",
         "Can they complete the ball against this secondary?"),
    Axis("rush", "Running the ball", "rush", "rush", 0.19, "rush",
         "Can they run it when the defense knows it is coming?"),
    Axis("protection", "Protection vs pass rush", "pass_protect", "pressure", 0.16, "pass",
         "Does the quarterback get time?"),
    Axis("explosive", "Explosive plays", "explosive", "pass", 0.12, "pass",
         "Who wins the chunk plays that decide one-score games?"),
    Axis("third_down", "Third down", "third_down", "third_down", 0.12, "neutral",
         "Who stays on the field?"),
    Axis("red_zone", "Red zone", "red_zone", "red_zone", 0.12, "neutral",
         "Touchdowns or field goals?"),
    Axis("turnovers", "Ball security vs takeaways", "ball_security", "takeaways", 0.07, "neutral",
         "Who gives the game away?"),
)


# --------------------------------------------------------------------------
# Rank arithmetic
# --------------------------------------------------------------------------


def strength(rank: float) -> float:
    """Rank 1..32 -> strength 1.0..0.0. Rank 1 is best, so it scores highest."""
    return max(0.0, min(1.0, (32.0 - float(rank)) / 31.0))


def blend_rank(
    prior: Optional[float],
    current: Optional[float],
    games: int,
    k: float = DEFAULT_K,
    carryover: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Credibility-weighted blend of last season and this season.

    Returns the blended rank and the three weights that produced it, so the
    report can show its work rather than asserting a number.
    """
    if prior is None and current is None:
        return LEAGUE_MID, {"current": 0.0, "prior": 0.0, "league": 1.0}
    if prior is None:
        return float(current), {"current": 1.0, "prior": 0.0, "league": 0.0}
    if current is None:
        # No current-season evidence: last year, pulled toward average by how
        # much of last year's team is gone.
        c = max(0.0, min(1.0, carryover))
        return c * float(prior) + (1 - c) * LEAGUE_MID, {
            "current": 0.0, "prior": c, "league": 1 - c,
        }

    g = max(0, int(games))
    w_cur = g / (g + k) if (g + k) > 0 else 0.0
    rest = 1.0 - w_cur
    c = max(0.0, min(1.0, carryover))
    w_prior = rest * c
    w_league = rest * (1 - c)

    blended = w_cur * float(current) + w_prior * float(prior) + w_league * LEAGUE_MID
    return blended, {"current": w_cur, "prior": w_prior, "league": w_league}


# --------------------------------------------------------------------------
# Resolved metrics
# --------------------------------------------------------------------------


@dataclass
class Metric:
    key: str
    label: str
    prior: Optional[float]
    current: Optional[float]
    blended: float
    #: After injury degradation.
    adjusted: float
    weights: Dict[str, float]
    confidence: float
    detail: str = ""
    prior_src: str = "estimate"
    current_src: str = "estimate"
    injury_shift: float = 0.0
    injury_causes: List[str] = field(default_factory=list)
    carryover: float = 1.0
    carryover_note: str = ""

    @property
    def strength(self) -> float:
        return strength(self.adjusted)

    @property
    def measured(self) -> bool:
        return self.prior_src == "measured" or self.current_src == "measured"

    @property
    def disagreement(self) -> float:
        """0..1 - how violently last season and this season contradict.

        This is the most honest number the model produces. A metric where the
        two seasons agree is a fact; a metric where they disagree by twenty
        rank spots is an open question that the blend papers over with an
        average nobody believes. Those are exactly the axes a game turns on,
        so the report surfaces them rather than hiding them inside a mean.
        """
        if self.prior is None or self.current is None:
            return 0.0
        return min(1.0, abs(float(self.prior) - float(self.current)) / 31.0)

    @property
    def trend(self) -> int:
        """-1 improving (rank fell), +1 declining, 0 flat."""
        if self.prior is None or self.current is None:
            return 0
        d = float(self.current) - float(self.prior)
        return 0 if abs(d) < 3 else (1 if d > 0 else -1)


@dataclass
class Injury:
    player: str
    pos: str
    status: str
    injury: str
    units: List[str]
    #: axis key -> how much of this player's absence lands on that axis.
    #: ``["pass_protect", "pass"]`` in the file is read as weight 1.0 each;
    #: ``{"pass_protect": 1.0, "pass": 0.4}`` says the left tackle wrecks
    #: protection and only bleeds through to the passing game from there.
    axes: Dict[str, float]
    impact: float
    note: str = ""

    @property
    def severity(self) -> float:
        return STATUS_SEVERITY.get(self.status.lower(), 0.0)

    @property
    def cost(self) -> float:
        """0..1 - how much of a full starter's worth is actually missing."""
        return max(0.0, min(1.0, self.impact)) * self.severity


@dataclass
class Team:
    abbr: str
    name: str
    record: str
    home: bool
    color: str
    color_dark: str
    coach: str
    play_caller: str
    dc: str
    qb: str
    last_game: str
    rest: Dict[str, Any]
    offense: Dict[str, Metric]
    defense: Dict[str, Metric]
    tendencies: Dict[str, Dict[str, Any]]
    players: List[Dict[str, Any]]
    injuries: List[Injury]
    carryover: Dict[str, Any]

    def tendency(self, key: str, games: int) -> float:
        """Blend a tendency the same way a rank is blended, on its own scale."""
        t = self.tendencies.get(key)
        if not t:
            return 0.0
        prior, current = t.get("prior"), t.get("current")
        if prior is None:
            return float(current or 0.0)
        if current is None:
            return float(prior)
        w = games / (games + 4.0)
        return w * float(current) + (1 - w) * float(prior)


# --------------------------------------------------------------------------
# Script projection
# --------------------------------------------------------------------------


@dataclass
class Script:
    team: str
    neutral_pass_rate: float
    projected_pass_rate: float
    projected_plays: float
    projected_points: float
    adjustments: List[Tuple[str, float, str]]

    @property
    def projected_run_rate(self) -> float:
        return 1.0 - self.projected_pass_rate

    @property
    def pass_attempts(self) -> float:
        return self.projected_plays * self.projected_pass_rate

    @property
    def rush_attempts(self) -> float:
        return self.projected_plays * self.projected_run_rate


def _weather_pass_penalty(weather: Dict[str, Any]) -> Tuple[float, str]:
    """Wind and rain push a play caller toward the run. Modestly."""
    if not weather:
        return 0.0, ""
    wind = float(weather.get("wind_mph") or 0.0)
    precip = float(weather.get("precip_chance") or 0.0)
    pen, why = 0.0, []
    if wind > 12:
        pen += 0.004 * (wind - 12)
        why.append(f"{wind:.0f} mph wind")
    if precip >= 0.5:
        pen += 0.030 * precip
        why.append(f"{precip * 100:.0f}% chance of rain")
    if float(weather.get("temp_f") or 60) < 30:
        pen += 0.02
        why.append("cold")
    return pen, ", ".join(why)


def project_script(
    team: Team,
    opponent: Team,
    *,
    games: int,
    expected_margin: float,
    total: float,
    weather: Dict[str, Any],
    rest_edge: float,
) -> Script:
    """Predict how a team will choose to play this specific game.

    ``expected_margin`` is positive when this team is favoured. Every
    adjustment is returned with its size and its reason so the report can
    print the arithmetic instead of the conclusion.
    """
    base = team.tendency("neutral_pass_rate", games) or NEUTRAL_PASS_RATE
    adjustments: List[Tuple[str, float, str]] = []
    rate = base

    # Script. Trailing teams throw; leading teams sit on it.
    script = -0.0080 * expected_margin
    if abs(script) >= 0.002:
        side = "favoured" if expected_margin > 0 else "an underdog"
        adjustments.append((
            "Game script", script,
            f"Priced as {side} by {abs(expected_margin):.1f}. "
            f"{'Leading teams run out the clock' if expected_margin > 0 else 'Trailing teams throw to catch up'}.",
        ))
        rate += script

    # Matchup. Attack the softer of the two defensive phases: a strong run
    # defence pushes you to throw, a strong pass defence pushes you to run.
    d_pass = opponent.defense["pass"].strength
    d_rush = opponent.defense["rush"].strength
    match = 0.16 * (d_rush - d_pass)
    if abs(match) >= 0.004:
        softer = "pass defense" if d_pass < d_rush else "run defense"
        adjustments.append((
            "Opponent's soft spot", match,
            f"{opponent.abbr}'s {softer} is the weaker unit "
            f"(run {opponent.defense['rush'].adjusted:.0f} vs pass "
            f"{opponent.defense['pass'].adjusted:.0f} in blended rank), "
            f"so {team.abbr} should attack it.",
        ))
        rate += match

    # Weather.
    pen, why = _weather_pass_penalty(weather)
    if pen:
        adjustments.append(("Weather", -pen, f"{why.capitalize()}."))
        rate -= pen

    # Rest. A tired team leans on the simpler plan.
    if abs(rest_edge) >= 1:
        rest_adj = -0.010 * rest_edge if rest_edge < 0 else 0.0
        if rest_adj:
            adjustments.append((
                "Short week", rest_adj,
                f"{abs(rest_edge):.0f} fewer days than the opponent. "
                "Thin practice weeks favour the run game and the base install.",
            ))
            rate += rest_adj

    # Protection. A team that cannot block cannot drop back all night.
    prot = team.offense["pass_protect"].strength
    rush_d = opponent.defense["pressure"].strength
    if rush_d - prot > 0.25:
        adj = -0.05 * (rush_d - prot)
        adjustments.append((
            "Protection risk", adj,
            f"{opponent.abbr}'s rush grades well ahead of {team.abbr}'s protection. "
            "Play callers shorten the dropback menu rather than feed the rush.",
        ))
        rate += adj

    rate = max(0.34, min(0.74, rate))

    pace = team.tendency("seconds_per_play", games) or 28.0
    plays = 125.0 * (28.0 / max(20.0, pace)) / 2.0
    plays = max(52.0, min(72.0, plays + (2.0 if expected_margin < 0 else -1.0)))

    points = total / 2.0 + expected_margin / 2.0

    return Script(
        team=team.abbr,
        neutral_pass_rate=base,
        projected_pass_rate=rate,
        projected_plays=plays,
        projected_points=points,
        adjustments=adjustments,
    )


# --------------------------------------------------------------------------
# Leverage
# --------------------------------------------------------------------------


@dataclass
class Leverage:
    axis: Axis
    attacker: str
    defender: str
    off_metric: Metric
    def_metric: Metric
    edge: float
    exposure: float
    confidence: float
    score: float

    @property
    def direction(self) -> str:
        return "offense" if self.edge >= 0 else "defense"

    @property
    def grade(self) -> str:
        a = abs(self.score)
        if a >= 5.0:
            return "decisive"
        if a >= 3.0:
            return "clear"
        if a >= 1.5:
            return "slight"
        return "even"


def _exposure_for(axis: Axis, pass_rate: float) -> float:
    """Scale an axis by how much of this game it will actually cover."""
    if axis.family == "pass":
        scale = pass_rate / NEUTRAL_PASS_RATE
    elif axis.family == "rush":
        scale = (1 - pass_rate) / (1 - NEUTRAL_PASS_RATE)
    else:
        scale = 1.0
    return axis.exposure * max(0.55, min(1.55, scale))


def leverage_board(
    attacker: Team, defender: Team, pass_rate: float
) -> List[Leverage]:
    """Every offensive phase of ``attacker`` against ``defender``'s answer."""
    out: List[Leverage] = []
    for axis in AXES:
        om = attacker.offense.get(axis.off_key)
        dm = defender.defense.get(axis.def_key)
        if om is None or dm is None:
            continue
        edge = om.strength - dm.strength
        exposure = _exposure_for(axis, pass_rate)
        conf = (om.confidence + dm.confidence) / 2.0
        out.append(
            Leverage(
                axis=axis,
                attacker=attacker.abbr,
                defender=defender.abbr,
                off_metric=om,
                def_metric=dm,
                edge=edge,
                exposure=exposure,
                confidence=conf,
                score=edge * exposure * conf * 100.0,
            )
        )
    return sorted(out, key=lambda l: -l.score)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _metric(key: str, raw: Dict[str, Any], games: int, carryover: float) -> Metric:
    k = STABILIZE_K.get(key, DEFAULT_K)
    prior = raw.get("prior")
    current = raw.get("current")
    # A unit-level carryover is the default, but continuity is rarely uniform
    # across a unit: a team can return its whole passing game and replace its
    # entire backfield. A per-metric override says so.
    carry = float(raw.get("carryover", carryover))
    blended, weights = blend_rank(prior, current, games, k=k, carryover=carry)
    p_src = raw.get("prior_src", "estimate")
    c_src = raw.get("current_src", "estimate")
    # Confidence follows the evidence that actually carries weight.
    conf = (
        weights["current"] * CONFIDENCE.get(c_src, 0.72)
        + weights["prior"] * CONFIDENCE.get(p_src, 0.72)
        + weights["league"] * 0.5
    )
    return Metric(
        key=key,
        label=raw.get("label", key.replace("_", " ").title()),
        prior=prior,
        current=current,
        blended=blended,
        adjusted=blended,
        weights=weights,
        confidence=max(0.4, min(1.0, conf)),
        detail=raw.get("detail", ""),
        prior_src=p_src,
        current_src=c_src,
        carryover=carry,
        carryover_note=raw.get("carryover_note", ""),
    )


def _axis_weights(raw: Any) -> Dict[str, float]:
    """Accept either ``["pass"]`` or ``{"pass": 0.4}`` in the game file."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    return {str(k): 1.0 for k in raw}


def _apply_injuries(team: Team) -> None:
    """Degrade the specific axes a missing player actually touches.

    A rank is not a property of a jersey - it is a property of the eleven men
    who will be on the field. A left tackle who is out does not make the whole
    offense worse by a constant; he makes *pass protection* worse, which makes
    *passing* worse, and leaves the run game alone.
    """
    for inj in team.injuries:
        cost = inj.cost
        if cost <= 0:
            continue
        shift = cost * INJURY_RANK_SWING
        for unit_name in inj.units:
            unit = team.offense if unit_name == "offense" else team.defense
            for axis_key, axis_weight in inj.axes.items():
                m = unit.get(axis_key)
                if m is None:
                    continue
                local = shift * float(axis_weight)
                if local <= 0:
                    continue
                m.adjusted = max(1.0, min(32.0, m.adjusted + local))
                m.injury_shift += local
                m.injury_causes.append(f"{inj.player} ({inj.status})")


def _team_from_dict(abbr: str, d: Dict[str, Any], games: int) -> Team:
    carry = d.get("carryover", {}) or {}
    off_carry = float(carry.get("offense", 1.0))
    def_carry = float(carry.get("defense", 1.0))

    offense = {
        k: _metric(k, v, games, off_carry) for k, v in (d.get("offense") or {}).items()
    }
    defense = {
        k: _metric(k, v, games, def_carry) for k, v in (d.get("defense") or {}).items()
    }

    injuries = [
        Injury(
            player=i.get("player", "?"),
            pos=i.get("pos", ""),
            status=str(i.get("status", "healthy")).lower(),
            injury=i.get("injury", ""),
            units=list(i.get("units") or []),
            axes=_axis_weights(i.get("axes")),
            impact=float(i.get("impact", 0.0)),
            note=i.get("note", ""),
        )
        for i in (d.get("injuries") or [])
    ]

    team = Team(
        abbr=abbr,
        name=d.get("name", abbr),
        record=d.get("record", ""),
        home=bool(d.get("home", False)),
        color=d.get("color", "#333333"),
        color_dark=d.get("color_dark", d.get("color", "#8899bb")),
        coach=d.get("coach", ""),
        play_caller=d.get("play_caller", ""),
        dc=d.get("dc", ""),
        qb=d.get("qb", ""),
        last_game=d.get("last_game", ""),
        rest=d.get("rest", {}) or {},
        offense=offense,
        defense=defense,
        tendencies=d.get("tendencies", {}) or {},
        players=list(d.get("players") or []),
        injuries=injuries,
        carryover=carry,
    )
    _apply_injuries(team)
    return team


# --------------------------------------------------------------------------
# The whole analysis
# --------------------------------------------------------------------------


@dataclass
class SituationFlag:
    label: str
    team: str
    weight: str  # "high" | "medium" | "low"
    body: str


@dataclass
class Matchup:
    meta: Dict[str, Any]
    weather: Dict[str, Any]
    home: Team
    away: Team
    scripts: Dict[str, Script]
    boards: Dict[str, List[Leverage]]
    flags: List[SituationFlag]
    narrative: List[Dict[str, Any]]
    games: int

    @property
    def teams(self) -> List[Team]:
        return [self.away, self.home]

    def team(self, abbr: str) -> Team:
        return self.home if self.home.abbr == abbr else self.away

    def opponent(self, abbr: str) -> Team:
        return self.away if self.home.abbr == abbr else self.home

    def net_leverage(self, abbr: str) -> float:
        return sum(l.score for l in self.boards[abbr])

    def top_exploits(self, n: int = 6) -> List[Leverage]:
        every = [l for b in self.boards.values() for l in b]
        return sorted(every, key=lambda l: -l.score)[:n]

    def open_questions(self, n: int = 5, floor: float = 0.30) -> List[Tuple[Team, str, Metric]]:
        """The metrics where last season and this season flatly disagree.

        These are the report's own caveats, stated up front instead of buried:
        the blend has produced a number for each of them, and that number is
        the least trustworthy thing on the page.

        Ordered by disagreement *weighted by confidence*, which matters more
        than it sounds. Two published figures that contradict each other are a
        real question about a real team. Two of the author's own estimates that
        contradict each other are mostly a question about the estimates, and
        promoting that to the top of the page would be dressing up the model's
        own noise as a finding.
        """
        out: List[Tuple[Team, str, Metric]] = []
        for t in self.teams:
            for unit_name, unit in (("offense", t.offense), ("defense", t.defense)):
                for m in unit.values():
                    if m.disagreement >= floor:
                        out.append((t, unit_name, m))
        return sorted(out, key=lambda r: -(r[2].disagreement * r[2].confidence))[:n]


def _situation_flags(
    home: Team, away: Team, weather: Dict[str, Any], meta: Dict[str, Any]
) -> List[SituationFlag]:
    flags: List[SituationFlag] = []

    for t, other in ((home, away), (away, home)):
        rest = t.rest or {}
        days = rest.get("days")
        other_days = (other.rest or {}).get("days")
        if rest.get("short_week"):
            delta = ""
            if days is not None and other_days is not None:
                delta = f" - {other_days - days} fewer day(s) than {other.abbr}"
            flags.append(SituationFlag(
                "Short week", t.abbr, "medium",
                f"{days} days between games{delta}. {rest.get('note', '')}".strip(),
            ))
        miles = float(rest.get("travel_miles") or 0)
        if miles >= 1500:
            flags.append(SituationFlag(
                "Long travel", t.abbr, "medium",
                f"{miles:.0f} miles, {abs(int(rest.get('tz_shift') or 0))} time zone(s). "
                f"{rest.get('note', '')}".strip(),
            ))
        elif miles > 0:
            flags.append(SituationFlag(
                "Travel", t.abbr, "low",
                f"{miles:.0f} miles. {rest.get('note', '')}".strip(),
            ))

    if weather:
        wind = float(weather.get("wind_mph") or 0)
        precip = float(weather.get("precip_chance") or 0)
        weight = "high" if (wind >= 18 or precip >= 0.7) else "medium" if (wind >= 12 or precip >= 0.4) else "low"
        flags.append(SituationFlag(
            "Weather", "both", weight,
            f"{weather.get('forecast', '')}. {weather.get('temp_f', '?')}F, "
            f"wind {weather.get('wind_mph', '?')} mph"
            + (f" gusting {weather.get('gust_mph')}" if weather.get("gust_mph") else "")
            + f", {precip * 100:.0f}% precipitation. {weather.get('note', '')}",
        ))

    return flags


def load_matchup(path: str) -> Matchup:
    with open(path, "r", encoding="utf-8") as fh:
        return build_matchup(json.load(fh))


def build_matchup(raw: Dict[str, Any]) -> Matchup:
    meta = raw.get("meta", {}) or {}
    games = int(meta.get("games_played", 0))
    weather = raw.get("weather", {}) or {}

    teams_raw = raw.get("teams", {}) or {}
    if len(teams_raw) != 2:
        raise ValueError("a matchup file needs exactly two teams")

    teams = {k: _team_from_dict(k, v, games) for k, v in teams_raw.items()}
    home = next((t for t in teams.values() if t.home), None)
    away = next((t for t in teams.values() if not t.home), None)
    if home is None or away is None:
        vals = list(teams.values())
        away, home = vals[0], vals[1]

    market = meta.get("market", {}) or {}
    total = float(market.get("total") or 44.0)
    spread = float(market.get("spread") or 0.0)
    fav = market.get("favorite")
    margin = {home.abbr: 0.0, away.abbr: 0.0}
    if fav in margin:
        margin[fav] = abs(spread)
        other = home.abbr if fav == away.abbr else away.abbr
        margin[other] = -abs(spread)

    rest_days = {
        home.abbr: float((home.rest or {}).get("days") or 7),
        away.abbr: float((away.rest or {}).get("days") or 7),
    }

    scripts: Dict[str, Script] = {}
    for t, o in ((home, away), (away, home)):
        scripts[t.abbr] = project_script(
            t, o,
            games=games,
            expected_margin=margin[t.abbr],
            total=total,
            weather=weather,
            rest_edge=rest_days[t.abbr] - rest_days[o.abbr],
        )

    boards = {
        home.abbr: leverage_board(home, away, scripts[home.abbr].projected_pass_rate),
        away.abbr: leverage_board(away, home, scripts[away.abbr].projected_pass_rate),
    }

    return Matchup(
        meta=meta,
        weather=weather,
        home=home,
        away=away,
        scripts=scripts,
        boards=boards,
        flags=_situation_flags(home, away, weather, meta),
        narrative=list(raw.get("narrative_flags") or []),
        games=games,
    )
