"""Core data model for halftime prop evaluation.

Everything the simulator needs is expressed here as plain dataclasses so that
state can be built from a live feed, a JSON fixture, or by hand in a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class Position(str, Enum):
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DEF = "DEF"


class InjuryStatus(str, Enum):
    """In-game or pre-existing designation.

    ``availability`` is the multiplier applied to a player's snap share; a
    player who is OUT is removed from the usage pool entirely and his share is
    redistributed to teammates.
    """

    HEALTHY = "healthy"
    PROBABLE = "probable"
    QUESTIONABLE = "questionable"
    LIMITED = "limited"          # in-game: playing but visibly hobbled
    DOUBTFUL = "doubtful"
    OUT = "out"

    @property
    def availability(self) -> float:
        return {
            "healthy": 1.0,
            "probable": 0.98,
            "questionable": 0.88,
            "limited": 0.65,
            "doubtful": 0.35,
            "out": 0.0,
        }[self.value]

    @property
    def efficiency(self) -> float:
        """Separate hit to per-touch effectiveness, not just volume."""
        return {
            "healthy": 1.0,
            "probable": 1.0,
            "questionable": 0.97,
            "limited": 0.88,
            "doubtful": 0.85,
            "out": 0.0,
        }[self.value]


class Alignment(str, Enum):
    """Where a pass catcher primarily lines up.

    Needed because defensive scheme funnels are alignment-specific: a two-high
    shell that concedes underneath pushes volume to the slot and the backfield
    while erasing perimeter deep shots, and "opponent pass defense rank" is far
    too blunt an instrument to capture that.
    """

    PERIMETER = "perimeter"
    SLOT = "slot"
    INLINE = "inline"        # in-line tight end
    BACKFIELD = "backfield"


class Precip(str, Enum):
    NONE = "none"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    SNOW = "snow"


class PropScope(str, Enum):
    """Whether a line settles on the whole game or only the remaining half.

    This is the single most common source of error when pricing props at the
    break: most books keep full-game markets live, but also post second-half
    derivatives. A full-game line must be compared against first-half actuals
    plus the simulated remainder.
    """

    FULL_GAME = "full_game"
    SECOND_HALF = "second_half"


class Side(str, Enum):
    OVER = "over"
    UNDER = "under"
    YES = "yes"          # anytime TD and similar one-way markets
    NO = "no"


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


@dataclass
class Weather:
    wind_mph: float = 0.0
    temp_f: float = 65.0
    precip: Precip = Precip.NONE
    dome: bool = False

    @property
    def effective_wind(self) -> float:
        return 0.0 if self.dome else self.wind_mph

    @property
    def effective_precip(self) -> Precip:
        return Precip.NONE if self.dome else self.precip


@dataclass
class Officials:
    """Crew tendencies. Effect sizes here are small by design.

    Referee influence on player props is real but second-order: an extra
    offensive holding call per game moves expected plays by well under one.
    Modelled as a drive-extension nudge rather than a headline factor.
    """

    crew: str = "league_average"
    penalties_per_game: float = 12.8      # league average, both teams
    offensive_holding_rate: float = 1.0   # multiplier vs league average
    dpi_rate: float = 1.0                 # multiplier vs league average


# --------------------------------------------------------------------------
# Players and teams
# --------------------------------------------------------------------------


@dataclass
class PlayerH1:
    """First-half box score. Added to simulated H2 output for full-game props."""

    pass_att: int = 0
    pass_cmp: int = 0
    pass_yds: float = 0.0
    pass_td: int = 0
    interceptions: int = 0
    rush_att: int = 0
    rush_yds: float = 0.0
    rush_td: int = 0
    targets: int = 0
    rec: int = 0
    rec_yds: float = 0.0
    rec_td: int = 0
    longest_rec: float = 0.0
    longest_rush: float = 0.0
    fg_made: int = 0
    xp_made: int = 0

    @property
    def kicking_points(self) -> float:
        return 3.0 * self.fg_made + 1.0 * self.xp_made


@dataclass
class Player:
    player_id: str
    name: str
    position: Position
    team: str

    # --- usage: the baseline floor for any prop -----------------------------
    target_share: float = 0.0        # share of team pass attempts drawing a target
    rush_share: float = 0.0          # share of team designed rushes
    rz_target_share: float = 0.0     # inside-20 target share (TD props)
    rz_rush_share: float = 0.0       # inside-20 carry share (TD props)
    snap_share: float = 1.0          # baseline snap rate when healthy

    # --- efficiency profile -------------------------------------------------
    adot: float = 8.0                # average depth of target
    catch_rate: float = 0.65         # on-target catch rate at this aDOT
    yac: float = 4.0                 # yards after catch per reception
    ypc: float = 4.3                 # yards per carry (rushers)

    # Explosiveness index in [0, 1]. 0 = pure possession/short-area profile,
    # 1 = boom-or-bust deep threat. Controls the *shape* of the per-touch
    # yardage distribution without changing its mean, which is what separates
    # a receiver who needs 8 catches from one who needs a single 60-yarder.
    explosiveness: float = 0.35

    alignment: Optional[Alignment] = None   # defaults from position

    # --- availability -------------------------------------------------------
    injury: InjuryStatus = InjuryStatus.HEALTHY
    star: bool = False               # eligible to be rested in a blowout

    # --- QB-specific --------------------------------------------------------
    is_starter_qb: bool = False
    sack_rate: float = 0.065
    int_rate: float = 0.023
    scramble_rate: float = 0.045     # dropbacks converted to QB runs

    h1: PlayerH1 = field(default_factory=PlayerH1)

    def __post_init__(self) -> None:
        if self.alignment is None:
            self.alignment = {
                Position.WR: Alignment.PERIMETER,
                Position.TE: Alignment.INLINE,
                Position.RB: Alignment.BACKFIELD,
            }.get(self.position, Alignment.PERIMETER)

    @property
    def available(self) -> bool:
        return self.injury is not InjuryStatus.OUT

    @property
    def expected_ypr(self) -> float:
        """Mean yards per reception implied by the aDOT/YAC profile."""
        return max(1.0, self.adot * 0.92 + self.yac)


@dataclass
class DefenseProfile:
    """Opponent adjustments as multipliers on offensive rates.

    Values are centred on 1.0 = league average. Above 1.0 means the offense
    does *better* than baseline against this unit, so a shutdown pass defense
    has ``pass_yds_mult < 1.0``.
    """

    pass_yds_mult: float = 1.0
    rush_yds_mult: float = 1.0
    comp_rate_mult: float = 1.0
    sack_rate_mult: float = 1.0
    int_rate_mult: float = 1.0
    td_rate_mult: float = 1.0

    # Scheme funnels: where the defense concedes volume. Applied as
    # multipliers on target share by alignment, then renormalised. A defense
    # that plays two-high and concedes underneath throws funnels targets to
    # the slot and backs rather than to outside deep threats.
    slot_funnel: float = 1.0
    perimeter_funnel: float = 1.0
    deep_funnel: float = 1.0
    rb_target_funnel: float = 1.0

    # Per-player shadow coverage: player_id -> multiplier on that player's
    # target share and efficiency (a true shutdown corner travelling with WR1).
    shadow: Dict[str, float] = field(default_factory=dict)


@dataclass
class TeamState:
    abbr: str
    name: str = ""
    score: int = 0
    players: List[Player] = field(default_factory=list)

    # --- pace and tendency --------------------------------------------------
    base_pass_rate: float = 0.575       # neutral-script pass rate

    # Game-clock seconds burned per snap when the clock runs after the play.
    # NOT the published "seconds between snaps" pace stat (~27s), which measures
    # snap-to-snap tempo only. This is the pre-discount figure: the engine
    # separately credits in-play stoppages (incompletions, out of bounds) and
    # bills dead time between drives (punts, kickoffs, scores) on its own, so
    # neither may be baked in here. Calibrated so a neutral game reproduces the
    # league's ~63 plays and ~11.5 drives per team.
    base_sec_per_play: float = 31.0
    no_huddle_rate: float = 0.08

    # --- offensive quality (multipliers vs league average) ------------------
    off_pass_eff: float = 1.0
    off_rush_eff: float = 1.0

    defense: DefenseProfile = field(default_factory=DefenseProfile)

    timeouts: int = 3

    # First-half team totals, used to shrink observed efficiency toward prior.
    h1_plays: int = 0
    h1_pass_att: int = 0

    def player(self, player_id: str) -> Optional[Player]:
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None

    def kicker(self) -> Optional[Player]:
        for p in self.players:
            if p.position is Position.K:
                return p
        return None

    def starting_qb(self) -> Optional[Player]:
        qbs = [p for p in self.players if p.position is Position.QB and p.available]
        if not qbs:
            return None
        for p in qbs:
            if p.is_starter_qb:
                return p
        return qbs[0]


# --------------------------------------------------------------------------
# Game state
# --------------------------------------------------------------------------


@dataclass
class GameState:
    """Snapshot of the game at the moment of evaluation (normally halftime)."""

    game_id: str
    home: TeamState
    away: TeamState

    quarter: int = 3                 # period that is about to begin
    seconds_remaining: int = 1800    # 30:00 left at the half
    possession: str = ""             # team abbr receiving the ball next

    weather: Weather = field(default_factory=Weather)
    officials: Officials = field(default_factory=Officials)

    # Pregame market context. The closing total/spread is the single best
    # public estimate of scoring environment and is used to sanity-check the
    # simulated pace rather than to drive it.
    pregame_total: Optional[float] = None
    pregame_spread: Optional[float] = None   # negative = home favoured

    neutral_site: bool = False

    def team(self, abbr: str) -> TeamState:
        if self.home.abbr == abbr:
            return self.home
        if self.away.abbr == abbr:
            return self.away
        raise KeyError(f"unknown team {abbr!r} in game {self.game_id}")

    def opponent(self, abbr: str) -> TeamState:
        return self.away if self.home.abbr == abbr else self.home

    def find_player(self, player_id: str) -> Optional[Player]:
        return self.home.player(player_id) or self.away.player(player_id)

    def all_players(self) -> List[Player]:
        return list(self.home.players) + list(self.away.players)

    @property
    def margin_home(self) -> int:
        return self.home.score - self.away.score


# --------------------------------------------------------------------------
# Props
# --------------------------------------------------------------------------


class Market(str, Enum):
    PASS_YDS = "pass_yds"
    PASS_TDS = "pass_tds"
    PASS_CMP = "pass_cmp"
    PASS_ATT = "pass_att"
    PASS_INT = "pass_int"
    RUSH_YDS = "rush_yds"
    RUSH_ATT = "rush_att"
    RUSH_TDS = "rush_tds"
    REC = "rec"
    REC_YDS = "rec_yds"
    REC_TDS = "rec_tds"
    RUSH_REC_YDS = "rush_rec_yds"
    PASS_RUSH_YDS = "pass_rush_yds"
    LONGEST_REC = "longest_rec"
    LONGEST_RUSH = "longest_rush"
    ANYTIME_TD = "anytime_td"
    KICKING_POINTS = "kicking_points"
    TEAM_TOTAL = "team_total"
    GAME_TOTAL = "game_total"


#: Markets that resolve on a team rather than an individual player.
TEAM_MARKETS = {Market.TEAM_TOTAL}
GAME_MARKETS = {Market.GAME_TOTAL}

#: Markets that are inherently one-way (no over/under, just yes/no).
BINARY_MARKETS = {Market.ANYTIME_TD}


@dataclass
class Prop:
    """A single offered line, as posted by the book."""

    prop_id: str
    market: Market
    side: Side
    line: float
    odds: int                        # American odds as posted
    scope: PropScope = PropScope.FULL_GAME

    player_id: Optional[str] = None
    team: Optional[str] = None       # for team/game markets
    player_name: str = ""
    book: str = "stake"

    #: Opposite side's odds when both are visible, enabling proper vig removal.
    opposing_odds: Optional[int] = None

    def __post_init__(self) -> None:
        if self.market in BINARY_MARKETS and self.side in (Side.OVER, Side.UNDER):
            # "Anytime TD over 0.5" is how some feeds express it; normalise.
            self.side = Side.YES if self.side is Side.OVER else Side.NO

    @property
    def label(self) -> str:
        who = self.player_name or self.player_id or self.team or "?"
        if self.market in BINARY_MARKETS:
            return f"{who} {self.market.value.replace('_', ' ')}"
        return f"{who} {self.side.value} {self.line:g} {self.market.value.replace('_', ' ')}"
