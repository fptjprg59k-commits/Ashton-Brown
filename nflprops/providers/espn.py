"""Live game state from ESPN's public scoreboard/summary JSON.

This supplies the half of the input that is not controversial to fetch: score,
clock, possession, and the first-half box score. It does **not** supply usage
priors (season-long target and carry shares), which are what separate a good
projection from a crude one - when no priors are provided, first-half usage is
used as its own prior, and the report says so.

Written defensively against a feed that is public but unversioned: every field
access tolerates absence, and a partial parse yields a usable state rather than
an exception.

Note: this module was authored against ESPN's documented response shape but
could not be exercised against the live endpoint in the environment it was
built in (outbound access to sports APIs was blocked there). Run
``nflprops doctor --espn`` on a machine with network access to confirm the
mapping before trusting it in-play.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from ..models import (
    GameState,
    InjuryStatus,
    Player,
    PlayerH1,
    Position,
    Precip,
    TeamState,
    Weather,
)
from .base import GameStateProvider, ProviderError

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

_POSITION_MAP = {
    "QB": Position.QB, "RB": Position.RB, "FB": Position.RB,
    "WR": Position.WR, "TE": Position.TE, "PK": Position.K, "K": Position.K,
}


class ESPNProvider(GameStateProvider):
    name = "espn"

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    # -- http --------------------------------------------------------------

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "nflprops/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"ESPN request failed ({url}): {exc}") from exc

    # -- public ------------------------------------------------------------

    def list_games_at_halftime(self) -> List[str]:
        data = self._get(SCOREBOARD)
        out = []
        for event in data.get("events", []):
            status = _dig(event, "status", "type", "name") or ""
            if "HALFTIME" in str(status).upper():
                out.append(str(event.get("id")))
        return out

    def scoreboard(self) -> List[Dict[str, Any]]:
        """Light summary of every game on the slate, for selection UIs."""
        data = self._get(SCOREBOARD)
        rows = []
        for event in data.get("events", []):
            comp = (event.get("competitions") or [{}])[0]
            teams = {}
            for c in comp.get("competitors", []):
                teams[c.get("homeAway", "?")] = {
                    "abbr": _dig(c, "team", "abbreviation"),
                    "score": _int(c.get("score")),
                }
            rows.append({
                "game_id": str(event.get("id")),
                "name": event.get("shortName"),
                "status": _dig(event, "status", "type", "name"),
                "period": _dig(event, "status", "period"),
                "clock": _dig(event, "status", "displayClock"),
                "home": teams.get("home", {}),
                "away": teams.get("away", {}),
            })
        return rows

    def fetch(self, game_id: str) -> GameState:
        data = self._get(SUMMARY, {"event": game_id})
        header = data.get("header") or {}
        comp = (header.get("competitions") or [{}])[0]

        home_raw = away_raw = None
        for c in comp.get("competitors", []):
            if c.get("homeAway") == "home":
                home_raw = c
            else:
                away_raw = c
        if home_raw is None or away_raw is None:
            raise ProviderError(f"could not identify teams for event {game_id}")

        home = TeamState(
            abbr=str(_dig(home_raw, "team", "abbreviation") or "HOME"),
            name=str(_dig(home_raw, "team", "displayName") or ""),
            score=_int(home_raw.get("score")),
        )
        away = TeamState(
            abbr=str(_dig(away_raw, "team", "abbreviation") or "AWAY"),
            name=str(_dig(away_raw, "team", "displayName") or ""),
            score=_int(away_raw.get("score")),
        )

        rosters = self._parse_boxscore(data)
        home.players = rosters.get(home.abbr, [])
        away.players = rosters.get(away.abbr, [])
        for team in (home, away):
            _derive_shares_from_first_half(team)

        period = _int(_dig(comp, "status", "period")) or 2
        seconds = _seconds_remaining(period, _dig(comp, "status", "displayClock"))

        state = GameState(
            game_id=str(game_id),
            home=home,
            away=away,
            quarter=max(3, period + 1) if seconds >= 1800 else max(period, 3),
            seconds_remaining=seconds,
            possession=away.abbr,  # receiving team after the half; override if known
            weather=_parse_weather(data),
        )
        _apply_injuries(state, data)
        return state

    # -- parsing -----------------------------------------------------------

    def _parse_boxscore(self, data: Dict) -> Dict[str, List[Player]]:
        out: Dict[str, List[Player]] = {}
        for team_block in _dig(data, "boxscore", "players") or []:
            abbr = str(_dig(team_block, "team", "abbreviation") or "?")
            players: Dict[str, Player] = {}

            for group in team_block.get("statistics") or []:
                kind = str(group.get("name", "")).lower()
                labels = [str(x).upper() for x in (group.get("labels") or [])]
                for ath in group.get("athletes") or []:
                    pid = str(_dig(ath, "athlete", "id") or "")
                    if not pid:
                        continue
                    player = players.get(pid)
                    if player is None:
                        pos = str(_dig(ath, "athlete", "position", "abbreviation") or "")
                        player = Player(
                            player_id=pid,
                            name=str(_dig(ath, "athlete", "displayName") or pid),
                            position=_POSITION_MAP.get(pos.upper(), Position.WR),
                            team=abbr,
                            h1=PlayerH1(),
                        )
                        players[pid] = player
                    _fill_stats(player, kind, labels, ath.get("stats") or [])

            out[abbr] = list(players.values())
        return out


def _fill_stats(player: Player, kind: str, labels: List[str], stats: List[str]) -> None:
    values = {labels[i]: stats[i] for i in range(min(len(labels), len(stats)))}
    h = player.h1

    if kind.startswith("passing"):
        comp_att = values.get("C/ATT", "0/0")
        if "/" in comp_att:
            c, _, a = comp_att.partition("/")
            h.pass_cmp, h.pass_att = _int(c), _int(a)
        h.pass_yds = _float(values.get("YDS"))
        h.pass_td = _int(values.get("TD"))
        h.interceptions = _int(values.get("INT"))
        if h.pass_att > 0:
            player.position = Position.QB
            player.is_starter_qb = True
    elif kind.startswith("rushing"):
        h.rush_att = _int(values.get("CAR"))
        h.rush_yds = _float(values.get("YDS"))
        h.rush_td = _int(values.get("TD"))
        h.longest_rush = _float(values.get("LONG"))
    elif kind.startswith("receiving"):
        h.rec = _int(values.get("REC"))
        h.rec_yds = _float(values.get("YDS"))
        h.rec_td = _int(values.get("TD"))
        h.longest_rec = _float(values.get("LONG"))
        h.targets = _int(values.get("TGTS")) or h.rec
    elif kind.startswith("kicking"):
        fg = values.get("FG", "0/0")
        if "/" in fg:
            h.fg_made = _int(fg.partition("/")[0])
        xp = values.get("XP", "0/0")
        if "/" in xp:
            h.xp_made = _int(xp.partition("/")[0])
        player.position = Position.K


def _derive_shares_from_first_half(team: TeamState) -> None:
    """Bootstrap usage shares when no season-long priors were supplied.

    Half a game is a poor prior, so this is a fallback, not a feature. Supply
    real target and carry shares whenever you have them.
    """
    total_targets = sum(p.h1.targets for p in team.players) or 1
    total_carries = sum(p.h1.rush_att for p in team.players if p.position is not Position.QB) or 1

    for p in team.players:
        if p.position is Position.QB:
            continue
        if p.target_share == 0.0:
            p.target_share = p.h1.targets / total_targets
        if p.rush_share == 0.0:
            p.rush_share = p.h1.rush_att / total_carries
        if p.h1.rec > 0 and p.h1.rec_yds:
            ypr = p.h1.rec_yds / p.h1.rec
            p.adot = max(2.0, min(18.0, ypr * 0.55))
            p.yac = max(1.5, ypr - p.adot * 0.92)
        if p.h1.rush_att >= 3:
            p.ypc = max(2.0, min(7.0, p.h1.rush_yds / p.h1.rush_att))
        p.rz_target_share = p.target_share
        p.rz_rush_share = p.rush_share
        p.star = p.target_share >= 0.22 or p.rush_share >= 0.45


def _parse_weather(data: Dict) -> Weather:
    w = _dig(data, "gameInfo", "weather") or {}
    display = str(w.get("displayValue", "")).lower()
    precip = Precip.NONE
    if "snow" in display:
        precip = Precip.SNOW
    elif "heavy rain" in display or "thunder" in display:
        precip = Precip.HEAVY_RAIN
    elif "rain" in display or "shower" in display or "drizzle" in display:
        precip = Precip.LIGHT_RAIN

    indoor = bool(_dig(data, "gameInfo", "venue", "indoor"))
    return Weather(
        wind_mph=_float(w.get("windSpeed")),
        temp_f=_float(w.get("temperature")) or 65.0,
        precip=precip,
        dome=indoor,
    )


def _apply_injuries(state: GameState, data: Dict) -> None:
    """Map ESPN's injury block onto player designations where names match."""
    from ..matching import PlayerResolver

    resolver = PlayerResolver(state.all_players())
    mapping = {
        "out": InjuryStatus.OUT,
        "doubtful": InjuryStatus.DOUBTFUL,
        "questionable": InjuryStatus.QUESTIONABLE,
        "probable": InjuryStatus.PROBABLE,
    }
    for block in data.get("injuries") or []:
        for item in block.get("injuries") or []:
            name = _dig(item, "athlete", "displayName")
            status = str(item.get("status", "")).lower()
            if not name or status not in mapping:
                continue
            pid = resolver.resolve(str(name))
            if pid:
                player = state.find_player(pid)
                if player is not None:
                    player.injury = mapping[status]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _dig(obj: Any, *keys: str) -> Any:
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _int(value: Any) -> int:
    try:
        return int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def _seconds_remaining(period: int, display_clock: Optional[str]) -> int:
    """Seconds left in regulation from period and displayed clock."""
    if period <= 2:
        return 1800  # halftime or earlier: a full second half remains
    mm, ss = 15, 0
    if display_clock and ":" in str(display_clock):
        parts = str(display_clock).split(":")
        mm, ss = _int(parts[0]), _int(parts[1])
    quarters_left_after = max(0, 4 - period)
    return max(0, quarters_left_after * 900 + mm * 60 + ss)
