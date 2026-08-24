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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..dashboard import Dashboard, GameCard, TeamLine
from ..status import GameStatus, SeasonType, derive_status
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

#: ESPN's integer encoding for the season type query parameter.
_SEASON_TYPE_PARAM = {
    SeasonType.PRESEASON: 1,
    SeasonType.REGULAR: 2,
    SeasonType.POSTSEASON: 3,
    SeasonType.OFFSEASON: 1,
}

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

    def _scoreboard_payload(
        self,
        season_type: Optional[SeasonType] = None,
        week: Optional[int] = None,
        date: Optional[str] = None,
        year: Optional[int] = None,
    ) -> Dict:
        """Fetch the slate, optionally pinned to a season type and week.

        Preseason is the reason these parameters exist. The bare scoreboard
        endpoint returns "the current slate", which during August can come back
        empty or fall through to regular-season week 1 depending on where the
        league is in its rollover. Asking for ``seasontype=1`` explicitly is
        what makes preseason games reliably appear, and the dashboard treats
        them like any other game because the status system does not care which
        part of the calendar a game sits in.
        """
        params: Dict[str, Any] = {}
        if season_type is not None:
            params["seasontype"] = _SEASON_TYPE_PARAM[season_type]
        if week is not None:
            params["week"] = int(week)
        if year is not None:
            params["year"] = int(year)
        if date:
            params["dates"] = date
        return self._get(SCOREBOARD, params or None)

    def list_games_at_halftime(self, **kwargs) -> List[str]:
        return [row["game_id"] for row in self.scoreboard(**kwargs) if row["at_halftime"]]

    def scoreboard(self, **kwargs) -> List[Dict[str, Any]]:
        """Light summary of every game on the slate, for selection UIs."""
        data = self._scoreboard_payload(**kwargs)
        rows = []
        for event in data.get("events", []):
            comp = (event.get("competitions") or [{}])[0]
            teams = {}
            for c in comp.get("competitors", []):
                teams[c.get("homeAway", "?")] = {
                    "abbr": _dig(c, "team", "abbreviation"),
                    "score": _int(c.get("score")),
                }
            status = derive_status(
                _dig(event, "status", "type", "name"),
                period=_dig(event, "status", "period"),
                clock=_dig(event, "status", "displayClock"),
                completed=bool(_dig(event, "status", "type", "completed")),
            )
            rows.append({
                "game_id": str(event.get("id")),
                "name": event.get("shortName"),
                "status": status.value,
                "at_halftime": status.is_actionable,
                "period": _dig(event, "status", "period"),
                "clock": _dig(event, "status", "displayClock"),
                "home": teams.get("home", {}),
                "away": teams.get("away", {}),
            })
        return rows

    # -- dashboard ---------------------------------------------------------

    def dashboard(
        self,
        season_type: Optional[SeasonType] = None,
        week: Optional[int] = None,
        date: Optional[str] = None,
        year: Optional[int] = None,
    ) -> Dashboard:
        """Build a full dashboard from one scoreboard call."""
        data = self._scoreboard_payload(
            season_type=season_type, week=week, date=date, year=year
        )
        return dashboard_from_payload(data, season_type=season_type, week=week)

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


# --------------------------------------------------------------------------
# Scoreboard event -> dashboard card
# --------------------------------------------------------------------------


def _card_from_event(event: Dict[str, Any]) -> Optional[GameCard]:
    """Map one scoreboard event onto a dashboard card.

    Tolerant by design: a slate where one game has a malformed competitor
    should still render the other fifteen, so an unusable event is dropped
    rather than raising.
    """
    comp = (event.get("competitions") or [{}])[0]
    home = away = None
    for c in comp.get("competitors") or []:
        line = TeamLine(
            abbr=str(_dig(c, "team", "abbreviation") or "?"),
            name=str(
                _dig(c, "team", "shortDisplayName")
                or _dig(c, "team", "displayName")
                or _dig(c, "team", "abbreviation")
                or "?"
            ),
            score=_int(c.get("score")) if c.get("score") not in (None, "") else None,
            record=_first_record(c),
            logo_url=_dig(c, "team", "logo"),
        )
        if c.get("homeAway") == "home":
            home = line
        elif c.get("homeAway") == "away":
            away = line

    if home is None or away is None:
        return None

    period = _int(_dig(event, "status", "period")) or None
    status = derive_status(
        _dig(event, "status", "type", "name"),
        period=period,
        clock=_dig(event, "status", "displayClock"),
        completed=bool(_dig(event, "status", "type", "completed")),
    )

    if status is GameStatus.SCHEDULED:
        home.score = away.score = None

    return GameCard(
        game_id=str(event.get("id")),
        home=home,
        away=away,
        status=status,
        period=period,
        clock=_dig(event, "status", "displayClock"),
        start_time=_parse_iso(event.get("date")),
        season_type=SeasonType.from_espn(_dig(event, "season", "type")),
        week=_int(_dig(event, "week", "number")) or None,
        broadcast=_broadcast(comp),
        venue=_dig(comp, "venue", "fullName"),
        situation=_dig(comp, "situation", "downDistanceText"),
        odds=_odds(comp),
        went_to_overtime=bool(period and period > 4),
    )


def _first_record(competitor: Dict[str, Any]) -> Optional[str]:
    for rec in competitor.get("records") or []:
        summary = rec.get("summary")
        if summary:
            return str(summary)
    return None


def _broadcast(comp: Dict[str, Any]) -> Optional[str]:
    for b in comp.get("broadcasts") or []:
        names = b.get("names") or []
        if names:
            return str(names[0])
    return None


def _odds(comp: Dict[str, Any]) -> Optional[str]:
    for o in comp.get("odds") or []:
        details = o.get("details")
        total = o.get("overUnder")
        if details and total:
            return f"{details} · O/U {total}"
        if details:
            return str(details)
    return None


def _parse_iso(value: Any) -> Optional[datetime]:
    """ESPN stamps kickoff in UTC with a trailing Z; render it locally."""
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone()


def dashboard_from_payload(
    data: Dict[str, Any],
    season_type: Optional[SeasonType] = None,
    week: Optional[int] = None,
) -> Dashboard:
    """Build a dashboard from a scoreboard payload.

    Shared by the live provider and by any saved snapshot, so an offline
    fixture exercises exactly the code path that runs in production rather
    than a parallel one that can drift.
    """
    cards = [
        card
        for card in (_card_from_event(ev) for ev in data.get("events", []))
        if card is not None
    ]

    resolved = season_type or SeasonType.from_espn(_dig(data, "season", "type"))
    wk = week or _int(_dig(data, "week", "number")) or None

    label = {
        SeasonType.PRESEASON: "Preseason",
        SeasonType.POSTSEASON: "Postseason",
    }.get(resolved, "")
    subtitle = " · ".join(p for p in (label, f"Week {wk}" if wk else "") if p)

    return Dashboard(
        games=cards,
        title="Today's Games",
        subtitle=subtitle or None,
        generated_at=datetime.now(),
    )


def load_dashboard(path: str) -> Dashboard:
    """Read a saved scoreboard snapshot from disk."""
    with open(path, "r", encoding="utf-8") as fh:
        return dashboard_from_payload(json.load(fh))
