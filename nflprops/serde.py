"""JSON <-> dataclass conversion for game state.

Deliberately tolerant on input: every field has a sensible default, so a
minimal snapshot (scores, clock, and a few players with target shares) runs,
while a fully specified one uses everything. Missing data degrades the model's
precision, it should never stop it from producing a number.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List

from .models import (
    Alignment,
    DefenseProfile,
    GameState,
    InjuryStatus,
    Officials,
    Player,
    PlayerH1,
    Position,
    Precip,
    TeamState,
    Weather,
)


# --------------------------------------------------------------------------
# Deserialization
# --------------------------------------------------------------------------


def _enum(cls, value, default):
    if value is None:
        return default
    if isinstance(value, cls):
        return value
    try:
        return cls(str(value).lower())
    except ValueError:
        try:
            return cls(str(value).upper())
        except ValueError:
            return default


def player_from_dict(d: Dict[str, Any], team_abbr: str) -> Player:
    h1_raw = d.get("h1") or d.get("first_half") or {}
    h1 = PlayerH1(**{k: v for k, v in h1_raw.items() if k in PlayerH1.__annotations__})

    pos = _enum(Position, d.get("position"), Position.WR)
    return Player(
        player_id=str(d.get("player_id") or d.get("id") or d["name"]),
        name=str(d.get("name", d.get("player_id", "unknown"))),
        position=pos,
        team=str(d.get("team", team_abbr)),
        target_share=float(d.get("target_share", 0.0)),
        rush_share=float(d.get("rush_share", 0.0)),
        rz_target_share=float(d.get("rz_target_share", 0.0)),
        rz_rush_share=float(d.get("rz_rush_share", 0.0)),
        snap_share=float(d.get("snap_share", 1.0)),
        adot=float(d.get("adot", 8.0)),
        catch_rate=float(d.get("catch_rate", 0.65)),
        yac=float(d.get("yac", 4.0)),
        ypc=float(d.get("ypc", 4.3)),
        explosiveness=float(d.get("explosiveness", 0.35)),
        alignment=_enum(Alignment, d.get("alignment"), None),
        injury=_enum(InjuryStatus, d.get("injury"), InjuryStatus.HEALTHY),
        star=bool(d.get("star", False)),
        is_starter_qb=bool(d.get("is_starter_qb", pos is Position.QB)),
        sack_rate=float(d.get("sack_rate", 0.065)),
        int_rate=float(d.get("int_rate", 0.023)),
        scramble_rate=float(d.get("scramble_rate", 0.045)),
        h1=h1,
    )


def defense_from_dict(d: Dict[str, Any]) -> DefenseProfile:
    d = d or {}
    return DefenseProfile(
        pass_yds_mult=float(d.get("pass_yds_mult", 1.0)),
        rush_yds_mult=float(d.get("rush_yds_mult", 1.0)),
        comp_rate_mult=float(d.get("comp_rate_mult", 1.0)),
        sack_rate_mult=float(d.get("sack_rate_mult", 1.0)),
        int_rate_mult=float(d.get("int_rate_mult", 1.0)),
        td_rate_mult=float(d.get("td_rate_mult", 1.0)),
        slot_funnel=float(d.get("slot_funnel", 1.0)),
        perimeter_funnel=float(d.get("perimeter_funnel", 1.0)),
        deep_funnel=float(d.get("deep_funnel", 1.0)),
        rb_target_funnel=float(d.get("rb_target_funnel", 1.0)),
        shadow={str(k): float(v) for k, v in (d.get("shadow") or {}).items()},
    )


def team_from_dict(d: Dict[str, Any]) -> TeamState:
    abbr = str(d["abbr"])
    team = TeamState(
        abbr=abbr,
        name=str(d.get("name", abbr)),
        score=int(d.get("score", 0)),
        base_pass_rate=float(d.get("base_pass_rate", 0.575)),
        base_sec_per_play=float(d.get("base_sec_per_play", 35.0)),
        no_huddle_rate=float(d.get("no_huddle_rate", 0.08)),
        off_pass_eff=float(d.get("off_pass_eff", 1.0)),
        off_rush_eff=float(d.get("off_rush_eff", 1.0)),
        defense=defense_from_dict(d.get("defense")),
        timeouts=int(d.get("timeouts", 3)),
    )
    team.players = [player_from_dict(p, abbr) for p in d.get("players", [])]
    team.h1_plays = int(d.get("h1_plays", 0))
    team.h1_pass_att = int(d.get("h1_pass_att", 0))
    return team


def weather_from_dict(d: Dict[str, Any]) -> Weather:
    d = d or {}
    return Weather(
        wind_mph=float(d.get("wind_mph", 0.0)),
        temp_f=float(d.get("temp_f", 65.0)),
        precip=_enum(Precip, d.get("precip"), Precip.NONE),
        dome=bool(d.get("dome", False)),
    )


def officials_from_dict(d: Dict[str, Any]) -> Officials:
    d = d or {}
    return Officials(
        crew=str(d.get("crew", "league_average")),
        penalties_per_game=float(d.get("penalties_per_game", 12.8)),
        offensive_holding_rate=float(d.get("offensive_holding_rate", 1.0)),
        dpi_rate=float(d.get("dpi_rate", 1.0)),
    )


def game_from_dict(d: Dict[str, Any]) -> GameState:
    home = team_from_dict(d["home"])
    away = team_from_dict(d["away"])
    return GameState(
        game_id=str(d.get("game_id", "game")),
        home=home,
        away=away,
        quarter=int(d.get("quarter", 3)),
        seconds_remaining=int(d.get("seconds_remaining", 1800)),
        possession=str(d.get("possession", away.abbr)),
        weather=weather_from_dict(d.get("weather")),
        officials=officials_from_dict(d.get("officials")),
        pregame_total=d.get("pregame_total"),
        pregame_spread=d.get("pregame_spread"),
        neutral_site=bool(d.get("neutral_site", False)),
    )


def load_game(path: str) -> GameState:
    with open(path, "r", encoding="utf-8") as fh:
        return game_from_dict(json.load(fh))


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def game_to_dict(state: GameState) -> Dict[str, Any]:
    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if hasattr(obj, "value"):  # str-backed enums
            return obj.value
        return obj

    return clean(asdict(state))


def dump_game(state: GameState, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(game_to_dict(state), fh, indent=2)
