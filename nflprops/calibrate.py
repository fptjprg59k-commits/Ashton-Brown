"""Calibration harness.

A model whose per-drive and per-play behaviour does not reproduce league
averages will misprice every line on the board, no matter how good its
game-script logic is. This module builds a deliberately neutral, league-average
matchup and reports the engine's output against published NFL rates, so drift
is visible before it reaches a probability.

Run it with ``nflprops calibrate``. Treat a red row as a bug in the priors, not
a rounding artefact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import Alignment, GameState, Player, Position, TeamState, Weather
from .priors import Priors
from .simulate import GameSimulator

#: League-average per-game targets, both teams combined unless noted.
#: Sourced from public NFL season aggregates; update alongside the priors.
LEAGUE_BASELINES: Dict[str, Tuple[float, float]] = {
    # metric: (target, tolerance)
    "points_total": (44.0, 4.0),
    "plays_per_team": (63.0, 5.0),
    "drives_per_team": (11.5, 1.2),
    "points_per_drive": (1.90, 0.25),
    "pass_yds_per_team": (230.0, 25.0),
    "rush_yds_per_team": (115.0, 18.0),
    "yards_per_play": (5.45, 0.45),
    "completion_pct": (0.645, 0.04),
    "td_rate_per_drive": (0.215, 0.030),
    "fg_rate_per_drive": (0.140, 0.030),
    "punt_rate_per_drive": (0.375, 0.045),
    "turnover_rate_per_drive": (0.110, 0.035),
}


def _generic_player(pid: str, name: str, pos: Position, team: str, **kw) -> Player:
    return Player(player_id=pid, name=name, position=pos, team=team, **kw)


def neutral_team(abbr: str) -> TeamState:
    """A league-average offense with a textbook usage distribution."""
    team = TeamState(
        abbr=abbr,
        name=abbr,
        score=0,
        base_pass_rate=0.575,
        # Pace deliberately left at the TeamState default so this harness
        # always calibrates whatever the shipped default actually is.
        off_pass_eff=1.0,
        off_rush_eff=1.0,
    )
    p = abbr.lower()
    team.players = [
        _generic_player(
            f"{p}_qb", f"{abbr} QB", Position.QB, abbr,
            is_starter_qb=True, rush_share=0.10, ypc=4.2, explosiveness=0.40,
        ),
        _generic_player(
            f"{p}_rb1", f"{abbr} RB1", Position.RB, abbr,
            rush_share=0.56, rz_rush_share=0.62, target_share=0.13,
            ypc=4.35, adot=1.2, yac=5.8, catch_rate=0.78,
            explosiveness=0.40, alignment=Alignment.BACKFIELD,
        ),
        _generic_player(
            f"{p}_rb2", f"{abbr} RB2", Position.RB, abbr,
            rush_share=0.34, rz_rush_share=0.28, target_share=0.07,
            ypc=4.20, adot=1.0, yac=5.4, catch_rate=0.76,
            explosiveness=0.38, alignment=Alignment.BACKFIELD,
        ),
        _generic_player(
            f"{p}_wr1", f"{abbr} WR1", Position.WR, abbr,
            target_share=0.24, rz_target_share=0.24, adot=11.0, yac=4.4,
            catch_rate=0.64, explosiveness=0.55, alignment=Alignment.PERIMETER,
        ),
        _generic_player(
            f"{p}_wr2", f"{abbr} WR2", Position.WR, abbr,
            target_share=0.18, rz_target_share=0.17, adot=10.0, yac=4.0,
            catch_rate=0.63, explosiveness=0.50, alignment=Alignment.PERIMETER,
        ),
        _generic_player(
            f"{p}_wr3", f"{abbr} WR3", Position.WR, abbr,
            target_share=0.13, rz_target_share=0.12, adot=8.0, yac=4.2,
            catch_rate=0.66, explosiveness=0.40, alignment=Alignment.SLOT,
        ),
        _generic_player(
            f"{p}_te1", f"{abbr} TE1", Position.TE, abbr,
            target_share=0.17, rz_target_share=0.20, adot=7.5, yac=4.3,
            catch_rate=0.68, explosiveness=0.33, alignment=Alignment.INLINE,
        ),
        _generic_player(
            f"{p}_te2", f"{abbr} TE2", Position.TE, abbr,
            target_share=0.08, rz_target_share=0.05, adot=6.5, yac=4.0,
            catch_rate=0.67, explosiveness=0.30, alignment=Alignment.INLINE,
        ),
        _generic_player(f"{p}_k", f"{abbr} K", Position.K, abbr),
    ]
    return team


def neutral_game(seconds: int = 1800) -> GameState:
    """Two league-average teams, 0-0, with one half left to play.

    Calibration runs a half rather than a whole game and doubles the counting
    stats, because a real game contains two end-of-half drives that die on the
    clock. Simulating 3600 continuous seconds produces only one, which quietly
    inflates every other drive-outcome rate by about three points.
    """
    return GameState(
        game_id="calibration",
        home=neutral_team("HOM"),
        away=neutral_team("AWY"),
        quarter=1,
        seconds_remaining=seconds,
        possession="AWY",
        weather=Weather(dome=True),
    )


@dataclass
class CalibrationRow:
    metric: str
    simulated: float
    target: float
    tolerance: float

    @property
    def ok(self) -> bool:
        return abs(self.simulated - self.target) <= self.tolerance

    @property
    def delta(self) -> float:
        return self.simulated - self.target


def measure(
    n_sims: int = 4000,
    priors: Optional[Priors] = None,
    seed: int = 11,
) -> List[CalibrationRow]:
    """Simulate a neutral full game and compare against league baselines."""
    priors = priors or Priors()
    state = neutral_game(1800)
    sim_engine = GameSimulator(state, priors)
    sim = sim_engine.run(n_sims=n_sims, seed=seed)

    teams = ("HOM", "AWY")
    # One half is simulated; counting stats double to a full-game equivalent.
    plays = 2 * np.mean([sim.stats[f"@{t}.plays"].mean() for t in teams])
    drives = 2 * np.mean([sim.stats[f"@{t}.drives"].mean() for t in teams])
    points_total = 2 * sim.game_points().mean()

    def team_stat(team_abbr: str, stat: str) -> float:
        total = np.zeros(sim.n_sims)
        for p in state.team(team_abbr).players:
            total = total + sim.get(p.player_id, stat)
        return float(total.mean())

    pass_yds = 2 * np.mean([team_stat(t, "pass_yds") for t in teams])
    rush_yds = 2 * np.mean([team_stat(t, "rush_yds") for t in teams])
    pass_att = np.mean([team_stat(t, "pass_att") for t in teams])
    pass_cmp = np.mean([team_stat(t, "pass_cmp") for t in teams])

    outcomes = sim_engine.drive_outcomes
    total_drives = max(1, sum(outcomes.values()))

    values = {
        "points_total": points_total,
        "plays_per_team": plays,
        "drives_per_team": drives,
        "points_per_drive": points_total / max(2 * drives, 1e-9),
        "pass_yds_per_team": pass_yds,
        "rush_yds_per_team": rush_yds,
        "yards_per_play": (pass_yds + rush_yds) / max(plays, 1e-9),
        "completion_pct": pass_cmp / max(pass_att, 1e-9),
        "td_rate_per_drive": outcomes.get("td", 0) / total_drives,
        "fg_rate_per_drive": outcomes.get("fg_good", 0) / total_drives,
        "punt_rate_per_drive": outcomes.get("punt", 0) / total_drives,
        "turnover_rate_per_drive": outcomes.get("turnover", 0) / total_drives,
    }

    return [
        CalibrationRow(m, values[m], *LEAGUE_BASELINES[m])
        for m in LEAGUE_BASELINES
    ]


def render(rows: List[CalibrationRow]) -> str:
    lines = [
        f"{'metric':<26}{'simulated':>11}{'target':>10}{'delta':>9}  status",
        "-" * 66,
    ]
    for r in rows:
        status = "ok" if r.ok else "OFF"
        lines.append(
            f"{r.metric:<26}{r.simulated:>11.3f}{r.target:>10.3f}"
            f"{r.delta:>+9.3f}  {status}"
        )
    failed = [r for r in rows if not r.ok]
    lines.append("-" * 66)
    lines.append(
        f"{len(rows) - len(failed)}/{len(rows)} within tolerance"
        + ("" if not failed else "  <- investigate: " + ", ".join(r.metric for r in failed))
    )
    return "\n".join(lines)
