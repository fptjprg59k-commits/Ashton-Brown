"""Model coefficients in one place.

These are literature-informed league-average defaults, **not** parameters fit
to a proprietary play-by-play database. They produce sensible, well-shaped
distributions out of the box, and `nflprops.calibrate` exists so they can be
refit against your own settled-prop history. Treat every number here as a
starting point you should tune, and read `docs/MODEL.md` for the reasoning.

Override any subset at runtime with a YAML file via `Priors.load(path)`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:  # PyYAML is optional; defaults work without it.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


DEFAULTS: Dict[str, Any] = {
    # ------------------------------------------------------------------
    # Game script: how score and clock bend play-calling
    # ------------------------------------------------------------------
    "script": {
        # Urgency driver is z = -point_diff / sqrt(minutes_remaining), the
        # standard "leverage" form: down 7 with 25 minutes left barely moves
        # play-calling, down 7 with 4 minutes left transforms it.
        "z_scale": 2.2,
        # Max deviation from a team's neutral pass rate, applied through
        # tanh(z / z_scale) so the effect saturates instead of running away.
        "pass_rate_beta": 0.20,
        "pass_rate_min": 0.15,
        "pass_rate_max": 0.92,
        # Pace: fraction by which seconds-per-play shrink when trailing hard
        # (and stretch when protecting a lead and bleeding clock).
        "pace_beta": 0.30,
        # Fraction of the tempo swing already in effect at the start of a half,
        # rising to 1.0 as the clock runs out. Clock management is a late-game
        # behaviour; without this, a leading team milks the play clock from the
        # opening snap of the third quarter.
        "pace_time_floor": 0.35,
        # Once inside this many seconds and trailing, the offense goes
        # effectively no-huddle regardless of its baseline tempo.
        "hurry_up_seconds": 300,
        "hurry_up_sec_per_play": 18.5,
        # Leading team in the last four minutes bleeds clock aggressively.
        "clock_kill_seconds": 240,
        "clock_kill_sec_per_play": 42.0,
    },
    # ------------------------------------------------------------------
    # Clock consumption
    # ------------------------------------------------------------------
    "clock": {
        "incompletion_seconds": 6.0,       # clock stops; just the snap-to-whistle
        "out_of_bounds_prob": 0.16,        # completions that stop the clock
        "sec_between_plays": 0.0,          # folded into team base_sec_per_play
        "kickoff_seconds": 8.0,
        "punt_seconds": 14.0,
        "score_reset_seconds": 20.0,       # PAT + kickoff dead time
    },
    # ------------------------------------------------------------------
    # Per-play outcomes
    # ------------------------------------------------------------------
    "play": {
        "league_pass_ypa": 7.05,           # yards per attempt, all attempts
        "league_rush_ypc": 4.35,
        "sack_yards_mean": 6.6,
        "fumble_lost_rate_rush": 0.0075,
        "fumble_lost_rate_sack": 0.055,
        "scramble_yards_mean": 6.8,
        # Gamma shape bounds for the per-touch yardage mixture. Lower shape =
        # fatter right tail. A possession receiver sits near `shape_possession`
        # (tight, high floor); a vertical threat near `shape_explosive`.
        # Chosen so that a league-average profile reproduces observed NFL
        # dispersion: ~8-9 yard SD per reception at a 12-yard mean, and ~5-6
        # yard SD per carry at a 4.35-yard mean.
        "shape_possession": 3.0,
        "shape_explosive": 0.70,
        "rush_shape_possession": 2.2,
        "rush_shape_explosive": 0.75,
        # Rushes lose yardage more often than receptions do; modelled as a
        # fixed negative shift applied before the gamma draw.
        "rush_shift": 2.5,
        "rec_shift": 0.8,
    },
    # ------------------------------------------------------------------
    # Fourth-down and kicking behaviour
    # ------------------------------------------------------------------
    "fourth_down": {
        # Field position (yards from own goal) beyond which a FG is attempted.
        "fg_range_yardline": 65.0,         # ~52-yard attempt
        "desperation_seconds": 360,        # trailing late: go for it far more
        # Fourth-down aggression, as go-for-it probability by distance and
        # field position. Modern play-callers go far more often than the
        # pre-analytics defaults suggest, and this materially affects drive
        # length - which is to say, every volume prop on the board.
        # Each entry: (max yards to go, min yardline, probability).
        "go_table": [
            [1.0, 40.0, 0.78],
            [1.0, 0.0, 0.34],
            [2.0, 45.0, 0.60],
            [2.0, 0.0, 0.14],
            [4.0, 52.0, 0.38],
            [4.0, 0.0, 0.06],
            [99.0, 55.0, 0.15],
            [99.0, 0.0, 0.02],
        ],
        # Fourth and goal from inside the five: take the points or take the
        # touchdown? Teams increasingly take the touchdown.
        "goal_line_go_prob": 0.45,
        # Logistic make curve, fitted to league-wide make rates by distance.
        "fg_midpoint": 58.5,               # distance at which a kick is 50/50
        "fg_scale": 7.7,
        "xp_make": 0.945,
        "two_point_rate": 0.09,
        "two_point_success": 0.475,
    },
    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------
    "weather": {
        "wind_threshold": 12.0,            # mph below which effect is ~nil
        "wind_comp_penalty": 0.0075,       # per mph over threshold
        "wind_deep_penalty": 0.016,        # extra penalty scaled by aDOT
        "wind_fg_penalty": 0.008,          # per mph over threshold
        "rain_comp_penalty": 0.018,
        "heavy_rain_comp_penalty": 0.042,
        "snow_comp_penalty": 0.038,
        "precip_explosive_damp": 0.12,     # shrinks the big-play tail
        "precip_fumble_mult": 1.45,
        "cold_threshold": 25.0,
        "cold_comp_penalty": 0.012,        # per 10 degrees under threshold
    },
    # ------------------------------------------------------------------
    # Halftime adjustments and regression
    # ------------------------------------------------------------------
    "adjustment": {
        # Weight on the pregame prior when blending with observed first-half
        # efficiency. A half is a tiny sample, so the prior dominates: a QB at
        # 11.0 yards per attempt in the first half is not an 11.0 YPA passer.
        "prior_weight": 0.74,
        # Per-simulation coordinator-adjustment shock (multiplicative, applied
        # to the side that was more efficient in the first half). This widens
        # the distribution rather than shifting it, which is the honest way to
        # represent "the defense might have solved them".
        "shock_sd": 0.10,
        "shock_regression_bias": -0.025,   # hot offenses cool slightly more often
        # Usage shares also regress: a 45% first-half target share on 11 pass
        # attempts is mostly noise.
        "usage_prior_weight": 0.65,
    },
    # ------------------------------------------------------------------
    # Blowout / rest risk
    # ------------------------------------------------------------------
    "blowout": {
        # Logistic hazard on the *simulated* margin, evaluated per drive in
        # the fourth quarter. Stars only.
        "margin_midpoint": 20.0,
        "margin_scale": 5.0,
        "max_pull_prob_per_drive": 0.42,
        "start_seconds": 900,              # only from the start of Q4
        "rb_multiplier": 1.35,             # backs get rested first
        "qb_multiplier": 1.10,
        "wr_multiplier": 0.85,
    },
    # ------------------------------------------------------------------
    # Officiating
    # ------------------------------------------------------------------
    "officials": {
        # Drive-extension probability per drive, per penalty over crew average.
        "extension_per_penalty": 0.006,
        "league_penalties_per_game": 12.8,
        "dpi_yards_mean": 16.0,
    },
    # ------------------------------------------------------------------
    # Simulation defaults
    # ------------------------------------------------------------------
    "sim": {
        "n_sims": 20000,
        "seed": 20260824,
        "max_drives": 40,                  # safety valve against runaway loops
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Priors:
    """Dotted-path accessor over the coefficient tree."""

    values: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULTS))

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Priors":
        if not path:
            return cls()
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load a priors override file")
        with open(path, "r", encoding="utf-8") as fh:
            override = yaml.safe_load(fh) or {}
        return cls(_deep_merge(DEFAULTS, override))

    def get(self, dotted: str) -> Any:
        node: Any = self.values
        for part in dotted.split("."):
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.values
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    # Convenience so hot loops can bind a whole section once.
    def section(self, name: str) -> Dict[str, Any]:
        return self.values[name]
