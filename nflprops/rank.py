"""Ranking, driver attribution, and correlation-aware parlay math.

A bare percentage is not actionable. Every ranked line carries the handful of
factors that actually moved it - script lean, remaining volume, usage share,
weather, rest risk, variance profile - so the number can be argued with rather
than taken on faith.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import gamescript
from .models import (
    BINARY_MARKETS,
    GameState,
    InjuryStatus,
    Market,
    Position,
    Prop,
    PropScope,
    Side,
)
from .pricing import Pricing, price
from .priors import Priors
from .props import PropEvaluation, evaluate, monte_carlo_stderr
from .simulate import SimResult

SORT_KEYS = ("probability", "edge", "ev", "kelly")


@dataclass
class GameDiagnostics:
    """Game-level context computed once and reused for every explanation."""

    seconds_remaining: int
    home_plays: float
    away_plays: float
    pass_rate: Dict[str, float]
    proj_points: Dict[str, float]
    win_prob: Dict[str, float]
    blowout_prob: float
    drives_left: Dict[str, float]

    def plays(self, abbr: str) -> float:
        return self.home_plays if abbr == self._home else self.away_plays

    _home: str = ""


@dataclass
class RankedProp:
    evaluation: PropEvaluation
    pricing: Pricing
    drivers: List[str] = field(default_factory=list)
    stderr: float = 0.0
    confidence: str = "medium"

    @property
    def prop(self) -> Prop:
        return self.evaluation.prop

    @property
    def probability(self) -> float:
        return self.evaluation.p_win

    def sort_value(self, key: str) -> float:
        return {
            "probability": self.evaluation.p_win,
            "edge": self.pricing.edge,
            "ev": self.pricing.ev_per_unit,
            "kelly": self.pricing.kelly_fraction,
        }[key]


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def diagnose(sim: SimResult, state: GameState, priors: Priors) -> GameDiagnostics:
    home, away = state.home.abbr, state.away.abbr

    home_pts = sim.team_points(home) + state.home.score
    away_pts = sim.team_points(away) + state.away.score
    margin = home_pts - away_pts

    def team_plays(abbr: str) -> float:
        return float(np.mean(sim.stats.get(f"@{abbr}.plays", np.zeros(sim.n_sims))))

    def team_pass_rate(team) -> float:
        att = np.zeros(sim.n_sims)
        rush = np.zeros(sim.n_sims)
        for p in team.players:
            att = att + sim.get(p.player_id, "pass_att")
            rush = rush + sim.get(p.player_id, "rush_att")
        total = att + rush
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(total > 0, att / np.maximum(total, 1), np.nan)
        return float(np.nanmean(r))

    avg_spp = (state.home.base_sec_per_play + state.away.base_sec_per_play) / 2.0
    drives = gamescript.possessions_remaining_estimate(
        state.seconds_remaining, avg_spp
    ) / 2.0

    diag = GameDiagnostics(
        seconds_remaining=state.seconds_remaining,
        home_plays=team_plays(home),
        away_plays=team_plays(away),
        pass_rate={home: team_pass_rate(state.home), away: team_pass_rate(state.away)},
        proj_points={home: float(np.mean(home_pts)), away: float(np.mean(away_pts))},
        win_prob={
            home: float(np.mean(margin > 0) + 0.5 * np.mean(margin == 0)),
            away: float(np.mean(margin < 0) + 0.5 * np.mean(margin == 0)),
        },
        blowout_prob=float(np.mean(np.abs(margin) >= 17)),
        drives_left={home: drives, away: drives},
    )
    diag._home = home
    return diag


# --------------------------------------------------------------------------
# Driver attribution
# --------------------------------------------------------------------------


_PASS_MARKETS = {
    Market.PASS_YDS, Market.PASS_TDS, Market.PASS_CMP, Market.PASS_ATT,
    Market.REC, Market.REC_YDS, Market.REC_TDS, Market.LONGEST_REC,
}
_RUSH_MARKETS = {Market.RUSH_YDS, Market.RUSH_ATT, Market.RUSH_TDS, Market.LONGEST_RUSH}


def explain(
    ev: PropEvaluation,
    state: GameState,
    diag: GameDiagnostics,
    priors: Priors,
) -> List[str]:
    """Short, specific reasons this line landed where it did."""
    prop = ev.prop
    out: List[str] = []

    player = state.find_player(prop.player_id) if prop.player_id else None
    team_abbr = player.team if player else prop.team
    if not team_abbr:
        return out

    team = state.team(team_abbr)
    opp = state.opponent(team_abbr)
    margin_now = team.score - opp.score

    # --- game script -----------------------------------------------------
    script_pass = diag.pass_rate.get(team_abbr, team.base_pass_rate)
    lean = script_pass - team.base_pass_rate
    if abs(lean) >= 0.025:
        direction = "pass-leaning" if lean > 0 else "run-leaning"
        state_word = (
            f"trailing by {abs(margin_now)}" if margin_now < 0
            else f"leading by {margin_now}" if margin_now > 0
            else "tied"
        )
        helps = (
            (lean > 0 and prop.market in _PASS_MARKETS)
            or (lean < 0 and prop.market in _RUSH_MARKETS)
        )
        verdict = "tailwind" if (helps == (prop.side is not Side.UNDER)) else "headwind"
        out.append(
            f"Script: {team_abbr} {state_word}, projected {script_pass:.0%} pass "
            f"({direction} vs {team.base_pass_rate:.0%} neutral) - {verdict}"
        )

    # --- volume ----------------------------------------------------------
    plays = diag.plays(team_abbr)
    out.append(
        f"Volume: ~{plays:.0f} projected H2 plays "
        f"({diag.drives_left.get(team_abbr, 0):.1f} drives, {diag.seconds_remaining // 60}:00 left)"
    )

    # --- usage share -----------------------------------------------------
    if player is not None:
        if prop.market in _PASS_MARKETS and player.position is not Position.QB:
            out.append(
                f"Usage: {player.target_share:.0%} target share, "
                f"{player.adot:.1f} aDOT, {player.catch_rate:.0%} catch rate"
            )
        elif prop.market in _RUSH_MARKETS:
            out.append(
                f"Usage: {player.rush_share:.0%} carry share, {player.ypc:.1f} ypc baseline"
            )

        # --- health ------------------------------------------------------
        if player.injury is not InjuryStatus.HEALTHY:
            out.append(
                f"Health: {player.injury.value} "
                f"({player.injury.availability:.0%} snaps, "
                f"{player.injury.efficiency:.0%} effectiveness)"
            )

        # --- matchup -----------------------------------------------------
        shadow = opp.defense.shadow.get(player.player_id)
        if shadow is not None and abs(shadow - 1.0) > 0.05:
            word = "shadowed by a shutdown corner" if shadow < 1 else "favourable coverage draw"
            out.append(f"Matchup: {word} ({shadow:.2f}x volume/efficiency)")
        elif prop.market in _PASS_MARKETS and abs(opp.defense.pass_yds_mult - 1.0) > 0.06:
            word = "stingy" if opp.defense.pass_yds_mult < 1 else "generous"
            out.append(
                f"Matchup: {opp.abbr} pass defense {word} "
                f"({opp.defense.pass_yds_mult:.2f}x yardage)"
            )
        elif prop.market in _RUSH_MARKETS and abs(opp.defense.rush_yds_mult - 1.0) > 0.06:
            word = "stingy" if opp.defense.rush_yds_mult < 1 else "generous"
            out.append(
                f"Matchup: {opp.abbr} run defense {word} "
                f"({opp.defense.rush_yds_mult:.2f}x yardage)"
            )

        # --- rest risk ---------------------------------------------------
        if player.star and diag.blowout_prob > 0.18:
            out.append(
                f"Blowout risk: {diag.blowout_prob:.0%} chance of a 17+ margin; "
                f"star rest risk truncates the top of this distribution"
            )

        # --- variance profile --------------------------------------------
        if prop.market in (Market.REC_YDS, Market.RUSH_YDS, Market.RUSH_REC_YDS):
            if player.explosiveness >= 0.6:
                out.append(
                    "Profile: boom-or-bust - clears mostly via one explosive play, "
                    "so the median sits well under the mean"
                )
            elif player.explosiveness <= 0.25:
                out.append(
                    "Profile: high-floor possession usage - accumulates steadily, "
                    "tight distribution"
                )

    # --- weather ---------------------------------------------------------
    w = state.weather
    if not w.dome:
        bits = []
        if w.wind_mph >= 12:
            bits.append(f"{w.wind_mph:.0f} mph wind")
        if w.precip.value != "none":
            bits.append(w.precip.value.replace("_", " "))
        if w.temp_f < 25:
            bits.append(f"{w.temp_f:.0f}F")
        if bits:
            out.append("Weather: " + ", ".join(bits) + " - depresses passing and kicking")

    # --- distance to the line --------------------------------------------
    if prop.market not in BINARY_MARKETS:
        gap = ev.median - prop.line
        scope = "full game" if prop.scope is PropScope.FULL_GAME else "H2 only"
        out.append(
            f"Line: median {ev.median:.0f} vs {prop.line:g} ({gap:+.0f}), "
            f"80% range {ev.p10:.0f}-{ev.p90:.0f} [{scope}]"
        )

    return out


def _confidence(ev: PropEvaluation, pricing: Pricing, stderr: float) -> str:
    """Qualitative confidence, dominated by how much is being extrapolated."""
    if stderr > 0.012:
        return "low"
    if not pricing.devig_exact and abs(pricing.edge) < 0.03:
        return "low"
    if ev.p_push > 0.06:
        return "medium"
    if abs(pricing.edge) >= 0.05:
        return "high"
    return "medium"


# --------------------------------------------------------------------------
# Top-level ranking
# --------------------------------------------------------------------------


def rank(
    sim: SimResult,
    state: GameState,
    props: Sequence[Prop],
    priors: Optional[Priors] = None,
    sort: str = "probability",
    devig_method: str = "shin",
    min_probability: float = 0.0,
    keep_values: bool = True,
    include_settled: bool = False,
) -> List[RankedProp]:
    """Evaluate, price, explain, and order every offered line.

    Default ordering is by raw hit probability, highest first. Note that this
    surfaces heavy favourites, which are the most *likely* bets rather than the
    most *profitable* ones - ``sort="edge"`` orders by model-vs-market
    disagreement instead.
    """
    priors = priors or Priors()
    diag = diagnose(sim, state, priors)

    ranked: List[RankedProp] = []
    for prop in props:
        try:
            ev = evaluate(sim, state, prop, keep_values=keep_values)
        except (ValueError, KeyError):
            continue

        # A line the first half already decided is not a prediction. Held out
        # by default so it cannot sit at the top of the board claiming a
        # 45-point edge over a price that no longer exists.
        if ev.settled is not None and not include_settled:
            continue

        if ev.p_win < min_probability:
            continue

        pr = price(
            model_prob=ev.p_win_excluding_push,
            odds=prop.odds,
            opposing_odds=prop.opposing_odds,
            p_push=ev.p_push,
            method=devig_method,
        )
        se = monte_carlo_stderr(ev.p_win, sim.n_sims)
        ranked.append(
            RankedProp(
                evaluation=ev,
                pricing=pr,
                drivers=explain(ev, state, diag, priors),
                stderr=se,
                confidence=_confidence(ev, pr, se),
            )
        )

    if sort not in SORT_KEYS:
        raise ValueError(f"sort must be one of {SORT_KEYS}, got {sort!r}")
    ranked.sort(key=lambda r: r.sort_value(sort), reverse=True)
    return ranked


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------


def parlay_probability(legs: Sequence[RankedProp]) -> float:
    """True joint probability of every leg winning.

    Because all legs are scored on the same simulated paths, this captures the
    correlation books charge for in same-game parlays. Multiplying individual
    probabilities would badly misprice a QB-to-WR stack (correlated, so the
    real number is higher) or two backs in the same backfield (substitutes, so
    the real number is lower).
    """
    masks = [r.evaluation.win_mask for r in legs]
    if not masks or any(m is None for m in masks):
        raise ValueError("parlay math requires rank(..., keep_values=True)")
    joint = masks[0].copy()
    for m in masks[1:]:
        joint &= m
    return float(np.mean(joint))


def independent_parlay_probability(legs: Sequence[RankedProp]) -> float:
    """What naive multiplication would say - useful only as a contrast."""
    p = 1.0
    for r in legs:
        p *= r.evaluation.p_win
    return p


def correlation_matrix(legs: Sequence[RankedProp]) -> np.ndarray:
    """Pairwise correlation of leg outcomes across simulated paths."""
    masks = [r.evaluation.win_mask for r in legs]
    if any(m is None for m in masks):
        raise ValueError("correlation requires rank(..., keep_values=True)")
    data = np.vstack([m.astype(float) for m in masks])
    # Guard against zero-variance legs (probability 0 or 1).
    sd = data.std(axis=1)
    safe = sd > 1e-9
    out = np.eye(len(masks))
    if safe.sum() >= 2:
        sub = np.corrcoef(data[safe])
        idx = np.where(safe)[0]
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                out[ia, ib] = sub[a, b]
    return out
