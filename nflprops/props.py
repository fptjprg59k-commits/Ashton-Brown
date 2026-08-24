"""Turn simulated stat lines into probabilities for specific offered props.

Two details here matter more than they look:

**Scope.** Books keep full-game markets live through halftime *and* post
second-half derivatives. A full-game line has to be compared against first-half
actuals plus the simulated remainder. Getting this backwards is the single
most expensive mistake available at the break - it prices a 65.5-yard receiving
line as though the receiver were starting from zero when he already has 44.

**Pushes.** Whole-number lines (``Over 65``) push on an exact hit and the stake
is returned. Yardage is therefore rounded to integers before comparison, the
way the box score does it, so pushes are counted rather than silently
distributed into wins and losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from .models import (
    BINARY_MARKETS,
    GameState,
    Market,
    Prop,
    PropScope,
    Side,
)
from .simulate import SimResult

#: Markets whose values are whole numbers in the box score.
_INTEGER_MARKETS = {
    Market.PASS_YDS, Market.REC_YDS, Market.RUSH_YDS,
    Market.RUSH_REC_YDS, Market.PASS_RUSH_YDS,
    Market.LONGEST_REC, Market.LONGEST_RUSH,
    Market.PASS_CMP, Market.PASS_ATT, Market.PASS_TDS, Market.PASS_INT,
    Market.RUSH_ATT, Market.RUSH_TDS, Market.REC, Market.REC_TDS,
    Market.KICKING_POINTS, Market.TEAM_TOTAL, Market.GAME_TOTAL,
}


@dataclass
class PropEvaluation:
    """Everything the ranker and the report need about one offered line."""

    prop: Prop
    p_win: float
    p_push: float
    p_loss: float
    mean: float
    median: float
    p10: float
    p90: float
    #: Distribution of the underlying stat, retained for correlation work.
    values: Optional[np.ndarray] = None
    #: Per-path win indicator. Because every prop is scored on the same set of
    #: simulated paths, ANDing these masks gives a correlation-aware parlay
    #: probability directly - no copula, no independence assumption.
    win_mask: Optional[np.ndarray] = None
    #: ``"won"`` / ``"lost"`` when first-half production alone already decides a
    #: full-game line, else None. At the break a real share of the board is
    #: already settled - a receiver past his yardage line, a back who has
    #: scored. Reporting those as "100% to hit" with a huge edge is noise, so
    #: they are detected and held out of the ranking.
    settled: Optional[str] = None

    @property
    def p_win_excluding_push(self) -> float:
        """Win probability conditional on the bet actually resolving."""
        live = 1.0 - self.p_push
        return self.p_win / live if live > 1e-12 else 0.0


# --------------------------------------------------------------------------
# Market -> simulated value
# --------------------------------------------------------------------------


def _player_sum(sim: SimResult, pid: str, *stats: str) -> np.ndarray:
    total = np.zeros(sim.n_sims)
    for s in stats:
        total = total + sim.get(pid, s)
    return total


def _h1_value(state: GameState, prop: Prop) -> float:
    """First-half contribution for a full-game line."""
    m = prop.market

    if m is Market.GAME_TOTAL:
        return float(state.home.score + state.away.score)
    if m is Market.TEAM_TOTAL:
        return float(state.team(prop.team).score) if prop.team else 0.0

    player = state.find_player(prop.player_id) if prop.player_id else None
    if player is None:
        return 0.0
    h = player.h1

    return {
        Market.PASS_YDS: h.pass_yds,
        Market.PASS_TDS: float(h.pass_td),
        Market.PASS_CMP: float(h.pass_cmp),
        Market.PASS_ATT: float(h.pass_att),
        Market.PASS_INT: float(h.interceptions),
        Market.RUSH_YDS: h.rush_yds,
        Market.RUSH_ATT: float(h.rush_att),
        Market.RUSH_TDS: float(h.rush_td),
        Market.REC: float(h.rec),
        Market.REC_YDS: h.rec_yds,
        Market.REC_TDS: float(h.rec_td),
        Market.RUSH_REC_YDS: h.rush_yds + h.rec_yds,
        Market.PASS_RUSH_YDS: h.pass_yds + h.rush_yds,
        Market.LONGEST_REC: h.longest_rec,
        Market.LONGEST_RUSH: h.longest_rush,
        Market.ANYTIME_TD: float(h.rush_td + h.rec_td),
        Market.KICKING_POINTS: h.kicking_points,
    }.get(m, 0.0)


_SECOND_HALF: Dict[Market, Callable[[SimResult, Prop], np.ndarray]] = {
    Market.PASS_YDS: lambda s, p: s.get(p.player_id, "pass_yds"),
    Market.PASS_TDS: lambda s, p: s.get(p.player_id, "pass_td"),
    Market.PASS_CMP: lambda s, p: s.get(p.player_id, "pass_cmp"),
    Market.PASS_ATT: lambda s, p: s.get(p.player_id, "pass_att"),
    Market.PASS_INT: lambda s, p: s.get(p.player_id, "interceptions"),
    Market.RUSH_YDS: lambda s, p: s.get(p.player_id, "rush_yds"),
    Market.RUSH_ATT: lambda s, p: s.get(p.player_id, "rush_att"),
    Market.RUSH_TDS: lambda s, p: s.get(p.player_id, "rush_td"),
    Market.REC: lambda s, p: s.get(p.player_id, "rec"),
    Market.REC_YDS: lambda s, p: s.get(p.player_id, "rec_yds"),
    Market.REC_TDS: lambda s, p: s.get(p.player_id, "rec_td"),
    Market.RUSH_REC_YDS: lambda s, p: _player_sum(s, p.player_id, "rush_yds", "rec_yds"),
    Market.PASS_RUSH_YDS: lambda s, p: _player_sum(s, p.player_id, "pass_yds", "rush_yds"),
    Market.ANYTIME_TD: lambda s, p: _player_sum(s, p.player_id, "rush_td", "rec_td"),
    Market.KICKING_POINTS: lambda s, p: (
        3.0 * s.get(p.player_id, "fg_made") + s.get(p.player_id, "xp_made")
    ),
    Market.TEAM_TOTAL: lambda s, p: s.team_points(p.team),
    Market.GAME_TOTAL: lambda s, p: s.game_points(),
}


def stat_values(sim: SimResult, state: GameState, prop: Prop) -> np.ndarray:
    """The distribution of the quantity this prop settles on."""
    m = prop.market

    # "Longest" markets are a maximum, not a sum, so the first half combines
    # with the second half through max() rather than addition.
    if m in (Market.LONGEST_REC, Market.LONGEST_RUSH):
        stat = "longest_rec" if m is Market.LONGEST_REC else "longest_rush"
        second = sim.get(prop.player_id, stat)
        if prop.scope is PropScope.SECOND_HALF:
            values = second
        else:
            values = np.maximum(second, _h1_value(state, prop))
        return np.rint(values)

    fn = _SECOND_HALF.get(m)
    if fn is None:
        raise ValueError(f"unsupported market: {m}")

    values = fn(sim, prop).astype(np.float64, copy=True)
    if prop.scope is PropScope.FULL_GAME:
        values = values + _h1_value(state, prop)

    if m in _INTEGER_MARKETS:
        values = np.rint(values)
    return values


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def settled_status(state: GameState, prop: Prop) -> Optional[str]:
    """Whether first-half production alone already decides a full-game line.

    Every stat these markets settle on is non-decreasing - yardage accumulates,
    receptions accumulate, and a "longest" is a running maximum. So a total
    already past the line can never come back under it.
    """
    if prop.scope is not PropScope.FULL_GAME:
        return None

    h1 = _h1_value(state, prop)

    if prop.market in BINARY_MARKETS:
        if h1 >= 1.0:
            return "won" if prop.side is Side.YES else "lost"
        return None

    if h1 > prop.line:
        return "won" if prop.side is Side.OVER else "lost"
    return None


def evaluate(
    sim: SimResult,
    state: GameState,
    prop: Prop,
    keep_values: bool = False,
) -> PropEvaluation:
    """Probability that ``prop`` wins, pushes, and loses."""
    values = stat_values(sim, state, prop)

    if prop.market in BINARY_MARKETS:
        # Yes/no markets: "did he score at all".
        hit = values >= 1.0
        win = hit if prop.side is Side.YES else ~hit
        p_win = float(np.mean(win))
        p_push = 0.0
        push = np.zeros(sim.n_sims, dtype=bool)
    else:
        line = prop.line
        push = values == line
        if prop.side is Side.OVER:
            win = values > line
        elif prop.side is Side.UNDER:
            win = values < line
        else:
            raise ValueError(f"side {prop.side} is not valid for market {prop.market}")
        p_win = float(np.mean(win))
        p_push = float(np.mean(push))

    p_loss = max(0.0, 1.0 - p_win - p_push)

    return PropEvaluation(
        prop=prop,
        p_win=p_win,
        p_push=p_push,
        p_loss=p_loss,
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p10=float(np.percentile(values, 10)),
        p90=float(np.percentile(values, 90)),
        values=values if keep_values else None,
        win_mask=np.asarray(win, dtype=bool) if keep_values else None,
        settled=settled_status(state, prop),
    )


def evaluate_all(
    sim: SimResult,
    state: GameState,
    props: list,
    keep_values: bool = False,
) -> list:
    out = []
    for prop in props:
        try:
            out.append(evaluate(sim, state, prop, keep_values=keep_values))
        except (ValueError, KeyError) as exc:  # skip unusable lines, keep going
            out.append(None)
            _warn_unusable(prop, exc)
    return [e for e in out if e is not None]


def _warn_unusable(prop: Prop, exc: Exception) -> None:
    import sys

    print(f"[warn] skipping {prop.label}: {exc}", file=sys.stderr)


def monte_carlo_stderr(p: float, n: int) -> float:
    """Standard error on a simulated probability.

    Reported alongside every number so nobody mistakes simulation noise for
    signal. At 20,000 paths a 50% probability carries roughly +/-0.35pp.
    """
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1)))
