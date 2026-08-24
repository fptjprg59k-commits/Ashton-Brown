"""Odds conversion, vig removal, and bet sizing.

A model probability is only half of the picture. Sorting purely by "most likely
to hit" surfaces heavy favourites - an over priced at -450 hits about 82% of
the time and is often still a bad bet. The de-vigged book probability is what
turns a hit rate into an assessment of whether the number is *wrong*.

Both are reported. The default sort honours the hit-rate view; ``--sort edge``
switches to the value view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


def american_to_decimal(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_to_prob(odds: int) -> float:
    """Implied probability *including* the book's margin."""
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def prob_to_american(p: float) -> int:
    """Fair American price for a probability, rounded the way books quote."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    # Exactly even money is quoted +100, never -100.
    if p > 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


def devig_multiplicative(odds_a: int, odds_b: int) -> Tuple[float, float]:
    """Proportional (power-free) vig removal across a two-way market.

    Simple and stable. It assumes the margin is spread proportionally, which
    slightly understates the favourite's true probability on lopsided markets -
    see ``devig_shin`` when that matters.
    """
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    total = pa + pb
    if total <= 0:
        raise ValueError("degenerate market")
    return pa / total, pb / total


def devig_shin(odds_a: int, odds_b: int, max_iter: int = 60) -> Tuple[float, float]:
    """Shin's method, which attributes the margin to informed money.

    Handles favourite-longshot bias better than proportional scaling, which is
    exactly the regime most player props sit in.
    """
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    total = pa + pb
    if total <= 1.0:
        return pa, pb

    lo, hi = 0.0, 0.5
    for _ in range(max_iter):
        z = (lo + hi) / 2.0
        s = sum(_shin_prob(p, z, total) for p in (pa, pb))
        if s > 1.0:
            lo = z
        else:
            hi = z
    z = (lo + hi) / 2.0
    qa = _shin_prob(pa, z, total)
    qb = _shin_prob(pb, z, total)
    norm = qa + qb
    return qa / norm, qb / norm


def _shin_prob(p: float, z: float, total: float) -> float:
    inner = z * z + 4.0 * (1.0 - z) * (p * p) / total
    return (math.sqrt(max(inner, 0.0)) - z) / (2.0 * (1.0 - z)) if z < 1.0 else p


def fair_prob(
    odds: int,
    opposing_odds: Optional[int] = None,
    method: str = "shin",
) -> float:
    """Book's true probability for a side, vig removed where possible.

    With only one side visible the margin cannot be identified, so a typical
    half-margin for the market is subtracted as an approximation. This is
    flagged in the report rather than hidden, because a one-sided de-vig is a
    guess and should not be mistaken for a measurement.
    """
    if opposing_odds is None:
        raw = american_to_prob(odds)
        return raw * (1.0 / 1.045)  # ~4.5% typical two-way hold on player props

    if method == "multiplicative":
        a, _ = devig_multiplicative(odds, opposing_odds)
    else:
        a, _ = devig_shin(odds, opposing_odds)
    return a


@dataclass
class Pricing:
    """Model vs. market for a single side."""

    odds: int
    decimal: float
    implied_prob: float          # with vig
    fair_prob: float             # vig removed (or approximated)
    model_prob: float            # our number, pushes excluded
    edge: float                  # model - fair
    ev_per_unit: float
    kelly_fraction: float
    devig_exact: bool


def price(
    model_prob: float,
    odds: int,
    opposing_odds: Optional[int] = None,
    p_push: float = 0.0,
    method: str = "shin",
    kelly_cap: float = 0.05,
) -> Pricing:
    """Combine a model probability with the offered price.

    ``model_prob`` should already be the win probability *excluding pushes*, so
    that it is comparable with the book's fair probability on the same basis.
    Expected value, however, is computed on the full three-outcome tree, since
    a push returns the stake rather than losing it.
    """
    dec = american_to_decimal(odds)
    implied = american_to_prob(odds)
    fair = fair_prob(odds, opposing_odds, method=method)

    p_win = model_prob * (1.0 - p_push)
    p_loss = max(0.0, 1.0 - p_win - p_push)
    profit = dec - 1.0
    ev = p_win * profit - p_loss  # push contributes zero

    # Kelly on the push-adjusted tree, capped so a single mispriced line
    # cannot recommend an absurd stake.
    live = 1.0 - p_push
    if live <= 1e-9 or profit <= 0:
        kelly = 0.0
    else:
        p = p_win / live
        kelly = (p * profit - (1.0 - p)) / profit
    kelly = max(0.0, min(kelly, kelly_cap))

    return Pricing(
        odds=odds,
        decimal=dec,
        implied_prob=implied,
        fair_prob=fair,
        model_prob=model_prob,
        edge=model_prob - fair,
        ev_per_unit=ev,
        kelly_fraction=kelly,
        devig_exact=opposing_odds is not None,
    )
