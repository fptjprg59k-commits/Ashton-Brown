"""Per-touch outcome samplers.

The important idea here is that **two players with the same projected mean can
have very different probabilities of clearing the same line.** A possession
receiver projected for 62 yards gets there on volume and rarely misses badly;
a vertical threat projected for the same 62 yards misses low most weeks and
occasionally posts 130 on two catches. Any model that compares a projection to
a line and stops has thrown that away.

So yardage is drawn from a shifted gamma whose *shape* is set by the player's
explosiveness index while its *mean is held fixed*. Low explosiveness gives a
high shape parameter (tight, symmetric, reliable); high explosiveness gives a
low shape (right-skewed, fat-tailed, boom-or-bust). Mean-preserving means
tuning explosiveness never silently smuggles in a projection change.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .priors import Priors


class Sampler:
    """Thin wrapper over ``random.Random`` holding the tuned shape constants.

    Pure-Python ``random`` is used deliberately: the drive engine draws scalars
    in a tight loop, where NumPy's per-call overhead is an order of magnitude
    worse than the stdlib Mersenne Twister.
    """

    def __init__(self, priors: "Priors", seed: int = 0) -> None:
        self.rng = random.Random(seed)
        p = priors.section("play")
        self.rec_shape_lo = p["shape_explosive"]
        self.rec_shape_hi = p["shape_possession"]
        self.rush_shape_lo = p["rush_shape_explosive"]
        self.rush_shape_hi = p["rush_shape_possession"]
        self.rec_shift = p["rec_shift"]
        self.rush_shift = p["rush_shift"]
        self.sack_mean = p["sack_yards_mean"]
        self.scramble_mean = p["scramble_yards_mean"]

    # -- primitives --------------------------------------------------------

    def random(self) -> float:
        return self.rng.random()

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def choice_index(self, weights: list, total: float) -> int:
        """Sample an index from unnormalised ``weights`` summing to ``total``."""
        r = self.rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r < acc:
                return i
        return len(weights) - 1

    # -- yardage -----------------------------------------------------------

    def _shifted_gamma(self, mean: float, shape: float, shift: float) -> float:
        """Draw with expectation exactly ``mean``, skew controlled by ``shape``."""
        target = max(mean + shift, 0.35)
        scale = target / shape
        return self.rng.gammavariate(shape, scale) - shift

    def reception_yards(self, mean: float, explosiveness: float) -> float:
        e = _clip01(explosiveness)
        shape = self.rec_shape_hi + (self.rec_shape_lo - self.rec_shape_hi) * e
        y = self._shifted_gamma(mean, shape, self.rec_shift)
        return max(y, -8.0)

    def rush_yards(self, mean: float, explosiveness: float) -> float:
        e = _clip01(explosiveness)
        shape = self.rush_shape_hi + (self.rush_shape_lo - self.rush_shape_hi) * e
        y = self._shifted_gamma(mean, shape, self.rush_shift)
        return max(y, -12.0)

    def sack_yards(self) -> float:
        # Sack losses are roughly exponential with a floor of a yard.
        return -max(1.0, self.rng.expovariate(1.0 / self.sack_mean))

    def scramble_yards(self) -> float:
        return max(-3.0, self._shifted_gamma(self.scramble_mean, 1.9, 2.5))


# --------------------------------------------------------------------------
# Deterministic probability helpers
# --------------------------------------------------------------------------


def completion_prob(base_catch_rate: float, adot: float, penalty: float) -> float:
    """Completion probability for a target, after environmental penalties.

    ``penalty`` arrives from the weather model already scaled by depth, so this
    only has to apply it and keep the result inside sane bounds.
    """
    return _clip(base_catch_rate - penalty, 0.02, 0.99)


def fg_make_prob(distance: float, priors: "Priors", wind_penalty: float = 0.0) -> float:
    """Field goal make probability by distance, with a wind adjustment.

    Logistic in distance, not linear. The real curve is flat out to about 40
    yards and then falls away steeply, so a straight line fit badly underprices
    routine kicks - a linear decay calibrated to hit 60-yarders correctly puts
    a 40-yarder near 70%, when kickers actually make those about 90% of the
    time. Midpoint and scale are fitted to the league make-rate curve:
    ~99% at 20, ~92% at 40, ~75% at 50, ~61% at 55, ~45% at 60.
    """
    fd = priors.section("fourth_down")
    p = logistic((fd["fg_midpoint"] - distance) / fd["fg_scale"])
    return _clip(p - wind_penalty, 0.01, 0.995)


def yardline_to_fg_distance(yardline: float) -> float:
    """``yardline`` is yards from the offense's own goal line (0-100)."""
    return (100.0 - yardline) + 17.0


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _clip01(x: float) -> float:
    return _clip(x, 0.0, 1.0)


def logistic(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
