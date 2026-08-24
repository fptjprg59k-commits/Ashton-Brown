"""Halftime NFL prop evaluation.

Simulates the remainder of a game thousands of times from its state at the
break, then reports how often each offered prop cashes across those paths.
"""

from .models import (
    Alignment,
    DefenseProfile,
    GameState,
    InjuryStatus,
    Market,
    Officials,
    Player,
    PlayerH1,
    Position,
    Precip,
    Prop,
    PropScope,
    Side,
    TeamState,
    Weather,
)
from .priors import Priors
from .props import PropEvaluation, evaluate
from .rank import RankedProp, correlation_matrix, parlay_probability, rank
from .simulate import SimResult, simulate

__version__ = "1.0.0"

__all__ = [
    "Alignment", "DefenseProfile", "GameState", "InjuryStatus", "Market",
    "Officials", "Player", "PlayerH1", "Position", "Precip", "Prop",
    "PropScope", "Side", "TeamState", "Weather",
    "Priors", "PropEvaluation", "evaluate", "RankedProp", "rank",
    "correlation_matrix", "parlay_probability", "SimResult", "simulate",
    "__version__",
]
