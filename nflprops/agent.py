"""End-to-end pipeline: state + offered lines -> a ranked board.

One call does acquisition-agnostic work: resolve names, simulate, evaluate,
price, explain, rank. Anything that needs a network lives in ``providers``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .models import GameState, Prop
from .priors import Priors
from .providers.base import resolve_props
from .rank import GameDiagnostics, RankedProp, diagnose, rank
from .simulate import GameSimulator, SimResult


@dataclass
class Analysis:
    state: GameState
    ranked: List[RankedProp]
    sim: SimResult
    diagnostics: GameDiagnostics
    n_sims: int
    #: Props whose player could not be matched to the roster. Reported, never
    #: silently dropped - an unmatched line usually means a roster gap that
    #: would also corrupt the usage shares of the players who *did* match.
    unresolved: List[Prop] = field(default_factory=list)
    #: Lines the first half already decided, held out of the ranking.
    settled: List[RankedProp] = field(default_factory=list)

    def top(self, n: int = 10) -> List[RankedProp]:
        return self.ranked[:n]


def analyze(
    state: GameState,
    props: Sequence[Prop],
    priors: Optional[Priors] = None,
    n_sims: Optional[int] = None,
    seed: Optional[int] = None,
    sort: str = "probability",
    devig_method: str = "shin",
    min_probability: float = 0.0,
    include_settled: bool = False,
) -> Analysis:
    """Simulate the rest of the game and rank every offered line."""
    priors = priors or Priors()
    n_sims = int(n_sims or priors.get("sim.n_sims"))

    resolved, unresolved = resolve_props(list(props), state)

    engine = GameSimulator(state, priors)
    sim = engine.run(n_sims=n_sims, seed=seed)

    diag = diagnose(sim, state, priors)
    ranked = rank(
        sim,
        state,
        resolved,
        priors=priors,
        sort=sort,
        devig_method=devig_method,
        min_probability=min_probability,
        keep_values=True,
        include_settled=include_settled,
    )

    settled = rank(
        sim, state, resolved, priors=priors, sort=sort,
        devig_method=devig_method, keep_values=False, include_settled=True,
    )
    settled = [r for r in settled if r.evaluation.settled is not None]

    return Analysis(
        state=state,
        ranked=ranked,
        sim=sim,
        diagnostics=diag,
        n_sims=n_sims,
        unresolved=unresolved,
        settled=settled,
    )
