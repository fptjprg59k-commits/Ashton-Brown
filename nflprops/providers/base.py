"""Provider interfaces.

The engine never talks to a network directly. Everything arrives through one
of these two interfaces, which means the same code path runs against a live
feed, a saved fixture, or a block of text pasted out of a betting app.

That indirection is not architectural decoration. Sportsbooks do not publish
a public odds API, their terms generally prohibit scraping, and any endpoint
that does work today will change without notice. Keeping acquisition behind a
narrow interface means the model stays useful no matter how the numbers are
obtained, and the ugly, brittle, legally-sensitive part stays quarantined in
one small class you control.
"""

from __future__ import annotations

import abc
from typing import List, Optional

from ..models import GameState, Prop


class PropsProvider(abc.ABC):
    """Source of offered lines."""

    name: str = "base"

    @abc.abstractmethod
    def fetch(self, game_id: Optional[str] = None) -> List[Prop]:
        """Return every prop currently offered for the game."""


class GameStateProvider(abc.ABC):
    """Source of live game state and first-half box score."""

    name: str = "base"

    @abc.abstractmethod
    def fetch(self, game_id: str) -> GameState:
        """Return the current state of the game."""

    def list_games_at_halftime(self) -> List[str]:
        """Game ids currently at the break. Optional for offline providers."""
        return []


class ProviderError(RuntimeError):
    """Raised when a provider cannot supply usable data."""


def resolve_props(props: List[Prop], state: GameState) -> tuple:
    """Bind free-text player names in ``props`` to roster ids.

    Returns ``(resolved, unresolved)``. Anything that cannot be matched
    confidently is returned in ``unresolved`` rather than attached to a
    best-guess player - a wrongly bound prop produces a confident, plausible,
    completely wrong probability, which is far worse than a visible gap.
    """
    from ..matching import PlayerResolver
    from ..models import GAME_MARKETS, TEAM_MARKETS

    resolver = PlayerResolver(state.all_players())
    resolved: List[Prop] = []
    unresolved: List[Prop] = []

    known_teams = {state.home.abbr.upper(), state.away.abbr.upper()}

    for prop in props:
        if prop.market in GAME_MARKETS:
            resolved.append(prop)
            continue

        if prop.market in TEAM_MARKETS:
            if prop.team and prop.team.upper() in known_teams:
                resolved.append(prop)
            else:
                unresolved.append(prop)
            continue

        if prop.player_id and state.find_player(prop.player_id) is not None:
            resolved.append(prop)
            continue

        pid = resolver.resolve(prop.player_name or prop.player_id or "", team=prop.team)
        if pid is None:
            unresolved.append(prop)
            continue

        prop.player_id = pid
        player = state.find_player(pid)
        if player is not None:
            prop.player_name = player.name
            prop.team = player.team
        resolved.append(prop)

    return resolved, unresolved
