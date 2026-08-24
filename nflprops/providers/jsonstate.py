"""Game state from a local JSON snapshot.

The offline path, and the one every test runs against. Also the right way to
work when you want to hand-tune a scenario - change the score, flip an injury
designation, raise the wind, and re-run to see what actually moves.
"""

from __future__ import annotations

from typing import List, Optional

from ..models import GameState
from ..serde import load_game
from .base import GameStateProvider


class JSONStateProvider(GameStateProvider):
    name = "json"

    def __init__(self, path: str) -> None:
        self.path = path

    def fetch(self, game_id: Optional[str] = None) -> GameState:
        return load_game(self.path)

    def list_games_at_halftime(self) -> List[str]:
        return [self.fetch().game_id]
