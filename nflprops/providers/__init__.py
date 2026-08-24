"""Data acquisition adapters."""

from .base import GameStateProvider, PropsProvider, ProviderError, resolve_props
from .jsonstate import JSONStateProvider
from .stake import MARKET_ALIASES, StakePropsProvider

__all__ = [
    "GameStateProvider",
    "PropsProvider",
    "ProviderError",
    "resolve_props",
    "JSONStateProvider",
    "StakePropsProvider",
    "MARKET_ALIASES",
    "ESPNProvider",
]


def __getattr__(name):
    # ESPN pulls in network code; import lazily so offline use never touches it.
    if name == "ESPNProvider":
        from .espn import ESPNProvider

        return ESPNProvider
    raise AttributeError(name)
