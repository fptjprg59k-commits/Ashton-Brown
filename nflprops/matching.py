"""Fuzzy player-name resolution.

Book feeds, box scores, and hand-typed notes never agree on how to spell a
name. "Ja'Marr Chase", "JaMarr Chase", "J. Chase" and "Chase, Ja'Marr" all have
to land on the same roster entry, and getting this wrong silently attaches a
prop to the wrong player - a failure mode that produces confident nonsense
rather than an error.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
#: Marks that sit *inside* a name and must be deleted, not spaced out:
#: "Ja'Marr" has to normalise to "jamarr" so it matches a feed that writes
#: "JaMarr". Spacing it into "ja marr" makes the two names look different and
#: silently fails to resolve the player.
_INTRAWORD = re.compile(r"['‘’ʼ`]")
_PUNCT = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")


def normalize(name: str) -> str:
    """Lowercase, strip accents and punctuation, drop generational suffixes."""
    if not name:
        return ""
    if "," in name:  # "Chase, Ja'Marr" -> "Ja'Marr Chase"
        last, _, first = name.partition(",")
        name = f"{first.strip()} {last.strip()}"
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _INTRAWORD.sub("", text.lower())
    text = _PUNCT.sub(" ", text)
    parts = [p for p in _WS.split(text) if p and p not in _SUFFIXES]
    return " ".join(parts)


def _tokens(name: str) -> List[str]:
    return [t for t in normalize(name).split() if t]


def similarity(a: str, b: str) -> float:
    """Score in [0, 1] for two names referring to the same person."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    if ta == tb:
        return 1.0

    # Surname must be compatible; it carries nearly all the identifying signal.
    surname_score = SequenceMatcher(None, ta[-1], tb[-1]).ratio()
    if surname_score < 0.82:
        return 0.0

    first_a, first_b = ta[0], tb[0]
    if first_a == first_b:
        first_score = 1.0
    elif len(first_a) == 1 or len(first_b) == 1:
        # Initial-only forms: "J. Chase" vs "Ja'Marr Chase".
        first_score = 0.9 if first_a[0] == first_b[0] else 0.0
    else:
        first_score = SequenceMatcher(None, first_a, first_b).ratio()

    return 0.65 * surname_score + 0.35 * first_score


class PlayerResolver:
    """Resolves free-text names against a known roster."""

    def __init__(self, roster: Iterable) -> None:
        # roster: iterable of objects with .player_id, .name, .team
        self._entries: List[Tuple[str, str, str]] = [
            (p.player_id, p.name, getattr(p, "team", "")) for p in roster
        ]
        self._exact: Dict[str, str] = {}
        for pid, name, _ in self._entries:
            self._exact.setdefault(normalize(name), pid)
            self._exact.setdefault(normalize(pid), pid)

    def resolve(
        self,
        name: str,
        team: Optional[str] = None,
        threshold: float = 0.86,
    ) -> Optional[str]:
        """Return a player_id, or None when no candidate is confident enough.

        Returning None is deliberate: an unmatched prop should be reported and
        skipped, never guessed onto the nearest name.
        """
        key = normalize(name)
        if key in self._exact:
            return self._exact[key]

        best_id, best_score = None, 0.0
        for pid, cand, cand_team in self._entries:
            if team and cand_team and cand_team.upper() != team.upper():
                continue
            score = similarity(name, cand)
            if score > best_score:
                best_id, best_score = pid, score

        return best_id if best_score >= threshold else None

    def resolve_all(
        self, names: Sequence[str], team: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        return {n: self.resolve(n, team=team) for n in names}
