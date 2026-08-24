"""Stake prop ingestion.

Three input modes, in order of how reliably they work:

1. **Paste** - copy the prop list out of the app and hand it over as text.
   Always works, needs no credentials, breaks only if the app's wording
   changes. This is the recommended path.
2. **File** - JSON or CSV in the documented schema, for a saved snapshot or
   an export from tooling you already run.
3. **HTTP** - a JSON endpoint you supply and are authorised to call.

On the third mode, plainly: there is no public Stake odds API, and scraping a
sportsbook generally violates its terms of service. No endpoint is hardcoded
here and none is discovered for you. If you have lawful programmatic access
(an affiliate or partner feed, or your own account data via a route the
operator permits), point ``url`` at it and the schema below applies. Otherwise
use paste mode, which is what the parser is built around.
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
from typing import Dict, List, Optional, Tuple

from ..models import Market, Prop, PropScope, Side
from .base import PropsProvider, ProviderError

# --------------------------------------------------------------------------
# Market vocabulary
# --------------------------------------------------------------------------

MARKET_ALIASES: Dict[str, Market] = {
    "passing + rushing yards": Market.PASS_RUSH_YDS,
    "pass + rush yards": Market.PASS_RUSH_YDS,
    "passing and rushing yards": Market.PASS_RUSH_YDS,
    "rushing + receiving yards": Market.RUSH_REC_YDS,
    "rush + rec yards": Market.RUSH_REC_YDS,
    "rushing and receiving yards": Market.RUSH_REC_YDS,
    "rush and receive yards": Market.RUSH_REC_YDS,
    "longest reception": Market.LONGEST_REC,
    "longest completion": Market.LONGEST_REC,
    "longest rush": Market.LONGEST_RUSH,
    "passing touchdowns": Market.PASS_TDS,
    "passing tds": Market.PASS_TDS,
    "pass tds": Market.PASS_TDS,
    "touchdown passes": Market.PASS_TDS,
    "passing yards": Market.PASS_YDS,
    "pass yards": Market.PASS_YDS,
    "pass yds": Market.PASS_YDS,
    "passing completions": Market.PASS_CMP,
    "completions": Market.PASS_CMP,
    "passing attempts": Market.PASS_ATT,
    "pass attempts": Market.PASS_ATT,
    "interceptions thrown": Market.PASS_INT,
    "interceptions": Market.PASS_INT,
    "receiving yards": Market.REC_YDS,
    "reception yards": Market.REC_YDS,
    "rec yards": Market.REC_YDS,
    "rec yds": Market.REC_YDS,
    "receiving touchdowns": Market.REC_TDS,
    "receiving tds": Market.REC_TDS,
    "receptions": Market.REC,
    "catches": Market.REC,
    "rushing yards": Market.RUSH_YDS,
    "rush yards": Market.RUSH_YDS,
    "rush yds": Market.RUSH_YDS,
    "rushing touchdowns": Market.RUSH_TDS,
    "rushing tds": Market.RUSH_TDS,
    "rushing attempts": Market.RUSH_ATT,
    "rush attempts": Market.RUSH_ATT,
    "carries": Market.RUSH_ATT,
    "anytime touchdown scorer": Market.ANYTIME_TD,
    "anytime touchdown": Market.ANYTIME_TD,
    "anytime td": Market.ANYTIME_TD,
    "to score a touchdown": Market.ANYTIME_TD,
    "kicking points": Market.KICKING_POINTS,
    "team total points": Market.TEAM_TOTAL,
    "team total": Market.TEAM_TOTAL,
    "total points": Market.GAME_TOTAL,
    "game total": Market.GAME_TOTAL,
}

# Longest aliases first so "passing yards" never shadows "passing + rushing yards".
_ALIASES_BY_LENGTH: List[Tuple[str, Market]] = sorted(
    MARKET_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True
)

_ODDS_RE = re.compile(r"(?<![\w.])([+-]\d{2,5})(?![\w.])")
_NUMBER_RE = re.compile(r"(?<![\w+-])(\d+(?:\.\d+)?)(?![\w])")
_SIDE_RE = re.compile(r"\b(over|under|o|u|yes|no)\b", re.IGNORECASE)
_SECOND_HALF_RE = re.compile(r"\b(2nd|second)\s+half\b", re.IGNORECASE)
_FIRST_HALF_RE = re.compile(r"\b(1st|first)\s+half\b", re.IGNORECASE)


class StakePropsProvider(PropsProvider):
    """Loads offered lines from pasted text, a file, or a JSON endpoint."""

    name = "stake"

    def __init__(
        self,
        text: Optional[str] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 15.0,
        default_scope: PropScope = PropScope.FULL_GAME,
    ) -> None:
        if not any((text, path, url)):
            raise ValueError("provide one of text=, path=, or url=")
        self.text = text
        self.path = path
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.default_scope = default_scope
        #: Lines the parser could not understand, surfaced rather than dropped.
        self.rejected: List[str] = []

    # -- public -----------------------------------------------------------

    def fetch(self, game_id: Optional[str] = None) -> List[Prop]:
        if self.text is not None:
            return self.parse_text(self.text, self.default_scope, self.rejected)
        if self.path is not None:
            return self._from_file(self.path)
        return self._from_url()

    # -- file / http ------------------------------------------------------

    def _from_file(self, path: str) -> List[Prop]:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        stripped = raw.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            return self._from_json(json.loads(raw))
        if path.lower().endswith(".csv") or _looks_like_csv(raw):
            return self._from_csv(raw)
        return self.parse_text(raw, self.default_scope, self.rejected)

    def _from_url(self) -> List[Prop]:
        req = urllib.request.Request(self.url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            raise ProviderError(f"stake feed request failed: {exc}") from exc
        return self._from_json(payload)

    def _from_json(self, payload) -> List[Prop]:
        """Schema: ``{"props": [{player, market, side, line, odds, ...}]}``."""
        rows = payload.get("props", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ProviderError("expected a list of props")

        out: List[Prop] = []
        for i, row in enumerate(rows):
            try:
                out.append(self._prop_from_row(row, i))
            except (KeyError, ValueError) as exc:
                self.rejected.append(f"{row!r}: {exc}")
        return out

    def _from_csv(self, raw: str) -> List[Prop]:
        reader = csv.DictReader(io.StringIO(raw))
        out: List[Prop] = []
        for i, row in enumerate(reader):
            try:
                out.append(self._prop_from_row(row, i))
            except (KeyError, ValueError) as exc:
                self.rejected.append(f"{row!r}: {exc}")
        return out

    def _prop_from_row(self, row: Dict, index: int) -> Prop:
        market = _coerce_market(row["market"])
        side_raw = str(row.get("side", "over")).lower()
        side = {
            "over": Side.OVER, "o": Side.OVER,
            "under": Side.UNDER, "u": Side.UNDER,
            "yes": Side.YES, "no": Side.NO,
        }.get(side_raw, Side.OVER)

        scope_raw = str(row.get("scope", self.default_scope.value)).lower()
        scope = (
            PropScope.SECOND_HALF
            if scope_raw in ("second_half", "2h", "h2")
            else PropScope.FULL_GAME
        )

        opposing = row.get("opposing_odds")
        return Prop(
            prop_id=str(row.get("prop_id") or f"{self.name}-{index}"),
            market=market,
            side=side,
            line=float(row.get("line", 0.5)),
            odds=int(row["odds"]),
            scope=scope,
            player_id=row.get("player_id"),
            player_name=str(row.get("player") or row.get("player_name") or ""),
            team=row.get("team"),
            opposing_odds=int(opposing) if opposing not in (None, "") else None,
            book=str(row.get("book", self.name)),
        )

    # -- text paste -------------------------------------------------------

    @staticmethod
    def parse_text(
        text: str,
        default_scope: PropScope = PropScope.FULL_GAME,
        rejected: Optional[List[str]] = None,
    ) -> List[Prop]:
        """Parse a block copied out of the app.

        Tolerant of ordering and separators. The key disambiguation rule is
        that American odds always carry an explicit sign (``-115``, ``+140``)
        while a line never does (``65.5``, ``249.5``), so the two can never be
        confused no matter where they appear.
        """
        rejected = rejected if rejected is not None else []
        out: List[Prop] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parsed = _parse_one(line, default_scope, len(out))
            if parsed is None:
                rejected.append(raw_line)
            else:
                out.append(parsed)

        return out


def _parse_one(line: str, default_scope: PropScope, index: int) -> Optional[Prop]:
    if _FIRST_HALF_RE.search(line):
        return None  # first-half markets are already settled at the break

    scope = PropScope.SECOND_HALF if _SECOND_HALF_RE.search(line) else default_scope

    odds_matches = _ODDS_RE.findall(line)
    if not odds_matches:
        return None
    odds = int(odds_matches[0])
    opposing = int(odds_matches[1]) if len(odds_matches) > 1 else None

    working = _ODDS_RE.sub(" ", line)

    market = None
    lowered = working.lower()
    alias_found = ""
    for alias, m in _ALIASES_BY_LENGTH:
        if alias in lowered:
            market, alias_found = m, alias
            break
    if market is None:
        return None

    start = lowered.index(alias_found)
    working = working[:start] + " " + working[start + len(alias_found):]

    side_match = _SIDE_RE.search(working)
    side = Side.OVER
    if side_match:
        token = side_match.group(1).lower()
        side = {
            "over": Side.OVER, "o": Side.OVER,
            "under": Side.UNDER, "u": Side.UNDER,
            "yes": Side.YES, "no": Side.NO,
        }[token]
        working = working[: side_match.start()] + " " + working[side_match.end():]

    numbers = _NUMBER_RE.findall(working)
    if numbers:
        value = float(numbers[0])
        working = _NUMBER_RE.sub(" ", working, count=1)
    else:
        value = 0.5  # anytime-TD style markets carry an implicit 0.5

    name = _clean_name(working)
    if not name:
        return None

    return Prop(
        prop_id=f"stake-{index}",
        market=market,
        side=side,
        line=value,
        odds=odds,
        scope=scope,
        player_name=name,
        opposing_odds=opposing,
        book="stake",
    )


def _clean_name(text: str) -> str:
    text = _SECOND_HALF_RE.sub(" ", text)
    text = re.sub(r"[|\-–—:()\[\],]+", " ", text)
    text = re.sub(r"\b(prop|player|props|line|odds|scorer)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_like_csv(raw: str) -> bool:
    """A header row naming the required columns, not merely a stray comma."""
    lines = raw.splitlines()
    if not lines:
        return False
    header = lines[0].lower()
    return "," in header and "odds" in header and "market" in header


def _coerce_market(value) -> Market:
    if isinstance(value, Market):
        return value
    text = str(value).strip().lower()
    try:
        return Market(text)
    except ValueError:
        pass
    if text in MARKET_ALIASES:
        return MARKET_ALIASES[text]
    for alias, m in _ALIASES_BY_LENGTH:
        if alias in text:
            return m
    raise ValueError(f"unrecognised market {value!r}")
