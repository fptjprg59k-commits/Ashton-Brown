"""Game status as a semantic system.

The rule this module exists to enforce: **green is not a colour, it is
HALFTIME.** Nothing downstream should ever style a card directly. A card asks
the status what tone it carries, what label to print, and whether it is
actionable, and renders that. Adding a new state means adding it here once,
and every surface picks it up.

That indirection is what makes the dashboard correct rather than merely
pretty. Feeds disagree about how to describe the break - some report a
dedicated halftime status, some report "end of period" with the period set to
2, some just leave the clock at zero - and all three mean the same thing to a
bettor. Resolving that in one place is the difference between a dashboard that
turns green at halftime and one that turns green *most* of the time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Tone(str, Enum):
    """Visual family a status belongs to. Surfaces map tones to tokens.

    Deliberately smaller than the status set: several statuses share a tone
    (postponed and canceled are both ``WARNING``), and no status invents its
    own colour.
    """

    UPCOMING = "upcoming"
    LIVE = "live"
    HALFTIME = "halftime"
    FINAL = "final"
    WARNING = "warning"


class GameStatus(str, Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    END_PERIOD = "end_period"      # end of Q1 or Q3 - live, but stopped
    HALFTIME = "halftime"
    DELAYED = "delayed"
    FINAL = "final"
    POSTPONED = "postponed"
    CANCELED = "canceled"

    @property
    def tone(self) -> Tone:
        return {
            GameStatus.SCHEDULED: Tone.UPCOMING,
            GameStatus.LIVE: Tone.LIVE,
            GameStatus.END_PERIOD: Tone.LIVE,
            GameStatus.HALFTIME: Tone.HALFTIME,
            GameStatus.DELAYED: Tone.WARNING,
            GameStatus.FINAL: Tone.FINAL,
            GameStatus.POSTPONED: Tone.WARNING,
            GameStatus.CANCELED: Tone.WARNING,
        }[self]

    @property
    def is_live(self) -> bool:
        """In progress in the broad sense - the ball is still to be played."""
        return self in (GameStatus.LIVE, GameStatus.END_PERIOD, GameStatus.HALFTIME)

    @property
    def has_score(self) -> bool:
        return self is not GameStatus.SCHEDULED and self is not GameStatus.POSTPONED

    @property
    def is_actionable(self) -> bool:
        """Whether a halftime prop board can be produced for this game.

        The single reason this whole application exists, expressed as a
        property of the status rather than as a check scattered through the UI.
        """
        return self is GameStatus.HALFTIME

    @property
    def sort_key(self) -> int:
        """Halftime first, then live, then upcoming, then done.

        A dashboard is read top-down under time pressure, so the games that can
        still be acted on sit above the ones that cannot.
        """
        return {
            GameStatus.HALFTIME: 0,
            GameStatus.END_PERIOD: 1,
            GameStatus.LIVE: 1,
            GameStatus.DELAYED: 2,
            GameStatus.SCHEDULED: 3,
            GameStatus.FINAL: 4,
            GameStatus.POSTPONED: 5,
            GameStatus.CANCELED: 5,
        }[self]


class SeasonType(str, Enum):
    PRESEASON = "preseason"
    REGULAR = "regular"
    POSTSEASON = "postseason"
    OFFSEASON = "offseason"

    @property
    def badge(self) -> Optional[str]:
        """Short label, or None when it needs no calling out."""
        return {
            SeasonType.PRESEASON: "PRE",
            SeasonType.REGULAR: None,
            SeasonType.POSTSEASON: "POST",
            SeasonType.OFFSEASON: None,
        }[self]

    @classmethod
    def from_espn(cls, value) -> "SeasonType":
        """ESPN encodes season type as an integer: 1 pre, 2 regular, 3 post."""
        try:
            n = int(value)
        except (TypeError, ValueError):
            return cls.REGULAR
        return {1: cls.PRESEASON, 2: cls.REGULAR, 3: cls.POSTSEASON, 4: cls.POSTSEASON}.get(
            n, cls.REGULAR
        )


# --------------------------------------------------------------------------
# Derivation
# --------------------------------------------------------------------------

#: Feed status strings that mean the game will not be played.
_ABANDONED = {
    "STATUS_POSTPONED": GameStatus.POSTPONED,
    "STATUS_CANCELED": GameStatus.CANCELED,
    "STATUS_CANCELLED": GameStatus.CANCELED,
    "STATUS_SUSPENDED": GameStatus.POSTPONED,
    "STATUS_FORFEIT": GameStatus.CANCELED,
}

_FINISHED = {
    "STATUS_FINAL",
    "STATUS_FULL_TIME",
    "STATUS_FINAL_OVERTIME",
    "STATUS_FINAL_PEN",
}

_DELAYED = {"STATUS_DELAYED", "STATUS_RAIN_DELAY", "STATUS_WEATHER_DELAY"}


def derive_status(
    raw_status: Optional[str],
    period: Optional[int] = None,
    clock: Optional[str] = None,
    completed: bool = False,
) -> GameStatus:
    """Resolve a feed's status description into one semantic state.

    Order matters. Abandonment and completion are checked before anything
    else, because a postponed game can still carry a stale period and clock.

    Halftime is then recognised three ways, because feeds are inconsistent
    about it:

    1. an explicit halftime status;
    2. "end of period" with the period at 2 - the break, described positionally;
    3. an in-progress game sitting at period 2 with the clock at zero, which is
       what a feed shows in the seconds before it switches to either of the
       above.

    Missing all three would leave the most important state in the application
    rendering as an ordinary live game.
    """
    name = (raw_status or "").strip().upper()
    if name and not name.startswith("STATUS_"):
        name = f"STATUS_{name}"

    if name in _ABANDONED:
        return _ABANDONED[name]

    if completed or name in _FINISHED:
        return GameStatus.FINAL

    if name in _DELAYED:
        return GameStatus.DELAYED

    at_period_end = _clock_is_zero(clock)

    if name == "STATUS_HALFTIME":
        return GameStatus.HALFTIME

    if name == "STATUS_END_PERIOD":
        return GameStatus.HALFTIME if period == 2 else GameStatus.END_PERIOD

    if name in ("STATUS_IN_PROGRESS", "STATUS_LIVE", "STATUS_END_OF_PERIOD"):
        if period == 2 and at_period_end:
            return GameStatus.HALFTIME
        if at_period_end and period:
            return GameStatus.END_PERIOD
        return GameStatus.LIVE

    if name in ("STATUS_SCHEDULED", "STATUS_PRE_GAME", "STATUS_PREGAME", ""):
        return GameStatus.SCHEDULED

    # Unknown status with a live-looking period: treat as live rather than
    # silently claiming the game has not started.
    if period:
        return GameStatus.HALFTIME if (period == 2 and at_period_end) else GameStatus.LIVE
    return GameStatus.SCHEDULED


def _clock_is_zero(clock: Optional[str]) -> bool:
    if clock is None:
        return False
    text = str(clock).strip()
    if not text:
        return False
    return text in {"0:00", "00:00", "0.0", "0", "0:00.0", "-"}


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------


def ordinal_period(period: Optional[int]) -> str:
    """Human name for a period, including overtime."""
    if not period:
        return ""
    if period <= 4:
        return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[period]
    if period == 5:
        return "OT"
    return f"{period - 4}OT"


@dataclass(frozen=True)
class StatusDisplay:
    """Everything a surface needs to render a status, and nothing more."""

    status: GameStatus
    label: str                     # the loud line: HALFTIME, FINAL, 2nd 08:42
    detail: Optional[str] = None   # the quiet line under it

    @property
    def tone(self) -> Tone:
        return self.status.tone


def describe(
    status: GameStatus,
    period: Optional[int] = None,
    clock: Optional[str] = None,
    start_time: Optional[datetime] = None,
    now: Optional[datetime] = None,
    went_to_overtime: bool = False,
) -> StatusDisplay:
    """Build the label and sub-label for a status.

    ``now`` is injected rather than read from the clock so the countdown is
    testable and so a rendered dashboard is reproducible.
    """
    if status is GameStatus.HALFTIME:
        return StatusDisplay(status, "HALFTIME", "2nd half to come")

    if status is GameStatus.FINAL:
        return StatusDisplay(status, "FINAL/OT" if went_to_overtime else "FINAL")

    if status is GameStatus.END_PERIOD:
        return StatusDisplay(status, f"END {ordinal_period(period)}".strip())

    if status is GameStatus.LIVE:
        pieces = [p for p in (ordinal_period(period), _clean_clock(clock)) if p]
        return StatusDisplay(status, " · ".join(pieces) or "LIVE")

    if status is GameStatus.DELAYED:
        return StatusDisplay(status, "DELAYED")

    if status is GameStatus.POSTPONED:
        return StatusDisplay(status, "POSTPONED")

    if status is GameStatus.CANCELED:
        return StatusDisplay(status, "CANCELED")

    # Scheduled
    if start_time is None:
        return StatusDisplay(status, "SCHEDULED")

    label = _format_kickoff(start_time)
    return StatusDisplay(status, label, _countdown(start_time, now))


def _clean_clock(clock: Optional[str]) -> str:
    text = (clock or "").strip()
    return "" if text in ("", "-", "0.0") else text


def _format_kickoff(start_time: datetime) -> str:
    """Local wall-clock kickoff, without a leading zero on the hour."""
    hour = start_time.hour % 12 or 12
    meridiem = "AM" if start_time.hour < 12 else "PM"
    return f"{hour}:{start_time.minute:02d} {meridiem}"


def _countdown(start_time: datetime, now: Optional[datetime]) -> Optional[str]:
    """"Starts in 42 min", but only while that is the useful thing to say."""
    if now is None:
        return None

    start = _as_aware(start_time)
    current = _as_aware(now)
    delta = (start - current).total_seconds()

    if delta <= 0:
        return "Starting now"
    minutes = int(delta // 60)
    if minutes < 1:
        return "Starts in under a minute"
    if minutes < 60:
        return f"Starts in {minutes} min"
    hours = minutes / 60.0
    if hours < 24:
        # Half-hour resolution reads better than "in 3.7 hours".
        rounded = round(hours * 2) / 2
        unit = "hour" if rounded == 1 else "hours"
        text = f"{rounded:g}"
        return f"Starts in {text} {unit}"
    days = int(hours // 24)
    return f"In {days} day{'s' if days != 1 else ''}"


def _as_aware(value: datetime) -> datetime:
    """Treat a naive datetime as UTC rather than raising mid-render."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
