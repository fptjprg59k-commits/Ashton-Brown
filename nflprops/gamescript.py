"""Game script: how the scoreboard and the clock rewrite play-calling.

This is the dominant factor in second-half prop outcomes. A running back's
carry prop and a receiver's yardage prop are largely a bet on which side of
the script their team ends up on, and the two are close to mirror images.

The driver is a leverage term

    z = -point_diff / sqrt(minutes_remaining)

which captures the interaction the raw deficit misses: down 7 with 28 minutes
left is a normal football game, down 7 with 3 minutes left is a two-minute
drill. Effects are passed through ``tanh`` so they saturate rather than run
off to absurd values in a blowout.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .models import TeamState
    from .priors import Priors


def urgency_z(point_diff: float, seconds_remaining: float, z_scale: float) -> float:
    """Leverage term from the offense's point of view.

    ``point_diff`` is offense score minus defense score, so a trailing team has
    a negative diff and a positive ``z``.
    """
    minutes = max(seconds_remaining / 60.0, 0.75)
    z = -point_diff / math.sqrt(minutes)
    return math.tanh(z / z_scale)


def pass_rate(
    team: "TeamState",
    point_diff: float,
    seconds_remaining: float,
    priors: "Priors",
) -> float:
    """Probability the next play is a dropback, given the current script."""
    s = priors.section("script")
    tz = urgency_z(point_diff, seconds_remaining, s["z_scale"])
    rate = team.base_pass_rate + s["pass_rate_beta"] * tz
    return _clip(rate, s["pass_rate_min"], s["pass_rate_max"])


def seconds_per_play(
    team: "TeamState",
    point_diff: float,
    seconds_remaining: float,
    priors: "Priors",
) -> float:
    """Expected clock burned per offensive snap, including pre-snap time.

    Two hard regimes override the smooth curve, because late-game behaviour is
    not a gentle gradient: a trailing offense inside five minutes goes
    no-huddle, and a leading offense inside four minutes milks the play clock.
    """
    s = priors.section("script")
    tz = urgency_z(point_diff, seconds_remaining, s["z_scale"])

    # Clock management intensity scales with time pressure. A team up 17 with
    # thirty minutes left does not milk the play clock - it just runs the ball,
    # and the run itself is what keeps the clock moving. Applying the full
    # tempo swing from the opening snap of the second half made leading teams
    # burn ~39 seconds a play all half, which starved both offenses of snaps.
    pressure = 1.0 - min(1.0, max(0.0, seconds_remaining) / 1800.0)
    scale = s["pace_time_floor"] + (1.0 - s["pace_time_floor"]) * pressure
    spp = team.base_sec_per_play * (1.0 - s["pace_beta"] * tz * scale)

    trailing = point_diff < 0
    leading = point_diff > 0

    if trailing and seconds_remaining <= s["hurry_up_seconds"]:
        spp = min(spp, s["hurry_up_sec_per_play"])
    elif leading and seconds_remaining <= s["clock_kill_seconds"]:
        spp = max(spp, s["clock_kill_sec_per_play"])

    return max(12.0, spp)


def is_desperate(point_diff: float, seconds_remaining: float, priors: "Priors") -> bool:
    """Whether the offense should abandon normal fourth-down and clock logic."""
    fd = priors.section("fourth_down")
    if seconds_remaining > fd["desperation_seconds"]:
        return False
    if point_diff >= 0:
        return False
    # Need at least two scores, or one score with very little time.
    return point_diff <= -9 or seconds_remaining <= 150


def possessions_remaining_estimate(
    seconds_remaining: float, avg_sec_per_play: float, plays_per_drive: float = 5.9
) -> float:
    """Rough count of remaining drives, used for reporting and sanity checks.

    The simulator does not rely on this; it is here so reports can explain
    *why* a volume prop looks unreachable ("only ~4.2 drives left").
    """
    drive_seconds = plays_per_drive * avg_sec_per_play + 18.0
    return max(0.0, seconds_remaining / max(drive_seconds, 1.0))


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x
