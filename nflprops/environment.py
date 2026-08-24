"""Weather and officiating adjustments.

Both are real but easy to overrate. Wind is the only weather variable with a
large, reliable effect on passing and kicking props; temperature and rain
matter much less than broadcast narrative suggests. Officiating is genuinely
third-order and is modelled as a small drive-extension nudge so that it can
never dominate a projection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import Precip

if TYPE_CHECKING:  # pragma: no cover
    from .models import Officials, Weather
    from .priors import Priors


def completion_penalty(weather: "Weather", adot: float, priors: "Priors") -> float:
    """Absolute reduction in completion probability for a target at ``adot``.

    Wind hurts deep throws far more than screens, so the depth-scaled term is
    separated from the flat term. A 10-yard dig in a 25 mph crosswind is a
    materially harder throw; a swing pass to a back is nearly unaffected.
    """
    w = priors.section("weather")
    penalty = 0.0

    wind = weather.effective_wind
    over = max(0.0, wind - w["wind_threshold"])
    if over > 0:
        depth_factor = max(0.0, (adot - 6.0) / 10.0)
        penalty += w["wind_comp_penalty"] * over
        penalty += w["wind_deep_penalty"] * over * depth_factor

    precip = weather.effective_precip
    if precip is Precip.LIGHT_RAIN:
        penalty += w["rain_comp_penalty"]
    elif precip is Precip.HEAVY_RAIN:
        penalty += w["heavy_rain_comp_penalty"]
    elif precip is Precip.SNOW:
        penalty += w["snow_comp_penalty"]

    if not weather.dome and weather.temp_f < w["cold_threshold"]:
        penalty += w["cold_comp_penalty"] * (w["cold_threshold"] - weather.temp_f) / 10.0

    return max(0.0, penalty)


def explosiveness_damp(weather: "Weather", priors: "Priors") -> float:
    """Multiplier on a player's explosiveness index.

    Poor footing compresses the big-play tail: receivers cannot separate deep
    and backs cannot bounce runs outside. This shrinks variance without moving
    the projection, which is the correct shape for bad-weather games.
    """
    w = priors.section("weather")
    damp = 1.0
    precip = weather.effective_precip
    if precip is not Precip.NONE:
        severity = {
            Precip.LIGHT_RAIN: 0.5,
            Precip.HEAVY_RAIN: 1.0,
            Precip.SNOW: 1.0,
        }[precip]
        damp -= w["precip_explosive_damp"] * severity

    over = max(0.0, weather.effective_wind - w["wind_threshold"])
    damp -= min(0.18, 0.008 * over)
    return max(0.55, damp)


def fumble_multiplier(weather: "Weather", priors: "Priors") -> float:
    w = priors.section("weather")
    if weather.effective_precip is Precip.NONE:
        return 1.0
    return w["precip_fumble_mult"]


def fg_penalty(weather: "Weather", priors: "Priors") -> float:
    """Absolute reduction in field goal make probability."""
    w = priors.section("weather")
    over = max(0.0, weather.effective_wind - w["wind_threshold"])
    penalty = w["wind_fg_penalty"] * over
    if weather.effective_precip in (Precip.HEAVY_RAIN, Precip.SNOW):
        penalty += 0.035
    return penalty


def drive_extension_prob(officials: "Officials", priors: "Priors") -> float:
    """Per-drive chance a penalty gifts the offense a fresh set of downs.

    Deliberately small. A crew calling three extra flags a game is worth a
    couple of percentage points of drive survival, not a reshaped projection.
    """
    o = priors.section("officials")
    excess = officials.penalties_per_game - o["league_penalties_per_game"]
    base = 0.115 * officials.offensive_holding_rate  # holdings also kill drives
    bump = o["extension_per_penalty"] * excess * officials.dpi_rate
    return max(0.0, min(0.25, base + bump))
