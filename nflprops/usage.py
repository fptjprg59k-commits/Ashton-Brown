"""Opportunity allocation: who gets the ball, and who stops getting it.

Usage share is the floor under every counting prop. A receiver with a 28%
target share has a very different distribution from one at 12% even if their
yards-per-target are identical, and an injury to a co-star is the fastest way
for that floor to move mid-game.

Three distinct mechanisms live here:

1. **Redistribution** - when a player is unavailable his share does not
   evaporate, it flows to teammates, weighted toward same-role replacements.
2. **Scheme funnels** - the defense reshapes *which* available player gets
   targeted, independent of the offense's intent.
3. **Rest risk** - in a decided game, stars stop accumulating in the fourth
   quarter, which truncates the top of their distribution exactly where an
   over would otherwise cash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

from .distributions import logistic
from .models import Alignment, Player, Position

if TYPE_CHECKING:  # pragma: no cover
    from .models import DefenseProfile, TeamState
    from .priors import Priors


@dataclass
class UsagePool:
    """Resolved, normalised opportunity weights for one team.

    Weight lists are index-aligned with ``players`` so the hot loop can sample
    with a single cumulative pass and no dictionary lookups.
    """

    players: List[Player] = field(default_factory=list)
    target_w: List[float] = field(default_factory=list)
    rush_w: List[float] = field(default_factory=list)
    rz_target_w: List[float] = field(default_factory=list)
    rz_rush_w: List[float] = field(default_factory=list)

    target_total: float = 0.0
    rush_total: float = 0.0
    rz_target_total: float = 0.0
    rz_rush_total: float = 0.0

    #: Per-player efficiency multipliers from coverage matchup, index-aligned.
    eff_mult: List[float] = field(default_factory=list)

    def index_of(self, player_id: str) -> int:
        for i, p in enumerate(self.players):
            if p.player_id == player_id:
                return i
        return -1


def _funnel_multiplier(player: Player, defense: "DefenseProfile") -> float:
    """How the defense's scheme reshapes this player's share of targets."""
    align = player.alignment or Alignment.PERIMETER
    base = {
        Alignment.SLOT: defense.slot_funnel,
        Alignment.INLINE: defense.slot_funnel,
        Alignment.PERIMETER: defense.perimeter_funnel,
        Alignment.BACKFIELD: defense.rb_target_funnel,
    }[align]

    # Deep funnels act on top of alignment, scaled by how vertical the player
    # actually is: a 4-yard aDOT slot man is unaffected by a defense that
    # takes away the deep third.
    depth_factor = max(0.0, min(1.0, (player.adot - 6.0) / 10.0))
    deep = 1.0 + (defense.deep_funnel - 1.0) * depth_factor

    shadow = defense.shadow.get(player.player_id, 1.0)
    return max(0.05, base * deep * shadow)


def build_pool(team: "TeamState", defense: "DefenseProfile") -> UsagePool:
    """Resolve a team's usage weights against a specific defensive profile.

    Unavailable players are dropped; the remaining weights are renormalised,
    which is exactly the redistribution effect - the surviving players'
    relative shares are preserved while their absolute shares rise.
    """
    pool = UsagePool()

    for p in team.players:
        if p.position in (Position.K, Position.DEF):
            continue
        if not p.available:
            continue

        avail = p.injury.availability * p.snap_share
        if avail <= 0:
            continue

        funnel = _funnel_multiplier(p, defense)
        shadow = defense.shadow.get(p.player_id, 1.0)

        pool.players.append(p)
        pool.target_w.append(p.target_share * avail * funnel)
        pool.rush_w.append(p.rush_share * avail)
        pool.rz_target_w.append(p.rz_target_share * avail * funnel)
        pool.rz_rush_w.append(p.rz_rush_share * avail)
        # Shadow coverage suppresses efficiency as well as volume; injury
        # severity degrades effectiveness beyond simple snap loss.
        pool.eff_mult.append(p.injury.efficiency * (0.55 + 0.45 * shadow))

    _finalise(pool)
    return pool


def _finalise(pool: UsagePool) -> None:
    pool.target_total = sum(pool.target_w)
    pool.rush_total = sum(pool.rush_w)
    pool.rz_target_total = sum(pool.rz_target_w)
    pool.rz_rush_total = sum(pool.rz_rush_w)

    # Fall back to plain target/rush weights if red-zone shares were not
    # supplied, rather than silently producing zero-TD projections.
    if pool.rz_target_total <= 0:
        pool.rz_target_w = list(pool.target_w)
        pool.rz_target_total = pool.target_total
    if pool.rz_rush_total <= 0:
        pool.rz_rush_w = list(pool.rush_w)
        pool.rz_rush_total = pool.rush_total


def shrink(observed: float, prior: float, n: float, prior_strength: float) -> float:
    """Blend a first-half observation toward its pregame prior.

    ``prior_strength`` is expressed as an equivalent sample size. With a half's
    worth of data (say 16 pass attempts) against a prior strength of 45, the
    observation gets about a quarter of the weight - which is the right answer
    for a sample that small, and the reason a receiver who caught everything in
    the first half should not be projected as a 90% catch-rate player.
    """
    if n <= 0:
        return prior
    w = n / (n + prior_strength)
    return w * observed + (1.0 - w) * prior


def apply_first_half_shrinkage(team: "TeamState", priors: "Priors") -> None:
    """Nudge usage shares toward what the first half actually showed.

    Mutates the team in place, so callers should work on a copy of the game
    state. The observed share gets modest weight; the pregame prior keeps most
    of it. This is where "an injury to a co-star spiked his share" enters the
    model empirically, without anyone having to hand-edit a projection.
    """
    a = priors.section("adjustment")
    strength = a["usage_prior_weight"] * 60.0

    total_targets = sum(p.h1.targets for p in team.players)
    total_carries = sum(p.h1.rush_att for p in team.players)

    for p in team.players:
        if total_targets >= 6 and p.position is not Position.QB:
            observed = p.h1.targets / total_targets
            p.target_share = shrink(observed, p.target_share, total_targets, strength)
        if total_carries >= 5 and p.position is not Position.QB:
            observed = p.h1.rush_att / total_carries
            p.rush_share = shrink(observed, p.rush_share, total_carries, strength)


def rest_hazard(
    player: Player,
    margin: float,
    seconds_remaining: float,
    priors: "Priors",
) -> float:
    """Per-drive probability a star is pulled with the game decided.

    ``margin`` is this player's team's lead (positive = winning). Only leading
    teams rest starters; a trailing team plays its stars to the whistle.
    """
    b = priors.section("blowout")
    if not player.star:
        return 0.0
    if seconds_remaining > b["start_seconds"]:
        return 0.0
    if margin <= 0:
        return 0.0

    z = (margin - b["margin_midpoint"]) / b["margin_scale"]
    p = logistic(z) * b["max_pull_prob_per_drive"]

    role_mult = {
        Position.RB: b["rb_multiplier"],
        Position.QB: b["qb_multiplier"],
        Position.WR: b["wr_multiplier"],
        Position.TE: b["wr_multiplier"],
    }.get(player.position, 1.0)

    # The clock also matters: a 24-point lead with 12 minutes left still sees
    # starters take a series; the same lead at 5 minutes does not.
    time_factor = 1.0 - (seconds_remaining / max(b["start_seconds"], 1.0)) * 0.55
    return max(0.0, min(0.95, p * role_mult * time_factor))


def redistribute_after_pull(pool: UsagePool, pulled: Dict[int, bool]) -> Dict[str, float]:
    """Recompute weight totals once some players have been benched.

    Returns the new totals; the caller keeps the per-index weights and simply
    skips pulled indices when sampling.
    """
    t = sum(w for i, w in enumerate(pool.target_w) if not pulled.get(i))
    r = sum(w for i, w in enumerate(pool.rush_w) if not pulled.get(i))
    rzt = sum(w for i, w in enumerate(pool.rz_target_w) if not pulled.get(i))
    rzr = sum(w for i, w in enumerate(pool.rz_rush_w) if not pulled.get(i))
    return {"target": t, "rush": r, "rz_target": rzt, "rz_rush": rzr}
