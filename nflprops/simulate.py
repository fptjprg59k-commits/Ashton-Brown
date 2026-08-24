"""Monte Carlo engine for the remainder of a game.

Design note: this simulates **drives and downs**, not an abstract projection.
That choice is what makes the listed factors fall out of the model instead of
being bolted onto it:

* Game script emerges because the simulated score feeds back into play-calling
  on the very next snap.
* Correlation between props is automatic - when a simulated path has the QB
  throwing for 340, his WR1's yardage is high on that same path. Every prop is
  evaluated against the same set of paths, so joint questions (parlays, "does
  the over on both sides of a stack hit") are answerable without a copula.
* Blowout truncation and rest risk act on the simulated margin, so a player
  gets benched on exactly the paths where his team ran away with it.

The output is a matrix of per-simulation second-half stat lines. Adding
first-half actuals and comparing against a line happens in ``props.py``.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import environment, gamescript, usage
from .distributions import Sampler, completion_prob, fg_make_prob, yardline_to_fg_distance
from .models import GameState, Player, Position, TeamState
from .priors import Priors

# Drive outcomes
TD = "td"
FG_GOOD = "fg_good"
FG_MISS = "fg_miss"
PUNT = "punt"
TURNOVER = "turnover"
DOWNS = "downs"
END_HALF = "end_half"

PLAYER_STATS = (
    "pass_att", "pass_cmp", "pass_yds", "pass_td", "interceptions",
    "rush_att", "rush_yds", "rush_td",
    "targets", "rec", "rec_yds", "rec_td",
    "longest_rec", "longest_rush",
    "fg_made", "xp_made",
)


@dataclass
class SimResult:
    """Per-simulation second-half output.

    ``stats`` maps ``"<player_id>.<stat>"`` to an array of length ``n_sims``.
    Team scoring lives under ``"@<team>.points"`` and the game total under
    ``"@game.points"``.
    """

    n_sims: int
    stats: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def get(self, player_id: str, stat: str) -> np.ndarray:
        key = f"{player_id}.{stat}"
        arr = self.stats.get(key)
        if arr is None:
            return np.zeros(self.n_sims)
        return arr

    def team_points(self, abbr: str) -> np.ndarray:
        return self.stats.get(f"@{abbr}.points", np.zeros(self.n_sims))

    def game_points(self) -> np.ndarray:
        return self.stats.get("@game.points", np.zeros(self.n_sims))


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------


@dataclass
class TeamPrep:
    team: TeamState
    pool: usage.UsagePool
    qbs: List[Player]
    kicker: Optional[Player]
    rec_mu: List[float]        # index-aligned with pool.players
    rush_mu: List[float]
    explosive: List[float]
    catch_rate: List[float]
    adot: List[float]
    comp_penalty: List[float]  # weather penalty, depth-scaled per player
    sack_rate: float
    int_rate: float
    base_pass_rate: float


def _blend_team_efficiency(state: GameState, team: TeamState, priors: Priors) -> None:
    """Regress observed first-half efficiency toward the pregame prior.

    Half a game is a very small sample. A quarterback at 11.0 yards per attempt
    before the break is not an 11.0 YPA quarterback for the next thirty
    minutes, and the single most common way to misprice a halftime prop is to
    extrapolate the hot half. The shrinkage weight is deliberately heavy.
    """
    a = priors.section("adjustment")
    play = priors.section("play")
    prior_w = a["prior_weight"]
    # Express the prior as an equivalent sample size relative to the observed
    # attempts, so a team that threw 30 times gets more credit than one that
    # threw 9.
    strength = prior_w / max(1e-6, 1.0 - prior_w) * 14.0

    att = sum(p.h1.pass_att for p in team.players)
    yds = sum(p.h1.pass_yds for p in team.players)
    if att >= 5:
        observed_ratio = (yds / att) / play["league_pass_ypa"]
        team.off_pass_eff = usage.shrink(observed_ratio, team.off_pass_eff, att, strength)

    car = sum(p.h1.rush_att for p in team.players)
    ryds = sum(p.h1.rush_yds for p in team.players)
    if car >= 5:
        observed_ratio = (ryds / car) / play["league_rush_ypc"]
        team.off_rush_eff = usage.shrink(observed_ratio, team.off_rush_eff, car, strength)


def prepare(state: GameState, priors: Priors) -> Tuple[GameState, Dict[str, TeamPrep]]:
    """Clone the state and precompute everything invariant across simulations."""
    st = copy.deepcopy(state)
    preps: Dict[str, TeamPrep] = {}

    for team in (st.home, st.away):
        usage.apply_first_half_shrinkage(team, priors)
        _blend_team_efficiency(st, team, priors)

    damp = environment.explosiveness_damp(st.weather, priors)

    for team in (st.home, st.away):
        opp = st.opponent(team.abbr)
        pool = usage.build_pool(team, opp.defense)

        rec_mu, rush_mu, expl, catch, adots, pens = [], [], [], [], [], []
        for i, p in enumerate(pool.players):
            eff = pool.eff_mult[i]
            rec_mu.append(
                p.expected_ypr * team.off_pass_eff * opp.defense.pass_yds_mult * eff
            )
            rush_mu.append(
                p.ypc * team.off_rush_eff * opp.defense.rush_yds_mult * eff
            )
            expl.append(max(0.0, min(1.0, p.explosiveness * damp)))
            catch.append(min(0.995, p.catch_rate * opp.defense.comp_rate_mult))
            adots.append(p.adot)
            pens.append(environment.completion_penalty(st.weather, p.adot, priors))

        qbs = [p for p in team.players if p.position is Position.QB and p.available]
        qbs.sort(key=lambda p: (not p.is_starter_qb, p.name))
        if not qbs:
            qbs = [
                Player(
                    player_id=f"{team.abbr}_QB_UNKNOWN",
                    name=f"{team.abbr} QB",
                    position=Position.QB,
                    team=team.abbr,
                    is_starter_qb=True,
                )
            ]

        starter = qbs[0]
        preps[team.abbr] = TeamPrep(
            team=team,
            pool=pool,
            qbs=qbs,
            kicker=team.kicker(),
            rec_mu=rec_mu,
            rush_mu=rush_mu,
            explosive=expl,
            catch_rate=catch,
            adot=adots,
            comp_penalty=pens,
            sack_rate=starter.sack_rate * opp.defense.sack_rate_mult,
            int_rate=starter.int_rate * opp.defense.int_rate_mult,
            base_pass_rate=team.base_pass_rate,
        )

    return st, preps


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------


class GameSimulator:
    def __init__(self, state: GameState, priors: Optional[Priors] = None) -> None:
        self.priors = priors or Priors()
        self.state, self.preps = prepare(state, self.priors)
        self.s_script = self.priors.section("script")
        self.s_clock = self.priors.section("clock")
        self.s_play = self.priors.section("play")
        self.s_fd = self.priors.section("fourth_down")
        self.fg_wind_penalty = environment.fg_penalty(self.state.weather, self.priors)
        self.fumble_mult = environment.fumble_multiplier(self.state.weather, self.priors)
        self.extend_prob = environment.drive_extension_prob(
            self.state.officials, self.priors
        )
        self.adj = self.priors.section("adjustment")
        #: Tally of simulated drive outcomes, for calibration against the
        #: league-wide per-drive distribution. Accumulates across ``run``.
        self.drive_outcomes: Counter = Counter()
        #: (drives reaching the red zone, of which ended in a touchdown)
        self.redzone_trips: int = 0
        self.redzone_tds: int = 0

    # -- public -----------------------------------------------------------

    def run(self, n_sims: Optional[int] = None, seed: Optional[int] = None) -> SimResult:
        sim_cfg = self.priors.section("sim")
        n = int(n_sims or sim_cfg["n_sims"])
        seed = int(seed if seed is not None else sim_cfg["seed"])

        keys: List[str] = []
        for team in (self.state.home, self.state.away):
            for p in team.players:
                for stat in PLAYER_STATS:
                    keys.append(f"{p.player_id}.{stat}")
            keys.append(f"@{team.abbr}.points")
            keys.append(f"@{team.abbr}.plays")
            keys.append(f"@{team.abbr}.drives")
        keys.append("@game.points")

        arrays = {k: np.zeros(n, dtype=np.float64) for k in dict.fromkeys(keys)}
        sampler = Sampler(self.priors, seed=seed)

        for i in range(n):
            box = self._simulate_half(sampler)
            for k, v in box.items():
                arr = arrays.get(k)
                if arr is not None:
                    arr[i] = v

        result = SimResult(n_sims=n, stats=arrays)
        result.meta = {
            "seconds_simulated": self.state.seconds_remaining,
            "home": self.state.home.abbr,
            "away": self.state.away.abbr,
            "h1_home_score": self.state.home.score,
            "h1_away_score": self.state.away.score,
            "seed": seed,
        }
        return result

    # -- one simulated half ------------------------------------------------

    def _simulate_half(self, rs: Sampler) -> Dict[str, float]:
        box: Dict[str, float] = {}
        st = self.state

        scores = {st.home.abbr: float(st.home.score), st.away.abbr: float(st.away.score)}
        clock = float(st.seconds_remaining)

        # Per-simulation coordinator-adjustment shock. This widens the outcome
        # distribution instead of shifting the projection: on some paths the
        # defense solves what worked in the first half, on others it does not.
        shocks = {}
        for abbr in (st.home.abbr, st.away.abbr):
            shocks[abbr] = max(
                0.55,
                1.0
                + self.adj["shock_regression_bias"]
                + rs.rng.gauss(0.0, self.adj["shock_sd"]),
            )

        pulled: Dict[str, Dict[int, bool]] = {st.home.abbr: {}, st.away.abbr: {}}
        qb_pulled: Dict[str, bool] = {st.home.abbr: False, st.away.abbr: False}

        offense = st.possession or st.away.abbr
        defense = st.home.abbr if offense == st.away.abbr else st.away.abbr
        start_yl = self._kickoff_start(rs)

        max_drives = self.priors.get("sim.max_drives")
        drives = 0

        while clock > 0 and drives < max_drives:
            drives += 1
            box[f"@{offense}.drives"] = box.get(f"@{offense}.drives", 0.0) + 1
            self._maybe_rest_starters(rs, offense, scores, clock, pulled, qb_pulled)

            outcome, end_yl, used, points = self._run_drive(
                rs, offense, defense, start_yl, clock, scores, box,
                pulled[offense], qb_pulled, shocks[offense],
            )
            clock -= used
            self.drive_outcomes[outcome] += 1
            if self._last_drive_reached_rz:
                self.redzone_trips += 1
                if outcome == TD:
                    self.redzone_tds += 1
            if points:
                scores[offense] += points
                box[f"@{offense}.points"] = scores[offense] - self._initial_score(offense)

            if clock <= 0:
                break

            if outcome in (TD, FG_GOOD):
                clock -= self.s_clock["score_reset_seconds"]
                start_yl = self._kickoff_start(rs)
            elif outcome == FG_MISS:
                start_yl = max(20.0, 100.0 - (end_yl - 8.0))
            elif outcome == PUNT:
                clock -= self.s_clock["punt_seconds"]
                start_yl = self._punt_result(rs, end_yl)
            else:  # turnover or downs
                start_yl = max(1.0, 100.0 - end_yl)

            offense, defense = defense, offense

        for abbr in (st.home.abbr, st.away.abbr):
            box.setdefault(f"@{abbr}.points", 0.0)
        box["@game.points"] = box[f"@{st.home.abbr}.points"] + box[f"@{st.away.abbr}.points"]
        return box

    def _initial_score(self, abbr: str) -> float:
        return float(self.state.team(abbr).score)

    # -- drive -------------------------------------------------------------

    def _run_drive(
        self,
        rs: Sampler,
        offense: str,
        defense: str,
        start_yl: float,
        clock: float,
        scores: Dict[str, float],
        box: Dict[str, float],
        pulled: Dict[int, bool],
        qb_pulled: Dict[str, bool],
        shock: float,
    ) -> Tuple[str, float, float, float]:
        prep = self.preps[offense]
        team = prep.team
        yl = start_yl
        down = 1
        to_go = 10.0
        used = 0.0
        points = 0.0
        self._last_drive_reached_rz = False

        # A generous penalty-driven reprieve, rolled once per drive.
        extension_available = rs.chance(self.extend_prob)

        while True:
            remaining = clock - used
            if remaining <= 0:
                return END_HALF, yl, used, points

            diff = scores[offense] - scores[defense]
            pr = gamescript.pass_rate(team, diff, remaining, self.priors)
            spp = gamescript.seconds_per_play(team, diff, remaining, self.priors)

            if down >= 4:
                # A defensive penalty (holding, DPI, roughing) gifting an
                # automatic first down has to be resolved *here*, at the point
                # where the series would otherwise end. Checking it only after
                # a failed fourth-down conversion makes it unreachable on the
                # ~95% of drives that punt instead of going for it.
                if extension_available:
                    extension_available = False
                    down = 1
                    to_go = min(10.0, 100.0 - yl)
                    continue

                action = self._fourth_down_choice(rs, yl, to_go, diff, remaining)
                if action == "fg":
                    used += spp
                    dist = yardline_to_fg_distance(yl)
                    made = rs.chance(fg_make_prob(dist, self.priors, self.fg_wind_penalty))
                    if prep.kicker is not None:
                        if made:
                            self._bump(box, prep.kicker.player_id, "fg_made", 1)
                    if made:
                        return FG_GOOD, yl, used, points + 3.0
                    return FG_MISS, yl, used, points
                if action == "punt":
                    return PUNT, yl, used, points
                # else: go for it, fall through and run a normal play

            gain, time_used, turnover, touchdown = self._run_play(
                rs, prep, offense, pr, spp, yl, box, pulled, qb_pulled, shock, remaining
            )
            used += time_used
            self._bump(box, f"@{offense}", "plays", 1)

            if turnover:
                return TURNOVER, yl + max(0.0, gain), used, points

            if touchdown:
                points += 6.0
                points += self._extra_point(rs, prep, box)
                return TD, 100.0, used, points

            yl += gain
            to_go -= gain
            if yl >= 80.0:
                self._last_drive_reached_rz = True

            if yl <= 0:  # safety territory; treat as a stalled drive
                return PUNT, max(1.0, yl), used, points

            if to_go <= 0:
                down = 1
                to_go = min(10.0, 100.0 - yl)
            else:
                down += 1
                if down > 4:
                    return DOWNS, yl, used, points

    def _run_play(
        self,
        rs: Sampler,
        prep: TeamPrep,
        offense: str,
        pass_rate: float,
        spp: float,
        yl: float,
        box: Dict[str, float],
        pulled: Dict[int, bool],
        qb_pulled: Dict[str, bool],
        shock: float,
        remaining: float,
    ) -> Tuple[float, float, bool, bool]:
        """Returns (yards, seconds_used, turnover, touchdown)."""
        pool = prep.pool
        to_goal = max(1.0, 100.0 - yl)
        red_zone = yl >= 80.0

        qb = prep.qbs[0]
        if qb_pulled[offense] and len(prep.qbs) > 1:
            qb = prep.qbs[1]

        if rs.chance(pass_rate):
            # --- dropback ---------------------------------------------------
            if rs.chance(prep.sack_rate):
                loss = rs.sack_yards()
                self._bump(box, qb.player_id, "rush_att", 0)  # sacks are not carries
                if rs.chance(self.s_play["fumble_lost_rate_sack"] * self.fumble_mult):
                    return loss, spp, True, False
                return loss, spp, False, False

            if rs.chance(qb.scramble_rate):
                gain = min(rs.scramble_yards(), to_goal)
                self._bump(box, qb.player_id, "rush_att", 1)
                self._bump(box, qb.player_id, "rush_yds", gain)
                self._max(box, qb.player_id, "longest_rush", gain)
                td = gain >= to_goal
                if td:
                    self._bump(box, qb.player_id, "rush_td", 1)
                return gain, spp, False, td

            self._bump(box, qb.player_id, "pass_att", 1)

            if rs.chance(prep.int_rate):
                self._bump(box, qb.player_id, "interceptions", 1)
                return 0.0, spp, True, False

            idx = self._sample_target(rs, pool, pulled, red_zone)
            if idx < 0:
                return 0.0, self.s_clock["incompletion_seconds"], False, False

            receiver = pool.players[idx]
            self._bump(box, receiver.player_id, "targets", 1)

            p_comp = completion_prob(
                prep.catch_rate[idx], prep.adot[idx], prep.comp_penalty[idx]
            )
            if not rs.chance(p_comp):
                return 0.0, self.s_clock["incompletion_seconds"], False, False

            mu = max(1.0, prep.rec_mu[idx] * shock)
            gain = min(rs.reception_yards(mu, prep.explosive[idx]), to_goal)

            self._bump(box, receiver.player_id, "rec", 1)
            self._bump(box, receiver.player_id, "rec_yds", gain)
            self._max(box, receiver.player_id, "longest_rec", gain)
            self._bump(box, qb.player_id, "pass_cmp", 1)
            self._bump(box, qb.player_id, "pass_yds", gain)

            td = gain >= to_goal
            if td:
                self._bump(box, receiver.player_id, "rec_td", 1)
                self._bump(box, qb.player_id, "pass_td", 1)
                return gain, spp, False, True

            oob = rs.chance(self.s_clock["out_of_bounds_prob"])
            return gain, spp * (0.55 if oob else 1.0), False, False

        # --- designed run -------------------------------------------------
        idx = self._sample_rusher(rs, pool, pulled, red_zone)
        if idx < 0:
            return 0.0, spp, False, False

        carrier = pool.players[idx]
        mu = max(0.6, prep.rush_mu[idx] * shock)
        gain = min(rs.rush_yards(mu, prep.explosive[idx]), to_goal)

        self._bump(box, carrier.player_id, "rush_att", 1)
        self._bump(box, carrier.player_id, "rush_yds", gain)
        self._max(box, carrier.player_id, "longest_rush", gain)

        if rs.chance(self.s_play["fumble_lost_rate_rush"] * self.fumble_mult):
            return gain, spp, True, False

        td = gain >= to_goal
        if td:
            self._bump(box, carrier.player_id, "rush_td", 1)
        return gain, spp, False, td

    # -- helpers -----------------------------------------------------------

    def _sample_target(
        self, rs: Sampler, pool: usage.UsagePool, pulled: Dict[int, bool], red_zone: bool
    ) -> int:
        weights = pool.rz_target_w if red_zone else pool.target_w
        if not pulled:
            total = pool.rz_target_total if red_zone else pool.target_total
            if total <= 0:
                return -1
            return rs.choice_index(weights, total)

        live = [0.0 if pulled.get(i) else w for i, w in enumerate(weights)]
        total = sum(live)
        if total <= 0:
            return -1
        return rs.choice_index(live, total)

    def _sample_rusher(
        self, rs: Sampler, pool: usage.UsagePool, pulled: Dict[int, bool], red_zone: bool
    ) -> int:
        weights = pool.rz_rush_w if red_zone else pool.rush_w
        if not pulled:
            total = pool.rz_rush_total if red_zone else pool.rush_total
            if total <= 0:
                return -1
            return rs.choice_index(weights, total)

        live = [0.0 if pulled.get(i) else w for i, w in enumerate(weights)]
        total = sum(live)
        if total <= 0:
            return -1
        return rs.choice_index(live, total)

    def _fourth_down_choice(
        self, rs: Sampler, yl: float, to_go: float, diff: float, remaining: float
    ) -> str:
        fd = self.s_fd
        desperate = gamescript.is_desperate(diff, remaining, self.priors)

        if desperate:
            # Trailing late: kick only if it is close and actually helps.
            if yl >= fd["fg_range_yardline"] and (diff >= -3 or remaining < 40):
                return "fg"
            return "go"

        # Fourth and goal from the shadow of the end zone: the touchdown is
        # often worth more than the near-certain three.
        if yl >= 95.0 and to_go <= 1.0 and rs.chance(fd["goal_line_go_prob"]):
            return "go"

        if yl >= fd["fg_range_yardline"]:
            return "fg"

        for max_togo, min_yl, prob in fd["go_table"]:
            if to_go <= max_togo and yl >= min_yl:
                return "go" if rs.chance(prob) else "punt"
        return "punt"

    def _extra_point(self, rs: Sampler, prep: TeamPrep, box: Dict[str, float]) -> float:
        fd = self.s_fd
        if rs.chance(fd["two_point_rate"]):
            return 2.0 if rs.chance(fd["two_point_success"]) else 0.0
        made = rs.chance(fd["xp_make"] - self.fg_wind_penalty * 0.4)
        if made and prep.kicker is not None:
            self._bump(box, prep.kicker.player_id, "xp_made", 1)
        return 1.0 if made else 0.0

    def _maybe_rest_starters(
        self,
        rs: Sampler,
        offense: str,
        scores: Dict[str, float],
        clock: float,
        pulled: Dict[str, Dict[int, bool]],
        qb_pulled: Dict[str, bool],
    ) -> None:
        """Roll rest risk for the leading team at the top of each drive."""
        other = self.state.home.abbr if offense == self.state.away.abbr else self.state.away.abbr
        margin = scores[offense] - scores[other]
        prep = self.preps[offense]

        for i, p in enumerate(prep.pool.players):
            if pulled[offense].get(i):
                continue
            if rs.chance(usage.rest_hazard(p, margin, clock, self.priors)):
                pulled[offense][i] = True

        if not qb_pulled[offense]:
            qb = prep.qbs[0]
            if rs.chance(usage.rest_hazard(qb, margin, clock, self.priors)):
                qb_pulled[offense] = True

    def _kickoff_start(self, rs: Sampler) -> float:
        # Touchback-heavy under the current kickoff rules, with a returned tail.
        if rs.chance(0.68):
            return 30.0
        return max(5.0, min(60.0, rs.rng.gauss(26.0, 8.0)))

    def _punt_result(self, rs: Sampler, yl: float) -> float:
        net = rs.rng.gauss(40.5, 9.0)
        spot = yl + net
        if spot >= 100.0:
            return 20.0  # touchback
        return max(1.0, 100.0 - spot)

    @staticmethod
    def _bump(box: Dict[str, float], pid: str, stat: str, value: float) -> None:
        key = f"{pid}.{stat}"
        box[key] = box.get(key, 0.0) + value

    @staticmethod
    def _max(box: Dict[str, float], pid: str, stat: str, value: float) -> None:
        key = f"{pid}.{stat}"
        if value > box.get(key, 0.0):
            box[key] = value


def simulate(
    state: GameState,
    priors: Optional[Priors] = None,
    n_sims: Optional[int] = None,
    seed: Optional[int] = None,
) -> SimResult:
    """Convenience wrapper: build a simulator and run it."""
    return GameSimulator(state, priors).run(n_sims=n_sims, seed=seed)
