# nflprops

Rank NFL player props by how likely they are to hit, given the state of a game
at halftime.

It takes the score, clock, first-half box score, and the lines you are looking
at, simulates the rest of the game 20,000 times, and prints a board sorted
highest-probability-first — with the reasons each number landed where it did.

```
========================================================================================================
  HALFTIME PROP BOARD  -  sample-bal-cin
========================================================================================================
  CIN 10  @  BAL 17   |   Q3, 30:00 remaining   |   CIN ball
  Projected final:  CIN 18.2  -  BAL 31.6   (CIN win 8%, BAL win 92%)
  Projected H2 pass rate:  CIN 73% (neutral 62%)   |   BAL 38% (neutral 55%)
  Projected H2 plays:  CIN 32   |   BAL 32   |   blowout risk (17+): 40%
  Conditions:  41F, wind 18 mph, none
--------------------------------------------------------------------------------------------------------
   #  PROP                                           HIT%                ODDS   FAIR    EDGE      EV  CONF
--------------------------------------------------------------------------------------------------------
   1  Joe Burrow over 244.5 pass yds               88.0% ██████████▋   -1100 89.4%  -1.5%  -0.04  medium
   2  Zay Flowers over 4.5 rec                     88.0% ██████████▋   -1233 90.2%  -2.3%  -0.05  medium
   3  Lamar Jackson over 189.5 pass yds            87.2% ██████████▌   -1023 88.8%  -1.7%  -0.04  medium
   4  Derrick Henry over 109.5 rush yds            83.4% ██████████     -673 84.8%  -1.4%  -0.04  medium
   5  Ja'Marr Chase over 99.5 rec yds              77.2% █████████▍     -422 78.6%  -1.4%  -0.04  medium
  ...
  12  Tee Higgins over 63.5 rec yds                48.8% █████▉         +119 43.4%  +5.4%  +0.07  high
```

---

## Quick start

```bash
pip install -e .          # or: pip install numpy pyyaml

# everything at once: the slate, with each halftime game clickable through
# to its own prop board
nflprops app --scoreboard fixtures/scoreboard_preseason.json \
             --manifest   fixtures/manifest_preseason.json \
             --out board.html

# or either surface on its own
nflprops dashboard --source fixtures/scoreboard_preseason.json
nflprops run --game fixtures/game_bal_cin_halftime.json \
             --props fixtures/props_paste.txt --verbose
```

Everything above runs offline against bundled sample data. Nothing needs a
network connection.

---

## The dashboard

A vertical feed of game cards, sorted so the games you can still act on sit at
the top. Card state is driven entirely by the game's status:

| Status | Card | Reads |
| --- | --- | --- |
| **Halftime** | Green surface, green rail | `HALFTIME` + *Prop board ready* |
| **Live** | Blue rail, pulsing dot | `2nd · 08:42` |
| **End of quarter** | Blue rail, steady dot | `END 1st` |
| **Upcoming** | Neutral, no score | `7:30 PM` + *in 42 min* |
| **Final** | Muted, dimmed | `FINAL` / `FINAL/OT` |
| **Delayed / postponed** | Amber | `DELAYED` |

```bash
nflprops dashboard                                  # live, today's slate
nflprops dashboard --season-type preseason --week 3 # pin the slate
nflprops dashboard --format html --out board.html --refresh 60
```

### Click a green card to get its board

`nflprops app` renders the slate and a full prop board for every analysed game
into **one self-contained page**. Click a green card, or press its link, and
the board opens; `Esc` or the back arrow returns to the slate. Boards are
addressable (`board.html#game-401780001`), so a deep link opens straight onto
one, and the browser's back button works normally.

There is no fetch and no loading state — every board is rendered ahead of time
into the same file — so it works offline, from a `file://` path, or published
as an artifact.

Which games are clickable is decided by the status, not the caller: only a game
that is *actually at halftime* can be linked, so a stale board for a game that
has since restarted can never be presented as live analysis. A halftime game
with no props loaded stays inert and says **No props loaded** rather than
offering a click that goes nowhere.

The manifest maps slate ids to input files:

```json
{"games": [
  {"game_id": "401780001",
   "state": "fixtures/game_bal_cin_halftime.json",
   "props": "fixtures/props_paste.txt"}
]}
```

### Green is a meaning, not a colour

`nflprops/status.py` owns the status system, and nothing downstream styles a
card directly. A card asks its status for a *tone*; the tone maps to design
tokens. Adding a state, or changing what halftime looks like, is a one-line
change that propagates to the HTML board, the terminal board, and anything
built later.

That indirection is what makes halftime detection correct rather than merely
pretty. Feeds disagree about how to describe the break — some report a
dedicated halftime status, some report "end of period" with the period set to
2, some just leave the clock at zero — and all three mean the same thing to a
bettor. Resolving that in one place is the difference between a dashboard that
turns green at halftime and one that turns green *most* of the time. The
bundled fixture contains two halftime games encoded two different ways, and a
test asserts both are caught, along with the inverse: an end-of-**first**
quarter must never turn the card green.

The status also carries `is_actionable`, which is true for exactly one state.
That is why a halftime card — and only a halftime card — offers the prop board.

### Preseason

Preseason is a calendar fact, not a different kind of football, so the status
logic is identical and games carry a `PRE Wk 3` badge. The one real difference
is fetching: the bare scoreboard endpoint returns "the current slate", which in
August can come back empty or fall through to regular-season week 1 depending
on where the league is in its rollover. Pass `--season-type preseason`
(optionally with `--week`) to pin it.

---

## Read this before you rely on it

Three things are true and worth knowing up front.

**Sorting by hit probability is what you asked for, and it is not the same as
sorting by good bets.** The top of the board is populated by short-priced
favourites. An over at −1100 hitting 88% of the time is *likely* and still a
bad price if its true probability is 89%. The `EDGE` column — model probability
minus the book's vig-free probability — is where the model claims the number is
actually wrong. `--sort edge` reorders the board that way.

**The priors are league-average defaults, not parameters fitted to a
proprietary play-by-play database.** The engine reproduces league-average
football closely (see Calibration below), but the individual coefficients —
how hard a 17-point deficit bends play-calling, how fast a star gets rested —
are informed estimates. Edges under about three points should be read as noise.
`nflprops calibrate` and `--priors my.yaml` exist so you can tune them against
your own settled-prop history.

**There is no public Stake odds API, and scraping a sportsbook generally
violates its terms of service.** No endpoint is hardcoded here and none is
discovered for you. The tool is built around pasting the lines in, which always
works and needs no credentials. See below.

---

## Getting the lines in

Three modes, in order of how reliably they work.

### 1. Paste (recommended)

Copy the props out of the app into a text file and point at it:

```
Ja'Marr Chase Over 121.5 Receiving Yards -105 -114
Joe Burrow Over 301.5 Passing Yards -107 -112
Derrick Henry Over 23.5 Rushing Attempts -113 -106
Chase Brown Anytime Touchdown +518
Zay Flowers Over 30.5 Receiving Yards 2nd Half -110
```

The parser tolerates ordering and separators — `Player Over 65.5 Receiving
Yards -115`, `Player - Receiving Yards - Over 65.5 (-115)`, and
`Receiving Yards | Player | Over 65.5 | -115` all work. The rule it rests on is
that American odds always carry an explicit sign (`-115`, `+140`) and a line
never does (`65.5`), so the two can never be confused.

A second price on the line is read as the other side of the market, which
enables exact vig removal. Without it the tool subtracts a typical half-margin
and marks the row's confidence down, because a one-sided de-vig is a guess.

Check what it understood before running:

```bash
nflprops parse --props my_props.txt
```

Anything it cannot parse is reported, never silently dropped.

### 2. File

JSON or CSV in the documented schema:

```json
{"props": [
  {"player": "Ja'Marr Chase", "market": "rec_yds", "side": "over",
   "line": 121.5, "odds": -105, "opposing_odds": -114, "scope": "full_game"}
]}
```

### 3. HTTP

If you have lawful programmatic access to an odds feed — an affiliate or
partner feed, or your own account data via a route the operator permits —
point `StakePropsProvider(url=...)` at it and the JSON schema above applies.

---

## Getting the game state in

### Offline (works anywhere)

A JSON snapshot. Every field has a sensible default, so a minimal file runs:

```json
{
  "game_id": "my-game", "seconds_remaining": 1800, "possession": "CIN",
  "home": {"abbr": "BAL", "score": 17, "players": [...]},
  "away": {"abbr": "CIN", "score": 10, "players": [...]}
}
```

See `fixtures/game_bal_cin_halftime.json` for a fully specified example with
usage shares, alignments, defensive profiles, shadow coverage, an injury
designation, and weather. Hand-editing it is the fastest way to ask "what
happens to this board if the wind picks up / if WR1 goes out".

### Live

```bash
nflprops scoreboard                    # today's games and their status
nflprops run --game espn:401671789 --props my_props.txt
```

The ESPN adapter supplies score, clock, and the first-half box score. It does
**not** supply season-long usage priors, which are what separate a good
projection from a crude one — without them it falls back to using first-half
usage as its own prior, which is a weak substitute. Supply real target and
carry shares whenever you have them.

> The ESPN adapter was written against the documented response shape but could
> not be exercised against the live endpoint in the environment it was built in
> (outbound access to sports APIs was blocked there). Run
> `nflprops doctor --espn` on a networked machine to confirm the mapping before
> trusting it in-play.

---

## How the factors are modelled

The engine simulates **drives and downs**, not an abstract per-player
projection. That choice is why the factors below fall out of the model rather
than being bolted onto it — the simulated score feeds back into play-calling on
the very next snap.

| Factor | How it enters |
| --- | --- |
| **Game script** | Play-calling responds to `z = −point_diff / √minutes_left`, so being down 7 with 28 minutes left is a normal game and down 7 with 3 minutes left is a two-minute drill. Effects pass through `tanh` so they saturate instead of running away in a blowout. |
| **Pace** | Seconds per snap shift with the same term, scaled by time pressure — clock-killing is a late-game behaviour, not something a team does up 17 in the third quarter. Hurry-up and clock-kill regimes override the smooth curve late. |
| **Opponent strength** | Multipliers on yardage, completion rate, sack and interception rate. |
| **Defensive scheme** | Alignment-specific funnels (slot / perimeter / inline / backfield) plus a depth-scaled deep funnel, because "opponent pass defense rank" is far too blunt to capture a two-high shell conceding underneath. Per-player shadow coverage models a corner travelling with WR1. |
| **Usage share** | Target, carry, and separate red-zone shares set every player's floor. An unavailable player's share flows to teammates rather than evaporating. |
| **Injuries** | Separate hits to snap share and to per-touch effectiveness, since playing hurt is not the same as playing less. |
| **Weather** | Wind reduces completion probability, scaled by target depth — a screen is nearly wind-proof, a vertical shot is not — and depresses field goals. Precipitation compresses the big-play tail without moving the projection. |
| **Halftime adjustments** | First-half efficiency is regressed hard toward the pregame prior (half a game is a tiny sample), plus a per-simulation coordinator-adjustment shock that widens the distribution rather than shifting it. |
| **Blowout / rest risk** | A logistic hazard on the *simulated* margin, evaluated per drive in the fourth quarter. Stars get benched on exactly the paths where their team ran away with it, which truncates the top of the distribution where an over would otherwise cash. |
| **Big-play reliance** | Per-touch yardage is drawn from a shifted gamma whose *shape* is set by an explosiveness index while its *mean is held fixed*. A possession receiver and a deep threat projected for the same yards get very different probabilities of clearing the same line. |
| **Referees** | A small per-drive chance a defensive penalty gifts an automatic first down. Deliberately third-order. |

`docs/MODEL.md` has the reasoning, the equations, and where each number came
from.

---

## Two things the board does that are easy to miss

**Full-game vs. second-half scope.** Books keep full-game markets live through
the break *and* post second-half derivatives. A full-game line has to be
compared against first-half actuals plus the simulated remainder. Getting this
backwards prices a 121.5-yard receiving line as though the receiver were
starting from zero when he already has 82. Lines default to full game; write
`2nd Half` on the line or pass `--scope second_half` to change it.

**Already-settled lines.** At halftime a real share of the board is already
decided — a receiver past his yardage line, a back who has scored. Those are
detected and held out of the ranking, because showing them at "100% to hit"
with a 45-point edge over a stale price is noise. `--include-settled` shows
them.

---

## Same-game parlays

Every prop is scored on the *same* set of simulated paths, so joint
probabilities come out correctly without any copula or independence
assumption:

```bash
nflprops parlay --game fixtures/game_bal_cin_halftime.json \
                --props fixtures/props_paste.txt \
                "Burrow over 244" "Chase over 99"
```

```
   88.1%  Joe Burrow over 244.5 pass yds
   77.5%  Ja'Marr Chase over 99.5 rec yds

  Correlation-aware joint probability : 72.89%
  Naive independent multiplication    : 68.27%
  Difference                          : +4.61%  - correlated (worth more than it looks)
```

A quarterback and his WR1 are correlated (+0.34 here), so the stack cashes
together more often than multiplying the two numbers suggests. A quarterback's
passing yards and his own running back's rushing yards are mild substitutes —
they sit on opposite sides of the same game script — so that parlay is worth
*less* than independence implies.

---

## Calibration

A model that does not reproduce league averages will misprice everything on the
board no matter how good its script logic is. The harness simulates a neutral
league-average matchup and checks the engine against published NFL rates:

```bash
$ nflprops calibrate
metric                      simulated    target    delta  status
------------------------------------------------------------------
points_total                   40.478    44.000   -3.522  ok
plays_per_team                 63.424    63.000   +0.424  ok
drives_per_team                11.149    11.500   -0.351  ok
points_per_drive                1.815     1.900   -0.085  ok
pass_yds_per_team             229.697   230.000   -0.303  ok
rush_yds_per_team             115.635   115.000   +0.635  ok
yards_per_play                  5.445     5.450   -0.005  ok
completion_pct                  0.660     0.645   +0.015  ok
td_rate_per_drive               0.195     0.215   -0.020  ok
fg_rate_per_drive               0.154     0.140   +0.014  ok
punt_rate_per_drive             0.412     0.375   +0.037  ok
turnover_rate_per_drive         0.099     0.110   -0.011  ok
------------------------------------------------------------------
12/12 within tolerance
```

Treat a failing row as a bug in the priors, not a rounding artefact. Override
any coefficient with a YAML file:

```yaml
script:
  pass_rate_beta: 0.24
blowout:
  margin_midpoint: 17.0
```

```bash
nflprops run --game g.json --props p.txt --priors my_priors.yaml
```

---

## Library use

```python
from nflprops.agent import analyze
from nflprops.serde import load_game
from nflprops.providers.stake import StakePropsProvider

state = load_game("fixtures/game_bal_cin_halftime.json")
props = StakePropsProvider(path="fixtures/props_paste.txt").fetch()

result = analyze(state, props, n_sims=20000, seed=1)

for r in result.top(5):
    print(f"{r.evaluation.p_win:.1%}  {r.prop.label}  (edge {r.pricing.edge:+.1%})")
    for driver in r.drivers:
        print("     ", driver)
```

`result.sim` holds the raw per-path stat matrix if you want to ask questions
the CLI does not, and `result.unresolved` / `result.settled` carry what was
held out and why.

---

## Output formats

```bash
nflprops app ...                      # slate + boards, one page
nflprops dashboard ... --format html  # the slate alone
nflprops run ... --format text        # one game's props (default)
nflprops run ... --format json --out board.json
nflprops run ... --format html --out board.html
nflprops run ... --verbose            # driver attribution under each line
nflprops run ... --sort edge          # order by model-vs-market disagreement
nflprops run ... --limit 10
```

---

## Limitations

Worth knowing before you trust a number:

- **Priors are not fitted.** See above. The shape of the model is the
  contribution; the constants need your data.
- **No play-by-play input.** The engine takes a box score, so it cannot see
  that a receiver's 82 yards came on one blown coverage.
- **Down-and-distance is not conditioned on.** Play-calling responds to score
  and clock, not to 3rd-and-9 specifically, so third-down passing rates are
  smoothed rather than modelled.
- **Special teams are coarse.** Punts and kickoffs are sampled from fixed
  distributions; return touchdowns and blocked kicks are not modelled.
- **Live weather is not fetched.** Set it in the state file.
- **In-game injuries need to be entered by hand** unless the live feed reports
  them, and feeds are usually slow to.

---

## Responsible use

This is an analysis tool, not advice, and not a prediction of what will happen.
Every number on the board is a model's estimate carrying both simulation error
and the uncertainty of unfitted priors. Sports betting loses money for most
people who do it; a model that is directionally right will not change that on
its own. Bet only what you can afford to lose, and check that sports betting is
legal where you are.

If gambling stops being fun: in the US, call or text **1-800-GAMBLER**.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q                # 157 tests
nflprops calibrate       # engine vs. league averages
```
