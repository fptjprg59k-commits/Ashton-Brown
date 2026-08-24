# The model

Why it is built this way, what each coefficient means, and where the numbers
came from.

---

## The central choice: simulate drives, not players

The obvious way to price a prop is to project a player's rest-of-game stat line
and compare it to the line. That approach throws away three things that matter
more than the projection itself.

**Game script is endogenous.** Whether a running back gets 18 carries or 8 is
mostly a question of whether his team is ahead. A projection made at halftime
has to assume a script; a simulation *produces* one, differently on every path,
and the play-calling on each path responds to the score it just created.

**Props on the same game are correlated.** When the quarterback throws for 340,
his WR1 is high on that same path. A per-player model has no way to represent
this, which is precisely why same-game parlays are mispriced. Here every prop is
evaluated against the same set of simulated paths, so joint probabilities fall
out by ANDing win indicators — no copula, no independence assumption.

**Distribution shape decides the bet, not the mean.** Two receivers projected
for 62 yards can have very different probabilities of clearing 59.5. Comparing
a point projection to a line discards exactly the information the bet turns on.

So: the engine plays out the remaining game drive by drive and down by down,
20,000 times, and reports how often each prop cashed.

---

## Structure of one simulated half

```
while clock remains:
    roll rest risk for the leading team's stars     (fourth quarter only)
    run a drive:
        while the drive lives:
            pass rate  ← f(score diff, clock, team tendency)
            sec/play   ← f(score diff, clock, team tempo)
            on 4th down: penalty reprieve? then field goal / punt / go
            run a play:
                pass → sack? scramble? interception? target → catch? yards
                run  → carrier → yards
            update down, distance, field position, clock
    resolve the drive: touchdown / field goal / punt / turnover / downs
    hand field position to the other team
```

Per-simulation stat lines accumulate into a matrix. Adding first-half actuals
and comparing to a line happens afterwards, in `props.py`.

---

## Game script

The driver is a leverage term:

```
z = −point_diff / √(minutes_remaining)
```

This captures the interaction a raw deficit misses. Down 7 with 28 minutes left
is a normal football game; down 7 with 3 minutes left is a two-minute drill.
The same seven points mean completely different things, and dividing by
`√minutes` is the standard way to express that.

Play-calling and tempo then respond through `tanh`, so effects **saturate**:

```
pass_rate = clip(base_pass + β · tanh(z / z_scale),  0.15, 0.92)
sec_per_play = base_spp · (1 − β_pace · tanh(z / z_scale) · pressure)
```

`tanh` matters. A linear response to a 35-point deficit produces a 130% pass
rate; saturation produces a team that throws about as much as a team can.

`pressure` ramps the tempo effect from 35% at the start of a half to 100% as
the clock expires. Without it, a team up 17 milks the play clock from the
opening snap of the third quarter — which is wrong, and in an earlier version
of this model it starved both offenses of about 12% of their snaps. Clock
management is a late-game behaviour. Running the ball with a lead is what keeps
the clock moving early; deliberately bleeding the play clock comes later.

Two hard regimes override the smooth curve, because late-game behaviour is not
a gentle gradient: trailing inside five minutes forces no-huddle, and leading
inside four minutes forces clock-killing.

---

## Per-touch yardage: the shape question

This is the part that separates a useful model from a projection with extra
steps.

Yardage is drawn from a **shifted gamma whose shape is set by the player's
explosiveness index while its mean is held fixed**:

```
shape = shape_possession + (shape_explosive − shape_possession) · e
Y = Gamma(shape, (μ + shift) / shape) − shift        E[Y] = μ  for all e
```

Because a gamma's coefficient of variation is `1/√shape`, a low shape gives a
right-skewed, fat-tailed distribution and a high shape gives a tight,
near-symmetric one. Holding the mean fixed is deliberate: tuning explosiveness
must never silently smuggle in a change to expected production.

At a 13-yard mean:

| Profile | `e` | shape | SD | median | P(>35 yds) |
| --- | --- | --- | --- | --- | --- |
| Possession | 0.05 | ~3.0 | ~7.5 | ~11.5 | low |
| Deep threat | 0.95 | ~0.8 | ~15 | ~8 | ~3× higher |

Same projection, very different bets. The deep threat misses low most weeks and
occasionally posts 130 on two catches — which is exactly the "big-play reliance"
problem, represented rather than averaged away.

Shape constants are chosen so a league-average profile reproduces observed NFL
dispersion: about 8–9 yards of SD per reception at a 12-yard mean, and 5–6 per
carry at 4.35.

---

## Halftime adjustments and regression

Half a game is a tiny sample. A quarterback at 11.0 yards per attempt before
the break is not an 11.0 YPA quarterback for the next thirty minutes, and
extrapolating the hot half is the most common way to misprice a halftime prop.

Observed first-half efficiency is therefore blended toward the pregame prior
with the prior weighted heavily, expressed as an equivalent sample size so a
team that threw 30 times gets more credit than one that threw 9:

```
weight_on_observation = n / (n + prior_strength)
```

Usage shares regress the same way — a 45% first-half target share on 11 team
pass attempts is mostly noise.

On top of that sits a **per-simulation coordinator-adjustment shock**: a
multiplicative draw applied to each offense for the whole simulated half. This
widens the outcome distribution rather than shifting it, which is the honest
way to represent "the defense might have solved them at the break". On some
paths the adjustment lands, on others it does not. A small negative bias
reflects that hot first halves cool more often than cold ones heat up.

---

## Usage, injuries, and rest

**Redistribution.** An unavailable player's share does not evaporate; the
remaining weights renormalise, so survivors' absolute shares rise. This is what
makes "an injury to a co-star spikes his teammate's share" fall out of the
model instead of needing a hand edit.

**Injury is two effects, not one.** A designation reduces snap share *and*
per-touch effectiveness separately, because playing hurt is not the same as
playing less.

**Scheme funnels** reshape *which* available player gets targeted, independent
of the offense's intent. Funnels are alignment-specific (slot / perimeter /
inline / backfield) with a depth-scaled deep funnel on top, because "opponent
pass defense rank" cannot express a two-high shell that concedes underneath
while erasing perimeter deep shots. Shadow coverage applies a per-player
multiplier to both volume and efficiency.

**Rest risk** is a logistic hazard on the *simulated* margin, evaluated per
drive in the fourth quarter, for stars only, on the leading team only:

```
p_pull = logistic((margin − midpoint) / scale) · max_prob · role_mult · time_factor
```

Once pulled, a player accumulates nothing for the remainder of that path. This
truncates the top of his distribution on exactly the paths where his team ran
away with it — which is where an over would otherwise have cashed. Backs get
rested first, receivers last.

---

## Weather

Wind is the only weather variable with a large, reliable effect on props.
Completion penalty separates a flat term from a depth-scaled term:

```
penalty = c₁ · excess_wind + c₂ · excess_wind · max(0, (aDOT − 6) / 10)
```

A swing pass to a back is nearly wind-proof; a deep shot is not. Treating wind
as a flat completion penalty gets this backwards for half the board.

Precipitation compresses the explosiveness index — poor footing means receivers
cannot separate deep and backs cannot bounce runs outside — which shrinks
variance without moving the projection. That is the correct shape for a
bad-weather game.

---

## Kicking

Field goal probability is **logistic in distance**, not linear:

```
p = logistic((58.5 − distance) / 7.7)
```

The real curve is flat out to about 40 yards and then falls away steeply. An
earlier linear version calibrated to get 60-yarders right priced a 40-yarder at
70%, when kickers make those about 92% of the time — which distorted every
fourth-down decision and, through them, drive length and every volume prop.

| Distance | Model | League |
| --- | --- | --- |
| 20 | 0.993 | ~0.99 |
| 40 | 0.917 | ~0.92 |
| 50 | 0.751 | ~0.75 |
| 55 | 0.612 | ~0.61 |
| 60 | 0.451 | ~0.45 |

---

## Fourth down

Modelled as a go-for-it probability table over (yards to go, field position).
Modern play-callers go far more often than pre-analytics defaults assume, and
this is not a detail: fourth-down aggression directly sets drive length, which
sets play volume, which sets every counting prop on the board.

A separate per-drive reprieve models a defensive penalty gifting an automatic
first down, at roughly the league rate of 1.6 defensive automatic first downs
per team per game. It is resolved *at the point where the series would end* —
an earlier version checked it only after a failed fourth-down conversion, which
made it unreachable on the ~95% of drives that punt instead, and cost the model
about three points of scoring per game.

---

## Odds, vig, and what "edge" means

A model probability is half the picture. The board reports both:

- **HIT%** — the model's probability, what was asked for.
- **FAIR** — the book's probability with vig removed, via Shin's method when
  both sides are visible (it handles favourite-longshot bias better than
  proportional scaling, which is the regime most player props sit in).
- **EDGE** — the difference. This, not the hit rate, is where the model claims
  the number is wrong.

With only one side visible the margin cannot be identified, so a typical
half-margin is subtracted as an approximation and the row's confidence is
marked down. A one-sided de-vig is a guess and is flagged rather than hidden.

**Pushes** are handled explicitly. Whole-number lines return the stake on an
exact hit, so yardage is rounded to integers the way a box score does it and
pushes are counted rather than distributed into wins and losses. Expected value
uses the full three-outcome tree; note that pushes pull EV *toward zero* in both
directions — a profitable bet loses edge to a push, a losing one recovers some.

---

## Calibration

`nflprops calibrate` simulates a neutral league-average matchup and compares
twelve aggregate metrics against published NFL rates. It runs one half and
doubles the counting stats, because a real game contains two end-of-half drives
that die on the clock; simulating 3600 continuous seconds produces only one and
quietly inflates every drive-outcome rate by about three points.

Current state: 12/12 within tolerance, with yards per play, completion rate,
and per-team passing and rushing yards essentially exact.

This is a necessary condition, not a sufficient one. Reproducing league
averages says the football engine is sound; it says nothing about whether the
*team-specific* coefficients you feed it are right.

---

## What would improve it most

Roughly in order of expected value:

1. **Fit the priors to settled props.** The shape of the model is the
   contribution here; the constants are informed defaults. A season of settled
   lines with the state at the time of pricing would let every coefficient be
   fitted and the model's calibration measured properly (Brier score, reliability
   curve) rather than asserted.
2. **Condition play-calling on down and distance**, not just score and clock.
   Third-and-9 is a different distribution from first-and-10 at the same score.
3. **Real usage priors from play-by-play** — route participation and air yards
   rather than target share, which conflates opportunity with efficiency.
4. **Drive-level starting field position from the actual game**, rather than
   sampled from fixed punt and kickoff distributions.
5. **Model the leading team forcing three-and-outs**, which is the main reason
   real trailing teams run more plays than this engine gives them.
