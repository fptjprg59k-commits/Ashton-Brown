"""Command line interface.

    nflprops run --game fixtures/game_bal_cin_halftime.json \\
                 --props fixtures/props_paste.txt --verbose

Live modes (``--game espn:<id>``, ``scoreboard``, ``watch``) need outbound
network access and are documented in the README.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from .agent import analyze
from .models import PropScope, Prop
from .priors import Priors
from .providers.stake import StakePropsProvider
from .rank import SORT_KEYS, correlation_matrix, independent_parlay_probability, parlay_probability
from .report import render_html, render_terminal, to_json
from .serde import load_game


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------


def _load_state(spec: str):
    """``path/to.json`` for an offline snapshot, ``espn:<event_id>`` for live."""
    if spec.startswith("espn:"):
        from .providers.espn import ESPNProvider

        return ESPNProvider().fetch(spec.split(":", 1)[1])
    return load_game(spec)


def _load_props(path: str, scope: PropScope) -> tuple:
    provider = StakePropsProvider(path=path, default_scope=scope)
    props = provider.fetch()
    return props, provider.rejected


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    priors = Priors.load(args.priors) if args.priors else Priors()
    state = _load_state(args.game)
    scope = PropScope.SECOND_HALF if args.scope == "second_half" else PropScope.FULL_GAME
    props, rejected = _load_props(args.props, scope)

    if rejected:
        print(f"[warn] {len(rejected)} line(s) could not be parsed:", file=sys.stderr)
        for line in rejected[:10]:
            print(f"       {line!r}", file=sys.stderr)

    if not props:
        print("No usable props found. Check the input format.", file=sys.stderr)
        return 2

    result = analyze(
        state,
        props,
        priors=priors,
        n_sims=args.sims,
        seed=args.seed,
        sort=args.sort,
        devig_method=args.devig,
        min_probability=args.min_prob,
        include_settled=args.include_settled,
    )

    if result.unresolved:
        print(
            f"[warn] {len(result.unresolved)} prop(s) had no roster match and were "
            f"skipped:",
            file=sys.stderr,
        )
        for p in result.unresolved[:10]:
            print(f"       {p.player_name or p.prop_id!r} ({p.market.value})", file=sys.stderr)

    if result.settled and not args.include_settled:
        print(
            f"[note] {len(result.settled)} full-game line(s) already decided by "
            f"first-half production, held out of the board "
            f"(--include-settled to show):",
            file=sys.stderr,
        )
        for r in result.settled[:6]:
            print(
                f"       {r.prop.label}  -> already {r.evaluation.settled}",
                file=sys.stderr,
            )

    if args.format == "json":
        out = to_json(result.ranked, state, result.diagnostics, result.n_sims)
    elif args.format == "html":
        out = render_html(result.ranked, state, result.diagnostics, result.n_sims)
    else:
        out = render_terminal(
            result.ranked,
            state,
            result.diagnostics,
            result.n_sims,
            verbose=args.verbose,
            limit=args.limit,
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"wrote {args.out}")
    else:
        print(out)
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    """Check what the prop parser makes of an input file before running it."""
    props, rejected = _load_props(args.props, PropScope.FULL_GAME)
    print(f"parsed {len(props)} prop(s), rejected {len(rejected)}\n")
    for p in props:
        opp = f" (opp {p.opposing_odds:+d})" if p.opposing_odds else ""
        print(
            f"  {p.player_name or p.team:<24} {p.market.value:<16} "
            f"{p.side.value:<5} {p.line:<7g} {p.odds:+d}{opp}  [{p.scope.value}]"
        )
    if rejected:
        print("\nrejected:")
        for line in rejected:
            print(f"  {line!r}")
    return 0 if props else 2


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibrate import measure, render

    priors = Priors.load(args.priors) if args.priors else Priors()
    rows = measure(n_sims=args.sims, priors=priors)
    print(render(rows))
    return 0 if all(r.ok for r in rows) else 1


def cmd_parlay(args: argparse.Namespace) -> int:
    """Correlation-aware pricing for a same-game parlay."""
    priors = Priors.load(args.priors) if args.priors else Priors()
    state = _load_state(args.game)
    props, _ = _load_props(args.props, PropScope.FULL_GAME)
    result = analyze(state, props, priors=priors, n_sims=args.sims, seed=args.seed)

    by_id = {r.prop.prop_id: r for r in result.ranked}
    by_index = list(result.ranked)
    legs = []
    for token in args.legs:
        if token in by_id:
            legs.append(by_id[token])
        elif token.isdigit() and 1 <= int(token) <= len(by_index):
            legs.append(by_index[int(token) - 1])
        else:
            # Board position shifts with the seed and the prop list, so a
            # substring of the label is the stable way to name a leg.
            matches = [r for r in by_index if token.lower() in r.prop.label.lower()]
            if len(matches) == 1:
                legs.append(matches[0])
            elif not matches:
                print(f"no prop matching {token!r}", file=sys.stderr)
                return 2
            else:
                print(f"{token!r} is ambiguous:", file=sys.stderr)
                for m in matches:
                    print(f"    {m.prop.label}", file=sys.stderr)
                return 2

    joint = parlay_probability(legs)
    naive = independent_parlay_probability(legs)
    print("Legs:")
    for leg in legs:
        print(f"  {leg.evaluation.p_win:6.1%}  {leg.prop.label}")
    print()
    print(f"  Correlation-aware joint probability : {joint:.2%}")
    print(f"  Naive independent multiplication    : {naive:.2%}")
    delta = joint - naive
    direction = "correlated (worth more than it looks)" if delta > 0 else "substitutes (worth less)"
    print(f"  Difference                          : {delta:+.2%}  - {direction}")

    if len(legs) > 1:
        print("\nCorrelation matrix:")
        m = correlation_matrix(legs)
        for i, leg in enumerate(legs):
            row = "  ".join(f"{m[i][j]:+.2f}" for j in range(len(legs)))
            print(f"  {row}   {leg.prop.label[:40]}")
    return 0


def cmd_scoreboard(args: argparse.Namespace) -> int:
    from .providers.espn import ESPNProvider

    for row in ESPNProvider().scoreboard():
        away, home = row.get("away", {}), row.get("home", {})
        print(
            f"  {row['game_id']:<12} {str(row.get('name')):<14} "
            f"{away.get('abbr','?')} {away.get('score',0)} @ "
            f"{home.get('abbr','?')} {home.get('score',0)}   "
            f"{row.get('status')} Q{row.get('period')} {row.get('clock')}"
        )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check that optional dependencies and live providers actually work."""
    ok = True
    try:
        import numpy  # noqa: F401

        print("  numpy            ok")
    except ImportError:
        print("  numpy            MISSING (required)")
        ok = False
    try:
        import yaml  # noqa: F401

        print("  pyyaml           ok (priors overrides available)")
    except ImportError:
        print("  pyyaml           absent (built-in priors only)")

    if args.espn:
        from .providers.base import ProviderError
        from .providers.espn import ESPNProvider

        try:
            games = ESPNProvider().scoreboard()
            print(f"  espn scoreboard  ok ({len(games)} games)")
            halftime = [g for g in games if "HALFTIME" in str(g.get("status", "")).upper()]
            print(f"  at halftime      {len(halftime)}")
        except ProviderError as exc:
            print(f"  espn scoreboard  FAILED: {exc}")
            ok = False
    else:
        print("  espn             skipped (pass --espn to test live access)")

    return 0 if ok else 1


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nflprops",
        description="Rank NFL player props by how likely they are to hit, "
        "given the state of the game at halftime.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--sims", type=int, default=None, help="simulation paths (default 20000)")
        sp.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
        sp.add_argument("--priors", default=None, help="YAML file overriding model coefficients")

    run = sub.add_parser("run", help="rank the board for one game")
    run.add_argument("--game", required=True, help="state JSON path, or espn:<event_id>")
    run.add_argument("--props", required=True, help="prop list: paste text, JSON, or CSV")
    run.add_argument("--sort", default="probability", choices=SORT_KEYS)
    run.add_argument("--format", default="text", choices=("text", "json", "html"))
    run.add_argument("--out", default=None, help="write to a file instead of stdout")
    run.add_argument("--verbose", action="store_true", help="show driver attribution")
    run.add_argument("--limit", type=int, default=None, help="show only the top N")
    run.add_argument("--min-prob", dest="min_prob", type=float, default=0.0)
    run.add_argument(
        "--include-settled",
        action="store_true",
        help="also show full-game lines the first half already decided",
    )
    run.add_argument("--devig", default="shin", choices=("shin", "multiplicative"))
    run.add_argument(
        "--scope",
        default="full_game",
        choices=("full_game", "second_half"),
        help="how to treat lines that do not say (default: full game)",
    )
    add_common(run)
    run.set_defaults(func=cmd_run)

    parse = sub.add_parser("parse", help="show how a prop file is interpreted")
    parse.add_argument("--props", required=True)
    parse.set_defaults(func=cmd_parse)

    cal = sub.add_parser("calibrate", help="check the engine against league averages")
    cal.add_argument("--sims", type=int, default=5000)
    cal.add_argument("--priors", default=None)
    cal.set_defaults(func=cmd_calibrate)

    par = sub.add_parser("parlay", help="correlation-aware same-game parlay pricing")
    par.add_argument("--game", required=True)
    par.add_argument("--props", required=True)
    par.add_argument("legs", nargs="+", help="board ranks (1 2 5) or prop ids")
    add_common(par)
    par.set_defaults(func=cmd_parlay)

    sb = sub.add_parser("scoreboard", help="list today's games (needs network)")
    sb.set_defaults(func=cmd_scoreboard)

    doc = sub.add_parser("doctor", help="check dependencies and live providers")
    doc.add_argument("--espn", action="store_true", help="also test live ESPN access")
    doc.set_defaults(func=cmd_doctor)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
