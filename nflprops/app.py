"""The two surfaces as one application: slate in, prop board out.

The dashboard answers *which game*, the board answers *which bet*. Keeping
them in one page makes the actual workflow a single gesture - see a card turn
green, click it, read the board - instead of copying a game id into a terminal
command while the second half kicks off.

Both views are rendered server-side into one self-contained file and swapped
by the URL fragment, so there is no fetch, no loading state, and the whole
thing works offline, from a file:// path, or published as an artifact. The
board markup is the *same* fragment the standalone report emits; this module
adds routing and chrome, never a second copy of the board.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from .dashboard import Dashboard, GameCard, _CSS as _DASH_CSS, _JS as _FILTER_JS
from .models import GameState
from .rank import GameDiagnostics, RankedProp
from .report import _COMPONENT_CSS as _BOARD_CSS, render_board_fragment
from .status import GameStatus
from .theme import FONT_DISPLAY, FONT_LINKS, FONT_MONO


@dataclass
class GameBoard:
    """A fully analysed game, keyed to the slate entry it belongs to."""

    game_id: str                      # the scoreboard's id, not the state's
    ranked: Sequence[RankedProp]
    state: GameState
    diag: GameDiagnostics
    n_sims: int
    settled: Optional[Sequence[RankedProp]] = None

    @property
    def anchor(self) -> str:
        return f"game-{self.game_id}"


# --------------------------------------------------------------------------
# Styles and behaviour unique to the combined app
# --------------------------------------------------------------------------

_APP_CSS = f"""
/* ---- view switching -------------------------------------------------- */
.view[hidden]{{display:none}}

/* ---- clickable cards -------------------------------------------------- */
a.card{{text-decoration:none; color:inherit; cursor:pointer}}
a.card:hover{{border-color:var(--st-half-edge)}}
a.card:hover .ready .arrow{{transform:translateX(3px)}}
a.card:focus-visible{{outline:2px solid var(--st-half-edge); outline-offset:3px}}
.ready .arrow{{display:inline-block; margin-left:6px;
  transition:transform .18s ease}}
.ready.muted{{color:var(--faint)}}

/* The view is focused on navigation so a screen reader starts reading the
   board rather than staying where the old view was. It is a container, not a
   control, so it takes the focus without drawing a ring around the page. */
.gameview:focus,.gameview:focus-visible{{outline:none}}

/* ---- board view chrome ------------------------------------------------ */
.backbar{{display:flex; align-items:center; gap:14px; flex-wrap:wrap;
  margin:26px 0 2px}}
.back{{
  display:inline-flex; align-items:center; gap:8px; text-decoration:none;
  font-family:{FONT_DISPLAY}; font-weight:600; font-size:12px;
  letter-spacing:.11em; text-transform:uppercase; color:var(--muted);
  background:var(--card); border:1px solid var(--rule);
  padding:8px 15px; border-radius:999px; transition:color .15s,
  border-color .15s, background .15s;
}}
.back:hover{{color:var(--ink); border-color:var(--st-up-edge)}}
.back .arrow{{display:inline-block; transition:transform .18s ease}}
.back:hover .arrow{{transform:translateX(-3px)}}
.atbreak{{display:inline-flex; align-items:center; gap:7px;
  font-family:{FONT_DISPLAY}; font-weight:600; font-size:12px;
  letter-spacing:.12em; text-transform:uppercase;
  color:#fff; background:var(--st-half-edge);
  padding:6px 12px; border-radius:999px}}
.atbreak .dot{{width:7px; height:7px; border-radius:50%;
  background:currentColor}}
.hint{{margin-left:auto; font-size:12.5px; color:var(--faint)}}
.hint kbd{{font-family:{FONT_MONO}; font-size:11px; padding:1px 5px;
  border:1px solid var(--rule); border-radius:3px; background:var(--card)}}

/* Sibling boards, so you can cross-check the other halftime game without
   returning to the slate first. */
.siblings{{display:flex; gap:8px; flex-wrap:wrap; align-items:center;
  margin:22px 0 0; padding-top:16px; border-top:1px solid var(--rule)}}
.siblings .lbl{{font-family:{FONT_DISPLAY}; font-weight:600; font-size:11px;
  letter-spacing:.14em; text-transform:uppercase; color:var(--faint)}}
.siblings a{{text-decoration:none; font-size:13.5px; color:var(--accent);
  border:1px solid var(--rule); border-radius:999px; padding:6px 13px;
  transition:border-color .15s, background .15s}}
.siblings a:hover{{border-color:var(--accent); background:var(--accent-soft)}}

@media (max-width:760px){{
  .hint{{display:none}}
  .backbar{{margin-top:18px}}
}}
"""

_ROUTER_JS = """
(function(){
  var views = Array.prototype.slice.call(document.querySelectorAll('.view'));
  var slate = document.getElementById('view-slate');

  function show(id){
    var target = id && document.getElementById(id);
    if (!target || !target.classList.contains('view')) target = slate;
    views.forEach(function(v){ v.hidden = v !== target; });
    document.title = target.dataset.title || document.title;
    // A view swap is a navigation, so it should start at the top.
    window.scrollTo(0, 0);
    if (target !== slate) target.focus({preventScroll:true});
  }

  function fromHash(){
    show((location.hash || '').replace(/^#/, ''));
  }

  window.addEventListener('hashchange', fromHash);

  // Escape closes the board and returns to the slate. Deliberately not
  // history.back(): after hopping board -> board via the sibling links, "back"
  // is the previous board, but "close" is always the slate. Conflating the two
  // strands you one level down with no obvious way out. The browser's own back
  // button still walks the history properly.
  document.addEventListener('keydown', function(ev){
    if (ev.key === 'Escape' && location.hash && location.hash !== '#') {
      location.hash = '';
    }
  });

  fromHash();
})();
"""


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _board_view(
    board: GameBoard,
    card: Optional[GameCard],
    siblings: List[GameBoard],
    cards_by_id: Dict[str, GameCard],
) -> str:
    e = html.escape
    state = board.state

    if card is not None:
        matchup = f"{card.away.display} at {card.home.display}"
    else:
        matchup = f"{state.away.abbr} at {state.home.abbr}"

    others = ""
    peers = [b for b in siblings if b.game_id != board.game_id]
    if peers:
        links = "".join(
            f'<a href="#{e(b.anchor)}">{e(_short_name(b, cards_by_id))}</a>'
            for b in peers
        )
        others = (
            f'<div class="siblings"><span class="lbl">Also at halftime</span>'
            f"{links}</div>"
        )

    body = render_board_fragment(
        board.ranked, state, board.diag, board.n_sims, settled=board.settled
    )

    return f"""<section class="view gameview" id="{e(board.anchor)}"
    data-title="{e(matchup)} — halftime board" tabindex="-1" hidden>
  <div class="backbar">
    <a class="back" href="#"><span class="arrow">&larr;</span>All games</a>
    <span class="atbreak"><span class="dot"></span>Halftime</span>
    <span class="hint">Press <kbd>Esc</kbd> to go back</span>
  </div>
  {body}
  {others}
</section>"""


def _short_name(board: GameBoard, cards_by_id: Dict[str, GameCard]) -> str:
    card = cards_by_id.get(board.game_id)
    if card:
        return f"{card.away.abbr} @ {card.home.abbr}"
    return f"{board.state.away.abbr} @ {board.state.home.abbr}"


def render_app(
    dash: Dashboard,
    boards: Sequence[GameBoard],
    now: Optional[datetime] = None,
    title: str = "NFL Halftime Props",
) -> str:
    """One page: the slate, plus a board view for every analysed game."""
    from .dashboard import _format_stamp, render_slate_fragment

    e = html.escape
    by_id = {b.game_id: b for b in boards}
    cards_by_id = {c.game_id: c for c in dash.games}

    # A board is only reachable if its game is actually at the break. A stale
    # board for a game that has since restarted must not be presented as live
    # analysis, so the status gate in the slate is the single arbiter.
    reachable = [
        b for b in boards
        if cards_by_id.get(b.game_id)
        and cards_by_id[b.game_id].status is GameStatus.HALFTIME
    ]

    slate = render_slate_fragment(
        dash,
        now=now,
        link_for=lambda gid: f"#{by_id[gid].anchor}" if gid in by_id else None,
    )

    views = "".join(
        _board_view(b, cards_by_id.get(b.game_id), reachable, cards_by_id)
        for b in boards
    )

    stamp = f"As of {_format_stamp(dash.generated_at)}" if dash.generated_at else ""
    n = len(reachable)
    lead = (
        f"{n} game{'s' if n != 1 else ''} at the break — open one for its board."
        if n
        else "No game is at halftime right now."
    )

    return f"""<title>{e(title)}</title>
{FONT_LINKS}
<style>{_DASH_CSS}
{_BOARD_CSS}
{_APP_CSS}</style>
<div class="wrap">
  <section class="view" id="view-slate" data-title="{e(title)}">
    {slate}
    <footer>
      <span>{e(stamp)}</span>
      <span>Green means <strong>halftime</strong>. {e(lead)}</span>
    </footer>
  </section>
  {views}
</div>
<script>{_FILTER_JS}</script>
<script>{_ROUTER_JS}</script>"""
