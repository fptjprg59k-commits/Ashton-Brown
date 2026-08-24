"""Shared design tokens for every rendered surface.

One source of truth, for the same reason ``status.py`` exists: the dashboard
and the prop board are one product, and a status tone must mean the same thing
on both. Components read tokens and never hard-code a colour, so the whole
system re-themes from this file.

Colour is assigned semantically, not decoratively:

* the **accent** blue carries structure and live games;
* **green** is reserved for HALFTIME - the one state that means "there is
  something to do here";
* red appears only for a negative edge on the prop board, never for status, so
  a red number always means "this bet is priced against you" and never "this
  game is live".
"""

from __future__ import annotations

FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Saira+Condensed:wght@500;600;700&"
    "family=Source+Sans+3:ital,wght@0,400;0,600;1,400&"
    'family=IBM+Plex+Mono:wght@400;500;600&display=swap">'
)

FONT_DISPLAY = '"Saira Condensed",ui-sans-serif,system-ui,sans-serif'
FONT_BODY = '"Source Sans 3",ui-sans-serif,system-ui,-apple-system,sans-serif'
FONT_MONO = '"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace'

#: Light palette on bare :root, redefined for the two dark states. Every token
#: is declared here in full so no colour exists only inside a media query - the
#: classic way an artifact ends up rendering one theme's text on the other
#: theme's ground.
_LIGHT = """
  --ground:#f5f7fa; --card:#ffffff; --raised:#fbfcfe;
  --ink:#151a22; --muted:#687184; --faint:#8b93a3;
  --rule:#e2e6ed; --rule-soft:#eef1f6; --track:#e8ebf1; --tick:#151a22;
  --accent:#2d4b8e; --accent-soft:#dbe3f4; --accent-ink:#20356a;
  --pos:#12734a; --pos-soft:#d9efe3; --neg:#a82f27; --neg-soft:#f7e0de;
  --st-up-fg:#5b6478; --st-up-bg:#eef1f6; --st-up-edge:#c8d0de;
  --st-live-fg:#2d4b8e; --st-live-bg:#dbe3f4; --st-live-edge:#2d4b8e;
  --st-half-fg:#0d6b48; --st-half-bg:#d5efe2; --st-half-edge:#12905f;
  --st-final-fg:#7b8397; --st-final-bg:#f1f3f7; --st-final-edge:#d6dbe5;
  --st-warn-fg:#8a5a12; --st-warn-bg:#fbeed6; --st-warn-edge:#d8a24a;
"""

_DARK = """
  --ground:#0f131b; --card:#171d28; --raised:#1d2431;
  --ink:#e6e9ef; --muted:#98a1b3; --faint:#798296;
  --rule:#28303e; --rule-soft:#1f2632; --track:#232b38; --tick:#e6e9ef;
  --accent:#7fa3e8; --accent-soft:#22304b; --accent-ink:#a9c3f2;
  --pos:#4fc98a; --pos-soft:#16352a; --neg:#f0817a; --neg-soft:#3a1f1d;
  --st-up-fg:#9aa3b6; --st-up-bg:#1e2531; --st-up-edge:#333c4c;
  --st-live-fg:#8fb0ee; --st-live-bg:#1b2740; --st-live-edge:#5f88d8;
  --st-half-fg:#57d69f; --st-half-bg:#122f24; --st-half-edge:#2f9d70;
  --st-final-fg:#79828f; --st-final-bg:#1a202a; --st-final-edge:#2b3340;
  --st-warn-fg:#e0b26a; --st-warn-bg:#2f2617; --st-warn-edge:#8a6a2c;
"""

TOKENS_CSS = f"""
:root{{{_LIGHT}}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{{_DARK}}}}}
:root[data-theme="dark"]{{{_DARK}}}
"""

#: Reset and typographic baseline shared by every surface.
BASE_CSS = f"""
*{{box-sizing:border-box}}
body{{
  margin:0; padding:0 0 72px; background:var(--ground); color:var(--ink);
  font-family:{FONT_BODY}; font-size:15px; line-height:1.55;
  -webkit-font-smoothing:antialiased;
}}
.wrap{{max-width:1180px; margin:0 auto; padding:0 20px}}
.eyebrow{{
  font-family:{FONT_DISPLAY}; font-weight:600; font-size:11px;
  letter-spacing:.16em; text-transform:uppercase; color:var(--faint);
}}
.num{{font-family:{FONT_MONO}; font-variant-numeric:tabular-nums}}
:focus-visible{{outline:2px solid var(--accent); outline-offset:2px;
  border-radius:3px}}
@media (prefers-reduced-motion:reduce){{
  *,*:before,*:after{{animation-duration:.001ms !important;
    animation-iteration-count:1 !important; transition-duration:.001ms !important}}
}}
"""

#: Maps a ``status.Tone`` value to its token trio. Surfaces call this instead
#: of branching on the status themselves.
TONE_TOKENS = {
    "upcoming": ("--st-up-fg", "--st-up-bg", "--st-up-edge"),
    "live": ("--st-live-fg", "--st-live-bg", "--st-live-edge"),
    "halftime": ("--st-half-fg", "--st-half-bg", "--st-half-edge"),
    "final": ("--st-final-fg", "--st-final-bg", "--st-final-edge"),
    "warning": ("--st-warn-fg", "--st-warn-bg", "--st-warn-edge"),
}
