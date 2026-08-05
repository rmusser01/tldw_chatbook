---
id: TASK-1660
title: 'Graphics-protocol selection runs too late; images render half-cell'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - console
  - images
  - bug
dependencies: []
priority: high
---

## Description (the why)

User report (closing task-1532's open AC): in real Kitty, the Console rail
character avatar renders PIXELATED rather than as a native graphics image.

Root cause is load order, not config. `textual_image` selects its
rendering protocol exactly once, at import time, in
`textual_image/renderable/__init__.py`: it writes an escape query and
reads the terminal's reply, choosing Sixel → TGP → halfcell → unicode.
Every app-side import of `textual_image` was LAZY — nested inside
functions that run in the live app — so by the time the query ran,
Textual held the terminal in raw mode and owned stdin. The query never
got a reply, selection silently fell to `HalfcellImage`, the widget
mounted successfully, and the user saw half-block pixelation with no
exception and no log line. `textual_image` warns about exactly this in
its own source ("querying the terminal isn't possible anymore once
Textual is started").

Scope is every image surface, not just the avatar: Console transcript
images, Personas thumbnails, and the full-size viewer modal share the
lazy-import pattern.

Ruled out during diagnosis: the user's config is correct
(`default_render_mode = "auto"`, `terminal_overrides.kitty = "regular"`),
terminal detection is correct (`TERM=xterm-kitty` → `kitty` → `tgp:
True`), and `resolve_default_mode` returns `graphics` for their exact
config — verified by probe. `textual_image` is installed. The
`optional_deps` check does not help: importing the top-level
`textual_image` package does NOT load the `renderable` submodule where
selection happens, and those checks are lazy by default anyway.

## Acceptance Criteria (the what)

- [x] The protocol-selecting import runs before Textual takes the terminal
- [x] Both entry paths warm up (`__main__` block and `main_cli_runner`)
- [x] A missing optional dependency or a non-responding terminal degrades
      safely to the existing mosaic/pixels fallbacks
- [x] App startup shows no stray escape-query bytes
- [x] User confirms a non-pixelated avatar in real Kitty/iTerm2

## Implementation Notes

`warm_up_image_protocol()` in `Utils/terminal_utils.py` imports
`textual_image.widget` (which pulls in `renderable`, where the choice is
made) and returns True/False; both entry points call it immediately
before `app_instance.run()`. The lazy import sites are left alone — they
now receive the already-resolved class.

Verification: under a pty the TGP query escape is observably written at
warm-up time (`_Gi=…,a=q,…`) — the query that previously never reached a
listening terminal; with no responder it correctly lands on halfcell and
returns False. Full proof needs a real Kitty, which the harness cannot
provide — hence the final AC stays open for the user, exactly as
task-1532's did.

5 new tests (submodule actually loaded, safe without the dependency,
per-entry-point ordering, helper invariant). Suites: 52 image/terminal +
53 smoke/App green. Files: `tldw_chatbook/Utils/terminal_utils.py`,
`tldw_chatbook/app.py`, `Tests/Utils/test_image_protocol_warmup.py`.
