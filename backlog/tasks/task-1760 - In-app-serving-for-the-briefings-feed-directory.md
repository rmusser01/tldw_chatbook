---
id: TASK-1760
title: 'In-app serving for the briefings feed directory'
status: To Do
assignee: []
created_date: '2026-08-01 20:05'
labels:
  - watchlists
  - briefings
  - web-server
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec #2's phase 3 design (`Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md`,
"Exports and feed") promised that "if the app's `[web_server]` is enabled it can serve that
directory over localhost for podcast clients; serving is a toggle." At phase 3 plan/implementation
time (task-1540, phase 3 close-out) that premise turned out to be false, and the project owner
decided to cut localhost serving from phase 3 rather than build on a mechanism that does not
exist. This task exists so a future implementer does not have to re-discover the same evidence.

**Why `[web_server]` cannot do this, in full:**

- `[web_server]` is **textual-serve**, not a general-purpose HTTP static-file server. Its
  `enabled` key is defined in the default config template (`tldw_chatbook/config.py:3665-3671`)
  but is **read by no code anywhere in the app** -- there is no `get_cli_setting("web_server",
  "enabled", ...)` call or equivalent gate on it at all. Toggling it currently does nothing.
- `[web_server]` is a **mutually exclusive process mode**, not something that runs alongside the
  TUI. `app.py`'s `main()` branches on the `--serve` CLI flag (`tldw_chatbook/app.py:9231-9254`):
  when set, it calls `run_web_server(...)` and then `return`s -- exiting the process after the web
  server stops -- instead of ever constructing and running `TldwCli` as a TUI. There is no code
  path where both the TUI and a web server are live in the same process; "if `[web_server]` is
  enabled" was never a real branch a running TUI session could take.
- Even when engaged via `--serve`, its only static route serves **textual-serve's own browser
  assets**, not arbitrary files from a user-chosen directory. `create_server`
  (`tldw_chatbook/Web_Server/serve.py:258-297`) builds a `textual_serve.server.Server` whose
  `command` re-launches `python -m tldw_chatbook.app` as a subprocess bridged to a browser
  terminal; `ChatbookWebServerMixin.handle_textual_js`
  (`tldw_chatbook/Web_Server/serve.py:213-220`) serves exactly one hardcoded file,
  `self.statics_path / "js" / "textual.js"` -- textual-serve's own JS bundle, patched for viewport
  resize. There is no route, hardcoded or configurable, that serves a directory the caller names.

In short: there was never a server running alongside the TUI for a "serve this feed folder"
toggle to flip. Phase 3 shipped the feed directory as the deliverable (the spec's own wording
already said "the directory is the deliverable") and documents a user-run static server
(e.g. `python -m http.server` from the exported folder) as the way to point a podcast client at
it today.

**Scope of this task: net-new work.** This is not "wire up `[web_server]`" -- it is a standalone,
purpose-built static file server that the TUI can start and stop on demand, independent of
textual-serve entirely:

- Its own bind-address and port settings (not reusing `[web_server]`'s config section, which
  belongs to a different, unrelated feature).
- Started/stopped from the UI (or a command), scoped to one exported feed directory at a time.
- A security review this scope implies, since this is new listening-socket surface: default bind
  scope (localhost-only unless the user deliberately widens it), hardening against path traversal
  out of the chosen directory, and an explicit posture on authentication (none by default, stated
  plainly to the user rather than implied).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user can serve an exported briefings feed folder over localhost and successfully point
      a podcast client at the resulting URL
- [ ] #2 Serving is opt-in and off by default -- no feed directory is ever served without an
      explicit user action to start it
- [ ] #3 The server cannot serve any file outside the one chosen directory (path traversal into
      parent directories or elsewhere on disk is rejected)
- [ ] #4 A security review documents the default bind scope (localhost vs. wider), and states the
      no-auth-by-default posture explicitly rather than leaving it implicit
<!-- AC:END -->
