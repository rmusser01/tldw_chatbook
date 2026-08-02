---
id: TASK-1760
title: 'In-app serving for the briefings feed directory'
status: In Progress
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
- [x] #1 A user can serve an exported briefings feed folder over localhost and successfully point
      a podcast client at the resulting URL
- [x] #2 Serving is opt-in and off by default -- no feed directory is ever served without an
      explicit user action to start it
- [x] #3 The server cannot serve any file outside the one chosen directory (path traversal into
      parent directories or elsewhere on disk is rejected)
- [x] #4 A security review documents the default bind scope (localhost vs. wider), and states the
      no-auth-by-default posture explicitly rather than leaving it implicit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. New `tldw_chatbook/Subscriptions/feed_server.py`: stdlib `ThreadingHTTPServer` on a daemon thread wrapping a locked-down `SimpleHTTPRequestHandler(directory=...)` subclass — GET/HEAD only, resolved-path prefix check on top of `translate_path` (symlink escapes rejected), no auth (posture stated), start()/stop() with ephemeral-or-configured port, bind 127.0.0.1 by default (`[briefings_feed_server]` config for port/bind; serving itself never auto-starts).
2. UI: Serve/Stop action beside the phase-3 feed export on the watchlists Artifacts pane; toast shows the URL and the plain no-auth statement; state is session-only.
3. Security review notes (AC #4) in the user guide's feed section + module docstring.
4. Tests: real server on an ephemeral localhost port (round-trip via httpx), traversal matrix (.. / absolute / symlink-out), GET-only, opt-in default off, stop closes the socket.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** `tldw_chatbook/Subscriptions/feed_server.py` is a small, standalone module: `_ContainedRequestHandler`
subclasses stdlib `http.server.SimpleHTTPRequestHandler` and overrides `translate_path` to add a resolved-path
containment check on top of it (`Path(candidate).resolve().relative_to(served_root)`) -- `translate_path` alone
already collapses literal `../`/absolute-path traversal in the URL down to a path inside the served directory (it
never joins a `..`-shaped or directory-shaped word onto the base), but does **not** stop a symlink planted inside the
served directory from resolving to a target outside it, since `open()`/`os.stat()` follow symlinks by default. A
candidate that fails the check is answered with a plain, non-existent filename (never a null byte, which raised an
uncaught `ValueError` in `open()` during development and tore the connection down instead of 404ing) so the stdlib's
own `open()`-fails-with-`OSError` -> 404 path handles it unchanged. `do_POST`/`do_PUT`/`do_DELETE`/`do_PATCH` are
overridden to `405`; request logging is routed to `loguru.debug` (paths only, never content). `FeedDirectoryServer`
wraps a `ThreadingHTTPServer` on a daemon thread; `start()` refuses (raises `FeedServerError`, naming the running
URL) rather than restarting when already running -- this is the "one directory at a time" decision the plan left
open, and refusing was chosen because it needs no reconciliation between an old server possibly mid-request and a
new one starting, at the cost of one extra Stop press. `configured_bind_and_port()` reads the optional
`[briefings_feed_server]` `bind`/`port` via the three-argument `get_cli_setting(section, key, default)` form (never
the two-argument dotted form, per the repo's TASK-1771 lesson) -- reading it never starts anything; only the UI's
Serve action calls `start()`.

**UI wiring.** Two new buttons, "Serve Feed" / "Stop Serving", added to `ArtifactsPane`'s existing
`#artifacts-toolbar` right after "Export Feed…" (same toolbar Generate/Refresh/Export/Keep already live in, so no new
row is spent). Two buttons rather than one toggling label, mirroring the pane's own Play/Stop audio pair: Serve is
disabled while already running OR with nothing exported yet; Stop is disabled while nothing is running. New
messages `ServeFeedRequested`/`StopFeedServerRequested` follow the pane's existing no-payload message shape (the
directory/state lives on the screen). `WatchlistsCollectionsScreen` owns one `FeedDirectoryServer` instance
(constructed in `__init__`, not a module singleton) plus `_last_feed_export_directory`, set in `_export_feed_directory`
on every successful export (full or partial -- both leave a real, servable `feed.xml`). Both handlers are
synchronous (no `run_worker`): `start()`/`stop()` are fast, non-blocking stdlib calls with no `await` boundary,
unlike every other action on this screen that moves real I/O off the UI thread. `_sync_feed_server_pane_state()`
patches the mounted pane's reactives in place after every state change, mirroring the existing picker-writer idiom
(`handle_briefing_mode_changed` et al.) rather than a full recompose. `on_unmount` stops the server if still
running, so navigating away or closing the app never leaves a wedged listening socket (verified live in
`test_screen_teardown_stops_a_still_running_feed_server`: the socket refuses connections after the screen tears
down, even when Stop was never pressed).

**Config.** New `[briefings_feed_server]` section added to `config.py`'s default template (`bind = "127.0.0.1"`,
`port = 0`), placed immediately before `[web_server]` with a comment stating explicitly that it is unrelated to that
section. Nothing reads it except `configured_bind_and_port()`, and nothing calls that except the Serve handler.

**One-directory-at-a-time choice: refuse, not restart.** `FeedDirectoryServer.start()` raises when already running,
naming the currently-served URL; the UI's Serve button is also disabled while running (so this path is reached only
via the handler's own defensive re-check, or a second in-process caller). Restarting in place was rejected as the
more complex of the two options the plan allowed for.

**Security posture (AC #4).** Documented in three places: the module docstring's "Security posture" section (the
canonical version), a new "Security posture" subsection in `Docs/User_Guide/watchlists.md`'s feed section
(loopback-by-default, no authentication stated plainly, path containment, session-only/opt-in), and the toast every
`ServeFeedRequested` success prints: `"Serving the exported feed at {url}. No authentication — anyone who can reach
this address can read the feed while it is serving."` (`markup=False`).

**Tests.** `Tests/Subscriptions/test_feed_server.py` (23 tests): round-trip GET/HEAD, POST/PUT/DELETE/PATCH -> 405,
the traversal matrix (`../`, percent-encoded `..%2f`, an absolute-path-shaped request, and the load-bearing symlink-
inside-pointing-outside case plus its sibling proving an in-bounds symlink still works), double-start refusal,
stop -> connection refused, missing/non-directory destination rejection, and `configured_bind_and_port` (defaults,
a stored section, a bad value falling back to ephemeral, and a spy proving reading config never opens a socket).
Two mutations verified by hand (temporarily removing the containment check and the GET/HEAD gate, confirming the
symlink and 405 tests REDed, then restoring via the Edit tool and re-confirming green; `git status --short` was
clean between). `Tests/Watchlists/test_watchlists_artifacts_pane.py` gained 7 UI tests: disabled-by-default (no
socket open), enabling after an export, a full Serve-then-Stop round trip through the real buttons (toast content,
a real `httpx` GET, connection refused after Stop), the teardown-without-Stop case, the running-refusal case, the
nothing-to-stop refusal, and a wiring proof that the handler actually calls `configured_bind_and_port`.

**Verification.** `Tests/Subscriptions/` (558 passed), `Tests/Watchlists/` (372 passed, including the 124 in
`test_watchlists_artifacts_pane.py`), plus targeted `test_config_*` files -- all green. No regressions found.

**Files added:** `tldw_chatbook/Subscriptions/feed_server.py`, `Tests/Subscriptions/test_feed_server.py`.
**Files modified:** `tldw_chatbook/config.py` (new `[briefings_feed_server]` section), `tldw_chatbook/UI/
Watchlists_Modules/artifacts_pane.py` (Serve/Stop buttons, reactives, messages), `tldw_chatbook/UI/Screens/
watchlists_collections_screen.py` (server ownership, handlers, `on_unmount`), `Tests/Watchlists/
test_watchlists_artifacts_pane.py` (new UI tests), `Docs/User_Guide/watchlists.md` (Serve Feed usage + Security
posture section).
<!-- SECTION:NOTES:END -->
