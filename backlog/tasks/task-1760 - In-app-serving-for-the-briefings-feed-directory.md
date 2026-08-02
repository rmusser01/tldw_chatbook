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

**Fix wave (whole-branch security review, `task-1760-verdict.md`).** Four Medium + one Low finding remediated,
no blockers found. **M1**: `log_message` read `self.command`/`self.path` directly; `BaseHTTPRequestHandler.
parse_request` sets `self.command = None` before `self.path` exists and calls `send_error` -> `log_error` ->
`log_message` on every malformed-request-line path (bare `GET`, a >64KiB request line, a 4-word line with a
parseable version) that runs before `self.path` is ever assigned -- raising `AttributeError`, which killed the
intended 400/414 response (client got zero bytes) and the handler thread, with no log line at all. Fixed with
`getattr(self, "command", "-")` / `getattr(self, "path", "-")`. **M2**: `stop()` measured ~501ms blocking the
Textual event loop (both callers -- the Stop handler and `on_unmount` -- call it synchronously), because
`serve_forever()` was started with no `poll_interval` (stdlib default 0.5s) and `shutdown()` blocks until the
loop's next poll returns. Fixed by starting the loop at `poll_interval=0.05`; measured `stop()` afterward at
well under 150ms in the new regression test (no change to `stop()` itself needed -- the bound comes entirely
from how the loop was started). The "fast, non-blocking stdlib calls" comment that justified skipping
`run_worker` is corrected to state the real, now-bounded cost instead of claims that measured false. **M3**:
`bind` was `str(...)`'d with no further validation -- a blank config value or a numeric typo (`bind = 0`, meant
for `port`) both resolve to `0.0.0.0` (every interface) at the socket layer, silently turning the loopback
default into a LAN-wide one with no signal anywhere. Added `_normalize_bind` (blank/whitespace/non-string ->
`"127.0.0.1"`) applied at both `configured_bind_and_port` (config-read layer) and `start()` itself (the actual
socket boundary, so a direct caller gets the same guarantee); `start()` now also logs a warning and the Serve
toast appends an explicit exposure sentence whenever the resolved bind is not loopback (new `is_loopback_bind`
helper, covering `127.0.0.0/8`, `::1`, and `localhost`). **M4**: the handler served the chosen directory
recursively with browsable directory listings enabled (`GET /` returned a full index) -- undersold in the
original posture text. Fixed in code, not just docs: `_ContainedRequestHandler.list_directory` now 404s
instead of rendering a listing (recursive FILE serving is unchanged -- an episode in a subfolder still
resolves). Module docstring, `config.py`'s template comment, and `Docs/User_Guide/watchlists.md` all now state
plainly that the server serves every file in the chosen directory **and every subdirectory**, and steer users
toward a dedicated export folder rather than a general-purpose one like `$HOME`. **L1**: re-exporting a feed
while a server is running on a *different*, older directory previously gave no signal beyond the Serve
button's own disabled state. `FeedDirectoryServer` now exposes `.directory` (the resolved directory the running
server actually serves); the export success/partial-export toasts append "Still serving the previously-exported
folder -- Stop Serving and Serve again to publish this export." when the server is running and pointed
elsewhere (silent when re-exporting into the *same* directory, since the server reads from disk per-request and
already reflects it). **Test-quality note**: `test_dotdot_traversal_does_not_escape_the_served_directory` was
confirmed near-vacuous (httpx normalizes a literal `../` client-side before it reaches the wire) -- kept, and
augmented with a new raw-socket parametrized test (`test_raw_socket_dotdot_forms_httpx_would_never_put_on_the_
wire`, 6 wire forms including literal `..`, doubled `../../`, `..%2f`, `..%5c`, leading `//`, and `./../`) that
puts each form directly on the wire, bypassing any client normalization. L2 (no HTTP Range/206 support) is a
deliberately out-of-scope non-goal, per the review.

Three mutations verified by hand with the Edit tool (git-status-clean between each): reverting `log_message`'s
`getattr` fallbacks reproduced the exact `AttributeError` from the review against all three M1 tests; reverting
`_normalize_bind`'s fallback to a bare `str(raw_bind)` REDed all three blank/numeric-bind tests; reverting
`list_directory` to `super().list_directory(path)` REDed both no-listing tests (200 with a real HTML index
instead of 404). All three restored byte-exact and reconfirmed green.

**Verification.** `Tests/Subscriptions/test_feed_server.py`: 41 passed (23 original + 18 new: 2 malformed-
request, 1 stop-latency, 5 bind-validation, 3 no-listing, 6 raw-socket traversal, 1 malformed-line-survives).
`Tests/Watchlists/test_watchlists_artifacts_pane.py`: 124 passed on a clean run; three earlier full-suite runs
each surfaced exactly one different, unrelated test failing (`test_export_feed_press_survives_an_os_error_from_
the_service`, then `test_citations_do_not_shrink_the_briefings_table_below_its_pinned_minimum`, then three more
at once) -- every one of them passed standalone and touches no code this fix wave changed, consistent with this
suite's known pre-existing full-run flakiness rather than a regression. Targeted `test_config_*` sweep (71
tests, covering the new config-template comment) also green.

**Fix wave (Qodo review).** Three findings remediated, no blockers found. **F1**: neither `start()` nor
`configured_bind_and_port()` range-checked `port` -- `int(...)` accepted any integer, so an out-of-range-but-
parseable value (e.g. `99999`, or a negative value) reached `ThreadingHTTPServer.__init__` -> `socket.bind` as a
bare `OverflowError`, a type the UI's Serve handler (`watchlists_collections_screen.py`, only catches
`FeedServerError`/`OSError`) does not catch -- it escaped as an unhandled exception instead of the toast every
other rejection produces. The two entry points now degrade differently on purpose: `configured_bind_and_port()`
(a config value, expected to sometimes be hand-edited wrong) falls back to ephemeral `0` with a type-only
`logger.warning`, mirroring its existing bad-bind precedent; `FeedDirectoryServer.start()` (the actual socket
boundary, and the one a direct caller -- script, test, future code -- can reach without going through config at
all) instead raises `FeedServerError` for a non-`int`/out-of-range `port`, the same type its neighboring
directory-validation checks already raise, which the UI already catches. **F2**: `start()` built
`self._url = f"http://{bind}:{port}/"` unconditionally, which is not a URL any client can parse for an IPv6
literal (`http://::1:8080/` is ambiguous) despite `is_loopback_bind` explicitly supporting `::1`. Added
`_format_host_for_url` (bracket an IPv6 literal, leave IPv4/hostname forms unchanged) and wired it into the URL
build. Verifying this end-to-end surfaced a second, deeper pre-existing defect: `http.server.HTTPServer`/
`ThreadingHTTPServer` hard-code `address_family = socket.AF_INET` and never infer it from the bind address, so
`start(bind="::1", ...)` failed with a bare `socket.gaierror` on every platform (not merely IPv6-disabled CI)
before the URL-formatting code ever ran -- the bracketing fix would otherwise have been unreachable dead code.
Added `_IPv6ThreadingHTTPServer` (the same class, `address_family` forced to `socket.AF_INET6`) and select it in
`start()` whenever `bind` is an IPv6 literal (`_is_ipv6_literal`, factored out and shared with
`_format_host_for_url` so the two can never disagree); IPv4/hostname binds are unaffected. **F3**:
`is_loopback_bind` gained a Google-style `Args`/`Returns` docstring section; no behavior change.

**Tests.** 21 new tests in `Tests/Subscriptions/test_feed_server.py` (51 total, up from 41): port validation (a
config port of `99999`/`-1` falls back to ephemeral with a warning; a valid configured port still passes
through; `start()` raises `FeedServerError` for a negative port, a port above 65535, and a non-`int` port, and
still accepts a legitimate nonzero port end to end) and IPv6 URL bracketing (`_format_host_for_url` unit tests
needing no socket at all, plus a live `start(bind="::1", port=0)` round trip via `httpx`, skipped -- not failed
-- only if the runner cannot bind `::1` at all).

**Mutation verification (Edit-tool revert, `git status --short` clean between each, all restored byte-exact and
reconfirmed green afterward):** dropping `start()`'s port range/type check REDed all three of its rejection
tests (each raised `OverflowError`/`TypeError` instead of the expected `FeedServerError`); dropping
`configured_bind_and_port()`'s range check REDed both its out-of-range fallback tests (`99999`/`-1` passed
through unchanged instead of falling back to `0`); reverting `_format_host_for_url` to return `bind` unmodified
REDed both the pure-format test and the live IPv6 round-trip test (`http://::1:PORT/`, unbracketed).

**Verification.** `Tests/Subscriptions/test_feed_server.py`: 51 passed. `Tests/Watchlists/
test_watchlists_artifacts_pane.py`: 124 passed on this run (the two feed-server UI tests,
`test_pressing_serve_then_stop_round_trips_through_a_real_server` and
`test_serve_reads_bind_and_port_from_configured_bind_and_port`, both green) -- no repeat of the known rotating-
victim flake this run. No regressions found.

**Files modified:** `tldw_chatbook/Subscriptions/feed_server.py`, `Tests/Subscriptions/test_feed_server.py`.
<!-- SECTION:NOTES:END -->
