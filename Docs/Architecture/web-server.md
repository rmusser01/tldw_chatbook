# The web server

This document describes served mode: running the entire Textual TUI in a browser through textual-serve, its auth model and fail-closed bind policy, the Canvas served-mode seam, and the artifact-share child server. The briefing feed server is a separate opt-in static directory server (see [subscriptions-watchlists.md](./subscriptions-watchlists.md)).

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Core | `Web_Server/serve.py` | the textual-serve subclass (`build_chatbook_web_server_class`), `serve()`, `_make_app()` (routes + auth/canvas middlewares), `_ServedCanvasAuthorityProxy`, `create_server()` (config resolution + policy validation), the viewport-JS patch, `ChatbookAppService` (child spawn env) |
| Auth core | `Canvas/web_auth.py` | token resolution (env → config → keyring), `WebAuthManager`, `build_web_auth_policy` (the fail-closed bind rules) |
| Artifact share | `Web_Server/artifact_share.py`, `artifact_share_server.py`, `artifact_share_manifest.py` | `ArtifactShareController` (one active share), the child aiohttp server (Basic auth, lockout, containment), staging (`<user_data>/share/<uuid>/`, 0600/0700, atomic bundle) |
| Shell | `Web_Server/static/served_shell.{html,css,js}` | the owned terminal-first shell (template placeholders filled at request time) |
| Share UI | `UI/Screens/artifact_share_dialog.py`, `UI/Library_Modules/library_artifacts_share_controller.py` | typed confirmation for unsafe binds; finite-local-worker capture |
| Reference | `Web_Server/README.md` | install/usage/auth/incident-response (verified consistent with code) |

Entry points: `tldw-serve`, `tldw-cli --serve`, `python -m tldw_chatbook.app --serve` — a dependency gate (`[web]` extra) exits with an install hint; served mode is a mutually exclusive process mode (the TUI is never also run).

## Routes served

`/auth/login` (+ one-use 60 s bootstrap), `/` (the owned shell, no-store), `/ws` (websocket, protocol `chatbook-v1` + a `csrf.<token>` subprotocol; one browser session ↔ exactly one AppService child), `/canvas/api/session` (bounded child-owned canvas state), `/download/{key}`, the auth-bridge shim, the two immutable shell assets, the patched `textual.js` (content-fingerprinted URL, 1 h public cache), textual-serve statics, and — only when Canvas is enabled — the canvas static route plus everything under `/canvas/gateway-<uuid>/…`, each request fenced to the exact outer browser session. Request bodies are capped (`client_max_size` 16 KiB; login POST 4 KiB).

## Auth and bind policy (fail-closed)

Token precedence: `TLDW_CHATBOOK_WEB_ACCESS_TOKEN` env → `[web_server].access_token` → OS keyring (failures read as absence; credential reprs redacted). Sessions are in-memory only, stored as SHA-256 digests, with bounded defaults (60 s bootstrap, 30 min idle, 8 h absolute, 8 login attempts/min/IP, capped sessions/bootstraps). Loopback by default gets automatic local login (same-origin fetch check); everything else redirects to login. CSRF proof is required for mutating methods; websockets require the CSRF subprotocol.

`build_web_auth_policy` refuses to listen when: non-loopback without a token; wildcard without a `public_url`; any remote exposure without a token; an https origin without direct TLS or a trusted proxy; remote without HTTPS unless explicitly allowed (which logs an INSECURE DEVELOPMENT MODE warning); direct TLS with a non-https origin. `localhost` is resolved via getaddrinfo — a hostname that resolves beyond loopback is **not** local-only. TLS requires both cert and key (only one raises). Trusted-proxy mode honors forwarded headers only from literal configured proxy IPs and only with a complete forwarded set.

## Canvas served mode (seam)

The server composes the loopback canvas control broker (started on startup, revoked on shutdown), the `CanvasGateway` with the served-authority proxy (every failure becomes a typed unavailable error; 2 s broker timeouts), the process-pinned runtime profile snapshot, and the kill switch with a **restart latch** — disabling canvas mid-run stays off until restart, and a 0.25 s policy watcher revokes Canvas previews/capabilities while retaining authenticated Console control and child bindings. Internals: [canvas.md](./canvas.md).

## Artifact share (dataflow)

1. The Artifacts share dialog (typed confirmation required for `0.0.0.0` without auth) → `ArtifactShareController.start_share` (one active share; starting a new one stops the old).
2. `stage_share` copies records into a private staging dir (opaque 128-bit keys, symlinks refused, atomic `bundle.zip`), then spawns the child server in its own session (stdout to a temp file — a pipe would wedge), which writes a `status.json` readiness marker.
3. The parent polls ≤ 15 s (failure kills the child and cleans staging); the emitted status carries loopback + LAN URLs.
4. Recipients browse `/` or `/index.json`, download `/artifact/{key}` (constant-time match; 404 unknown / 410 gone) or `/bundle.zip`; optional Basic auth (PBKDF2 in an executor, per-IP lockout, verified-pair digest cache).
5. Stop is SIGTERM→SIGKILL to the process group plus staging removal; the child self-exits when orphaned (ppid watchdog) and the next app start sweeps crash residue with two-step liveness (pid check + command-line check so recycled PIDs can't pin dirs).

## Config keys (`[web_server]`)

`host` (localhost), `port` (8000), `title`, `font_size` (12, clamped 6–32), `debug`, `access_token`, `public_url`, `tls_certificate` / `tls_private_key`, `trusted_proxy_addresses` ([]), `allow_insecure_remote_http` (false). `public_url` derives from host:port when TLS is set and it's absent. **`enabled` (default true) is read by no code** — documented but inert.

## Governing decisions and tests

ADR-123 (`123-artifact-share-web-export.md` — the staged-snapshot child server); canvas ADRs 115/121/124 for the served sandbox. No dedicated superpowers spec exists for served mode itself. Tests: `Tests/Web_Server/` (auth integration, served shell, canvas spawn/kill-switch, viewport patch, dependency gate, artifact share controller/manifest/server). Future items (explicitly non-goals today): binary distribution for `--serve`; per-recipient TLS/upload for share.

## Verified gotchas

1. `[web_server].enabled` is inert — nothing reads it.
2. The master-password unlock refuses to run in a served child (plain-language error, not a hang).
3. The textual-serve viewport patch fails closed on bundle-shape mismatch; the content-fingerprinted URL keeps a stale cache from resurrecting the unpatched bundle.
4. Websocket spawn failures log only the exception **type** — the spawn frame carries a per-child secret.
5. Share usernames may not contain `:` (Basic-auth split); staged sources must be non-symlink regular files contained in staging.

## Related docs

- [canvas.md](./canvas.md) — served canvas internals
- [artifacts.md](./artifacts.md) — the share entry point
- [backup-recovery.md](./backup-recovery.md) — the startup fence that served mode shares

## Shared Console execution fence (ADR-244 foundation)

The executing native App or served child owns the physical run-store OS lease,
not its routing parent. Contention refuses execution before AgentRuns schema or
recovery work. Non-owning readers use the explicit read-only store path. The
legacy mobile engine temporarily acquires the same kind of lease for its existing
run-store path; its idle shutdown releases the lease only after controller/fleet retirement;
bounded teardown retains that lease until process exit when
physical retirement cannot be proven. Shared canonical storage/routing cutover
is subsequent work; this foundation does not yet change served attachments.

## Independent served control and bounded projections (ADR-244, Task 2)

The private loopback broker starts before children regardless of Canvas/mobile
exposure. `issue_child` installs lifecycle-owned `ConsoleEventBuffer` state before
returning spawn credentials; the authenticated connection reader continuously
folds child projection frames into it with no browser-owned drain. Browser or
mobile absence never prevents draining. The broker stays alive across Canvas
disablement; gateway close, launch revocation and the restart latch still refuse
all Canvas capabilities/routes. Real channel failure, malformed envelopes and
revoked/replayed child credentials remain failures, with child disconnect visible
to lifecycle supervision. No reconnect is manufactured.

Private control protocol **3** is separate from HTTP `/api/v1/mobile` and browser
`chatbook-v1`; private versions 1/2 refuse. The closed Console method envelopes are:

| Method | Request payload | Response payload |
| --- | --- | --- |
| `console.snapshot` | `{"request": {...}}` | `{"snapshot": {...}}` |
| `console.events` | `{"request": {...}}` | `{"batch": {...}}` |
| `console.ticket` | `{"request": {...}}` | `{"ticket": {...}}` |
| `console.command` | `{"request": {...}}` | `{"receipt": {...}}` |
| `console.receipt` | `{"request": {...}}` | `{"receipt": {...}}` or `{"receipt": null}` |

Every method uses `.request`/`.response` suffixes. Existing recursive frame,
value/depth/collection limits apply; extra outer fields and wrong container types
refuse. Later native adapters strictly validate the inner arguments. These methods
currently refuse `authority_unavailable` without constructing an engine.
`console.changed` is body-free `{sequence: int >= 0, reset_required?: bool}` metadata.
Correlated responses resolve directly through the pending request map; projections
never own command outcomes, receipts or native approval resolution.

`ConsoleHostIdentity(server_epoch, console_id)` is frozen and extra-forbidden;
a retained child keeps its ID, and replacement under the same server epoch gets
a fresh ID. `ConsoleTarget` adds exact session/incarnation and optional
turn/run/handle/round IDs. All non-null fields use `validate_opaque_identifier`
plus path/whitespace rejection; UUID/hex and colon-separated native run forms
remain valid. Spelling alone grants no authority: later adapters must resolve
identities against the current native owner.

The parent buffer retains at most 64 projections by default and one Event wake.
Overflow and child-publisher loss set sticky reset state without closing control.
Reads are non-consuming, expose a monotonic parent sequence and detect historical
gaps even after an acknowledgement. Only a snapshot acknowledgement at the
current parent sequence clears the sticky marker. This parent transport cursor is
separate from native projection revision: a later snapshot adapter must register
its parent cursor **before** native capture, replay events received during capture,
and never acknowledge past that capture merely because transport advanced.

The legacy mobile engine remains until its later cutover. Its relay observes
lifecycle state using its own cursor, drops decision projections on loss and
refuses stale/replacement-child decisions even while its observer is paused.
Per-browser projection queues also cap at 64; overflow retains one reset and
suppresses further projections for that subscription. The existing phone status
line reports unavailable and clears/fences approval cards until page refresh
obtains a new subscription/current native state. This affects projection traffic
only; native execution, decisions and HTTP/control command responses retain their
existing owners. The child publisher uses a bounded thread-safe deque, one
coalesced loop wake and one async sender, with reset metadata on dropped history.

Verification: `Tests/Web_Server/test_served_console_events.py` covers zero
subscribers, overflow, current snapshot acknowledgement, independent wake readers,
Canvas/mobile-disabled control, identity replacement, paused decision fencing,
bounded browser subscriptions and real Chromium reset behavior. Existing broker,
publisher, auth, Canvas kill-switch and mobile relay tests retain their native
security assertions. The auth module and config-reading kill-switch node use the
documented `bootstrap_profile` marker so guarded config keeps its admitted private
source; test DB/session stores remain temporary.
