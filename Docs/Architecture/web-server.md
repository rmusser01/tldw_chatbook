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

`/auth/login` (+ one-use 60 s bootstrap), `/` (the owned shell, no-store), `/ws` (websocket, protocol `chatbook-v1` + a `csrf.<token>` subprotocol; one replaceable terminal attachment to the retained profile AppService child), `/canvas/api/session` (bounded child-owned canvas state), `/download/{key}`, the auth-bridge shim, the two immutable shell assets, the patched `textual.js` (content-fingerprinted URL, 1 h public cache), textual-serve statics, and — only when Canvas is enabled — the canvas static route plus everything under `/canvas/gateway-<uuid>/…`, each request fenced to the exact outer browser session. Request bodies are capped (`client_max_size` 16 KiB; login POST 4 KiB).

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
is backed by the retained-host lifecycle described below.

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
refuse. Native snapshot/event adapters strictly validate the inner arguments. Ticket,
command and receipt methods dispatch through the retained runtime's native
remote ticket/command/receipt owner (`app_served.py`); unavailable authority still
refuses. No request constructs another engine. The separate public mobile-service
engine retirement and HTTP route cutover remain unfinished Task9 work.
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
line reports unavailable and clears/fences approval cards. A new subscription
inherits any outstanding child projection reset even while the child is idle;
page refresh cannot clear that warning or recover a missing native snapshot.
The phone reports that current approvals cannot be verified. A subscription-only
queue overflow may recover on page refresh when parent/native projections remain
current. This affects projection traffic
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


## Retained Console host and terminal attachment (TASK-36761, ADR-244)

`Web_Server/served_console_hosts.py` owns one shielded child start per server-admitted
run-store identity. Browser disconnect, takeover, revocation and output pressure
only detach that terminal generation. The child keeps running and draining output;
its bounded terminal queue (64 frames / 1 MiB) never waits on a browser socket.
Independent phone observers discover the retained child even with no terminal.
Canvas capabilities remain scoped to the exact attached browser; takeover revokes
the old binding rather than transferring its preview capabilities.

`ConsoleWebDriver` uses the installed WebDriver metadata seam. A reattach request
runs on the App thread, emits a generation-tagged begin boundary, restores terminal
modes, applies dimensions and forces a full compositor render before matching
ready. It never restarts the input thread or repeats startup framing. The parent
drops output before begin; both parent and browser refuse input before ready.
`served_shell.js` does not queue or replay disconnected input. The driver touches
Textual's internal resize/compositor methods; focused driver and actual-App browser
tests qualify this seam against the supported Textual version.

Shutdown closes host admission and drains lifecycle-owned retirement tasks even
if an observer is cancelled. The service uses its own process group for bounded
TERM/KILL escalation and waits for actual process exit. A closed host can be
replaced only after its process reports a return code; a timeout cannot prove
owner death. The child separately watches genuine private-control loss, closes
native admission/wakes, invokes native runtime disposal and exits. Disabling Canvas
alone preserves control and terminal work.

Host `reserve_saved_conversation` and `register_saved_conversation` deduplicate
native session identity and concurrent hydration. These are the Task3 seam; Task9
wires remote create/open and native save/hydration callbacks. Ordinary terminal
persistence/hydration remains native. Remote saved-open is not qualified here.

Evidence lives in `Tests/Web_Server/test_served_console_hosts.py`,
`test_console_web_driver.py`, `test_served_shell_browser.py` and focused
`Tests/Canvas/browser/test_canvas_served_flow.py` cases. Actual App tests use the
normal runtime/admission/persistence with a deterministic provider. Passive xterm
public-buffer observation verifies rendered output without changing the renderer;
child receipts verify one input thread, resize, mouse and multiline paste.


## Retained native read protocol (TASK-36763, private v3)

Authenticated `auth.response` requires exactly `status` and `host_identity`;
the broker supplies the frozen `{server_epoch, console_id}` identity. The child
client exposes it read-only after authentication; missing/malformed identities
fail closed before publisher or handler admission. It is never browser supplied.
The served publisher now polls retained runtime revision witnesses at its existing
cadence and emits body-free `console.changed`; mounted approval callbacks are no
longer its projection source. Correlated responses retain their independent
pending-request path. Console methods work when Canvas is disabled.

Within the outer `{"request": {...}}`, snapshot accepts only optional `target`
(null for discovery), `cursor` and `limit` (1–100, default 50). Event requests use
these closed operation-specific fields:

| Operation | Fields besides `operation` |
| --- | --- |
| `read` (default) | optional `after_sequence` (nonnegative integer, default 0) |
| `register`, `disconnect` | `client_id` |
| `answerable` | `client_id`, exact `target`, boolean `answerable` |
| `acknowledge` | `sequence` (nonnegative integer) |

Responses remain `{"snapshot": ...}` or `{"batch": {sequence, reset_required,
events}}`, with `events` a JSON list. Registration is a private parent lifecycle
operation, not browser authority. Task9 wires authenticated WebSocket/revocation
lifetime and must mint a different ID per live subscription. The inner Console
payload budget is checked before serialization; recursive/depth limits also apply.

There are two distinct sequence domains. The child snapshot/batch sequence is
its **native invalidation buffer** boundary, captured before reading detached
native messages and lock-owned round/revision witnesses. A native witness change
during capture publishes a new reset invalidation; the old capture cannot clear
it. Only a completed page chain can nominate a child boundary for acknowledgment,
and the native buffer clears a reset only if that boundary is still current.
The child publisher frame's sequence is only a native invalidation hint.

The parent's `ConsoleEventBuffer` sequence is the **transport receive cursor**;
it advances independently when child frames arrive, including overflow reset
frames. No Task5 code equates or acknowledges those two counters. Task9's HTTP
snapshot adapter must register the parent cursor before requesting child capture,
reconcile frames received during capture, and acknowledge only the reconciled
parent boundary. A later transport cursor must never be copied from the child's
snapshot `sequence` or acknowledged merely because capture returned. Native and
transport sticky resets each retain their respective owner.


## Private Console command tickets and receipts (ADR-244)

The child uses `ConsoleRemoteReceipts` around its existing
`ConsoleRemoteCommands`. `console.ticket.request` accepts exactly
`action`, `target`, `payload` inside `request` and returns `ticket` containing
opaque `token` plus `command_id` before any effect. A fresh child-only HMAC key
binds the immutable five-minute monotonic deadline, server and Console epochs,
exact target/action and keyed payload fingerprint. It is independent of browser
CSRF and child-launch secrets. Tickets never bypass native admission.

`console.command.request` accepts exactly `ticket`, `payload`; its body-free
`receipt` follows native outcomes. Send is pending/Preparing until native
custody observes acceptance or queue admission. Queued guidance is not consumed;
accepted Stop does not establish physical exit. Approval/question settlement
records the winning command under the existing native lock and returns
`accepted`/`answer_recorded`; a later native final outcome (after deadline,
cancellation and revocation checks) advances it to `applied`, `refused` or
`outcome_unknown`.
A losing answer returns refused, even when the same round is already resolved.
Interrupted cross-database handoffs stay unknown unless an exact native observer supplies a later fact.
`console.receipt.request` accepts exactly `command_id` and returns `receipt` or
null; absence does not authorize replay. Prior-epoch receipts remain readable in
the canonical profile while old tickets cannot execute. At full capacity, a new
claim can replace one oldest known prior-owner receipt; pending/unknown evidence
is protected. Verification and same-owner pruning share one clock reading to
preserve a duplicate at the expiry boundary. A new claimed command or capacity
Stop fallback rechecks expiry immediately before its native handoff; time spent
in repository work cannot extend its ticket.

Snapshots include bounded `command_receipts` and `next_receipt_cursor` fields;
pass the latter as `receipt_cursor` independently of transcript `cursor`.
Both page cursors must finish before the native snapshot is acknowledged.
No subscription or reconnect automatically resends a command.

At the 4,096-row bound, authenticated exact native Stop/subagent Stop can still
request cancellation through the idempotent native path without allocating a
row. Its transient result explicitly reports
`stop_requested_exit_unconfirmed_receipt_unavailable_capacity` (or a refusal).
Its query can return null. It is never reported as durable completion; clients
must inspect native state and retain uncertainty instead of retrying on reconnect.
The public API and client cutover remain separate implementation steps.
