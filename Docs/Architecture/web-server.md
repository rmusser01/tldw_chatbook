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

`/auth/login` (+ one-use 60 s bootstrap), `/` (the owned shell, no-store), `/ws` (websocket, protocol `chatbook-v1` + a `csrf.<token>` subprotocol; one authenticated login + terminal page ↔ exactly one AppService child), `/canvas/api/session` (bounded child-owned canvas state), `/download/{key}`, the auth-bridge shim, the content-fingerprinted owned shell scripts and immutable shell CSS, the patched `textual.js` (content-fingerprinted URL, 1 h public cache), textual-serve statics, and — only when Canvas is enabled — the canvas static route plus everything under `/canvas/gateway-<uuid>/…`, each request fenced to its exact terminal and owning authenticated login. Request bodies are capped (`client_max_size` 16 KiB; login POST 4 KiB).

## Auth and bind policy (fail-closed)

Token precedence: `TLDW_CHATBOOK_WEB_ACCESS_TOKEN` env → `[web_server].access_token` → OS keyring (failures read as absence; credential reprs redacted). Sessions are in-memory only, stored as SHA-256 digests, with bounded defaults (60 s bootstrap, 30 min idle, 8 h absolute, 8 login attempts/min/IP, capped sessions/bootstraps). Loopback by default gets automatic local login (same-origin fetch check); everything else redirects to login. CSRF proof is required for mutating methods; websockets require the CSRF subprotocol.

`build_web_auth_policy` refuses to listen when: non-loopback without a token; wildcard without a `public_url`; any remote exposure without a token; an https origin without direct TLS or a trusted proxy; remote without HTTPS unless explicitly allowed (which logs an INSECURE DEVELOPMENT MODE warning); direct TLS with a non-https origin. `localhost` is resolved via getaddrinfo — a hostname that resolves beyond loopback is **not** local-only. TLS requires both cert and key (only one raises). Trusted-proxy mode honors forwarded headers only from literal configured proxy IPs and only with a complete forwarded set.

## Terminal pages beneath a login (ADR-251)

Every no-store HTML render supplies a fresh random 32-hex `terminal_id`. The owned
WebSocket shim appends it after textual-serve adds its terminal-size query, and
the shell includes it in `/canvas/api/session` polls. The server validates one
supplied ID and composes it with the authenticated login ID before binding a child.
A page label grants no authentication; the same label under another login selects
an independent scope. Missing IDs retain the legacy single-login scope.

At most 512 terminal scopes may be active. Page renders allocate no pending
state, and only the bound child incarnation releases its slot. A fresh page can
start while an old page's child is still stopping. Logout, device revocation and
expiry continue to close every terminal channel registered beneath that login.
The auth shim and shell script URLs are content-fingerprinted to avoid a cached
older script discarding the page identity.

Mounted Canvas requests must match an active terminal scope belonging to the
authenticated login, then pass the gateway's exact shell, cookie and capability
checks. Disabling Canvas closes the broker/gateway and discards launches while
retaining terminal ownership until its WebSocket retires; terminals remain usable
and continue consuming their bounded slots. Exact-child terminal teardown fully
revokes its Canvas session, shell and capabilities, immediately releasing gateway
capacity for the next page rather than retaining an unreachable event lane.
Every awaited Canvas snapshot rechecks its captured terminal child, broker and
enabled state before publishing a launch, so a late poll cannot recreate a
retired page's shell after tab closure or Canvas revocation.

## Canvas served mode (seam)

The server composes the loopback canvas control broker (started on startup, revoked on shutdown), the `CanvasGateway` with the served-authority proxy (every failure becomes a typed unavailable error; 2 s broker timeouts), the process-pinned runtime profile snapshot, and the kill switch with a **restart latch** — disabling canvas mid-run stays off until restart, and a 0.25 s policy watcher revokes children. Internals: [canvas.md](./canvas.md).

## Artifact share (dataflow)

1. The Artifacts share dialog (typed confirmation required for `0.0.0.0` without auth) → `ArtifactShareController.start_share` (one active share; starting a new one stops the old).
2. `stage_share` copies records into a private staging dir (opaque 128-bit keys, symlinks refused, atomic `bundle.zip`), then spawns the child server in its own session (stdout to a temp file — a pipe would wedge), which writes a `status.json` readiness marker.
3. The parent polls ≤ 15 s (failure kills the child and cleans staging); the emitted status carries loopback + LAN URLs.
4. Recipients browse `/` or `/index.json`, download `/artifact/{key}` (constant-time match; 404 unknown / 410 gone) or `/bundle.zip`; optional Basic auth (PBKDF2 in an executor, per-IP lockout, verified-pair digest cache).
5. Stop is SIGTERM→SIGKILL to the process group plus staging removal; the child self-exits when orphaned (ppid watchdog) and the next app start sweeps crash residue with two-step liveness (pid check + command-line check so recycled PIDs can't pin dirs).

## Config keys (`[web_server]`)

`host` (localhost), `port` (8000), `title`, `font_size` (12, clamped 6–32), `debug`, `access_token`, `public_url`, `tls_certificate` / `tls_private_key`, `trusted_proxy_addresses` ([]), `allow_insecure_remote_http` (false). `public_url` derives from host:port when TLS is set and it's absent. **`enabled` (default true) is read by no code** — documented but inert.

## Governing decisions and tests

ADR-123 (`123-artifact-share-web-export.md` — the staged-snapshot child server); canvas ADRs 115/121/124 for the served sandbox and ADR-251 (`251-served-terminal-page-isolation.md`) for terminal ownership beneath shared logins. No dedicated superpowers spec exists for served mode itself. Tests: `Tests/Web_Server/` (auth integration, served shell, canvas spawn/kill-switch, viewport patch, dependency gate, artifact share controller/manifest/server). Future items (explicitly non-goals today): binary distribution for `--serve`; per-recipient TLS/upload for share.

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
