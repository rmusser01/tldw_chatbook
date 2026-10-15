# Served-Mode Authentication and Mobile Companion Surface — Design

- **Date:** 2026-10-15
- **Status:** Approved in brainstorming; pending implementation planning
- **Scope:** `--serve` web mode (`tldw_chatbook/Web_Server/`, `tldw_chatbook/Canvas/web_auth.py`)
- **Users:** single operator, multiple devices (phone, laptop). Not a multi-user system.

```text
ADR required: yes
ADR path: backlog/decisions/NNN-served-web-session-persistence-and-device-pairing.md (auth) and
          backlog/decisions/NNN-served-mobile-companion-surface.md (mobile API contract)
Reason: authentication/session-lifetime changes and a new long-lived authenticated service
        contract + UX surface. Both ADRs are created before implementation begins and linked
        from the Backlog tasks and implementation plan.
```

## 1. Context and motivation

Served mode (`tldw-cli --serve` / `tldw-serve`) runs the full Textual TUI in the browser via
`textual-serve` 1.1.3: an xterm.js character grid over a WebSocket, wrapped in the custom
`served_shell.html` toolbar + Canvas layout. Two complaints drive this design:

1. **Auth friction.** Remote admission uses one shared long access token resolved from
   `TLDW_CHATBOOK_WEB_ACCESS_TOKEN` → `[web_server].access_token` → OS keyring. Sessions are
   opaque and **in-memory only**: 30-minute idle / 8-hour absolute expiry, and every process
   restart revokes every device. Typing a long token on a phone keyboard, repeatedly, is the
   reported pain.
2. **Mobile usability.** The served experience is a keyboard-first desktop terminal. There is
   no mobile-specific handling anywhere in `Web_Server/`. On a phone this means tiny text, a
   soft keyboard fighting the grid, and modifier-key bindings that cannot be pressed. Users
   want the phone to be usable and enjoyable, not painful.

Deployment reality (confirmed with the operator): the phone reaches the server over a public
domain with HTTPS and over VPN/LAN with TLS; a LAN-with-shaky-TLS path is expected next.
Mobile use-cases are: chat with agents, reading/browsing, and check-ins/approvals — **not**
full app usage (settings, ingestion, persona management stay on the desktop terminal).

Grounding facts from the code:

- `Canvas/web_auth.py::WebAuthManager` keys sessions by cookie **digest** (`_cookie_digest`),
  mints cookies via a token factory, enforces login rate limits, and already has a one-use
  60-second bootstrap nonce (`issue_bootstrap` / `exchange_bootstrap`) exposed as
  `issue_browser_bootstrap_url()`.
- The Console engine — `Chat/console_chat_controller.py` and `Chat/console_chat_store.py` —
  has no *direct* Textual imports and is headless-capable; conversations persist in SQLite.
  Two caveats found in review: the controller transitively imports a Textual widget module
  via `Widgets/Chat_Widgets/chat_approval_card.py` (for one pure constant,
  `TOOL_DESCRIPTION_CAPTURE_CAP`), and `Chat/console_runtime.py::ensure_chat_controller`
  already supports **viewless-from-birth** runtimes but assembles them from a real App's
  context. Both caveats shape §4.3.
- `DB/ChaChaNotes_DB.py` performs automatic schema creation/checking/migration (V81) when
  opened — relevant to any parent-process DB access (§4.3).
- Existing tests: `Tests/Canvas/test_web_auth.py`, `Tests/Web_Server/`.
- Design-token governance covers `tldw_chatbook/css/` TCSS only; served web CSS
  (`Web_Server/static/`) is plain CSS and stays so.

## 2. Goals and non-goals

**Goals**

- End "type the long token on the phone" as a recurring event: device sessions survive
  restarts and last weeks, not hours.
- Provide a zero-typing pairing path (QR) that works on **every** network path, including
  LAN-by-IP where passkeys are impossible.
- Provide logout and per-device/all-device revocation (closes the README's "no logout or
  individual-session revoke UI" gap).
- Add passkey (WebAuthn) login as the premium path for stable HTTPS-domain origins.
- Provide a phone-first companion surface (`/m`) for chat, mobile-initiated approvals, and
  (phase 2) read-only browsing — installable as a PWA, with the full terminal one tap away.

**Non-goals**

- Multi-user identity, accounts, or per-user isolation (single operator).
- Replacing the terminal surface; the TUI remains the full application.
- Live cross-surface sync in v1 (desktop activity is not pushed to the phone; see §5.3).
- A service worker / offline mode for the mobile PWA in v1.
- Making the terminal itself mobile-friendly beyond a discovery link (possible cheap
  follow-up, deliberately out of scope to keep this design tight).

## 3. Part A — Authentication

### 3.1 Persistent device sessions

A new module `Web_Server/session_store.py` persists `BrowserSession` records as JSON at
`get_user_data_dir() / "web_sessions.json"` (atomic write via temp file + rename; file mode
`0600`). Records are keyed by cookie digest; **raw cookie values are never written to disk**
(stored fields: cookie digest, CSRF token, client IP, created/last-seen timestamps, expiry,
device label). `WebAuthManager` loads the store on startup and writes through on every
grant, revoke, and touch (touch writes are throttled, e.g. at most once per few minutes per
session, to avoid disk churn).

Lifetimes become config under `[web_server]`:

- `session_idle_timeout_days = 30` (sliding; replaces the 30-minute idle rule)
- `session_absolute_timeout_days = 180` (replaces the 8-hour absolute rule)

**Clock basis (correctness requirement found in review).** `WebAuthManager` currently
defaults to `time.monotonic`, and `serve.py` derives cookie `max_age` from the loop clock;
monotonic values are meaningless across restarts. The manager's default clock switches to
`time.time()` (the clock stays injectable for tests). Wall-clock timeouts tolerate NTP
steps at day-scale granularity — accepted and stated in the ADR.

**Cookie lifetime.** The cookie's `max_age` is set to the session's **absolute** expiry;
sliding idle revocation is enforced server-side (a cookie outliving its server session is
harmless because every request re-validates). No per-request cookie rewriting.

**Single owner.** Exactly one process — the served parent — reads or writes
`web_sessions.json`; native mode never touches it, and children do not either. Atomic
rename writes make the file crash-safe without locking. Session pruning and
capacity-driven eviction persist their deletions; throttled touch writes may lag
`last_seen` by minutes after a crash, which only shortens a session — harmless.

Loopback auto-login, the login rate limiter, TLS/proxy fail-closed rules, `SameSite=Strict`
+ CSRF posture, and the session capacity cap are unchanged.

### 3.2 Device manager and logout

A new authenticated plain-HTML page `/auth/devices` (same middleware, same CSRF rules):

- Lists devices: auto-derived label from User-Agent at grant time (e.g. "iPhone · Safari"),
  editable label, first-seen / last-seen, current-device marker.
- **Sign out this device** — revoke the current session, clear the cookie (logout).
- **Revoke** per device, and **Revoke all** — the incident-response replacement for
  "stop the process" (stopping no longer revokes; see §3.4).

### 3.3 QR pairing

- The `/auth/devices` page gains **Add a device**: the server issues a bootstrap nonce URL
  (existing machinery) and the page renders it as a QR code client-side using a vendored
  single-file JS QR renderer served from `Web_Server/static/` (no CDN; CSP unchanged).
- `tldw-serve --pair` prints an ASCII QR of a bootstrap URL to the terminal for first-run
  pairing when no desktop browser session is handy (uses the `qrcode` Python package,
  added to the `web` extra). It requires a configured `public_url`; when unset it logs a
  skip notice instead.
- The phone scans → `GET /auth/bootstrap?nonce=…` → session granted with a persistent
  (long-lived) cookie → redirect to `/`.
- Bootstrap nonce TTL for QR issuance is 120 seconds (up from 60); nonces remain one-use,
  exchange-rate-limited, and removed from the visible URL as today.
- Token-form login at `/auth/login` also issues persistent sessions; a "remember this
  device" checkbox (default checked) downgrades to the old short session when unchecked.

### 3.4 Security posture deltas (recorded in the auth ADR)

- **Stopping the process no longer revokes sessions.** Incident response becomes: use
  **Revoke all** on `/auth/devices`, or stop the process **and** delete
  `web_sessions.json` (or rotate the access token, which also invalidates pairing).
  `Web_Server/README.md` incident-response section is updated accordingly.
- **A stolen cookie is long-lived.** Mitigations: digest-only disk storage, `0600` file,
  per-device revocation with visibility into last-seen. No client-IP binding: roaming
  between Wi-Fi and cellular would break sessions; the trade-off is accepted and stated.
- The served host remains a single-user full-trust surface; nothing in this design adds
  multi-user isolation.

### 3.5 Passkeys — Phase 2 (separate task/PR)

- Dependency: `python-fido2` in the `web` extra. Discoverable (resident) credentials for a
  usernameless flow.
- RP ID is derived from `public_url`'s host: works for the public-domain path and for
  VPN/LAN origins with a stable hostname; **LAN-by-IP cannot use passkeys** (the WebAuthn
  spec forbids IP-address RP IDs) — QR pairing remains the fallback there.
- Registration: a button on `/auth/devices` available from an authenticated session.
  Login: "Use passkey" on `/auth/login`.
- Credential public keys persist in the session store. Amends the auth ADR when it lands.

## 4. Part B — Mobile companion surface

### 4.1 Surface

`GET /m` serves a mobile page from `Web_Server/static/mobile/` (`index.html`, `app.js`,
`app.css`, `manifest.webmanifest`, icon assets) — vanilla JS, no build step, plain CSS
matching the existing served-shell convention. It sits behind the existing auth middleware
and shares the browser session cookie. PWA manifest: `display: standalone`, theme color,
viewport with safe-area insets — installable to the home screen. No service worker in v1.

Discovery without User-Agent sniffing: the login page and the served shell render a
"Phone view" link visible only on narrow viewports (CSS media query). The mobile page links
back to `/` ("Open terminal") for everything else.

### 4.2 API

Authenticated routes under `/api/v1/mobile/`, CSRF-protected for state-changing calls:

| Route | Purpose |
| --- | --- |
| `GET /api/v1/mobile/conversations` | List conversations (paginated) |
| `GET /api/v1/mobile/conversations/{id}/messages?after=` | Cursor-paginated history from the DB |
| `POST /api/v1/mobile/conversations/{id}/messages` | Send a text prompt (attachments later) |
| `POST /api/v1/mobile/conversations` | Start a new conversation |
| `WS /api/v1/mobile/stream` | JSON events: stream deltas, message completion, approval requests, errors |
| `POST /api/v1/mobile/stream/cancel` | Stop the in-flight response |
| `GET /api/v1/mobile/approvals` | Pending approvals for mobile-initiated chats |
| `POST /api/v1/mobile/approvals/{key}` | Approve / deny |

The `/stream` WebSocket authenticates through the same `WebAuthManager.authenticate_request`
upgrade/origin validation the terminal WebSocket uses today (cookie + origin +
`Sec-Fetch-Site` checks already in `RequestFacts`) — no new admission surface.

Phase 2 additions (read-only browsing): `GET /api/v1/mobile/notes`, `GET /api/v1/mobile/library…`.

### 4.3 Engine binding (spike-gated decision)

The Console engine is headless-capable, but review surfaced two viable bindings, and the
choice is decided by a feasibility spike — the first step of Phase 1b, before the API shape
is frozen:

- **Binding 1 — parent-process service (lighter).** A `MobileChatService` instantiated
  lazily in the served parent process constructs `ConsoleChatController` +
  `console_chat_store` directly with a minimal context. Risk: the constructor takes ~50
  injected seams, and the agent/tool paths (agent bridge, run hooks, project-instruction
  callbacks, scratch spaces) are assembled in `console_runtime.py` from a real App's
  context — reconstructing them App-free is unproven.
- **Binding 2 — headless child App (heavier, reuses everything).** The mobile browser
  session spawns its own child running a full App headless, assembled through
  `console_runtime.ensure_chat_controller`'s existing **viewless-from-birth** path —
  the same process-per-session model as the terminal, with the mobile page as an
  alternate frontend via a parent↔child bridge (the control-broker pattern exists).
  Reuses all agent/tool/approval wiring at the cost of process management and bridge
  plumbing.

**Spike criteria:** stand up Binding 1 against a scratch DB and complete one
**agent-mode** round-trip (tools + one approval decision) with a deterministic fake
provider — plain-chat-only success does not qualify, because agent chat is a core
use-case. If agent seams demand App context, evaluate Binding 2 before freezing the API;
if neither lands cheaply, v1 ships mobile send scoped to plain chat and says so loudly.

Shared requirements either way:

- **Targeted refactor:** extract the pure constant `TOOL_DESCRIPTION_CAPTURE_CAP` (and any
  siblings) out of `Widgets/Chat_Widgets/chat_approval_card.py` into a non-widget module,
  so importing the controller no longer transitively imports Textual widget code into a
  non-UI process.
- **DB ownership:** the DB layer auto-creates/migrates schema (V81) on open. Children
  remain migration owners; a parent-process service (Binding 1) opens its connections
  lazily — first mobile request, well after child startup — and the spike verifies
  multi-open behavior.
- **Concurrency:** one mobile service instance; sends serialized per conversation.
  Concurrent sends to the same conversation from two live surfaces (phone + desktop) are
  a documented limitation with DB-level interleaving — semantically identical to two
  desktop tabs today.
- History reads always come fresh from the DB, so a conversation started at the desk is
  immediately readable on the phone.

**Stated v1 limit:** desktop activity is not live-pushed to the phone, and approvals are
surfaced only for **mobile-initiated** conversations. Approving a desktop-running session's
escalation from the phone requires a parent↔child event relay — Phase 2 plumbing, out of
v1 scope.

### 4.4 Errors and reconnect

- WebSocket drop → client reconnects with backoff and resyncs via `GET …/messages?after=`.
- Auth expiry mid-stream → the socket closes with a redirect-to-login signal; the composer
  draft is preserved in `sessionStorage`.
- Engine failure mid-stream → an error event; transcript state remains intact.

### 4.5 Configuration

`[web_server] mobile_surface = true` (default on — the surface is behind auth; the flag is
the escape hatch). Part A adds the two `session_*_timeout_days` keys (§3.1).

## 5. Sequencing

1. **Phase 1a — auth foundation:** session store + persistence, `/auth/devices` + logout +
   revocation, QR pairing (page + `--pair`), README incident-response update, auth ADR.
2. **Phase 1b — mobile v1:** engine-binding spike (§4.3 — gates the API shape), then the
   `/m` surface, mobile API (conversations/messages/send/stream/cancel/approvals),
   `MobileChatService` or headless-child binding per the spike, PWA manifest, discovery
   link; mobile-surface ADR.
3. **Phase 2:** passkeys (§3.5); read-only notes/library on `/m`; parent↔child relay for
   desktop approvals and live cross-surface updates.

Each phase is its own Backlog task(s) and PR; Phase 1a lands before 1b so the phone never
needs the token form again.

## 6. Testing strategy

- **Auth (extend `Tests/Canvas/test_web_auth.py`, `Tests/Web_Server/`):** store
  round-trip (atomic write, digest-only contents, `0600`), restart reload, sliding-idle and
  absolute expiry, revoke/logout/revoke-all, QR bootstrap → persistent cookie, nonce TTL +
  one-use + rate limits, `--pair` output, config key parsing, and **wall-clock migration**
  (a session persisted before the clock-basis switch is not resurrected with garbage
  timestamps).
- **Mobile API (`Tests/Web_Server/`):** route auth/CSRF denial; conversations list/history
  pagination with in-memory SQLite; send → stream round-trip with a deterministic fake
  provider; cancel; approvals list/decide round-trip; WebSocket upgrade rejected without a
  valid session.
- **Live verification** per `backlog/docs/lessons-live-verification.md`: phone-sized
  viewport via browser automation plus a real-phone manual pass over the operator's actual
  HTTPS origin.

## 7. Self-review notes

- Post-review revisions (2026-10-15, from the design review pass): wall-clock requirement
  for persistence (§3.1), cookie-max-age policy (§3.1), WebSocket auth clarification
  (§4.2), engine binding re-opened as a spike-gated decision with two candidate bindings
  and explicit agent-mode success criteria (§4.3), widget-constant extraction, lazy DB
  open with children as migration owners, and the two-surfaces-concurrent-send limitation.
- No TBDs; approvals scope, network-path constraints, and v1 sync limits are stated
  explicitly rather than left ambiguous. The engine-binding choice is deliberately left
  open — gated on spike criteria, not on an unexamined assumption.
- The two parts are separable: Part A is independently valuable and ships first.
- Out-of-scope items are listed in §2 so the implementation plan does not accrete them.
