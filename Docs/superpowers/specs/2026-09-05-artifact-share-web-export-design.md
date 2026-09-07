# Artifact Share Web Export — Design

- **Date:** 2026-09-05 (rev 2 — post design-review corrections)
- **Base branch:** `origin/dev` (artifacts screen and importer have evolved substantially vs `main`; this design reads `origin/dev`'s `UI/Screens/artifacts_screen.py` and `Chatbooks/` modules)
- **Status:** Implemented — ADR-123 (see the appendix for implementation notes/deviations)
- **ADR required:** yes — new network serving boundary, security/auth surface, and child-process lifecycle. Path: `backlog/decisions/123-artifact-share-web-export.md`.

## Problem

Users want to hand Chatbook artifacts to other people (colleagues, other machines, other tldw_chatbook instances). Today the only paths are manual file transfer of `.zip` bundles from the private chatbooks directory. The Artifacts screen should let a user select one artifact or a set of them and temporarily host a small web page from which recipients browse and download exactly those artifacts — optionally protected by a single shared username/password. Recipients import downloaded bundles through the existing Chatbook Import Wizard.

## Goals

1. From the Artifacts screen, share any selection of local Chatbook artifacts (including "all") over HTTP.
2. Recipients get a plain, mobile-friendly HTML library page with per-artifact download and a download-all bundle.
3. Exposure is exactly the chosen artifacts' bytes at share time — nothing else on disk is reachable, and later library changes don't alter a running share.
4. Optional single username/password (HTTP Basic auth), shared by all recipients.
5. Sharing is an explicit, bounded session: started and stopped from the UI; ends with the app.

## Non-goals (v1)

- Persistent share configurations, saved named collections, per-recipient accounts, or revocation lists (stop the share = revoked).
- HTTPS/TLS termination (documented reverse-proxy path instead; dialog warns that HTTP is plaintext).
- Sharing daily reports/briefings (future: same manifest + route pattern).
- Upload, comments, analytics, bandwidth limiting.
- Re-exporting/"refreshing" stale bundles at share time (see Snapshot semantics; recorded as future work).
- The browser-terminal (Textual-in-browser) recipient experience (future upgrade path, see Approaches).

## Decisions made without live user input

These were open clarifying questions; the user was unavailable, so each carries a recorded default for review:

| Question | Default taken |
| --- | --- |
| Recipient experience | Plain HTML page on an aiohttp server (not a browser terminal) |
| Share lifecycle | Ephemeral session; no persisted share config; single active share at a time |
| Artifact scope | Local Chatbook `.zip` artifacts that exist on disk |
| Collection semantics | Ad-hoc multi-select at share time; "select all" covers whole-library sharing |
| Default bind | Local network (all interfaces), with a mandatory typed confirmation if no password is set; loopback selectable for testing |

## Approaches considered

**A. Standalone share-server subprocess serving a staged snapshot (chosen).** The TUI *copies* each selected artifact's zip into a private staging directory, pre-builds the download-all bundle there, writes a manifest, then spawns `python -m tldw_chatbook.Web_Server.artifact_share_server <manifest.json>` (own process group) as a slim aiohttp app that only ever touches the staging directory.
- Pros: crash isolation from the TUI; immutable share content (later deletion/mutation of library files cannot change or break a running share); the child never needs access to the private chatbooks directory; bundle pre-building avoids streaming-zip pitfalls; standalone-testable and runnable headless in CI; mirrors textual-serve's own subprocess serving model.
- Cons: process supervision and a tiny stdout/status protocol; staging disk usage (chatbook zips are JSON-bundle sized, so small).

**B. In-process aiohttp server in a TUI worker thread.** Same routes, run inside a dedicated thread and event loop via `run_worker`.
- Pros: no IPC; shared state.
- Cons: loop teardown ordering against screen unmount/app exit is fragile; server exceptions can destabilize the TUI; orphaned listeners on crash.

**C. textual-serve `Server` subclass serving a dedicated read-only Textual viewer app.** The "purist" reading of "via textual-serve": recipients get the app-in-a-browser-terminal.
- Pros: most native textual-serve use; rich TUI browsing.
- Cons: textual-serve's `Server.__init__` requires a `command` and its value is the terminal-serving machinery; recipients pay websocket/JS terminal cost and get poor mobile UX; we would build and maintain a second Textual app; auth still has to be layered on the aiohttp side anyway.

**Positioning of textual-serve:** the full-app web mode (`Web_Server/serve.py`, `ChatbookWebServer`) remains the textual-serve surface and is unchanged. The share server reuses textual-serve's *stack and semantics* — the same `[web]` optional-dependency gate (`check_web_server_deps()`: textual-serve + aiohttp) and the same streamed-download behavior as textual-serve's `handle_download` (`Content-Type`, `Content-Disposition: attachment`, chunked `StreamResponse`). This keeps the dependency story unchanged while giving recipients the lighter page. If a browser-terminal viewer is wanted later, the manifest/auth/selection work below is reused as-is (Approach C becomes an additional route).

## Snapshot semantics (design-review finding)

`LocalChatbookService` records may carry `file_path=None` (registry-only, no bundle on disk), and `update_chatbook` mutates registry records without ever rewriting the zip — the on-disk bundle is a frozen snapshot that can drift from the record's name/description. Consequences, by decision:

- **The zip bytes are the artifact.** What a recipient imports is the zip; sharing the zip is exactly consistent with manually handing over that file. v1 therefore stages *existing zip bytes*, and does not attempt to re-export from the live DBs (reconstructing an export payload from a record generically is a new feature surface — recorded as future work).
- **Selection-time honesty:** the dialog lists all local records but disables ones with no on-disk bundle ("No exported bundle on disk"), so users aren't surprised at share time.
- **Cosmetic drift is accepted and documented:** the page shows the record's name/description; the zip's internal manifest may carry an older name if the record was renamed after export. The sha256 in `index.json` is computed over the staged bytes, so recipients always verify what they actually download.
- **Mid-share library changes:** deleting or overwriting the source zip after share start does not affect the running share (staging copy), which is both safer and more intuitive than 410-ing mid-session. The 410 path remains only for staging-dir corruption/cleanup races.

## Design

### Sharer flow (Artifacts screen)

1. New single-letter action (htop-style per ADR-031; exact key chosen at implementation to avoid collisions) opens `ArtifactShareDialog(ModalScreen)`.
2. Dialog contents: multi-select list (DataTable) of local Chatbook artifacts (name, kind, size, updated; records without an on-disk bundle shown disabled with reason); select-all; share-name text input (becomes the page title); auth toggle with single username/password pair; bind selector (Loopback only / Local network); port (default: auto-pick a free port).
3. Guardrails: typed confirmation required to bind non-loopback with auth disabled; password field minimally validated (non-empty with username; no complexity rules — it's a shared gate, not an account system).
4. On confirm, an app-level `ArtifactShareController` (owned by the app alongside `local_chatbook_service`, **not** the screen — see Lifecycle) runs a background worker that: creates `<user_data>/share/<share_id>/`, copies each selected zip into it, computes sha256 per staged file, builds `bundle.zip` there, writes the manifest (`0600`, atomic), and spawns the server child. The dialog shows preparation progress ("Staging N artifacts…").
5. While active, the Artifacts screen shows a share status banner: URL(s) (loopback + primary LAN address), artifact count, and Stop. Because the controller is app-owned, the share **survives navigation** between screens; the banner re-renders on Artifacts screen resume, and share start/stop also raise notifications to guard against forgotten shares. Only one share may be active at a time — starting a new one stops the previous (dialog states this when a share is already running).

### Recipient flow

- `GET /` — HTML library page: share name, artifact cards (name, description, kind, size, updated), per-artifact Download, and a Download-all button. Plain HTML + inline CSS, no JavaScript, no build tooling (repo gotcha); artifact names/descriptions HTML-escaped (they are user text; reuse the screen's `DANGEROUS_TEXT_PATTERNS` posture); `Cache-Control: no-store` so a restarted share never shows a stale cached page; basic a11y (`lang`, viewport meta).
- `GET /artifact/{key}` — streamed staged-zip download (textual-serve `handle_download` semantics; `Cache-Control: no-store`).
- `GET /bundle.zip` — the pre-built staged bundle (no runtime zip construction).
- `GET /index.json` — machine-readable manifest (keys, names, sizes, sha256) so future tldw_chatbook tooling can offer one-click import; today recipients still use Chatbooks → Import Wizard; `Cache-Control: no-store`.
- Footer hint: "Import into tldw_chatbook via Chatbooks → Import".
- **Auth applies to every route** (HTML, artifact, bundle, index.json, HEAD variants) — there are no unauthenticated metadata side-channels.

### Server architecture (`Web_Server/artifact_share_server.py`)

- `ArtifactShareServer`: loads the manifest, builds an aiohttp `web.Application` with the routes above plus `HEAD` support, optional auth middleware, and lifecycle logging through loguru.
- **Opaque keys:** each artifact gets a `secrets.token_urlsafe(16)` key (128-bit bearer token — in no-auth mode the URL itself is the secret; `Referrer-Policy: no-referrer` is set, and the dialog notes that recipient browser history will retain the URL). Requests never carry paths. The manifest maps key → staging-relative filename.
- **Path containment:** every request resolves the key to a path that must stay inside that share's staging directory and be a regular existing file, else 404/410. The child never reads the private chatbooks directory, so containment is single-root and trivially auditable (mirrors `Subscriptions/feed_server.py`'s posture).
- **Auth:** HTTP Basic when enabled. Verifier in the manifest as salt + PBKDF2-HMAC-SHA256 hash (patterns from `Subscriptions/security.py`), derived once at startup into memory; `hmac.compare_digest` verification; `401` + `WWW-Authenticate` on failure; failed-attempt lockout is per-IP with a 10-failure threshold and 30 s duration — deliberately gentle because multiple recipients commonly share one NAT address and must not lock each other out; credentials never logged.
- **Headers:** `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'` (inline CSS only), and `Content-Disposition: attachment` with sanitized filenames on downloads — including RFC 5987 `filename*=UTF-8''…` encoding so non-ASCII artifact names survive (plain `filename=` is ASCII-only).
- **Binding:** `127.0.0.1` or `0.0.0.0` (IPv4-only in v1; dual-stack `::` deferred — noted because macOS LAN discovery may prefer AAAA); port defaults to 0 (ephemeral, avoids collisions, including with the app's own `--serve` web port when running in web mode); explicit override offered. On startup the server prints `ARTIFACT_SHARE_READY <url>` on stdout (machine-readable for the TUI) and writes the resolved URL to a `status.json` beside the manifest.
- **URL display:** the TUI computes human-usable URLs without new dependencies — loopback directly, and the primary outbound-route IP via the connect-a-UDP-socket trick (no packets sent). The child's ready line reports the bound host/port; interface enumeration is TUI-side.
- **Process model:** spawned with `start_new_session=True` (own process group). The controller stops it with SIGTERM → SIGKILL escalation; the child polls its PPID and exits when orphaned (parent died without cleanup). The child writes its PID into the staging dir (`pid` file) to support startup sweeps.
- **Startup sweep:** at app start, `<user_data>/share/*` entries whose recorded PID is dead are removed (best-effort), so hard crashes don't leave staging copies and auth verifiers behind. Live-PID entries are left alone (a second app instance must not destroy a first instance's active share).
- **Optional-deps gate:** module import and controller paths are gated on `check_web_server_deps()` exactly like `Web_Server/serve.py`; UI shows the "install `tldw_chatbook[web]`" guidance when unavailable.

### Manifest schema (v1)

`<user_data>/share/<share_id>/manifest.json`, permissions `0600`, written atomically (`Utils/atomic_file_ops`):

```json
{
  "schema": 1,
  "share_id": "<uuid4>",
  "share_name": "…",
  "created_at": "2026-09-05T…Z",
  "auth": {"username": "…", "pbkdf2_salt": "…", "pbkdf2_hash": "…"} | null,
  "artifacts": [
    {"key": "<opaque>", "display_name": "…", "description": "…",
     "kind": "chatbook", "size_bytes": 123, "sha256": "…",
     "source_chatbook_id": 7, "staged_name": "7-<slug>.zip"}
  ]
}
```

Pydantic model per repo validation conventions. Paths in the manifest are staging-relative (`staged_name` resolves against the manifest's own directory), so the staging dir is relocatable and the manifest leaks no absolute private paths. The staging directory additionally contains the copied zips, the pre-built `bundle.zip`, and `status.json` (bound URL, PID) — deleted best-effort on stop and by the startup sweep.

### Screen integration

- `ArtifactShareController` (UI-agnostic, `Web_Server/artifact_share.py`): builds staging + manifest from `LocalChatbookService` records with on-disk verification, spawns/supervises the child, exposes status/stop; app-owned, created alongside `local_chatbook_service` in `app.py`. The Artifacts screen keeps only dialog + banner wiring — thin-screen pattern, controller unit-testable without Textual.
- Lifecycle events follow the repo's post_message → `@on()` pattern; staging/share work runs under `run_worker` with `exclusive=True` for the controller's single share.
- No config.toml section in v1 (session state lives in the manifest; dialog choices are not persisted).

### Error handling

- Port in use → child exits non-zero with a machine-readable stdout line; dialog surfaces it and offers auto-pick.
- Staging copy fails (source zip vanished between dialog and confirm) → share start aborts with the artifact named; nothing is exposed.
- Staged file missing/corrupt at request time → 410 Gone for that key (cleanup race only — library-file deletion no longer affects a running share).
- Child crash → banner shows "share stopped unexpectedly" with restart action.
- App exit → SIGTERM/kill + staging cleanup; orphan guard (PPID poll) and startup sweep cover hard kills.
- Corrupt/incomplete manifest → child refuses to start (fail closed).

### Testing

- Unit: staging + manifest building (selection, on-disk verification, disabled-record exclusion, sha256, 0600 perms, staging-relative paths); auth middleware (valid/invalid/lockout thresholds); route handlers via `aiohttp.test_utils` (index escaping, key opacity, staging containment, headers incl. RFC 5987 names, HEAD, no-store); bundle integrity (zip-of-zips opens; members match staged files).
- Integration: subprocess smoke test — start, `GET /`, `GET /artifact/{key}`, auth challenge, stop; staging cleanup on stop; startup sweep removes dead-PID dirs. PPID orphan-guard covered as a best-effort test (PID-reuse makes strict assertions flaky).
- UI wiring tests under `Tests/UI/` following existing console/artifacts test conventions (dialog state, banner lifecycle across screen resume, single-share enforcement).
- Per repo rule: targeted runs only; full sweep only on explicit request.

## Open items deliberately deferred

- Re-export/refresh of stale bundles at share time (registry-record ↔ zip drift beyond cosmetic naming).
- Saved share collections and share history.
- Access token/session auth aligned with task-31230 (Chatbook web auth) once it lands — the middleware seam is ready for it.
- Daily-report sharing; Canvas artifact sharing (tasks 31230/31003).
- TLS via documented reverse-proxy example; dual-stack IPv6 binding.
- Browser-terminal viewer experience (Approach C) layered on the same manifest/auth.

## Implementation notes (post-implementation appendix)

The feature shipped on branch `codex/artifact-share-web-export` under ADR-123.
Deviations from the design text above, discovered during implementation:

- **LAN-IP discovery order** ("URL display", above): the hostname is resolved
  first (`socket.getaddrinfo(socket.gethostname(), …)`, no socket egress) and
  the UDP route probe is kept only as a fallback for hosts whose name maps to
  loopback. Reason: the repo's test-suite network guard (task-15111) records
  even a swallowed UDP connect, which would fail loopback-marked tests at
  teardown. Production behavior is unchanged.
- **aiohttp 3.14 middleware markers:** both middlewares are plain
  new-style `(request, handler)` methods with `__middleware_version__ = 1`
  set directly (what `@web.middleware` does) — aiohttp 3.14 ignores unmarked
  new-style middleware; doing it manually keeps aiohttp lazily imported so
  the module loads without the `[web]` extra.
- **App-exit idempotency:** `TldwCli._shutdown_artifact_share` nulls the
  controller reference in a `finally`, making shutdown strictly once per
  controller (safe re-entry from a late `on_unmount`); screen-side consumers
  already read the attribute with `getattr(..., None)`.
- **Dialog widget choice:** the multi-select is a Textual `SelectionList`
  (not a `DataTable`) with per-option disabled state — records without an
  on-disk bundle carry the reason inline in their label. No separate
  select-all control was added; the list's built-in selection interactions
  cover multi-select.
- **Manifest field names:** the auth verifier fields are
  `pbkdf2_salt_hex`/`pbkdf2_hash_hex` (the design sketch wrote
  `pbkdf2_salt`/`pbkdf2_hash`), and the child's PID lives in `status.json`
  (alongside the bound URL) rather than a separate `pid` file; the startup
  sweep reads it from there.
- **Child-crash UX (known gap vs "Error handling", above):** there is no
  mid-share watcher or "share stopped unexpectedly / restart" banner action.
  A dead child is detected and cleaned up on the next stop/start or by the
  startup sweep, and the child's own PPID guard exits it if the app dies
  first; until then the banner can show a share whose server is already
  gone. Recorded here rather than silently diverging; a crash watcher
  remains future work.
- **Port-in-use handling (simplification of "Error handling", above):** v1
  does not emit a machine-readable stdout line or offer auto-pick on that
  error; the controller surfaces the child's captured error tail verbatim,
  and port 0 (ephemeral auto-pick) is the default dialog path, so an
  in-use explicit port is the only way to hit the failure at all.
- **Staging-preparation progress (simplification of the dialog design,
  above):** the "Staging N artifacts…" preparation progress state is not
  implemented in v1 — the dialog dismisses immediately on confirm and the
  completion (or named-artifact failure) arrives via the completion
  notification; there is no intermediate progress surface.

- **Deferred controller creation and sweep timing (CI boot-budget, ADR-097):**
  the controller is created — and the stale-share startup sweep runs — on the
  first share interaction, not during app boot. Importing the Web_Server share
  chain at boot breaches `MAX_TLDW_MODULES_AT_UI_READY` (972), whose policy is
  that the budget never rises and new imports must be deferred. The sweep
  therefore still runs before any new share can start; only its timing moved.
- **Task numbering:** the backlog task for this feature is task-31978; the
  original task-31758 was renumbered after dev claimed the number.
