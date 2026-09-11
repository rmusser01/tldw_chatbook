# ADR-123: Artifact share web export via staged snapshot child server

- **Status:** Accepted
- **Date:** 2026-09-05
- **Scope:** Web serving, security surface, child-process lifecycle
- **Amends/relates:** none (task-31230 web auth remains scoped to the full-app web server)

## Context

Users need to hand Chatbook artifacts to other people. The only existing paths are
manual file transfer of `.zip` bundles from the private chatbooks directory.
`Web_Server/serve.py` serves the full TUI via textual-serve — far more exposure
and machinery than a download page needs.

## Decision

1. Sharing is an explicit, ephemeral session started from the Artifacts screen:
   the app stages immutable copies of the selected artifact zips (plus a
   pre-built bundle.zip) into `<user_data>/share/<id>/` with a `0600` manifest,
   then spawns `tldw_chatbook.Web_Server.artifact_share_server` as a child
   process (own process group) that serves ONLY the staging directory.
2. The recipient experience is a plain HTML page (no JavaScript) with per-file
   and bundle downloads plus `index.json`; aiohttp comes from the existing
   `[web]` extra. textual-serve remains the stack for full-app web mode and is
   unchanged; the share server reuses its dependency gate and download
   semantics (`Content-Disposition: attachment`, streamed `FileResponse`).
3. Auth is optional HTTP Basic (single shared username/password), PBKDF2
   verifier in the manifest, constant-time compare, per-IP lockout
   (10 failures / 30 s) tuned gentle because recipients may share a NAT IP.
   Non-loopback bind without a password requires typed confirmation.
   Auth covers every route; there are no unauthenticated metadata channels.
4. Lifecycle: the controller is app-owned (share survives navigation), one
   active share at a time, SIGTERM→SIGKILL escalation on stop, PPID orphan
   guard in the child, and a dead-PID startup sweep clears crash residue.

## Consequences

- Staging freezes content: later library edits/deletions cannot alter or break
  a running share; the child never reads the private chatbooks directory, so
  containment is single-root.
- The zip IS the artifact: registry records may drift cosmetically from bundle
  bytes (`update_chatbook` never rewrites the zip); v1 shares existing bytes
  and `index.json` hashes what recipients actually download. Re-export at
  share time is future work.
- Plain HTTP: Basic auth is an access gate, not confidentiality. Hostile
  networks require a reverse proxy (documented in the user guide).
- No per-recipient accounts, persistence, TLS, or upload — stop = revoke.

## Alternatives considered

- In-process aiohttp worker thread: rejected — event-loop teardown against
  screen/app exit is fragile; crashes destabilize the TUI.
- textual-serve `Server` subclass with a read-only viewer Textual app:
  rejected for v1 — heavy browser-terminal recipient UX and a second Textual
  app to maintain; the manifest/auth/staging work is reusable if wanted later.
- Serving live library paths: rejected — registry records may lack
  `file_path`, mid-share mutation/deletion breaks shares, and the child would
  need private-directory access.
