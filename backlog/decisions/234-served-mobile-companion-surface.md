# ADR-234: Served mobile companion surface

Date: 2026-10-15
Status: Accepted (implements spec Docs/superpowers/specs/2026-10-15-served-mode-auth-and-mobile-surface-design.md §4; binding per spike Docs/superpowers/spikes/2026-10-15-mobile-engine-binding-spike.md)
Superseded by: [ADR-244](244-shared-console-ownership-and-mobile-supervision.md) for separate mobile execution/storage (design only; implementation pending)
Amended: 2026-10-16 (PR #3128 review — the engine runs over ISOLATED DBs at
  `user_data_dir()/mobile`, not the shared chat DB; see the amendment notes
  under Decision and Consequences)

## Context

The served terminal is a keyboard-first character grid — painful on a phone.
The operator's mobile use-cases are chat with agents, check-ins/approvals for
mobile-initiated conversations, and reading; full app usage stays on desktop.
The spike proved Binding 1 (parent-process service driving the existing
Console engine App-free) works for agent-mode including tool approvals.

## Decision

1. A phone-first page at `/m` (vanilla JS, PWA-installable, no service
   worker) served by the same process and the same Phase-1a auth; the
   terminal remains one tap away.
2. `MobileChatService` in the served parent process constructs
   ConsoleChatStore + ConsoleAgentBridge + ConsoleChatController directly
   with a duck-typed App context (call_from_thread + headless
   UnifiedMCPControlPlaneService + workspace_registry_service). One service
   instance; sends serialized per session.
   **Amended 2026-10-16 (PR #3128 review):** the engine is constructed over
   ISOLATED databases under `user_data_dir()/mobile` (`chat.db`,
   `workspaces.db`, `runs.db`) — NOT the shared chat DB the desktop/child
   processes open. Conversations started at the desk are therefore not
   readable on the phone in v1.
3. Authenticated API under `/api/v1/mobile/`: REST for conversation list,
   cursor-paginated history, send, new conversation, cancel, approvals; one
   WebSocket (`/api/v1/mobile/stream`) carrying run/delta/approval/error
   JSON events. The store is the streaming source of truth; the service
   polls run-state and message content on a tick.
   **Amended 2026-10-16:** the store and the polling above are scoped to
   the isolated mobile DB — i.e. to mobile-initiated conversations only.
4. Approval surface: pending rounds captured from the controller's
   set_pending_approval seam, pushed as events, decided via
   resolve_pending_approval. Default posture fail-closed ("ask").
5. v1 limits (stated, not accidental): approvals only for mobile-initiated
   conversations; desktop activity reaches the phone via client cursor
   polling (completion-level), never live push; project-instruction
   dispatch stays disabled for mobile sessions; default-workspace sessions
   expose no fs_*/git_* tools.
   **Amended 2026-10-16 (PR #3128 review):** the "desktop activity reaches
   the phone via client cursor polling" clause above is wrong as written.
   With the isolated mobile DB, conversation polling cannot see desktop
   activity at all: the client polling's actual scope is cross-surface
   completion sync within mobile-initiated conversations. Desktop activity
   reaches the phone only via the Phase 2 parent↔child relay (spec §5:
   desktop approvals + run-status/live-update pushes — the "Phase 2c"
   relay), not via conversation polling. The spec §4.3 "a conversation started at the desk is
   immediately readable on the phone" sentence and the §4.4 desktop-
   completion framing are superseded by this amendment; the spec carries a
   matching dated post-implementation amendment note.
6. Config: [web_server] mobile_surface = true (default on).

## Consequences

- New authenticated HTTP/WS surface reusing the existing session + CSRF
  middleware — no second auth path to audit.
- The parent process now opens the chat DBs (children already do; WAL
  multi-open verified by the spike). Children remain migration owners.
  **Amended 2026-10-16 (PR #3128 review):** the parent opens only the
  ISOLATED mobile DBs under `user_data_dir()/mobile` (created/migrated by
  the service on first use); it never opens the shared chat DB. There is
  no shared conversation DB in v1 — desk-started conversations stay on the
  desktop surfaces, and mobile conversations live only in the mobile data
  dir.
- The controller's ~50 constructor seams stay defaulted except the five
  the spike named; future Console changes may need the service updated.
- Phase 2 (passkeys, read-only library/notes, desktop->phone approval
  relay) is explicitly out of scope.
  **Amended 2026-10-16:** that relay is not merely an enhancement — it is
  the only planned path by which desktop activity (approvals, run-status)
  reaches the phone, since v1's conversation polling cannot cross the
  isolated-DB boundary.
- Amended 2026-10-16: read-only notes + library browsing landed on /m
  (Phase 2b); write paths remain out of scope.

## Alternatives considered

- Binding 2 (headless child App per mobile session): reserved for the
  Phase-2 cross-surface relay; v1 needs no process management.
- Making the terminal mobile-friendly: ceiling is "less painful"; rejected
  as the primary path for the stated use-cases.
