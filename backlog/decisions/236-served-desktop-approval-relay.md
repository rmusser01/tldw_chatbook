# ADR-236: Served desktop approval relay

Date: 2026-10-16
Status: Accepted (implements spec Docs/superpowers/specs/2026-10-15-served-mode-auth-and-mobile-surface-design.md §4.3 stated-v1-limit removal + §5 Phase 2; extends ADR-233/ADR-234)
Amended by: [ADR-244](244-shared-console-ownership-and-mobile-supervision.md) for runtime-owned decisions, retained hosts and Canvas-independent control (design only; implementation pending)

## Context

The mobile companion surface (ADR-234) stated a v1 limit: approvals were
surfaced only for mobile-initiated conversations, and desktop activity
reached the phone only via client cursor polling. Approving a
desktop-running session's escalation from the phone requires a
parent↔child event relay — explicitly deferred to Phase 2 (spec §4.3/§5).
The spike's Binding 2 (headless child App per mobile session) was reserved
for exactly this relay but is only warranted if v2 wants phone-side
decisions to originate a desktop session; Phase 2c instead relays from the
already-running terminal child.

## Decision

1. Relay message types ride the existing loopback control broker
   (`Canvas/control_protocol.py`), extending `_MESSAGE_FIELDS` and
   `_REQUEST_REPLY` only — no transport, framing, secret, or frame-limit
   changes:
   - child→parent events (queued per child like `canvas.events`):
     `relay.approval.raised` {round_id, session_id, calls,
     timeout_seconds}, `relay.approval.cleared` {round_id, resolution},
     `relay.run.status` {session_id, status};
   - parent→child request: `relay.decide.request` {round_id, decisions} →
     `relay.decide.response` {round_id, delivered}.
2. Child side: a served-child-only publisher wraps the Console approval
   seam (call-through-first; TUI behavior unaltered) and forwards rounds
   plus run status through the child's `CanvasControlClient`. Native TUI
   runs never publish (the existing `CHATBOOK_SERVED_CHILD` detection
   gates activation).
3. Parent side: the parent's `MobileChatService` subscribes to broker
   events, merges desktop rounds into `list_approvals()` (marked
   `surface: "desktop"` + child_id; mobile rounds stay `surface:
   "mobile"`), and routes decisions back via `broker.request`. Unknown or
   dead child → round dropped, `False` returned.
4. Fail-closed on disconnect: child disconnect with pending rounds clears
   the parent's copies; the child's own timeout/deny semantics govern.
   With the broker absent (canvas disabled) the relay is inert and
   desktop approvals are invisible on the phone (documented).
5. Relay payloads carry tool names/arguments/options only — never
   credentials, cookies, or file contents beyond what the approval card
   already shows. Phone decisions ride the authenticated mobile API
   (auth + CSRF as in ADR-233) and the loopback + child-secret broker hop.

## Consequences

- The phone gains desktop approval authority: same single-operator trust
  posture as ADR-233/234 (the phone user is the desktop user).
- No cross-process secret handoff: relay frames cross the existing
  loopback, child-secret-authenticated channel; frame and queue limits
  are unchanged, so relay abuse cannot exceed existing broker bounds.
- The broker remains inert when canvas is disabled — desktop approvals
  then stay invisible on the phone; this is a documented limitation, not
  an error path.
- Child event classification widens from `canvas.events` to a small
  closed set of child→parent event types; anything else from a child is
  still `unexpected_child_message`.
- No live streaming mirroring of desktop transcripts: run completions
  push as `relay.run.status` events and the phone resyncs (update-level,
  not token-level).

## Alternatives considered

- Polling the child's DB from the parent: rejected — approvals are
  in-process Console state (the pending-approval seam), not persisted
  rows to poll.
- A second broker/transport for relay traffic: rejected — one loopback
  transport with per-child state, secrets, and backpressure already
  exists; duplicating it doubles the audit surface for zero isolation
  gain (relay frames are same-trust-class as canvas events).
- Binding 2 headless child App per mobile session (spike's reserved
  path): rejected for this phase — it buys process management cost that
  relaying from the existing terminal child does not need; revisit only
  if v2 wants phone-originated desktop sessions.
