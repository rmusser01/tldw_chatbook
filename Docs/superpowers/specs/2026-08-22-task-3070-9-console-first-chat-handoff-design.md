# TASK-3070.9 Console First-Chat Handoff Ownership Design

**Status:** User-approved design; pending specification review
**Date:** 2026-08-22
**Approved design base:** `0da426e1e4c2846f13671690b8f981f72e673359`
**Latest-dev Task 0 baseline:** `ede2162143331e324c44832ff6a3910e1185cf58`
**Task:** `TASK-3070.9`
**Parent design:** `Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md`

## Goal

Move the post-baseline first-chat handoff policy out of `ChatScreen` and into
the existing `ConsoleSessionController` without changing claim identity,
configuration-generation fencing, pristine-session eligibility, rollback,
acknowledgement, retry, privacy, mounted resynchronization, or focus behavior.

This is an ownership-only extraction. It does not redesign first-run setup or
the Console session model.

## Baseline and Scope

At the immutable approved design base, `chat_screen.py` has 19,995 physical
lines and `ChatScreen` has 640 direct methods. The reviewed first-chat family
is exactly eight methods and 328 definition lines:

- `_first_chat_defaults_match`
- `_current_first_chat_defaults`
- `eligible_console_first_chat_session_id`
- `_release_first_chat_claim`
- `_log_first_chat_handoff_exception`
- `_resync_console_after_first_chat_rollback`
- `_resync_mounted_console_after_first_chat_rollback`
- `consume_pending_console_first_chat_intent`

Task 0 rebased the reviewed documentation series without change onto latest
`origin/dev` `ede2162143331e324c44832ff6a3910e1185cf58`. The rebased candidate
retains the same 19,995 / 640 screen measurements, exact eight-method / 328-line
membership, and normalized family AST digest
`3a2968883c63dc89de430ee72b40444ebd97fb9b36c1dbc8a46e19d063a715ee` as
the immutable design base. Latest-dev test and diagnostic results are execution
baseline evidence only; they do not replace or rewrite the approved oracle.

All eight methods move to `ConsoleSessionController`. Their only mutable owned
state, `_first_chat_handoff_notified_revision`, moves with them. No compatibility
method or descriptor remains on `ChatScreen`.

Production callers are rewired to the owner:

- ChatScreen mount/resume consumption calls `self._session.consume_pending_...`.
- The first-run wizard asks the mounted Console screen's `_session` controller
  for `eligible_console_first_chat_session_id()`.
- Tests that intentionally exercise the policy call the controller owner.

The task does not change the first-run wizard's staging payload, reserve-new
decision, dismissal behavior, or error copy.

## Selected Architecture

Extend the existing `ConsoleSessionController`. This is the smallest design and
matches the parent Wave 6 decision that first-chat behavior is session policy.
Creating a fifteenth controller would duplicate session/store dependencies and
add an unnecessary lifetime boundary. Keeping screen delegates would fail the
single-owner inventory and method ratchet.

The moved policy continues to use the existing session controller's late-bound
store and UI synchronization dependencies. New dependencies are narrow,
keyword-only, late-bound callbacks for presentation observations/actions only:

- whether the screen is mounted;
- capture of the current control provider/model and an opaque focus token;
- projection of provider/model into the mounted Console controls;
- restoration of focus when the captured target is still mounted.

The moved methods and new first-chat seams do not query the DOM, retain a focus
widget, reach through `ChatScreen` to another controller, or capture a bound
screen method at construction. They do not change the session controller's
older, explicitly documented `_screen` framework-service reference. The focus
token is opaque to first-chat policy code and is passed directly back to the
presentation callback. Existing chat-core, settings-summary, control-bar,
native-chat UI, and worker callbacks remain the only repaint seams.

Runtime configuration helpers and the first-chat handoff types move to
`session.py` as ordinary module dependencies. The existing stable
`app_instance` reference remains the source of the pending-handoff store and
notification service.

## State and Ownership Boundary

`ConsoleSessionController` owns:

- exact claim/release/acknowledgement decisions;
- current-claim and configuration-generation fences;
- pristine global-session eligibility;
- reserved-target creation versus existing-target refresh;
- rollback of created or refreshed pristine sessions;
- retry notification deduplication by claim revision;
- allowlisted metadata-only failure logging;
- the decision to request presentation projection or rollback resynchronization.

`ChatScreen` and wiring-owned callbacks own only:

- reading and writing the control provider/model presentation mirrors;
- checking mounted state;
- scheduling/performing mounted repaint work;
- capturing and restoring the current focus target.

The session store remains authoritative for active session identity and
settings. The pending-handoff store remains authoritative for claim identity and
replacement safety.

## Processing Flow

1. Claim `CONSOLE_FIRST_CHAT`; return `False` if absent.
2. Validate the intent and resolve defaults only when provider, model, and
   configuration generation still match.
3. Capture prior active-session identity and presentation state.
4. Either create the exactly reserved target or refresh the exact pristine
   existing target. Never adopt an unreserved or concurrently claimed target.
5. Recheck configuration generation, active-session identity, and exact current
   claim after each mutation boundary.
6. Project the selected provider/model and request the existing mounted UI
   synchronization callbacks.
7. Recheck the fence and acknowledge the exact current claim under the same
   configuration generation.
8. On success, clear notification deduplication state and return `True`.
9. On any retryable failure, roll back only this attempt's mutation, request
   mounted resynchronization/focus restoration, release the exact claim without
   disturbing a replacement, emit at most one warning per current revision, and
   return `False`.

Ordering is part of the contract. Rollback precedes release; synchronous control
projection precedes the asynchronous native-chat repaint; focus restoration
occurs only after that repaint and only when both the screen and focus target are
still mounted.

## Error Handling and Privacy

The exact existing lifecycle boundaries remain unchanged. Notification,
release, rollback, and guarded acknowledgement exceptions are contained so
those retry paths do not fail mount/resume. Initial claim, configuration
inspection/default construction, and core session-store mutation exceptions do
not gain new broad containment in this extraction.

Logs include only an allowlisted operation category and exception type. They do
not include exception text, provider credentials, endpoints, handoff values, or
session contents. A failing instance-level release wrapper may use the existing
`PendingHandoffStore.release` fallback, preserving exact-claim and replacement
invariants.

No broad exception swallowing is added around claim acquisition, configuration
inspection, or core session-store mutations; their current propagation and
rollback/retry semantics remain unchanged.

## Testing Strategy

Testing is focused and layered:

1. Add controller-only, no-mount tests for eligibility, absent/invalid claims,
   reserved creation, existing pristine refresh, session/config/claim races,
   rollback-before-release, exact acknowledgement, replacement preservation,
   notification deduplication, and privacy-safe logging.
2. Keep mounted integration coverage for provider/model restoration, native UI
   resynchronization, and focus restoration after rollback.
3. Update first-run wizard tests to exercise the `_session` owner and preserve
   producer-side staging behavior.
4. Extend wiring tests to prove all new callbacks resolve late and return only
   presentation observations/actions.
5. Strengthen the Wave 6 inventory so the eight methods have exactly one owner,
   none remain on `ChatScreen`, and the moved bodies contain no DOM queries or
   sibling-controller reach-through.
6. Run mutation probes for a generation fence, active-session fence,
   rollback-before-release ordering, guarded acknowledgement, replacement-safe
   release, privacy logging, and focus restoration.
7. Run affected-only pytest selections, targeted Ruff lint/format, isolated
   compile checks, `git diff --check`, and the production diagnostic inventory
   gate. Any baseline failure must be reproduced against the frozen base and
   documented; task-caused failures are blockers.

## Ratchet and Diagnostics

The extraction must remove all eight direct `ChatScreen` methods. The expected
projection from the frozen base is at most 19,667 physical lines and 632 direct
methods before ordinary import/wiring adjustments. Final closeout records actual
post-rebase measurements; it never raises an existing ceiling.

Diagnostic calls move only when their source methods move. The canonical
inventory is updated exactly once after a three-way checked/base/candidate
comparison proves method/digest preservation and unchanged persistent sink
topology.

## Non-Goals

- No new controller class.
- No handoff schema or pending-store changes.
- No change to provider/model default resolution.
- No change to session creation, refresh, or rollback algorithms.
- No first-run wizard UX or copy change.
- No generic session-controller cleanup or typing sweep outside touched seams.
- No compatibility shims on `ChatScreen`.
- No auto-speak work from TASK-3070.10.

## ADR Check

**ADR required:** no
**ADR path:** N/A
**Reason:** This is a mechanical implementation of the already approved Wave 6
ownership boundary. It introduces no new storage, sync, service, security,
dependency, or long-lived application architecture decision.

## Acceptance Mapping

- **AC1:** exact eight-method ownership inventory, controller-owned revision
  state, direct caller rewiring, and presentation-only screen callbacks.
- **AC2:** no-mount and mounted tests lock fencing, rollback, acknowledgement,
  retry, privacy, session identity, control projection, and focus ordering.
- **AC3:** controller-focused tests, wizard/mounted integration tests,
  architecture/wiring guards, mutation probes, and targeted static/diagnostic
  gates complete before closeout.
