# PR 2427 Library entry live-focus guard

## Approved scope

The user approved the proposed bounded live-focus guard on 2026-09-12.
This document records that design for written-spec review before implementation.
Corresponding task: TASK-31932, AC3; existing behavior: TASK-2856.

ADR required: no.
ADR path: N/A.
Reason: close an event-ordering gap in the existing user-focus protection;
no new state owner, persistence, service boundary or focus policy.

## Evidence and intended behavior

On published checkpoint `2c79e598ae`, the entry continuation checks its arm's
pending flag and generation, but foreign focus disarms it only when a queued
DescendantFocus event is processed. The real retry timer can run in between.
The observed Retry widget remains attached with valid geometry throughout:
`/private/tmp/pr2427-retry-trace-native.apr8fF/trace.jsonl` shows Retry receiving
focus at 463517.535491 and the timer taking it at 463517.542918. The stale Retry
focus event then correctly has no authority to disarm on behalf of an old owner.

TASK-2856 promises that idle entry retries do not override user navigation.
Once a newer, attached foreign widget owns focus, an ordinary entry continuation
must preserve that exact widget and disarm the old request, even before its
focus event is processed. Both timer and after-refresh continuations must obey
the same check.

## Existing boundary and exceptions

Change only the ordinary, receipt-free branch of
`LibraryScreen._focus_library_list_entry_if_current`, after the existing pending
and generation admission and before `_focus_library_list_entry` is called.
Use existing fields and the existing `_disarm_library_list_entry_focus` method;
do not invent another generation, timer, retry loop or focus owner.

Ordinary entry remains eligible when focus is absent or no longer attached,
when the framework has fallen back to the existing adaptive pane grip, when
focus is still the original arm anchor, or when it belongs to a row of the
armed list under the existing row-class mapping. Preserve the existing exact
programmatic-target identity exception while that one-shot target remains
current; an unrelated or stale target pointer must not permit a different
focused widget. Do not treat arbitrary Buttons or same-ID replacement widgets
as the original anchor/programmatic object.

Otherwise, disarm and return without moving focus. Retain the existing stale
DescendantFocus rejection and user-key handling. This is not a global focus
override and must not change the direct entry algorithm, fallback order, row
selection, arm duration or retry interval.

Any non-null Media-return receipt stays under its current receipt eligibility,
live-focus and successful-settlement checks. Do not apply the new ordinary-entry
exception set to semantic returns, including revised/empty-result returns.
Gate the new check explicitly on `receipt is None`: non-candidate, non-null
receipts retain their current fallback path rather than acquiring this new
ordinary-entry check. Verify those existing paths with receipt regressions.

## Verification

First add a mounted deterministic regression using the actual retry timer
callback: capture that callback, use synchronous `Screen.set_focus` to give
the real attached Retry widget focus, assert it is the live focus owner, and
invoke the callback before yielding to DescendantFocus. `Widget.focus()` is
deferred and cannot establish that ordering. Assert the same Retry object
still owns focus and the old request is disarmed. Observe the intended focus
failure on unchanged runtime before implementing the guard.

Add controls for unchanged anchor, owned list row, absent/detached focus,
adaptive grip and current programmatic target; stale programmatic-target
identity, foreign non-row and foreign-list row must not retain authority.
Disarmed and superseded callbacks must neither move focus nor disarm a newer
request. Preserve both original stale-callback parameters and the adjacent
initial-error Retry recovery case. Exercise existing semantic Media-return
controls, including exact scroll/focus, revised origin and newer-user takeover.

Use real mounted widgets for focus/attachment claims; lightweight controls may
supplement, not replace, that regression. No longer waits, hidden timer
suppression, relaxed focus assertions or raised caps. Review actual changed
code independently, then freeze and run complete affected test files with the
unchanged native observer. Report behavior and final native resources separately.

## Alternatives and non-goals

Waiting for canvas readiness or holding timers only in the original test would
not fix the demonstrated live-owner race. Intercepting all set_focus calls or
adding a new intent ledger would broaden ownership unnecessarily. The existing
guarded continuation is the shared boundary for both scheduling paths.

No unrelated test cleanup, size paydown, CSS change, rebase resolution, automation
authorization or merge is included in this repair. Those remain separate
TASK-31932 gates. Preserve the unrelated untracked reader-paydown plan untouched.
