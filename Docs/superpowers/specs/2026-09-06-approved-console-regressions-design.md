# Approved Console regression repairs

The user approved these five bounded production fixes on 2026-09-06 after the
root-cause investigation recorded in `backlog/docs/dev-test-review-checkpoint-2026-09-05.md`.
This document records that approval; it does not expand the scope.

1. TASK-31817: retain failed Parakeet audio only while the exact retry dialog
   owned by that dictation controller covers Console. Snapshot that exception
   for the suspend transition; ordinary navigation, other overlays and unmount
   still abandon. Confirmation retries the retained audio once, rejection or
   cancellation clears it, and the current mounted mic reflects canonical state.
2. TASK-31928: include visible Redirect in the existing action-row width accounting,
   retaining idle space. Reflow after run-state geometry changes and bound the
   existing optional attachment label to the remaining narrow-row budget.
   Preserve control semantics, labels and ordering. Verify at 160x48 with the
   production CSS, ordinary composer focus before synthetic Send, and a real
   visible Stop click. No forced scrolling, Stop pre-focus or widened deadline.
3. TASK-31929: add the existing CHAT claimant to the tracked ordinary-resume
   timers. Preserve first-mount and ordered saved-chat startup; suspend cancels
   outstanding timers, and acknowledgement/release retains exact-slot ownership.
4. TASK-31930: reject late ContentsRebuilt events when the screen stack is empty
   before dereferencing the current screen. Matching active-screen events retain
   normal reconciliation, and stale-screen events stay ignored.
5. TASK-31931: classify unsaved synthesized rows in the contiguous leading system
   slice as RENDERED_SYSTEM provenance. Saved revision descriptors win; ordinary
   unsaved active rows remain ACTIVE_REQUEST. Keep capture enabled and fail-closed
   provenance validation intact. Do not reclassify nonleading system rows.

Alternatives rejected: weakening tests, forced layout scrolling, disabling trace
capture, suppressing all suspend cleanup, or making all modal lifetimes preserve
audio. No new service, schema, provider boundary, or storage policy is introduced.

ADR required: no.
ADR paths: `backlog/decisions/033-application-session-state-ownership.md` and
`backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md`.
Reason: routine regression fixes restore existing lifetime, handoff and provenance
contracts (with cached-screen behavior already established by TASK-31520).

Verification requires RED/GREEN regression evidence, complete affected-file runs,
scoped lint/format checks and independent review. Existing resource cleanup stays
enabled. Screen-size ceilings remain unchanged; no full-repository sweep, merge,
new UX redesign or unrelated architectural paydown is authorized by this approval.
