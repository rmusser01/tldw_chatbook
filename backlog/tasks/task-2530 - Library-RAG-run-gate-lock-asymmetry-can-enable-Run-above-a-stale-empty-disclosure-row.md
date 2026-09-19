---
id: TASK-2530
title: >-
  Library RAG run-gate lock asymmetry can enable Run above a stale empty
  disclosure row
status: Done
assignee:
  - '@codex'
created_date: '2026-08-06 12:00'
updated_date: '2026-09-17 03:37'
labels:
  - library
  - rag
  - bug
  - paid-moments
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the whole-branch re-review of PR-T2 (`feat/rag-truth-paid-moments`, at `5f6be61b3`) while verifying the F1 fix. This is the same defect class as F1 — Run enabled while the paid-mode disclosure row is empty — reached through a narrower race rather than the snapshot path F1 closed.

`_sync_library_rag_scope_toggle_and_run_gate_widgets` does not take `_library_rag_panel_refresh_lock`, so it can run inside the locked full refresh's own yield window. `_refresh_library_rag_query_status_widgets` writes `run_button.disabled` BEFORE it `await`s the quiet line's removal:

1. A refresh captures stale state `S0` (no scope) → Run disabled.
2. It `await`s the quiet-line removal → yields.
3. The snapshot worker runs the sync → Run **enabled**; the sync's quiet-line `query_one` raises `NoMatches` (the row is mid-removal) → guarded `pass`.
4. The refresh resumes and mounts the quiet line from the stale `S0` → **empty row**.

Net: Run enabled above an empty disclosure row, persisting until the next refresh event. Reachable when a keystroke-driven refresh coincides with the first real snapshot after a zero-count compose.

This is a pre-existing lock asymmetry, not introduced by PR-T2 — but PR-T2's F1 fix closed the common path, making this the remaining one. It was parked deliberately rather than fixed, because the safe remedy lives in the *other* method and the branch was at its final gate.

**Remedy (one line):** move the `run_button.disabled` write in `_refresh_library_rag_query_status_widgets` to AFTER its remove/mount loop, so the losing side of the race fails safe (stale-**disabled** button) instead of stale-enabled. Consider also whether the sync should take the panel refresh lock, but note RAG-27's no-yield constraint on that method.

Related, parked no-action by the same review (R2): the inverse ordering — a loud blocker at compose, lifted to ready by a snapshot — can leave a stale `Blocked | Select a provider/model…` callout beside an enabled Run. It cannot spend undisclosed money (the notice IS updated) and requires provider config to change without a navigation-triggered recompose, which the current flow does not permit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A refresh that loses the race against a landing snapshot preserves the latest coherent Run/disclosure pair, never enabling Run above a stale or empty disclosure row; if the disclosure widget is unavailable, Run remains disabled.
- [x] #2 The invariant `Run enabled in RAG Answer mode ⇒ the quiet row renders the paid-mode notice naming the provider` holds under the interleaving described above.
- [x] #3 A regression test drives the interleaving (refresh yields mid-removal, snapshot sync runs, refresh resumes) and would fail against today's ordering.
- [x] #4 RAG-27's no-yield constraint on `_sync_library_rag_scope_toggle_and_run_gate_widgets` is preserved (no `await`, no recompose, no mount/remove added there).
- [x] #5 At 80×24 and wide sizes in dark/light themes, the ready disclosure visibly names the recipient and outgoing question/evidence without shifting Run when the quiet gate changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR.
ADR path: backlog/decisions/003-settings-library-rag-defaults.md; backlog/decisions/150-design-token-system-and-design-language.md.
Reason: repair synchronization of the existing Library query controls and provider disclosure without changing provider authority, billing, or the search/answer boundary.
1. Reproduce the source-snapshot/query-refresh interleaving with mounted production CSS and assert the Run/disclosure invariant at the yield boundary and after settlement.
2. Preserve synchronous snapshot updates and make the always-present disclosure and Run gate update together; retain existing conditional recovery behavior.
   Native inspection also exposed compact clipping of the provider at the end of the existing sentence. Keep the reserved one-line geometry and shorten the ready copy to `To {provider}: question + evidence`; the mode-toggle tooltip retains the Search-stays-local explanation. Add a production-CSS painted-text regression before changing the copy.
3. Verify targeted query/status, keyboard, gate, and token regressions; inspect a private native wide/compact and dark/light journey without invoking a model.
4. Obtain independent review, record evidence and limitations, update the continuation ledger, and make a local commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retained the reserved quiet line and updated it with Run through a shared synchronous helper. An older full refresh now rebuilds only conditional recovery widgets, preserving the newest snapshot pair; missing disclosure disables Run. The original proposed final disabled write was replaced because it did not protect the intermediate yield window.
Native 80-column review exposed a second issue: the old sentence clipped before the provider. Recipient-first `To {provider}: question + evidence` copy now fits the unchanged one-line geometry; Search-local explanation remains on the mode toggle. The task AC/plan was expanded before this copy repair.
Tests cover deterministic arrival/loss/provider races, missing disclosure, full compositor text, both modes/themes/sizes, long provider names, and stable Run position. The stale unmounted inline-height assertion is replaced by mounted production-CSS checks. Final targeted run: 275 passed, 2 unrelated known failures deselected (TASK-15390 heading and TASK-4111 result opening); Open also fails on unchanged HEAD. No full suite. No added lint diagnostics; independent review found no actionable issues.
Four final native cells passed with explicit synthetic display-state injection, real TTY/LinuxDriver, and no Run/Send activation. Default profile hashes unchanged, ten private databases healthy, normal exit and PID absence verified. Evidence and reproduction: Docs/superpowers/qa/2026-09-16-rag-query-gate/README.md. Controller/model, related tests, continuation ledger, and native verification lesson updated.
ADR required: no new ADR; implements backlog/decisions/003-settings-library-rag-defaults.md and backlog/decisions/150-design-token-system-and-design-language.md. Provider authority, billing, CSS tokens, and retrieval boundaries are unchanged.
<!-- SECTION:NOTES:END -->
