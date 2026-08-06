---
id: TASK-2530
title: Library RAG run-gate lock asymmetry can enable Run above a stale empty disclosure row
status: To Do
assignee: []
created_date: '2026-08-06 12:00'
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

- [ ] A refresh that loses the race against a landing snapshot leaves the Run button **disabled** (fail-safe), never enabled above a stale or empty disclosure row.
- [ ] The invariant `Run enabled in RAG Answer mode ⇒ the quiet row renders the paid-mode notice naming the provider` holds under the interleaving described above.
- [ ] A regression test drives the interleaving (refresh yields mid-removal, snapshot sync runs, refresh resumes) and would fail against today's ordering.
- [ ] RAG-27's no-yield constraint on `_sync_library_rag_scope_toggle_and_run_gate_widgets` is preserved (no `await`, no recompose, no mount/remove added there).
