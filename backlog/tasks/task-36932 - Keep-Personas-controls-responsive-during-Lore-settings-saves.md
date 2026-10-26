---
id: TASK-36932
title: Keep Personas controls responsive during Lore settings saves
status: Done
created_date: 2026-10-25 23:38
labels:
- textual-book
- personas
- lore
- ui
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-25-textual-book-after-pr3248.md
assignee:
- '@codex'
updated_date: 2026-10-26 03:39
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let unrelated enabled Personas controls respond while a Lore settings save waits for its existing threaded storage operation.

### Verified scope

Verified on merged dev `7a5e2513c5d7e00ef57f3d19cd8e81a01e006d05` after PR #3248. During a controlled hold at the existing unconditional to_thread service boundary, Save settings retains the Personas screen message pump: the independent App callback runs but a screen callback and enabled rail-collapse action wait about 0.6 seconds until release. The actual manager then saves the typed name to real SQLite. The matched ordinary case processes both callbacks and collapse. This simulates storage latency; the separate real-writer-contention hypothesis was refuted by a prompt SQLite snapshot-upgrade failure. No whole-app freeze or data-loss claim.

Affected source: tldw_chatbook/UI/Screens/personas_screen.py:6957-7008; tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py; tldw_chatbook/Character_Chat/world_book_manager.py.

Book: Chapter 7 PDF145-146/158 (printed136-137/149), awaiting threaded work inside the screen handler still retains that screen message queue.

### Existing ownership and decisions

TASK-15469 moved backend work off the loop; TASK-2834 checks the thread boundary. TASK-33910.3 owns list windows and TASK-33910.8/.14 guard/draft/transition behavior. None requires the Lore Settings Save handler to release the screen message pump while its service operation remains pending. Existing other-form pending-save tasks remain separate.

Related tasks: [TASK-15469](./task-15469%20-%20Personas-dictionaries---indexed-attachment-lookup-and-a-threaded-backend.md), [TASK-2834](./task-2834%20-%20Guard-against-blocking-IO-reachable-from-a-Textual-message-handler.md), [TASK-33910.3](./task-33910.3%20-%20Roleplay-frame-B2a-line-based-items-list-marks-bar-and-one-delete-modal.md), [TASK-33910.8](./task-33910.8%20-%20Roleplay-frame-B5b-1-one-guard-dialog-per-domain-executors-and-the-load-fence.md), [TASK-33910.14](./task-33910.14%20-%20Roleplay-frame-B9a-entry-and-Options-draft-domains-no-layout-change.md), [TASK-36455](./task-36455%20-%20Keep-newer-Dictionary-Settings-drafts-when-an-earlier-Save-completes.md), [TASK-36097](./task-36097%20-%20Keep-newer-Character-and-Persona-edits-dirty-when-earlier-saves-finish.md).

ADR required: no.
ADR paths: [ADR-007](../decisions/007-personas-workbench-route-consolidation.md), [ADR-011](../decisions/011-chatbook-workbench-ui-system.md), [ADR-033](../decisions/033-application-session-state-ownership.md).
Reason: A screen-owned operation using the current Lore manager and draft contracts; no new service/storage boundary or long-lived UX structure is proposed.

[Post-merge audit and reproducible evidence](../../Docs/superpowers/qa/2026-10-25-textual-book-after-pr3248.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 During a pending Lore settings save, unrelated enabled Personas rail controls and screen callbacks complete without waiting for the storage response.
- [x] #2 An admitted save retains the submitted Lore identity, version and payload; repeated activation does not create duplicate writes or silently retarget it.
- [x] #3 Save completion reports the actual result for the captured Lore book and cannot retarget or overwrite a replacement selection/panel; existing dirty-state and transition ownership remains intact.
- [x] #4 Mounted controlled slow-save and ordinary success/failure tests verify responsive admission and correct real SQLite results, without claiming natural latency from a synthetic hold.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add controlled regression tests and observe the reported defect before production edits.
2. Release the screen handler promptly by admitting one screen-owned, noncancelling Lore settings save worker. Capture manager, book ID, expected version, payload and original panel before dispatch. Guard publication against retired selection/panel, preserve newer edits, and report captured-book success/failure truthfully. Test mounted held and ordinary saves using actual manager/private SQLite, repeated activation, conflicts, replacement and stale failures. Preserve existing dirty/transition owners.
3. Run focused regression and adjacent owner tests, lint/format checks, update Docs/Architecture/character-chat.md, and self-review.
4. Complete Superpowers review, register new UI coverage in the existing gate and verify CI before closure.

ADR required: no
ADR path: existing canonical ADRs 007, 011, 033 (linked in Description).
Reason: bounded implementation of existing UI/session/store ownership; no new contract or architecture.

Implementation plan: Docs/superpowers/plans/2026-10-25-textual-followup-fixes.md (Task 3).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Lore Settings Save synchronously admits one screen-owned worker with the captured manager, book identity/version, payload, original record/panel and raw submitted controls. The handler releases the screen queue; completion preserves newer raw edits and rejects retired selections/panels, including after list refresh. Explicit pre-entry cancellation permits later retry, while running native persistence retains admission through physical settlement. Fixed worker name/description keep captured Lore text out of diagnostic metadata; a safe optional post-await library-pane query preserves completion safety without growing the worker census.

Changed UI/Screens/personas_screen.py, UI/Persona_Modules/persona_save_completion.py, Widgets/Persona_Widgets/personas_lore_detail.py and Docs/Architecture/character-chat.md; added Tests/UI/test_personas_lore_save_responsiveness.py. Final safe-query source passes all 9 mounted cases and 9 actual private child reports; the existing optimistic-version regression also passed. Syntax and changed-block formatting pass with unchanged baseline E721 diagnostics. The Personas module ceiling was lowered to16818. Existing dirty/transition ownership is preserved.

Reviewed all changed diagnostic statements: two moved save/reload warnings and one new ID-only presentation warning, without new name/body/secret/path/URL interpolation or sink changes. Regenerated Docs/security/production-diagnostic-inventory.json; exact diagnostic and Textual worker guards pass on the combined dev tree. Existing sanitization and diagnose=False remain intact.

ADR required: no. Existing [ADR-007](../decisions/007-personas-workbench-route-consolidation.md), [ADR-011](../decisions/011-chatbook-workbench-ui-system.md) and [ADR-033](../decisions/033-application-session-state-ownership.md) govern this existing-boundary correction.

Superpowers task, whole-branch, fix-wave and CI-delta reviews are complete with no open findings; Qodo was replaced as requested. Current implementation-head hosted CI is green: [40261190139](https://github.com/rmusser01/tldw_chatbook/actions/runs/40261190139); [40261190218](https://github.com/rmusser01/tldw_chatbook/actions/runs/40261190218). Its 17 UI shards executed all 36 new exact cases once; maximum wall time 1094s under the unchanged 1200s cap. The gate preserves all 899 live-dev selectors plus 36 new cases (floor935). Rebase onto dev e554f25812 preserved all nine reviewed source/test blobs.

[Verification and explicit scope decisions](../../Docs/superpowers/qa/2026-10-25-textual-followup-fixes.md) records exact targeted evidence and the retained baseline profile/bootstrap, Chat-size and initial Chunking timing qualifications. No full local suite or native/live-server qualification is claimed.

[PR #3251](https://github.com/rmusser01/tldw_chatbook/pull/3251) targets dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
