---
id: TASK-36932
title: Keep Personas controls responsive during Lore settings saves
status: To Do
created_date: 2026-10-25 23:38
labels:
- textual-book
- personas
- lore
- ui
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-25-textual-book-after-pr3248.md
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
- [ ] #1 During a pending Lore settings save, unrelated enabled Personas rail controls and screen callbacks complete without waiting for the storage response.
- [ ] #2 An admitted save retains the submitted Lore identity, version and payload; repeated activation does not create duplicate writes or silently retarget it.
- [ ] #3 Save completion reports the actual result for the captured Lore book and cannot retarget or overwrite a replacement selection/panel; existing dirty-state and transition ownership remains intact.
- [ ] #4 Mounted controlled slow-save and ordinary success/failure tests verify responsive admission and correct real SQLite results, without claiming natural latency from a synthetic hold.
<!-- AC:END -->
