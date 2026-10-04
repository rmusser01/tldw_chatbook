---
id: TASK-34352
title: >-
  Console controller finds its staged-evidence hooks by attribute name, and two of the three are missing in production
status: To Do
assignee: []
created_date: '2026-10-03 18:37'
labels:
  - console
  - rag
  - follow-up-33940
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33940.4 found that ConsoleRuntime wired a three-argument capture into a two-argument seam, so every send raised a `TypeError` that a broad `except` hid. The same coupling is still in place elsewhere. `Chat/console_chat_controller.py` reaches its staged-evidence collaborators by taking `self._rag_capture_provider.__self__` and calling `getattr` on it for private method names, in `_has_explicit_staged_evidence`, `_snapshot_staged_evidence` and `_capture_frozen_rag_context`. A missing name returns "unsupported" silently.

On dev `01a2020981` the only controller the app builds is ConsoleRuntime's, and its capture provider is bound to ConsoleRuntime. ConsoleRuntime defines `_has_staged_evidence` and `_capture_frozen_console_staged_rag`, but not `_snapshot_console_staged_evidence` or `_release_frozen_console_staged_rag`. Those two are defined only on `ConsoleRetrievalController` (`UI/Console_Modules/retrieval.py`). So for an ordinary composer send, `_snapshot_staged_evidence()` reports nothing to freeze, and the turn's evidence is taken later through the live capture seam instead of being pinned at dispatch.

It is not yet established whether that loses a guarantee. For example, evidence staged after dispatch but before capture could be consumed by the earlier turn. It is the same silent-mismatch failure that hid the TASK-33940.4 `TypeError` for weeks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test against the runtime-owned controller establishes whether an ordinary composer send pins its staged evidence at dispatch, and the finding is recorded in the task notes
- [ ] #2 If evidence is not pinned at dispatch, it is pinned there, or the notes record why live capture gives the same guarantee
- [ ] #3 The controller receives its staged-evidence collaborators as explicit constructor dependencies; no `__self__` or private-name `getattr` lookup remains for them
- [ ] #4 A controller built without a required collaborator fails loudly at construction or first use, rather than silently treating the feature as unsupported
- [ ] #5 Existing staged-evidence, queue-custody and dispatch-recovery tests pass, compared against dev `01a2020981` for any suite that already fails locally
<!-- AC:END -->
