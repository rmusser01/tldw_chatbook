---
id: TASK-15101
title: Prompt and recipe delete leaves an Undo receipt per ADR-055
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 06:20'
labels:
  - library
  - ux
  - adr-055
dependencies: []
priority: medium
---

## Description

ADR-055 (task-14901) sets one reversibility rule across Library destructive
actions: soft-deleting persisted user data leaves a receipt at the point of
action offering Undo, with the durable recovery story named. Prompt/Recipe
deletion is a soft delete (`local_prompt_service.delete_prompt` →
`Prompts_DB.soft_delete_prompt`, versioned, with retained version history per
ADR-049), so the deleted row is recoverable at the store level — but the UI
today goes modal-confirm-then-silence: after `_delete_library_prompt`
succeeds, the editor resets to the list with no receipt and no way back. The
modal copy now states permanence honestly ("This cannot be undone from
Library.", task-14901), which ADR-055 accepts only as the interim state; the
owed pattern is the media one (task-4022): a "✓ deleted" receipt with
Undo/Dismiss in the prompts list, restoring through a service-level un-delete
seam (to be added — `Prompts_DB` has no undelete today; the version-history
machinery of ADR-049 is the natural substrate). Covers both the single-item
modal delete and any selection-based bulk delete the modal already fronts.
Update the modal copy to promise the Undo once it exists. The PR-1473 one-flag
lesson applies to every mutator of the shared prompts list/count/receipt
state.

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A successful prompt/recipe deletion leaves an in-place receipt in the prompts list naming what was deleted, with Undo and Dismiss, matching the media receipt grammar
- [x] #2 Undo restores the artifact through a service seam (no raw SQL) and the list and rail Prompts count update in place
- [x] #3 Undo and any concurrent prompt delete cannot race on shared list/count/receipt state (one shared in-flight interlock, PR-1473 pattern)
- [x] #4 The delete modal copy promises exactly what exists (Undo), replacing "This cannot be undone from Library."
<!-- AC:END -->

## Implementation Plan

ADR required: no
ADR path: `backlog/decisions/055-library-destructive-action-reversibility-rule.md` and `backlog/decisions/049-local-prompt-retained-version-history.md`
Reason: This task implements the already-approved Library recovery and Prompt history boundaries without changing storage ownership, sync policy, or service architecture.

1. Characterize the existing single-item and selection-based Prompt/Recipe delete paths and their list/count state ownership.
2. Add a version-checked Prompt undelete operation in `PromptsDatabase`, expose it through the local and scope service seams, and cover conflict and fidelity behavior.
3. Add one shared Prompt mutation interlock plus an in-place delete receipt with Undo/Dismiss, list/count refresh, and stable focus recovery.
4. Update confirmation copy so it promises the available Undo behavior and add mounted UI regression coverage for the rendered receipt and action flow.
5. Run focused database, service, state, modal, and mounted Library tests plus static checks; document results and complete the task record.

## Implementation Notes

- Added a version-checked `restore_deleted_prompt` store operation and routed it through the local and scope service seams. Delete tombstones now retain canonical keyword membership so Undo restores the exact Prompt/Recipe, keywords, FTS row, version lineage, and sync events atomically; retained-history restore remains a separate ADR-049 operation.
- Made Library Prompt deletion itself conditional on the editor's captured version, then added a named in-list receipt with Undo/Dismiss. Delete and Undo use one Prompt mutation admission flag and one worker group, while the exact browse page and rail count refresh through their existing owners.
- Updated single, dirty, and reusable bulk confirmation copy to promise the available list Undo. The receipt uses literal text, bounded names, familiar toolbar styling, and stable focus recovery rather than a transient toast.
- Qodo follow-up: completed Google-style `Args`, `Returns`, and `Raises` documentation across every modified database, interop, local/scope service, and Undo/Dismiss handler callable; no acceptance-criteria or architecture scope changed.
- Verification: 313 focused database/service/state/modal tests passed; 9 mounted Prompt-delete/receipt tests passed; 2 independently rerun adjacent retained-history tests passed; `ruff check --ignore E721` passed for every changed Python file (E721 is pre-existing repository style in these modules); `git diff --check` passed. A complete mounted Prompt-canvas sweep exceeded its 10-minute bound after progressing through the file; its reported order-dependent history failures passed when rerun as exact nodes. No new generalized lesson or ADR was required.
