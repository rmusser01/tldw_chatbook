---
id: TASK-31822
title: Convert remaining super().on_mount() calls under Textual MRO dispatch
status: Done
assignee:
  - '@claude'
created_date: '2026-09-06 04:14'
updated_date: '2026-09-06 17:16'
labels:
  - textual
  - cleanup
  - reliability
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from close-out burndown Task 4 (31418) review. 31418 converted every on_unmount site to the no-super convention, but ~19 live super().on_mount() calls remain repo-wide -- each a latent double-fire of a separately-MRO-dispatched base on_mount (the same bug class, mount side). Notable: change_review_screen.py's ChangeGitCommitModal/ChangeGitPushModal call super().on_mount() with a docstring that misdescribes the mechanism as 'ordinary attribute lookup ... SHADOWS the mixin' (Textual walks the MRO and dispatches both, so SafeModalDismissMixin.on_mount double-fires); and console_session_switcher_modal.py:247 + console_workspace_files_modal.py:400. Harmless while the bases are idempotent, but a non-idempotent base on_mount teardown would double-fire everywhere at once. Convert to the no-super convention (or the BaseWizard _post_mount_hook plain-method pattern where an explicit call is genuinely needed), and correct the misleading change_review docstrings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No super().on_mount() call remains that reaches a separately-MRO-dispatched base on_mount handler (audited repo-wide, allowlist any genuine plain-method exceptions)
- [x] #2 change_review_screen.py's on_mount docstrings describe the MRO-walk mechanism correctly, not 'ordinary attribute lookup / shadowing'
- [x] #3 An AST guard (mirroring the on_unmount guard) fails if a super().on_mount() to a dispatched handler is re-introduced
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-grep all live super().on_mount() sites (AST-based, excluding docstring/comment mentions).
2. For each site, resolve the actual base class in the MRO and check whether its on_mount is defined in its own class __dict__ (dispatched) vs a plain/inherited method.
3. Remove redundant super() calls and add the standard no-super comment; check for the leading-super ordering hazard (base now fires after, not before, the rest of the override) against each site's own state reads.
4. Fix the two change_review_screen.py docstrings that misdescribe the mechanism as attribute-lookup shadowing.
5. Add Tests/UI/test_on_mount_mro_convention.py mirroring the on_unmount guard (runtime count + AST scan), revert-check both.
6. Run the touched widgets' existing test suites; attribute any new failures vs base via a throwaway detached worktree.
7. Close out: AC ticks, Implementation Notes with per-site table, lessons-textual.md update, single commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audited every live `super().on_mount()` call repo-wide via an AST scan (mirroring the
guard's own detector, not just grep) and found 19 sites — matching the ~19 estimate.
Classified each by resolving the actual MRO base and checking whether its `on_mount`
is defined in its own class `__dict__` (separately MRO-dispatched, so `super()` is
redundant) vs a plain/inherited method (would be load-bearing). All 19 resolved to a
dispatched handler (`SafeModalDismissMixin` x18, `LibraryAdaptiveReaderShell` x1) —
zero load-bearing sites, zero refactors to the `_post_mount_hook` pattern needed.

Ordering-dependency check (mount-specific risk beyond the on_unmount precedent):
Textual's `_get_dispatch_methods` walks the MRO most-derived-first, so removing a
*leading* `super().on_mount()` doesn't just drop a redundant call — it also reorders
the base's body to run strictly *after* the whole subclass method returns, not inline
before the subclass's later statements. Audited every site's remaining on_mount body
for reads of `SafeModalDismissMixin`'s internal state
(`_safe_cancel_pending`/`_safe_opener_focus_ref`/`_safe_opener_focus_id`/
`_safe_mount_generation`/`_safe_backdrop_event_in_attempt`) and of anything
`LibraryAdaptiveReaderShell.on_mount` sets (`sync_layout`, a deferred
`AdaptiveReaderShellResized` post_message) — none found, so no ordering fix was
needed anywhere; every site was a straight redundant-call removal.

Per-site classification:

| # | Site | Base resolved | Dispatched? | Action |
|---|------|----------------|-------------|--------|
| 1 | UI/Library_Modules/skill_import_choice_modal.py:83 | SafeModalDismissMixin | yes | removed, comment |
| 2 | UI/Screens/profile_interview_screen.py:198 (ProfileInterviewScreen) | SafeModalDismissMixin | yes | removed, comment |
| 3 | UI/Screens/change_review_screen.py:4474 (ChangeGitCommitModal) | SafeModalDismissMixin | yes | removed, comment + docstring fixed |
| 4 | UI/Screens/change_review_screen.py:4780 (ChangeGitPushModal) | SafeModalDismissMixin | yes | removed, comment + docstring fixed |
| 5 | UI/Screens/scheduling/forms/new_task_choice_modal.py:92 | SafeModalDismissMixin | yes | removed, comment |
| 6 | Widgets/Persona_Widgets/actor_pack_import_review.py:187 | SafeModalDismissMixin | yes | removed, comment |
| 7 | Widgets/Library/library_media_reader_shell.py:64 | LibraryAdaptiveReaderShell | yes | removed, comment |
| 8 | Widgets/Settings_Widgets/tool_pack_import_review.py:180 (ToolPackImportOptionsModal) | SafeModalDismissMixin | yes | removed, comment |
| 9 | Widgets/Settings_Widgets/tool_pack_import_review.py:405 (ToolPackImportReviewModal) | SafeModalDismissMixin | yes | removed, comment |
| 10 | Widgets/Settings_Widgets/tool_pack_import_review.py:583 (ToolPackExportReviewModal) | SafeModalDismissMixin | yes | removed, comment |
| 11 | Widgets/Settings_Widgets/tool_pack_import_review.py:779 (ToolProfileFirstBindReviewModal) | SafeModalDismissMixin | yes | removed, comment |
| 12 | Widgets/Settings_Widgets/personal_context_review_modal.py:551 (PersonalContextReviewModal) | SafeModalDismissMixin | yes | whole no-op override deleted (body was only `super().on_mount()`) |
| 13 | Widgets/Console/console_fork_chat_modal.py:325 | SafeModalDismissMixin | yes | removed, comment |
| 14 | Widgets/Console/console_session_switcher_modal.py:247 | SafeModalDismissMixin | yes | removed, comment |
| 15 | Widgets/Console/trace_export_dialog.py:188 | SafeModalDismissMixin | yes | removed, comment |
| 16 | Widgets/Console/console_prompt_comparison_modal.py:138 | SafeModalDismissMixin | yes | removed, comment |
| 17 | Widgets/Console/console_workspace_files_modal.py:400 | SafeModalDismissMixin | yes | removed, comment |
| 18 | Widgets/Console/console_library_access_modal.py:152 | SafeModalDismissMixin | yes | removed, comment + docstring reworded |
| 19 | Widgets/Console/console_project_instructions.py:530 (ProjectInstructionSetupModal) | SafeModalDismissMixin | yes | removed, comment |

Totals: 19 sites, 19 redundant removals, 0 kept-with-reason, 0 refactored to
`_post_mount_hook`. `BaseWizard._post_mount_hook` remains the repo's only genuine
plain-method case and was left untouched (it was never a `super().on_mount()` call).

Guard: added `Tests/UI/test_on_mount_mro_convention.py`, a sibling of
`test_on_unmount_mro_convention.py` (kept intact, not modified) — a runtime count
test pinning the base `on_mount` firing exactly once under the no-super convention,
plus an AST scan (with an empty, documented `_ALLOWED_OFFENDERS` escape hatch) that
fails on any re-introduced `super().on_mount()` to a dispatched handler. Both guards
were revert-checked live (re-added a `super().on_mount()` at one real site -> AST
scan failed with the offending line; reintroduced the bug in the runtime test's
`_Child` -> count assertion failed as 2), then reverted back to green.

Fixed the two `change_review_screen.py` docstrings (`ChangeGitCommitModal`,
`ChangeGitPushModal`) that claimed Textual resolves `on_mount` by "ordinary
attribute lookup" and that a subclass override "SHADOWS" the mixin's handler —
corrected to describe the actual MRO walk and why `super()` would double-fire it.
Also rewrote `console_library_access_modal.py`'s docstring, which implied the
override itself captured opener focus (that's the mixin's job, now clearly
separate).

lessons-textual.md's `on_unmount` entry updated to record the on_mount conversion,
the classification result (19/19 redundant), and the MRO-dispatch-order ordering
hazard that had to be checked (base now runs after, not before, a converted
override's later statements) since it didn't apply the same way on the unmount side.

Verification: `python3 -m py_compile` on all 15 touched files; new guard tests run
standalone and paired with the existing on_unmount guard (4 passed); both guards
revert-checked to confirm they actually catch a reintroduced `super()` call. Ran the
touched widgets' existing suites in foreground batches (~700 tests total): all green
except 3 pre-existing failures, each confirmed identical on an unmodified
`f811d6903` throwaway detached worktree (removed after use) and unrelated to any
touched file/mechanism: `test_schedules_new_button.py::test_new_button_row_flattens_to_one_line_in_compact_mode`
(CSS layout height, `schedules_workbench.py` untouched here), the 6 known
`test_console_modal_dismissal.py` reds (all `ConsoleModelPopover` constructor-arity
drift, unrelated file), and 2 `test_library_media_return_settlement.py` failures
(a scroll-offset assertion, `(0, 45) == (0, 42)` — the "Local prompt backend is
unavailable" ValueError seen alongside is unrelated setup noise; root cause
corrected per the task review, which reproduced both failures identically at
base `f811d6903` — pre-existing either way, not a mount-order issue).

Modified files: skill_import_choice_modal.py, profile_interview_screen.py,
change_review_screen.py, new_task_choice_modal.py, actor_pack_import_review.py,
library_media_reader_shell.py, tool_pack_import_review.py,
personal_context_review_modal.py, console_fork_chat_modal.py,
console_session_switcher_modal.py, trace_export_dialog.py,
console_prompt_comparison_modal.py, console_workspace_files_modal.py,
console_library_access_modal.py, console_project_instructions.py,
backlog/docs/lessons-textual.md. Added: Tests/UI/test_on_mount_mro_convention.py.
<!-- SECTION:NOTES:END -->
