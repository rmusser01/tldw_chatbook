---
id: TASK-31822
title: Convert remaining super().on_mount() calls under Textual MRO dispatch
status: To Do
assignee: []
created_date: '2026-09-06 04:14'
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
- [ ] #1 No super().on_mount() call remains that reaches a separately-MRO-dispatched base on_mount handler (audited repo-wide, allowlist any genuine plain-method exceptions)
- [ ] #2 change_review_screen.py's on_mount docstrings describe the MRO-walk mechanism correctly, not 'ordinary attribute lookup / shadowing'
- [ ] #3 An AST guard (mirroring the on_unmount guard) fails if a super().on_mount() to a dispatched handler is re-introduced
<!-- AC:END -->

## Renumbering provenance

Renumbered from **TASK-31742 → TASK-31822** on 2026-09-06 per the TASK-19601 owner rule (older arrival keeps the id). `Address-Qodo-review-findings-for-Canvas-V1` (created 2026-09-05 21:37) is the older TASK-31742 and keeps that id on dev; this task (created 2026-09-06 04:14) is the younger and renumbers. Surfaced when dev was merged into the schedules close-out burndown branch (PR #2454). No dependency/reference updates were needed — the burndown tasks reference this follow-up by description, not by number.
