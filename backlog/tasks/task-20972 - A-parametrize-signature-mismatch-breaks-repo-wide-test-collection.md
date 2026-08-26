---
id: TASK-20972
title: A parametrize-signature mismatch breaks repo-wide test collection
status: Done
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-23 23:17'
labels:
  - bug
  - testing
  - test-integrity
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: found while baselining **TASK-19561** against a clean `origin/dev`
worktree. Re-verified at `684c6aba4`.

`Tests/UI/test_library_file_notes_workspace.py` errors during **collection**:

> `In test_wide_files_task_return_restores_database_browse_receipt: function
> uses no argument 'push_phase'`

`test_wide_files_task_return_restores_database_browse_receipt`
(`Tests/UI/test_library_file_notes_workspace.py:1263-1272`) carries two
`@pytest.mark.parametrize` decorators — one on `("save_state", "save_copy")`
and one on `("push_phase", "push_copy", "git_count")` — while its signature
takes no arguments at all. pytest reports the second one and stops.

The cost is out of proportion to the mistake. A collection error aborts the run
for the whole invocation, so `pytest --collect-only` over the repository exits
non-zero and cannot be used as a clean gate. That sweep is the cheapest check
this programme has: it is how a rebase is confirmed not to have broken an
import, and it is run before nearly every merge. While it is red by default,
every user of it has to hold a list of failures that are "expected", which is
the state in which a genuinely new breakage goes unnoticed.

Two further facts about the test itself should be settled rather than assumed.
Neither parametrize set is consumed, so the six cases the decorators describe
have never run in any form; and because the arguments are unused, whatever
`push_phase` / `save_state` were meant to vary is not being varied. The
question is whether the intended coverage was lost or was never written.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `pytest --collect-only` over the repository completes without a
      collection error from this module
- [x] #2 `test_wide_files_task_return_restores_database_browse_receipt` either
      consumes the parameters it is parametrized on, or is no longer
      parametrized on parameters it does not use
- [x] #3 It is established and recorded whether the parametrized cases represent
      coverage that was intended and lost, or decorators that were never wired
      up — and if coverage was intended, it exists and passes
- [x] #4 The test passes in every configuration it declares
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on `dev` by **`1796ff16a`** ("test(library): restore path transition
parametrization", 2026-08-22), not by this branch. Closing the task to match the code.

**AC#3 — intended-and-lost, or never wired up? Intended, and displaced.** The two decorator
blocks had slid up by exactly one test. They belong to
`test_path_transition_authority_names_file_operation_and_settles`, whose signature takes
`save_state`, `save_copy`, `push_phase`, `push_copy` and `git_count` by name. `1796ff16a`
moves them down onto it, restoring four parametrized cases.

Recording the wrong answer as well, because the two are hard to tell apart. This branch
first *deleted* the decorators, reasoning that
`test_wide_files_task_return_restores_database_browse_receipt` was introduced without them
(`1bda754fa`) and references none of the five names in its body. Both facts hold; the
inference did not. Deleting would have left collection green, the module passing and four
real parametrized cases silently gone. **A displaced decorator block and an abandoned one
present identically from the signature mismatch alone — check the function below before
concluding a decorator is debris.** The lesson is recorded in
`backlog/docs/test-health-baseline-2026-08-23.md` §4.

Verification at `73a43c71f`: `pytest Tests --collect-only` exits 0 with **58,066 tests
collected** and no collection errors, so AC#1 and AC#4 hold on current dev. The four
`Tests/UI/test_settings_*` collection errors this task's Notes predicted as a second
blocker do not reproduce either.
<!-- SECTION:NOTES:END -->

## Notes

Filed medium rather than low because of what it costs the gate, not because of
what it costs this one module. The same run also surfaced four further
collection errors in `Tests/UI/test_settings_*`; those have a different root
cause and are TASK-20970. Both must be fixed before a repo-wide
`--collect-only` is green.
