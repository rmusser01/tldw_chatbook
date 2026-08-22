---
id: TASK-20972
title: >-
  A parametrize-signature mismatch breaks repo-wide test collection
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - bug
  - testing
  - test-integrity
priority: medium
dependencies: []
---

## Description

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

## Acceptance Criteria

- [ ] `pytest --collect-only` over the repository completes without a
      collection error from this module
- [ ] `test_wide_files_task_return_restores_database_browse_receipt` either
      consumes the parameters it is parametrized on, or is no longer
      parametrized on parameters it does not use
- [ ] It is established and recorded whether the parametrized cases represent
      coverage that was intended and lost, or decorators that were never wired
      up — and if coverage was intended, it exists and passes
- [ ] The test passes in every configuration it declares

## Notes

Filed medium rather than low because of what it costs the gate, not because of
what it costs this one module. The same run also surfaced four further
collection errors in `Tests/UI/test_settings_*`; those have a different root
cause and are TASK-20970. Both must be fixed before a repo-wide
`--collect-only` is green.
