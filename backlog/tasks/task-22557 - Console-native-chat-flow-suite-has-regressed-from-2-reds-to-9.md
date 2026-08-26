---
id: TASK-22557
title: >-
  Console native chat flow suite has regressed from 2 reds to 9
status: To Do
assignee: []
created_date: '2026-08-26'
labels:
  - console
  - tests
  - regression
priority: high
dependencies: []
---

## Description

`Tests/UI/test_console_native_chat_flow.py` is the behavioural suite for the Console send
surface — dispatch, streaming render, model resolution, workspace conversation resume, and
collapsed-layout state. TASK-21590 repaired this file's harness and left it at **2 known
reds**. It is now at **9**, so the surface it covers has regressed further since that repair
and nothing caught it: no CI job has produced a verdict for the UI shards, and the reds do
not announce themselves in any gate a PR author sees.

This is filed as a diagnosis-first task rather than a fix: nine failures across five distinct
behaviours are unlikely to share one cause, and the first job is to partition them. Found
incidentally while triaging Qodo review comments on merged PRs (chore/qodo-console); the
triage itself changed nothing in this file.

## Evidence

Measured with `.venv/bin/python -m pytest ... -p no:randomly -q`, cwd = a worktree at
dev `732105c2d`, `tldw_chatbook.__file__` asserted to resolve inside that worktree.

Whole-file run (with `Tests/UI/test_console_setup_backdrop_repaint_cost.py`):
**302 collected, 293 passed, 9 failed, 0 errors** in 314 s.

The nine failing node ids:

```
Tests/UI/test_console_native_chat_flow.py::test_improvement_disclosure_is_pinned_and_drift_before_click_is_blocked[selection]
Tests/UI/test_console_native_chat_flow.py::test_improvement_disclosure_is_pinned_and_drift_before_click_is_blocked[config_endpoint]
Tests/UI/test_console_native_chat_flow.py::test_console_native_generic_provider_send_renders_completed_message
Tests/UI/test_console_native_chat_flow.py::test_console_native_send_button_click_dispatches_message
Tests/UI/test_console_native_chat_flow.py::test_console_successful_send_does_not_leave_empty_send_tooltip
Tests/UI/test_console_native_chat_flow.py::test_console_configured_model_reaches_gateway_when_ui_model_is_unset
Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_row_resumes_persisted_conversation
Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_resume_uses_real_local_services
Tests/UI/test_console_native_chat_flow.py::test_console_collapsed_layout_follows_cross_workspace_tab_state
```

**A/B control.** The nine were re-run after swapping `tldw_chatbook/Chat/console_chat_store.py`
to its `732105c2d` blob (identical command, identical selection), to rule out the branch's own
edits: **the same 9 node ids failed, none more, none fewer** — `9 failed in 34.06 s`. The
working tree was then restored and diff-verified byte-identical. So these are dev reds, not
artefacts of the branch that observed them.

Per `lessons-testing-evidence.md` ("Compare failure *sets* from identical commands, never
counts"), the comparison above is of sets, not counts.

## Acceptance Criteria

- [ ] Each of the nine node ids is partitioned into: production regression, stale test double,
      or harness/environment artefact — with the evidence for the call, not an assertion
- [ ] For every one classified as a production regression, the user-visible defect is stated
      in product terms (what a user does, what they see instead)
- [ ] Any test found to be pinning a defect is corrected rather than deleted or relaxed, and
      the production behaviour it was pinning is filed or fixed
- [ ] The file runs green, or every remaining red has a filed task id recorded in this task
- [ ] The delta from TASK-21590's 2 known reds to 9 is explained: which merges introduced
      which failures (`git log` bisection over the suspect commits is acceptable evidence)
- [ ] No `:memory:`-to-file-backed flips in `Tests/UI/app_factory.py`'s `attach_chachanotes_db`
      (TASK-21590: `:memory:` is load-bearing — a file-backed DB moves these tests onto the
      agent loop and changes their subject)
- [ ] Teardown errors are checked explicitly, not inferred from a green summary line
      (TASK-21590: 16 tests reached tiktoken's BPE download while *passing*; the current
      302-item run reports 0 errors, so that is the baseline to hold)
