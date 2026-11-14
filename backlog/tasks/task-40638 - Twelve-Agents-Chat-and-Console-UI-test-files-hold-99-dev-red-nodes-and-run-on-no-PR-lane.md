---
id: TASK-40638
title: "Twelve Agents, Chat and Console UI test files hold 99 dev-red nodes and run on no PR lane"
status: To Do
assignee: []
created_date: '2026-11-12 15:21'
labels:
  - follow-up
  - tests
  - console
  - dev-red
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviving PR #3446 (TASK-33620.15.53) on 2026-11-12. Its blast-radius run (every test file that names a module the PR touches) found 104 red nodes. 99 of them are red on origin/dev 4b90eb860b too. Each file's red nodes were rerun in one fresh process per file, twice on a detached dev worktree and twice on the rebased branch, and failed all four times (for `test_console_turn_library_authority.py`, a sample of five; see below). The other five flip between pass and fail on both sides. TASK-40637 tracks them, with three more flaky nodes from the same file. None of these twelve files is listed in `.github/workflows/*.yml` or `scripts/ui_pr_gate_census.txt`, so no PR check has seen them fail.

Failure signatures, per file (dev and branch identical):

- `Tests/Chat/test_console_turn_library_authority.py` (27 of 32): sends refused with "Hooks unavailable; review or disable hooks" (15), `RecoveryRequired: raw_source_selection_changed` (5), pytest-timeout at 300 s (6; the file takes 30 min). On dev, five sampled nodes failed 4/4. The other 22 were not rerun there because of the timeouts.
- `Tests/UI/test_console_hands_free_wiring.py` (41): `save_setting_to_cli_config('console', 'agent_runtime', False)` returns False (20), `RecoveryRequired: raw_source_selection_changed` (20), and one call-shape drift (`('dictation.acoustic_barge_in', None, False)` against the pinned `('dictation.acoustic_barge_in', False)`).
- `Tests/UI/test_console_headless_approval.py` (8): rounds or cards never arm or mount, and `RecoveryRequired: raw_source_selection_changed`.
- `Tests/Performance/test_first_send_warmup_gc_overlap.py` (8): test drift. Its script starts `preimport_first_send_modules(app)` on a thread, but the function now also requires `on_complete`, so the child dies with a `TypeError` and the test reports "actual import not entered".
- `Tests/integration/test_console_agent_marker_anchoring_e2e.py` (3): the submit is refused with "Hooks unavailable".
- `Tests/Chat/test_console_agent_swap.py` (3): a Stop persists the run as `error`, not `cancelled` (2), and `RecoveryRequired: console_capture_source_not_ready` (1).
- `Tests/UI/test_console_code_review_reports.py` (2): the `U review reports` footer key is missing, and a `SimpleNamespace` host lacks `_start_console_attach_reconciliation`.
- `Tests/UI/test_console_dictionary_send_integration.py` (2): `RuntimeError: Original MCP capture source changed.`
- `Tests/UI/test_console_poll_reconciliation.py` (2): "Pending demand reached neither original full-replay owner".
- `Tests/Tools/test_admitted_root_consumer_census.py` (1): new unpinned `.root` reads in `Agents/local_tool_dispatch.py`, `Agents/ordinary_tree.py`, `Agents/workflow_task_scope.py` and `Chat/console_provider_composition.py`.
- `Tests/Agents/test_code_review_boundary_ledger.py::test_v24_upgrade_and_reopen_preserve_ordinary_history` (1): `sqlite3.DatabaseError: unsupported legacy workflow schema`.
- `Tests/Agents/test_agent_service_review_containment.py::test_failed_spawn_returns_its_slot_only_when_no_child_ran[after_return-True-1]` (1): the scripted `ExecutionOwner.finish_root` failure never raises.

Some of these may be real product regressions rather than test drift. For example, the Stop that persists `error`, the missing footer key and the census's new root reads all describe behaviour.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the 99 nodes is classified as test drift (naming the commit that drifted it) or a product defect (fixed with a test that was red first), and passes on dev
- [ ] #2 Every node classified as a product defect has its own task or fix commit named in this task
- [ ] #3 Each of the twelve files runs on a PR lane, or this task records why it should not
<!-- AC:END -->
