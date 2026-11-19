# Final review I1 fix report

Status: ready for the controller's one scoped rereview and current-head CI. TASK-37032 remains In Progress. Reviewed/fix base: `0e844e6e3a7c6d3314a3f424f66ceea0c802bcc0`; branch `codex/textual-task-37031-37034`. Explicit worktree: `C:/Users/GDesktop-1/.codex/worktrees/textual-settings-patterns/tldw_tui`.

## Repair and ownership

The real callback and SchedulingService response publisher confirmed I1. Native nonexclusive workers for the same reminder could publish responses in reverse arrival order; `_persist_server_reminder_response` also clears that record's queued mutation. No service changes were necessary.

`_on_reminder_form_result` now derives a group from its captured owner/reminder ID, snapshots ALL existing same-node/group workers before submitting the new native worker, and awaits each predecessor's native `wait()` before invoking the service. Capturing every predecessor avoids relying on the manager's set/iteration admission order for three or more Saves. Each later admission includes the earlier waiting workers, so the ordering remains transitive. It retains the copied payload, captured service, owner and ID. Normal save failure remains recoverable, and completed failed writes do not cancel later workers. The two full-form entry/submit paths share this callback.

Different IDs remain independent. Creates bypass predecessor waiting even when their owner/group matches; separate creates remain admitted independently, and existing actual repeated Enter controls retain one write per create form. No global queue, lock map, state machine, new imports, service/runtime/schema boundary, durable lifetime, retry or remote rollback policy was added. Existing native screen teardown remains the lifetime boundary; inline edits/global service callers remain outside this full-form repair.

ADR required: no. ADR path: N/A. Reason: bounded implementation of the existing mounted UI owner and service contracts in `backlog/decisions/018-local-server-hybrid-scheduled-tasks.md`, `033-application-session-state-ownership.md`, `036-application-service-composition-lifecycle.md` and `031-tui-keybinding-and-footer-hint-conventions.md`. No new ADR was required. TASK-37032 AC5 and its implementation plan/ADR assessment were updated through the specified UTF-8 Backlog CLI BEFORE product/test changes; AC5 was checked and final notes appended after fresh verification. Architecture documentation records the actual policy and null transport limit.

## TDD and runtime evidence

All commands below used explicit WT cwd and the unchanged native private runner. TEMP/TMP/bootstrap roots precede pytest import; the existing two pytest plugins and 180-second per-case allowance are unchanged. `assert_worktree_sources` verifies actual workbench/form/service/SQLite/replay classes resolve inside this worktree. Gates release and native work drains in finally inside `run_test`; no test sleeps, timeouts, skips/xfails or production state-machine replacements were added.

```text
C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted-stream.py final-fix-same-red01 Tests/UI/test_schedules_reminder_form_saves.py::test_same_reminder_saves_settle_before_the_next_update

C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted-stream.py final-fix-forms-green01 Tests/UI/test_schedules_reminder_form_saves.py

C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted-stream.py final-fix-guards01 "Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py]" "Tests/Architecture/test_module_size_ratchet.py::test_budget_is_not_left_slack[tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py]" Tests/Performance/test_screen_preimport_payload_budget.py Tests/Performance/test_app_import_weight.py::test_app_import_own_module_count_stays_at_the_post_diet_size -rP
```

| Run | Parent cases/failures/errors/skips | Actual child cases/failures/errors/skips | Output |
|---|---|---|---|
| final-fix-same-red01 | 2/2/0/0 | 2/2/0/0; 2 XML files | Both expected RED failures: `AssertionError: assert 'First' == 'Second'` at final SQLite title, 60.23s |
| final-fix-forms-green01 | 11/0/0/0 | 11/0/0/0; 11 XML files | `11 passed, 1 warning in 339.34s` |
| final-fix-guards01 | 5/0/0/0 | No private-profile child wrappers | `5 passed, 4 warnings in 51.48s` |

These are eleven form scenarios, not twenty-two. The eleven comprise the nine original serial/different-ID/create/repeated-Enter/failure/retry/transfer/deleted/offline-replay controls plus two same-ID online/newer-offline cases. The new cases drive actual mounted Queue `e`, ReminderForm title editing, and visible Save Enter twice with the SAME captured reminder. A controlled client records an accepted first external effect and holds its response; the real service, SQLite writes, pending mutation and Queue repaint remain in place. On GREEN the second request has not started before first release; both truthful information notices publish; captured server ownership survives a change to the shared active owner; neither response is cancelled. Newest title `Second` remains in SQLite and visible Queue. The online controlled server effect is `Second`; the offline variant retains exactly one queued mutation with title `Second` after first acknowledgement and no later deletion.

Every actual RED/GREEN child XML was parsed and hashed; canonical parent/child classname+name sets are exactly equal. RED has only the intended stale-title failures, no setup/teardown errors/skips. GREEN has no child setup/teardown errors/skips. Source hashes were checked unchanged after both final runs. Retained commands, logs, parent/child XML and runner `verification.json` are under `.test-tmp/textual-37031-37034/<run>/`. Detailed child receipt paths/SHA256 and exact node equality: `final-fix-static01/child-receipt-proof.json`; guard receipt SHA256: `final-fix-static01/guard-proof.json`.

## Static/source/import checks

```text
C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 -m ruff check tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py Tests/UI/test_schedules_reminder_form_saves.py
C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 -m ruff format --check tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py Tests/UI/test_schedules_reminder_form_saves.py
git -c safe.directory=<WT> -c gc.auto=0 diff --check
```

All exit 0; Ruff reports `All checks passed!`, format reports `2 files already formatted`. Native static receipts: `final-fix-static01/static-checks.json`. Exact FIX_BASE vs current AST comparison changes only `SchedulesWorkbench._on_reminder_form_result` and its nested `_save_and_refresh`; every other function AST and all imports are identical. Scoped native worker detectors run on both in-memory ASTs give identical results: no ungrouped exclusive worker findings, no callback refresh finding, and the inherited `SchedulesWorkbench._request_tasks_refresh._load` inline-loader finding. No source restoration or broad worker-suite rerun was needed. Source proof: `final-fix-static01/source-proof.json`.

Workbench: 5530 physical lines under UNCHANGED5532 cap; 2-line slack passes existing 50-line tolerance. Evals4211 and its cap untouched. No source cap/guard file changed. Existing fresh import guards: preimport558/558 modules,424405/425347 LOC; fattest Library route130691/135111 LOC; first-send93/94 modules; boot680/686 modules with inherited snapshot drift+5/-0. All allowances are unchanged. No new boot imports. Unchanged stylesheet/token/design inputs were not rerun; this callback changes no visual values.

Native source SHA256 (physical checkout bytes):

| File | Lines | SHA256 |
|---|---:|---|
| `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py` | 5530 | `3fde4e3c8b6e6c3ecf04604f039ee41e60955d1cee691cab736aa54c10877caa` |
| `Tests/UI/test_schedules_reminder_form_saves.py` | 428 | `560d995176516eda4230e461fb9db7f7aaa0538494015f9a9ba4058037b61425` |
| `Tests/Architecture/test_module_size_ratchet.py` | 647 | `364c3cb3ad4db8e8da0bdd1a0e81ccbb07e60dc820611d07430c3c3c7ce8dfe6` |
| `tldw_chatbook/UI/Screens/evals_screen.py` | 4211 | `71f910ebe96bbbeec8baa0b8ab57661afdde0400188d7f424b05083d168f4cb0` |
| `tldw_chatbook/UI/Screens/scheduling/forms/reminder_form.py` | 860 | `ffca1af4a79f7077677bcde4d63f60e30c4c12b069e38b29053b865fd90f8c37` |
| `tldw_chatbook/Scheduling/services/scheduling_service.py` | 3164 | `390214ee673d2087f48c5194d697cfcdcc3f68eb6ce3e79ad6bafdd096235ee3` |
| `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` | 3844 | `f2c3fa6ce547b525e3fcfdfbde4fcbade6d29ed3b7c23388f68221d3fdc2233e` |
| `tldw_chatbook/Scheduling/services/sync_engine.py` | 2020 | `55eafc7a0b094d192045985c8fc0a2d018f0ab6e9abf24196b8fd0ab073b7a30` |
| `tldw_chatbook/Scheduling/services/server_client.py` | 812 | `ffb3792116c02791741b074ae83ddbfd6ffcb37a95b9b2d927bfc08bda0e21ba` |
| `tldw_chatbook/Notifications/server_notifications_service.py` | 631 | `95f2a813aaf4ac386ef6f6c7aa1a0f953da05fc9418f7fa7d15d056645583b6d` |

## Concerns, limits and self review

I1 is addressed by the focused mounted RED/GREEN and native ordering mechanism. M1 remains explicit: durable queue null retention and real replay do not prove remote explicit-null clearing; unchanged `ServerNotificationsService` omits `None`. M2 remains explicit: the broader branch has inherited seven E402 plus seven E721 findings, two format failures, worker/MRO inventory and Windows visual/path debt. Only the two owned Python files are claimed clean here. Existing Pydantic Field/exclude_if and pytest-asyncio fixture-loop deprecation warnings remain; import-budget tests emit their existing census UserWarnings. No output-pristine or whole-repository clearance claim is made.

Self review traced both callback callers, captured owner/service/ID/payload, all predecessor admission (including third-and-later transitive waits), create bypass, native worker error/cancellation scope, existing failure/refresh flow, source AST/string integrity, and complete four-file diff. No external reviewer, subagent, remote/PR/CI action, primary-checkout edit, new dependency or full suite occurred. No new reusable lesson beyond the existing ownership/child-receipt lessons was invented. Root owns the one scoped rereview, two new census selectors, current-head hosted CI and final delivery. This report/proofs are controller evidence and are not included in the scoped product commit.

## Commit

Commit `372c20559c5271b0f6356ae95a533b5260d64009` (`fix(schedules): serialize same-reminder form saves`). Verified exact four committed paths: workbench, form regression module, TASK-37032 and scheduling architecture document. No shared CI/census/QA/plan/proof files were committed.
