# TASK-37032 implementation report

Status: implementation ready for controller review/CI with disclosed baseline concerns. Backlog remains In Progress. Base: `a7af77a99aab2da5132fbcbbedbcfbcdeacfd6cc`.

## Implementation and contract

Full ReminderForm callbacks now submit independent native workers in the existing explicit `schedules-save-reminder` group (`exclusive=False`). Every accepted callback captures its SchedulingService reference, task ID, selected owner and copied form dictionary before dispatch. Form fields are strings, booleans, `None` and immutable datetimes; no nested mutable form value needs a deep copy. Create callbacks with `task_id=None` are also independent. No per-ID group can accidentally collapse separate creates. Real repeated Enter controls prove each dismissed create form still produces exactly one service write.

A `None` update result means deletion/transfer refusal rather than successful acknowledgement. It now enters the existing recoverable failure notification and Queue refresh path. Service exceptions retain existing handling. Full-form workers retain their existing screen lifetime. Cancelling a client waiter never establishes remote rollback. No new remote atomicity, same-reminder ordering, retry, idempotency or durable owner policy was introduced.

Real offline forms exposed an inherited failure: schedule recomputation adds a datetime `next_run_at`; the DB row updates, then `record_pending_mutation` fails JSON encoding, leaving no queued edit. The controller authorized a bounded `SchedulingService.update_reminder` correction after its plan was updated. Queued fields now use the existing vetted server-write allowlist and `datetime.isoformat()` representation. This omits local derived clocks, retains explicit nulls and preserves partial update shape. The strict real `ServerNotificationsService.update_reminder` signature accepts the same field set; ISO-encoding the derived clock alone would still be unreplayable.

Unchanged `ScheduledTasksDB.record_pending_mutation` injects top-level idempotency metadata. The full form never supplies that metadata in its fields. Unchanged `SyncEngine` forwards it, and unchanged `SchedulingServerClient` strips it only at the notification seam. The corrected one-time test verifies a nonempty stable key across durable DB reads and absence from the real API request. Existing stable-across-retry control also passes. Both offline schedules use the real reminder-only sync transaction, real client, real notification method and real request schema with a controlled underlying API transport; successful replay removes the durable queue entry. A subsequent partial/null mutation is exactly `{"body": null}` in queued fields.

ADR required: no. ADR path: N/A. Reason: these bounded fixes implement existing contracts in `backlog/decisions/018-local-server-hybrid-scheduled-tasks.md`, `033-application-session-state-ownership.md`, `036-application-service-composition-lifecycle.md`, and `031-tui-keybinding-and-footer-hint-conventions.md`. ADR-150/161 design rules remain unchanged; no stylesheet, visual token, CSS value, runtime styles or key map changed. Architecture notes and an incident-backed testing lesson were updated.

Bounded duplicate-owner search found no task specifically naming offline reminder next_run_at/datetime serialization. The only matching reminder/offline/sync title was historical TASK-299.2 (Done), whose existing ADR-018 contract is reused.

## TDD and actual XML results

Audit probe promoted from `Docs/superpowers/qa/2026-10-27-textual-book-after-pr3251/probes/workflows-schedules/test_full_form_admission.py.txt`. No vanished raw audit run is cited as fresh evidence.

- `task2-red01`: serial full edits passed; overlap failed `assert ['A', 'EditedB'] == ['EditedA', 'EditedB']` after both literal controlled external-effect assertions passed. Both external mutations occurred; A's SQLite/Queue acknowledgement was lost. All gates released and native workers drained before leaving run_test.
- `task2-red02`: serial creates/repeated Enter passed; overlap-create retained only EditedB. Transfer and deletion after opening the real form both failed because the old callback falsely emitted success. Zero unrelated setup/teardown errors or skips.
- `task2-green01`: seven behaviors passed, but the new recurring offline form found the genuine inherited datetime/queue failure (local title changed; pending queue length 0 rather than 1). This is not reported as a wholly green module run.
- `task2-offline-green01`: all refusal/failure/retry cases and real recurring replay passed. `-k offline` also matched the outcome function name, selecting all five outcome cases. The one-time replay itself succeeded and cleared its queue, but the observer wrongly expected `Z` instead of the existing `+00:00` ISO representation. That failed attempt remains retained.
- `task2-offline-one-time02`: corrected exact one-time case passed, including stable stored idempotency metadata, real API request and subsequent partial/null queue assertion. No product change was needed for the observer correction.

All nine new scenarios have passing actual private-process receipts across these qualified runs. The four serial/overlap cases' function AST is unchanged after their passing run; the workbench's final formatting is AST-identical, and the later service edit changes only the offline fallback queue projection, below the online-return/local-routing branches those four cases exercise. Existing identity/bulk source qualification is similarly limited: local writes or successful controlled server responses do not enter that changed offline block. No broad batch was rerun solely for formatting or the corrected equivalent ISO observer, per controller direction. The final one-time, 12 focused legacy/service controls and all guards ran against final product bytes.

Every child XML is inspected separately from its wrapper. Counts below are actual testcase/failure/error/skip counts; zero children means no private-profile wrapper, not zero live legacy bodies. Legacy app nodes use the authorized lifetime-private bootstrap source.

| Run | Parent cases / failures / errors / skips | Actual private child cases / failures / errors / skips |
|---|---|---|
| task2-red01 | 2/1/0/0 | 2/1/0/0 (2 XML files) |
| task2-red02 | 4/3/0/0 | 4/3/0/0 (4 XML files) |
| task2-green01 | 8/1/0/0 | 8/1/0/0 (8 XML files) |
| task2-legacy-admission01 | 3/0/2/0 | 0/0/0/0 (0 XML files) |
| task2-identity-bulk01 | 14/0/0/0 | 14/0/0/0 (14 XML files) |
| task2-offline-green01 | 5/1/0/0 | 5/1/0/0 (5 XML files) |
| task2-legacy-admission02 | 2/0/0/0 | 0/0/0/0 (0 XML files) |
| task2-offline-one-time02 | 1/0/0/0 | 1/0/0/0 (1 XML files) |
| task2-final-controls01 | 12/0/0/0 | 0/0/0/0 (0 XML files) |
| task2-guards01 | 25/3/0/0 | 0/0/0/0 (0 XML files) |

Raw commands, parent output/XML, actual child pytest.log/XML and runner verification.json are retained below `.test-tmp/textual-37031-37034/<label>/`. Labels are outside pytest numbered-temp rotation. Each private integration case asserts worktree source identity for the workbench, form, real SQLite/service and real replay classes. All held response gates release inside run_test cleanup; no native worker or fixture teardown errors are hidden behind a passing parent.

## Legacy fixture qualification

The initial exact form-title and transfer-key nodes did not run: `Tests/UI/conftest.py:193` imports `tldw_chatbook.app` to patch model catalog refresh; `app.py:693` invokes collection-bound `load_settings`, whose guarded config participant rejects the later per-test redirect with `RecoveryRequired(raw_source_selection_changed)`. The service title-only offline control passed. This rejected setup is retained as 2 errors / zero executed UI bodies.

After confirming neither module deliberately changes config/profile/env selection, the controller authorized ONLY `Tests/UI/test_reminder_form.py` and `Tests/UI/test_schedules_transfer_actions.py` to use the existing `bootstrap_profile` opt-in and per-test reset. One exact node per module passed first (`task2-legacy-admission02`); then only four affected form validation/ordinary/owner nodes and two transfer controls ran alongside six exact service/sync controls. Existing markers and DB fixtures are preserved. No global fixture/admission change, arbitrary skip, timeout increase or blanket plugin mark was used.

## Static, import and inherited concerns

Final scoped format check passes all six touched Python files. Full scoped Ruff retains seven inherited E402 findings: service.py lines41,46,55,56,64,65; test_reminder_form.py's existing later import (now595). The unchanged original sources produce those same seven findings. Boot import placements were preserved; excluding only E402 from that scoped lint command passes. Existing Pydantic Field/exclude_if and pytest-asyncio fixture-loop-scope deprecations remain; optional Sounddevice/WebRTC/HuggingFace setup messages remain. Worker synthetic-source checks also emit existing invalid-escape SyntaxWarnings. Output is not pristine.

Final scheduling size is5532, and only its own row was lowered from5536. Evals4211 is preserved. Both size controls and both unchanged import guards pass: preimport558/558 modules,424352/425347 LOC; fattest library route130690/135111 LOC; first-send93/94 modules. No ceiling, allowance or snapshot was raised.

Unchanged worker inventory: 22/25 pass; three exact inherited failures remain:

1. `test_no_ungrouped_exclusive_workers`: eight sites, listed in worker-baseline-proof.json. The vendored directory-navigation exemption is missed on native Windows backslash paths; seven other unrelated existing sites remain flagged.
2. `test_mutation_refreshes_dispatch_through_loader_group`: existing `_request_tasks_refresh._load` loader wrapper and two Watchlists loader sites.
3. `test_allowlist_has_no_stale_entries`: the same native vendored-path mismatch.

`worker-baseline-proof.json` runs the unchanged detector on current source and exact base source substituted only in memory for the two changed product modules. All findings are exactly equal (8 ungrouped,3 refresh,1 stale). No product source was restored, no guard changed, no failure hidden. The callback's existing single refresh dispatch remains valid.

Five inherited native Windows visual/path failures from Task1 were not rerun under the shared protocol: design-token `test_active_css_extension_is_in_hex_floor`; component-pattern `test_canonical_classes_defined_only_in_owning_sheet`, `test_dimension_literal_ratchet`, `test_active_css_extension_is_in_dimension_floor`, and `test_pin_rejects_active_css_extension_without_writing_baseline`. `ast-and-baseline-proof.json` confirms their exact source hashes match Task1 receipts, all CSS/registry inputs have empty Git diff from the exact base, and this task changes no visual values. This is source qualification of existing evidence, not a new green visual claim.

## Self review and final-byte evidence

Only `_on_reminder_form_result` and `SchedulingService.update_reminder` change product method ASTs. Four inherited workbench formatting sites were mechanically normalized with whole-module AST equality before/after formatting. Every function AST in the two legacy test modules is unchanged; their only behavior setup change is the approved bootstrap marker. The shared size guard changes only the scheduling row. Source snapshots/proofs/lint/format receipts are in `task2-static/`.

Both callback callers (create and captured edit ID) were traced. Real Queue e/title/visible Enter Save and create n/chooser/Enter paths use real form callbacks. Real SQLite and authoritative DataTable row text are asserted, not only internal worker state. Notification observation delegates to the real notify method. External response control and scoped SQLite failure injection are below the production ownership/state machine. No remote connections, service retries, transaction/schema/interface changes, new dependencies, Qodo, PR, CI, or full suite were run. Root-owned scripts/ui_pr_gate_census.txt remains unstaged and unmodified by this task.

Final SHA256 values refer to native checkout bytes (CRLF and Git LF are not conflated):

| File | Lines | SHA256 |
|---|---:|---|
| tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py | 5532 | `236933a0b492ac1f5ef68548b6a4339c2fe5b538916dcea51a5a25b2b4deabd8` |
| tldw_chatbook/Scheduling/services/scheduling_service.py | 3164 | `390214ee673d2087f48c5194d697cfcdcc3f68eb6ce3e79ad6bafdd096235ee3` |
| Tests/UI/test_schedules_reminder_form_saves.py | 362 | `5ea682fb502cefe48f659c99549eeafe141ef49eef9b8f2319eb3592beb0053a` |
| Tests/UI/test_reminder_form.py | 830 | `008564286db9791aec641893f543b8a89e2fdfa0bc99d7c59d187546978fea76` |
| Tests/UI/test_schedules_transfer_actions.py | 432 | `9c560cc6759a605b1e2684795135ce9e5267020d2b3cdef078add986ccc4c3ae` |
| Tests/Architecture/test_module_size_ratchet.py | 647 | `364c3cb3ad4db8e8da0bdd1a0e81ccbb07e60dc820611d07430c3c3c7ce8dfe6` |

## Full runner commands

All commands used explicit worktree cwd `C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui`, the primary venv Python with `-X utf8`, and the shared native private runner. Its fixed plugin/time budget/profile setup is unchanged.

### task2-red01

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task2-red01 Tests/UI/test_schedules_reminder_form_saves.py
```

### task2-red02

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task2-red02 Tests/UI/test_schedules_reminder_form_saves.py -k "transfer or deleted or create"
```

### task2-green01

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task2-green01 Tests/UI/test_schedules_reminder_form_saves.py
```

### task2-legacy-admission01

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task2-legacy-admission01 Tests/UI/test_reminder_form.py::test_reminder_form_requires_title Tests/UI/test_schedules_transfer_actions.py::test_key_bound_edit_refuses_on_a_transferring_row Tests/Scheduling/test_scheduling_service.py::test_update_reminder_server_falls_back_local_on_unavailable
```

### task2-identity-bulk01

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task2-identity-bulk01 Tests/UI/test_schedules_reminder_edit_identity.py Tests/UI/test_schedules_bulk_admission.py::test_ordinary_batch_completes_and_clears_its_marks Tests/UI/test_schedules_bulk_admission.py::test_mixed_success_refusal_and_failure_reports_actual_count
```

### task2-offline-green01

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task2-offline-green01 Tests/UI/test_schedules_reminder_form_saves.py -k offline
```

### task2-legacy-admission02

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task2-legacy-admission02 Tests/UI/test_reminder_form.py::test_reminder_form_requires_title Tests/UI/test_schedules_transfer_actions.py::test_key_bound_edit_refuses_on_a_transferring_row
```

### task2-offline-one-time02

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task2-offline-one-time02 Tests/UI/test_schedules_reminder_form_saves.py::test_full_form_refusal_failure_and_offline_outcomes[offline_one_time]
```

### task2-final-controls01

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task2-final-controls01 Tests/UI/test_reminder_form.py::test_reminder_form_requires_run_at_for_one_time Tests/UI/test_reminder_form.py::test_reminder_form_rejects_invalid_cron Tests/UI/test_reminder_form.py::test_runs_on_is_disabled_when_editing_an_existing_task Tests/UI/test_reminder_form.py::test_reminder_form_submits_when_valid_one_time Tests/UI/test_schedules_transfer_actions.py::test_workbench_locks_lifecycle_for_a_transferring_row Tests/UI/test_schedules_transfer_actions.py::test_clearing_the_lifecycle_lock_restores_the_buttons Tests/Scheduling/test_scheduling_service.py::test_create_reminder_server_falls_back_local_on_unavailable Tests/Scheduling/test_scheduling_service.py::test_update_reminder_server_falls_back_local_on_unavailable Tests/Scheduling/test_scheduling_service.py::test_update_reminder_targets_another_owner_without_flipping_owner_id Tests/Scheduling/test_scheduling_service.py::test_update_reminder_refused_while_transferring Tests/Scheduling/test_sync_engine.py::test_sync_pushes_update_mutation Tests/Scheduling/test_sync_engine.py::test_idempotency_key_stable_across_push_retries
```

### task2-guards01

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task2-guards01 Tests/Architecture/test_worker_exclusive_group_inventory.py Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py] Tests/Architecture/test_module_size_ratchet.py::test_budget_is_not_left_slack[tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py] Tests/Performance/test_screen_preimport_payload_budget.py
```

## Commit and remaining closure

Commit: `8a91bd5d85cdc7ab50190592d4a4439201d20759` (`fix(schedules): settle independent reminder form saves`). Final precommit and postcommit SHA256 verification confirms all six tested Python sources match final-source-hashes.json; each compiles, scoped git diff --check passes, and the root census SHA256 is preserved. Backlog remains In Progress for root independent review/current-head CI; no hosted validation is claimed.
