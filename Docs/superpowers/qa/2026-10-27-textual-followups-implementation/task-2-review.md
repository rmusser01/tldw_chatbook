# Spec Compliance

✅ Spec compliant for TASK-37032, at base `a7af77a99aab2da5132fbcbbedbcfbcdeacfd6cc` to head `8a91bd5d85cdc7ab50190592d4a4439201d20759`, with the controller's explicitly approved queue-only and bootstrap-profile scope. Task-quality gate only; no broad CI or merge approval.

- AC1: `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py:2958` captures the service, `:2970` copies the payload before removing routing metadata, and `:3018` submits the existing named group non-exclusively. ID, owner and payload remain callback-local. `Tests/UI/test_schedules_reminder_form_saves.py:159` covers serial/overlap edits and creates; the owner changes before A is released, and both acknowledged local rows retain the original owner.
- AC2: `schedules_workbench.py:2994` checks the update result before success; `:3003` retains failure notification and `:3009` refresh dispatch. Transfer/deletion refusal, controlled SQLite failure and successful retry are exercised in `Tests/UI/test_schedules_reminder_form_saves.py:233`. Cancellation is not presented as rollback.
- AC3: `tldw_chatbook/Scheduling/services/scheduling_service.py:732` projects only queued update fields onto the existing server-write allowlist and ISO-encodes datetimes. Local derived next-run calculation stays intact; nulls and partial fields are preserved in the queued object. `Tests/UI/test_schedules_reminder_form_saves.py:280` checks recurring/one-time durable queue/replay behavior. Final controls XML contains 12 passing form validation, owner, transfer, offline and sync controls.
- AC4: `Tests/UI/test_schedules_reminder_form_saves.py:113` uses the real create chooser/edit key, mounted form and focused Save Enter. `:140` drives repeated Enter for creates. Real SchedulingService and SQLite remain beneath controlled external response gates; assertions check external effects, SQLite, Queue cells and delegated notifications. Cleanup releases the gate before draining native workers inside run_test (`:226`, `:358`).
- Requested file coverage is complete: workbench, new regression module, architecture note, task record; the additional service correction and two exact legacy-module profile markers match controller rulings in `.superpowers/sdd/2026-10-27-textual-37031-37034/progress.md:45`, `:49`, `:50`. The size row decreases at `Tests/Architecture/test_module_size_ratchet.py:546`; no new dependencies, visual values, guard allowances or global fixtures appear in the diff.

# Strengths

- The worker fix addresses admission at the original cancellation boundary without adding locks, queues, lifetime owners or a new service API (`schedules_workbench.py:3014`). It also preserves independent creates rather than grouping all `task_id=None` submissions into one replaceable operation.
- The real replay chain checks more than JSON encodability: `Tests/UI/test_schedules_reminder_form_saves.py:307` instantiates actual SyncEngine, SchedulingServerClient and ServerNotificationsService, observes the validated transport request, and checks queue removal. This exercises the strict signature that a derived `next_run_at` would violate.
- The tests establish separate external-effect and local-acknowledgement facts (`Tests/UI/test_schedules_reminder_form_saves.py:49`, `:209`) and retain expected red evidence rather than treating a cancelled waiter as a rollback.
- No unnecessary abstraction or implementation-size growth: only two product method ASTs change, and the final workbench is four lines shorter (frozen `task2-static/ast-and-baseline-proof.json:1`, `format-equivalence.json:1`).

# Issues

## Critical

None found in this task's changes.

## Important

None found in this task's changes under the approved queue-only scope.

## Minor

1. **Inherited API null omission limits the replay evidence.** `tldw_chatbook/Notifications/server_notifications_service.py:251` drops every None before request construction. `Tests/UI/test_schedules_reminder_form_saves.py:290` proves null schedule fields exist in the queue; `:329` proves a later partial body-null queue record. Neither establishes that a null reaches transport or clears an existing remote value. This is unchanged API behavior, not a regression from the queue projection. The controller explicitly confirmed this is outside Task 2 and will qualify task/QA/PR wording; any end-to-end clearing change needs its own interface/ADR decision. Do not describe these controls as remote null-clear verification.
2. **Inherited validation noise remains.** `.test-tmp/textual-37031-37034/task2-final-controls01/run.log:4` contains Pydantic Field deprecation; `:23` contains pytest-asyncio fixture-loop-scope deprecation. `task2-guards01/run.log:19` includes optional audio dependency warnings, and its warning summary contains synthetic-source invalid-escape warnings. Three worker inventory failures remain in `task2-guards01/result.xml:1`; the frozen exact-base comparison reports identical 8 ungrouped/3 refresh/1 stale findings. These are disclosed baseline debt, not a clean guard run. Retain explicit qualification rather than suppressing warnings or relaxing guards.

# Evidence inspected

All paths in this section are below `.test-tmp/textual-37031-37034/`. I parsed every retained Task 2 parent and private-child XML; no test was rerun.

| Run | Parent cases/failures/errors/skips | Actual private-child cases/failures/errors/skips |
| --- | --- | --- |
| task2-red01 | 2/1/0/0 | 2/1/0/0 |
| task2-red02 | 4/3/0/0 | 4/3/0/0 |
| task2-green01 | 8/1/0/0 | 8/1/0/0 |
| task2-legacy-admission01 | 3/0/2/0 | no private child |
| task2-identity-bulk01 | 14/0/0/0 | 14/0/0/0 |
| task2-offline-green01 | 5/1/0/0 | 5/1/0/0 |
| task2-legacy-admission02 | 2/0/0/0 | no private child |
| task2-offline-one-time02 | 1/0/0/0 | 1/0/0/0 |
| task2-final-controls01 | 12/0/0/0 | no private child |
| task2-guards01 | 25/3/0/0 | no private child |

- Red overlap child failure text confirms both controlled external titles were asserted before SQLite failed with `['A', 'EditedB']` versus `['EditedA', 'EditedB']` (`task2-red01/pytest/test_full_forms_keep_both_acce1/private-profile-process/pytest.xml:1`).
- The initial offline failure is zero queue rows after the local title write (`task2-green01/pytest/test_full_form_refusal_failure3/private-profile-process/pytest.xml:1`). The later one-time failure occurs at the observer's Z-versus-+00:00 assertion after replay and queue removal (`task2-offline-green01/pytest/test_full_form_refusal_failure4/private-profile-process/pytest.xml:1`); corrected child passes in `task2-offline-one-time02`.
- The four serial/overlap cases pass in the actual green01 children; recurring passes in offline-green01; one-time passes in offline-one-time02. All nine new scenarios therefore have passing receipts across the disclosed versions, not one wholly green final-byte module run.
- Read frozen `task2-static/source-evidence-initial.json`, `format-equivalence.json`, `ast-and-baseline-proof.json`, `worker-baseline-proof.json`, `commit-verification.json`, and final-source-hashes.json. Independently compared current SHA256 for all six listed Python files: all match final-source-hashes.json. No source, index or HEAD mutation occurred.

# Focused checks outside the diff

1. **Cutoff method / owner-routing risk:** completed `SchedulingService.update_reminder` at `scheduling_service.py:631` and the callback preamble at `schedules_workbench.py:2951`, because the diff starts inside those functions. The early online return, transfer refusal and local/offline fallback are distinct; the queue projection affects only the fallback. Checked the two callback registrations (`schedules_workbench.py:2872`, `:3059`) for create/edit identity capture.
2. **Mutable form-payload risk:** checked `ReminderForm._save` payload construction at `forms/reminder_form.py:827`. Actual form fields are immutable scalar/datetime values; a detached shallow dict is sufficient here. No deep-copy abstraction is warranted.
3. **Replay-field/metadata contract risk:** compared `_REMINDER_SERVER_CREATE_FIELDS` at `scheduling_service.py:104` with the strict update signature at `server_notifications_service.py:220`; they match. Checked unchanged `ScheduledTasksDB.record_pending_mutation` at `scheduled_tasks_db.py:3550`, `SyncEngine._push_mutation` at `sync_engine.py:1553`, and client stripping at `server_client.py:161`. DB owns top-level idempotency metadata; replay forwards it separately from fields; client strips it at the existing notification seam. No idempotency-policy change is hidden in the projection.
4. **Leaked gate / mocked state-machine risk:** checked imported ReminderHost and until/settle helpers at `Tests/UI/test_schedules_reminder_edit_identity.py:47`, `:60`, `:66`. They instantiate the real service and bound drain waiting to 15 seconds. The new tests release their gate before the drain.

The immutable diff was read once; an output-size truncation required retrieval of only the missing new-test/legacy-test/lesson section. No git command, broad source crawl, subagent, CI operation, external message or test rerun was used.

# Cannot verify / controller checks

- Final combined branch execution of every new case remains the root's planned integration check; current hashes match the final task snapshot, while the available passing receipts span qualified earlier sources. This review does not substitute for that final-byte run.
- Exact scoped Ruff/format command output was not present among the retained static JSON files I inspected. Seven inherited E402 findings and six-file format success remain report claims supported only indirectly by frozen baseline/AST summaries, not a freshly executed or directly inspected lint receipt. Controller should retain/locate the original output if a clean-format/baseline-lint assertion is needed; do not rerun unrelated sweeps merely for this review.
- Five visual/path baseline failures are source-qualified by the frozen proof only; I did not independently replay Task 1's visual runs. No changed visual input appears in this diff.
- Controlled transport establishes real local client/schema traversal and durable queue settlement, not a live-server contract. Same-reminder write ordering, navigation departure/durable worker lifetime, physical remote rollback, retry policy and remote null clearing remain outside this task and unverified here. The controller confirmed the null-clearing boundary explicitly during review.

# Assessment

**Task quality: Approved.** The changes are small and directly repair independent admitted-save settlement and the full-form offline queue serialization failure. The meaningful remaining qualifications concern inherited API/validation behavior and root-owned final integration evidence; they do not justify widening this task's runtime or interface scope.
