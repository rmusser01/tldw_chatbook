# PR3346 UI6 fix handoff

Product base HEAD79e659dcde6f5360edb0070f9d184713bd8584d1. Parent explicitly approved the minimal fix after the read-only causal report `ui6-investigation.md`. No commits, pushes, manifests or unrelated files were edited.

## Change

`ConsolePromptMixinService.accept_attempt` now runs its real receipt callback on the current loop for memory-backed DBs, matching the existing native turn-commit treatment. File-backed DBs still use `run_owned_db_call`. The existing task, shield/drain cancellation loop, exception translation, post-write session/store/persistence ownership checks and accepted publication remain intact. This fixes real SQLite thread-local `:memory:` connection loss, not the10-second wait.

The new parameterized service regression writes empty and nonempty receipts with two actual SQLite message anchors, reads the classified durable receipt, verifies exact snapshot and live marker, and verifies clean transcript content. No source/provider/DB write is stubbed.

## Verification

- New regression RED before product edit:2 failed at the real save with `mixin_attempt_save_failed` (underlying thread-local SQL failure already independently observed).
- Covering run:47 passed,2 failed in50.00s. All four requested original UI nodes PASSED, with original timeouts and assertions:
  - UI6 paste token:4.85s call.
  - UI9 question_in_front_of_paste:4.90s call.
  - UI14 palette Stop:13.02s call (its own private-profile child completed).
  - UI14 saved-chat missing Library policy:6.51s call.
- The two failures were the new regression's read-back lookup: a pending assistant has no published persisted_message_id, and the operational write uses its native id. Corrected the test lookup to the same existing `persisted_message_id or id` rule. The real write had succeeded. No product change followed the four UI passes.
- Final entire service test file `Tests/Chat/test_console_prompt_mixin_admission.py`:45 passed in16.08s.
- Scoped Ruff:0 diagnostics before and after in both owned files; format check passes; scoped git diff --check passes. An initial Ruff invocation tried its default read-only cache and failed; `--no-cache` checks then completed. A formatter mutation attempt likewise was denied and its small diff was applied with apply_patch; final formatter output confirms both files already formatted.

No full suite, dev comparison or timeout change. These tests qualify the named failing routes and service regression, not all CI lanes or external providers.

## Explicit owned artifacts

Repository files:
- `tldw_chatbook/Chat/console_prompt_mixin_service.py` (+9/-1)
- `Tests/Chat/test_console_prompt_mixin_admission.py` (+60)

Private reports in parent task context:
- `ui6-investigation.md` (initial read-only findings; implementation authorization came afterward)
- `ui6-fix-report.md` (this final handoff)

Private evidence logs:
- `/private/tmp/mixins-ui6-original.log`
- `/private/tmp/mixins-ui6-observe.log`
- `/private/tmp/mixins-ui6-db-observe.log`
- `/private/tmp/mixins-ui6-memory-control.log`
- `/private/tmp/mixins-ui6-service-red.log`
- `/private/tmp/mixins-ui6-fix-covering.log`
- `/private/tmp/mixins-ui6-service-debug.log`
- `/private/tmp/mixins-ui6-service-green.log`
- `/private/tmp/mixins-ui6-lint-comparison.json`

Owned pytest basetemp directories:
- `/private/tmp/mixins-ui6-original`
- `/private/tmp/mixins-ui6-observe`
- `/private/tmp/mixins-ui6-db-observe`
- `/private/tmp/mixins-ui6-memory-control`
- `/private/tmp/mixins-ui6-service-red`
- `/private/tmp/mixins-ui6-fix-covering`
- `/private/tmp/mixins-ui6-service-debug`
- `/private/tmp/mixins-ui6-service-green`

Parent-provided CI logs and all other existing/parallel repository changes are not mine.
