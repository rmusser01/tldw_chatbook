### Finding Verdicts

- **I1: Retain and enforce source identity until uncustodied direct admission — ADDRESSED.** The service now exposes one retained currentness predicate for exact session/incarnation/binding/preference revision and app/scope/backend/DB/path identity (`tldw_chatbook/Chat/console_prompt_mixin_service.py:162`, `:194`). Direct Queue captures it before awaiting, checks after hook preparation, and checks again immediately before coordinator admission (`tldw_chatbook/Chat/console_chat_controller.py:6253`, `:6263`, `:6302`). Direct manual submission retains the predicate with its private snapshot/session witness and checks after hooks, configuration/library/provider preparation, and before acceptance (`tldw_chatbook/Chat/console_chat_controller.py:9395`, `:10081`, `:10131`, `:10324`, `:11155`, `:11269`).
- **Later durable waits — ADDRESSED.** Native acceptance invokes the predicate on the owning event loop after `drain_agent_handoff` and before issuing the native commit (`tldw_chatbook/Chat/console_native_commit.py:680`). The callback is an invocation-only private argument; it is not part of the durable acceptance object, frozen configuration, or serialized state (`tldw_chatbook/Chat/console_native_commit.py:555`, `:680`).
- **Paused pending sends and legal stale cleanup — ADDRESSED.** `_PreparedSendContinuation.prompt_mixin_capture` is representation-hidden and retains the same predicate, with no library recapture (`tldw_chatbook/Chat/console_chat_controller.py:3108`, `:10667`). Resume checks before and after provider resolution, and lifecycle checks after the resumed hook (`:7856`, `:7890`, `:9407`). Stale optimistic echoes are failed and COMMITTING work takes the existing legal rollback path (`:9504`; existing helper `:8636`). Native ownership releases in `finally` before the outer lifecycle cleanup (`tldw_chatbook/Chat/console_native_commit.py:958`).
- **Complete supplied and accepted custody compatibility — PRESERVED.** Only uncustodied direct manual/queue capture creates the predicate (`tldw_chatbook/Chat/console_chat_controller.py:6252`, `:9391`). The pending resume guard explicitly excludes durable postcommit continuation (`:7858`); the native callback is not retained into postcommit. The focused evidence includes supplied-empty compatibility and accepted queue edit/delete/drain retention.

### New Breakage in the Fix Diff

None found. No new Critical or Important issue in the scoped fix diff.

### Checks

- Reviewed fix package `review-32df824beb..a0c2c4bc32.diff` and the round-1 appendix in `task-4-report.md`. Initial combined tool output truncated its middle; read only the omitted diff portions afterward. No diff regeneration, git mutation, or product edit.
- Named risk: stale callback cleanup could leave a COMMITTING preparation wedged behind native ownership. Inspected the cut-off native acceptance context/finally and the existing `_rollback_committing_preparation` / `_abandon_preparation` helpers. Ownership releases before the legal COMMITTING → PAUSED → cancelled cleanup.
- Named risk: source checks could leak into accepted postcommit re-entry. Completed the cut-off `_continue_prepared_submission` context; trace postcommit re-entry uses `resume_durable_postcommit`, with the explicit postcommit guard exemption above. No callback was added to frozen/durable state.
- Verified retained RED output: all eight real source replacement/path-retarget tests fail before the fix (`task-4-evidence/fix1-red.log`). Reviewed the new tests' Event barriers and assertions for refusal, no provider invocation, unchanged draft/attachment ownership, failed echoes, and cleared preparations/native owners (`Tests/Chat/test_console_prompt_mixin_admission.py:831`, `:962`, `:1058`).
- Verified retained focused output: 15 passed in 5.04s with no warning summary (`task-4-evidence/fix1-focused-green.log:5`). Verified latest stale-cleanup output: 4 passed in 2.44s, no warning summary (`task-4-evidence/fix1-cleanup-green.log:5`).
- Verified connected output: 141 passed and one new resumed-hook test assertion failed (`task-4-evidence/fix1-final.log`). The fix diff now asserts the existing refused `ConsoleSubmitResult` contract; the later focused/cleanup outputs cover that corrected case. No suite rerun was needed.

### Out-of-Scope Observations

None. Previously qualified aggregate FD noise remains outside this fix and is not a new blocker.

### Verdict

**Fix round: All findings addressed, no new Critical/Important breakage.**

Skipped: suite reruns, whole-slice review, and deferred Tasks 5/7/8. No additional material risk found in this fix diff.
