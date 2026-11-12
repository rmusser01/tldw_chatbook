# Independent review: same-screen exact Open

## Spec Compliance

- ✅ Spec compliant for the narrow two-file correction, base `2f28e6d6d6b7c5000b57881a4e859292d5f563de` to head `f83f8975c88b762eb61e927247d76a20c466ea2f`. The production change is confined to the shared exact-resume producer at `tldw_chatbook/UI/Console_Modules/archive.py:185`; it reuses the existing navigation completion callback and consumer instead of adding a lifecycle event or action framework. The only other changed file adds one production-dispatch regression at `Tests/UI/test_console_conversation_archive_flow.py:350`.
- ✅ The change preserves the brief's original-ID, active-branch, unrelated-draft and zero-send behavior: the regression dispatches the typed request through the real App at `Tests/UI/test_console_conversation_archive_flow.py:415`, awaits acknowledged revision completion, and asserts persisted identity, active leaf, source draft and empty destination composer at lines 424–427. The cross-screen control at line 432 checks the reused Console, settled second revision, retained branch/draft and message count; lines 439–443 inspect agent run/step rows when that lazy store exists.
- ✅ No restoration writes, confirmation predicates, lifecycle checks, view-hook declarations, screen styling, storage contracts or dependencies change in this diff. This implements the existing original-ID handoff contract in ADR-147 and ADR-166, so no new ADR is needed.
- ⚠️ The full live native-receipt journey is outside this patch's automated proof. The regression begins at the shared request boundary; its cross-screen control uses Home rather than Library. The report explicitly assigns live native qualification to the controller and identifies the existing Library constructor issue separately. This is an evidence boundary, not a missing requirement for the narrowly dispatched fix.

## Strengths

- `tldw_chatbook/UI/Console_Modules/archive.py:189` gates the new scheduling on successful navigation, the captured screen still owning the visible screen, and `request_current()`. The latter retains both the app request token and caller-supplied `request_if`; it does not broaden receipt authority.
- The callback calls the existing `_consume_pending_conversation_resume` rather than implementing a second resume path (`archive.py:197`). That wrapper checks visible-screen ownership and a pending handoff (`tldw_chatbook/UI/Screens/chat_screen.py:17312`); the shared consumer checks durable lifecycle state, exact claim revision and ownership through awaited activation (`archive.py:247`).
- The new regression uses a fresh subprocess, sets HOME/XDG/config/data paths before importing production modules, and uses real scratch SQLite (`Tests/UI/test_console_conversation_archive_flow.py:361`). It leaves `post_message`, navigation dispatch and ScreenResume behavior intact. The target includes an inactive sibling, so the leaf assertion checks branch selection rather than merely message presence.

## Named risks and focused checks

1. **Successful completion ownership:** Checked `NavigateToScreen.commit_target_ownership` / `report_completion` at `tldw_chatbook/UI/Navigation/main_navigation.py:180` and App navigation at `tldw_chatbook/app.py:13423` and `:14078`. Completion is one-shot and success is committed when the destination takes the stack; pre-commit rejection/cancellation reports failure. The callback's `succeeded` check therefore uses the existing destination-ownership boundary.
2. **Same versus cross-screen identity:** Checked reusable target selection and stack transfer at `app.py:13948` and `:14078`, plus the existing ScreenResume consumer timer at `chat_screen.py:23366`. A Library/Home source differs from the destination Console and cannot take the callback path. The already-current Console can take it without relying on an absent ScreenResume event. Empty capture stacks simply leave ordinary lifecycle consumption unchanged.
3. **Stale/cancelled handoff and source guards:** Checked the receipt caller's `request_if` at `tldw_chatbook/UI/Console_Modules/archive_recovery.py:477`, its `is_answerable` predicate at `:89`, the Library request route at `tldw_chatbook/UI/Library_Modules/library_conversations_controller.py:1737` and `app.py:8868`, and the consumer/claim contract at `archive.py:247` and `tldw_chatbook/UI/Navigation/pending_handoff_store.py:279`. The new scheduling preserves request freshness before dispatch; consumer ownership remains exact-claim/screen based, cancellation releases its claim, and newer staged revisions invalidate old activation guards. It does not weaken existing confirmation/write guards.
4. **Draft/identity preservation and zero sends:** Checked the two existing activation paths called by the consumer: `tldw_chatbook/UI/Console_Modules/session.py:2515` and `workspace.py:5846`. They capture the outgoing draft and activate/hydrate the original identity under the consumer predicate. The callback adds no submit/provider/agent call. Checked lazy AgentRunsDB ownership at `tldw_chatbook/Chat/console_runtime.py:3283` to interpret the regression's conditional run-row assertion correctly.
5. **Regression fidelity:** Inspected the submitted review package once. Its changed producer hunk starts and ends mid-function, so read the full `request_conversation_resume` function to judge its nested callback and unchanged guard boundaries; read the shared consumer because the callback invokes it. Inspected the recorded final GREEN log at `/private/tmp/live-open-final-green.log`: **31 passed in 28.22s**, no warnings reported. Inspected the RED evidence at `/private/tmp/live-open-final-red.log`: the new test fails at its first post-request settlement with `exact conversation resume did not reach its mounted destination`, **1 failed in 14.15s**. The report records the exact final regression bytes tested against baseline producer bytes and scoped lint/format/whitespace passes. No tests were repeated: no concrete new concern required another run.

## Issues

### Critical

None.

### Important

None.

### Minor

None.

## Assessment

**Task quality:** Approved.

**Reasoning:** The small producer change closes the absent same-screen lifecycle dispatch while delegating storage, claim settlement and guarded activation to existing code. The production-dispatch regression fails on the baseline and passes with the correction, with direct assertions for the required identity, branch and draft behavior.

The completed whole-branch review was not repeated. Existing capture successor/retry provenance limitations, the Library constructor mismatch, and the two baseline `_submit_console_native_draft` send-harness failures are outside this diff and are not attributed to this fix. No source/task/plan/QA metadata edits, subagents, model calls, server/tmux operations or full-suite runs were performed; this report is the only requested write.
