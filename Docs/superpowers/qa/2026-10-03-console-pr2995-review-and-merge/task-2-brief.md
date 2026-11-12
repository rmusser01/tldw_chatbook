### Task 2: Repair final review close-ticket and child-tool documentation findings

**Files:**
- Modify: tldw_chatbook/Chat/console_chat_controller.py (finalize_session_close only).
- Modify: Docs/User_Guide/console/agent-runs-and-tools.md (Chat creation tools paragraph).
- Test: Tests/Chat/test_console_runtime_shutdown.py (existing meaningful controls).

**Interfaces:**
- Consumes: runtime-owned ConsoleSessionCloseTicket, current generation and remembered session grant.
- Produces: stale/mismatched/generation-refused tickets retain live grants; valid session close clears grants. Accurate child tool documentation.

- [ ] Read final-branch-review.md. Reproduce existing rejected-close grant controls and the valid-ticket cleanup control on unchanged source before repair; preserve RED evidence.
- [ ] Remove only the new prevalidation chat-create grant pop/comment at finalize_session_close. Preserve the existing validated post-ticket/generation grant cleanup and all runtime teardown behavior.
- [ ] Correct the guide: child agents may fork a chat or create a same-workspace draft; child requests require fresh confirmation. Casual destinations and bounded starts remain primary-only. Verify current prepare/bridge contracts before wording.
- [ ] Run complete Tests/Chat/test_console_runtime_shutdown.py with private profile and exact recorded command; no full suite/new markers/suppression. If baseline setup fails, diagnose/verify immutable base before repairing unrelated expectations.
- [ ] Run fatal Ruff, formatter ratchet for the touched Python file against recorded BASE, and git diff --check. Commit only these two owned paths; report exact SHA, results and retained warnings.
- [ ] Scoped independent re-review of these two findings and this fix diff; no second whole-branch review or repeated tests without a concrete doubt.

## External review and publication

- [ ] Independent task review of Task1, then whole-branch review of the rebased PR. Package diffs before dispatch; reviewers do not repeat completed test runs.
- [ ] Push with force-with-lease pinned to the verified old remote head; update the PR description and mark ready to enable reviews.
- [ ] Retrieve all PR issue/review comments and Qodo suggestions. Verify each actionable finding; dispatch concrete follow-up fixes with reproductions and covering tests. Reply and resolve addressed threads with commit/test evidence.
- [ ] Wait for current-head checks and Qodo completion. If future waiting is required, create a quiet thread heartbeat that continues review fixes and merge under the user's authorization.
- [ ] Fetch latest dev again; update/requalify if it advanced. Merge using normal repository gates and exact reviewed head, without admin bypass.
- [ ] Confirm merged state/SHA, close current Backlog criteria, archive this plan's evidence and remove only its disposable SDD directory after completion.
