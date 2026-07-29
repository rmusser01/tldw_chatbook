---
id: TASK-1231
title: File denial teaches recovery; approvals pre-flight the roots check
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 09:30'
updated_date: '2026-07-28 18:12'
labels:
  - console
  - agents
  - ux
  - uat
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expert UAT F3: on an unbound workspace (every fresh install), an approved read_file fails with "outside every allowed root" -- truncated in the transcript, no route to the fix (create a workspace, bind a folder in Settings > Workspaces, work in that workspace; Default cannot hold bindings). The model then keeps calling other root-gated file tools (captures show list_directory("."), not a retry of the identical path -- corrected post-Qodo-review, see Implementation Notes) and the user is asked to approve another doomed request with no new information, until the loop guard kills the run with jargon. First-run users cannot succeed at file access and nothing tells them why or what to do.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The outside-allowed-roots tool error appends the concrete recovery route (Settings > Workspaces folder binding; workspace-scoped sessions).
- [x] #2 The approval card pre-flights the roots check for file tools and warns when the path will be rejected regardless of approval (never auto-denies; the user can still approve).
- [x] #3 (Corrected post-Qodo-review; the original text claimed a retried IDENTICAL request, which the captures do not show -- see Implementation Notes) Post-denial approval fatigue is addressed by informed consent, not suppression: every approval ask in the same run for a path that fails the roots check carries the AC#2 pre-flight warning (whichever root-gated file tool it is -- the observed follow-up was list_directory("."), a DIFFERENT tool than the failed read_file, not a retry of the same path), and the tool error itself (AC#1) names the recovery route. No history-based auto-suppression of a repeated ask is required or implemented, since that would be auto-denial by another name.
- [x] #4 Loop-guard termination copy is user-comprehensible.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Utils/path_validation.py validate_path_multi: append a concrete recovery-route sentence to the outside-every-allowed-root ValueError, placed BEFORE the (open-ended-length) consulted-roots list so it survives the Console transcript step-marker 160-char truncation.
2. Agents/agent_runtime.py run_agent_loop: replace the "loop detected: ... N-cycle (Rx)" jargon in the STEP_ERROR summary with user-comprehensible copy naming the tool(s) and repeat count; keep the technical period/repeats detail as a debug log line.
3. Agents/mcp_tool_provider.py MCPPendingCall: add a path_precheck_failed bool field (default False).
4. Tools/file_operation_tools.py: add path_precheck_failed(tool_name, args) -- pre-flights the SAME allowed_file_roots/validate_path_multi check for read_file/list_directory/write_file, fail-closed to False on any error; never gates, only informs.
5. Chat/console_chat_controller.py build_tool_review_hook: compute path_precheck_failed per builtin file-tool row and thread it through the approval payload dict.
6. Widgets/Chat_Widgets/chat_approval_card.py: render a warning suffix on the row header when path_precheck_failed is set (combinable with the existing risk_floored/config_changed badges).
7. Correct report.md F3 and this tasks AC#3 per the Qodo finding (real observed follow-up ask was list_directory("."), a different tool, not an identical retry).
8. TDD: failing tests first for the error-copy unit test, the approval-payload precheck (controller-level + card-header-level), and the loop-guard message; then implement.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: four small, independently-testable changes plus one evidence correction, all landing together since they share one root cause (an unbound Default workspace denial with no recovery path).

AC1 -- Utils/path_validation.py: validate_path_multi's "outside every allowed root" ValueError now appends ROOT_DENIAL_RECOVERY_HINT (a Settings > Workspaces recovery sentence), placed BEFORE the consulted-roots list. Rationale for the ordering: the Console transcript's live tool-step marker (console_agent_bridge._STEP_MARKER_RESULT_LIMIT, 160 chars) truncates a tool's error preview, and the pre-fix UAT observed the consulted-roots list alone already ate the whole budget ("... (+64 chars)", no recovery text at all). Putting the recovery sentence first means it is the LAST thing truncation can reach, not the first.

AC2 -- Tools/file_operation_tools.py adds path_precheck_failed(tool_name, args): pre-flights the same allowed_file_roots/validate_path_multi check read_file/list_directory/write_file run at dispatch, fails closed to False on any error, and is WARN-ONLY (never gates or auto-denies). Agents/mcp_tool_provider.MCPPendingCall gained a path_precheck_failed bool field (default False, so MCP rows and non-file builtins are untouched). Chat/console_chat_controller.build_tool_review_hook computes it per builtin file-tool row and threads it through the approval payload dict. Widgets/Chat_Widgets/chat_approval_card._format_row_header appends a " -- path outside allowed folders; will fail even if approved" suffix when set, combinable with the existing risk_floored/config_changed badges.

AC3 (corrected) -- Qodo (external review) caught that the original AC#3 and report.md F3 both claimed the model retried "the identical path" after a read_file denial, generating a repeat approval ask for the SAME request. The actual captures show the next ask was list_directory("."), a DIFFERENT root-gated tool -- not a retry of the identical path. Corrected report.md's F3 paragraph inline and reworded this task's AC#3: the real phenomenon is post-denial approval fatigue across SUCCESSIVE gated asks (not one identical retry), and AC2's per-row pre-flight warning already covers every such ask since it is computed fresh per row/call. No history-based ask-suppression was implemented -- that would be auto-denial by another name; the fix is informed consent (AC1 + AC2 together), not blocking.

AC4 -- Agents/agent_runtime.py run_agent_loop: the STEP_ERROR summary for a detected cycle no longer reads "loop detected: X repeated in a 1-cycle (3x)". Period-1 trips now read "Agent stopped: it kept calling X with the same arguments (N times) without making progress."; period>1 trips read "Agent stopped: it kept repeating the same sequence of tool calls (X, Y) without making progress." The technical period/repeats detail is preserved as a logger.debug() line for anyone actually debugging the cycle detector.

Tests: new/updated unit tests in Tests/Utils/test_path_validation_multi.py (recovery hint present + precedes the consulted-root list), Tests/Chat/test_console_chat_controller.py (path_precheck_failed True/False/scope-guard on build_tool_review_hook, mirroring the existing builtin-review-hook test pattern), Tests/UI/test_chat_approval_card.py (_format_row_header warning suffix, combinable with risk_floored), and Tests/Agents/test_agent_runtime.py (loop-guard summary text updated for both period-1 and period>1 cycles). Full gate run: Tests/Agents/test_builtin_provider_workspace_binding.py + test_agent_runtime.py, Tests/Tools/test_file_tool_sandbox.py + test_file_tools_workspace_roots.py + test_glob_grep_files.py, Tests/Utils/test_path_validation*.py, Tests/Chat/test_console_chat_controller.py, Tests/UI/test_chat_approval_card.py + test_console_mcp_approval.py (342 passed; 2 pre-verified pre-existing failures unrelated to this change -- CSS-geometry zero-size assertion and an MCP execution-log error-string assertion, both reproduced identically on unmodified HEAD via git stash), then Tests/UI/test_console_parallel_runs.py (28 passed).

Files changed: tldw_chatbook/Utils/path_validation.py, tldw_chatbook/Agents/agent_runtime.py, tldw_chatbook/Agents/mcp_tool_provider.py, tldw_chatbook/Tools/file_operation_tools.py, tldw_chatbook/Chat/console_chat_controller.py, tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py, Docs/superpowers/qa/fleet-ux-expert-review-2026-07-28/report.md (F3 correction), plus the test files listed above.

Round 1 review fixes (two Criticals, both computed against the merged code):

CRITICAL 1 -- path_precheck_failed resolved the WRONG workspace. It called allowed_file_roots outside any run_workspace scope, so it fell back to registry.get_active_workspace() (the UI's active workspace) rather than the reviewed run's own workspace -- build_tool_review_hook's review provider was even built without a workspace_id at all. Fixed by: (1) Tools/file_operation_tools.path_precheck_failed gained a workspace_id keyword-only param and now wraps its allowed_file_roots call in workspace_file_roots.run_workspace(workspace_id), mirroring BuiltinToolProvider.invoke's own binding around the real dispatch call; (2) build_tool_review_hook gained a workspace_id keyword-only param threaded into every path_precheck_failed call; (3) _run_agent_reply now resolves self.store.session_workspace_id(session_id) (same lookup ConsoleAgentBridge.run_reply already makes for dispatch) and passes it through. New regression tests use a REAL two-workspace registry (ws-a bound to folder A, ws-b bound to folder B and set ACTIVE) and assert the precheck follows the RUN's workspace (ws-a) in both directions, not whatever is active.

CRITICAL 2 -- the recovery hint never survived transcript truncation. ROOT_DENIAL_RECOVERY_HINT (169 chars) plus the tool/path prefix routinely exceeded the transcript's 160-char tool-step-marker budget before "Settings > Workspaces" ever appeared -- measured: for every realistic 45-60-char path across all three file tools, the phrase was fully truncated away. Fixed by splitting the recovery text in two: ROOT_DENIAL_RECOVERY_POINTER, an ultra-short lead ("Fix: Settings > Workspaces -- create a workspace + bind a folder.", 67 chars) placed FIRST -- before the path is even repeated -- and the fuller ROOT_DENIAL_RECOVERY_HINT explanation after it (which, along with the path and consulted-roots list, may still be truncated away without losing the actionable route). New parametrized test runs the REAL composed tool error through the actual console_agent_bridge._truncate_step_text/_STEP_MARKER_RESULT_LIMIT pipeline for 40-60-char paths across all three tool-name prefixes and asserts the pointer survives.

Minors also fixed: path_precheck_failed gained a roots_cache keyword-only param (a fresh dict built once per review_tool_calls invocation, never reused across turns) memoizing allowed_file_roots across every file-tool row in one batch. _agent_failure_visible_copy now skips the "Agent run stuck: " lead-in when the STEP_ERROR summary already reads as a complete sentence (the loop-guard's own "Agent stopped: ..." copy), avoiding "Agent run stuck: Agent stopped: ...".

New/updated tests: Tests/Chat/test_console_chat_controller.py (two-workspace precheck regression tests, _agent_failure_visible_copy double-lead-in tests), Tests/Utils/test_path_validation_multi.py (pointer-precedes-everything ordering test, real-pipeline truncation-survival parametrized test). Full gate re-run: same suites as round 0 -- 352 passed (up from 342; +10 new tests), same 2 pre-existing failures (CSS-geometry, MCP execution-log); Tests/UI/test_console_parallel_runs.py 28 passed.
<!-- SECTION:NOTES:END -->
