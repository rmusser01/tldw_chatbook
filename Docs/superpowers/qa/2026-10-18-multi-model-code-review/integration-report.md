# TASK-35582 bounded delivery integration

Candidate: `/Users/macbook-dev/.codex/worktrees/task-35582-delivery/tldw_chatbook`, branch `codex/task-35582-delivery`, base `e3d07afa65927b9d477fe9f13af619550da9e005`.

## Conflict choices

- `Docs/Architecture/console-file-authority.md`: retain current dev optional folder index authority plus task captured review authority; append both sections.
- `backlog/docs/lessons-live-verification.md`: retain dev incidents and task incidents; append both sides without rewriting either history.
- `backlog/docs/lessons-testing-evidence.md`: retain dev incidents and task incidents at both conflict sites.
- `tldw_chatbook/Chat/console_agent_bridge.py`: retain both `received_item` (dev Stop/stall accounting) and `call_continuation_target` (task child destination identity) as nonlocals in the streaming consumer.
- `tldw_chatbook/Chat/console_chat_controller.py`: retain the captured-loop direct callback path and the current-dev native-thread identity guard; Textual dispatcher is used only from another thread. No authority gate removed.
- `tldw_chatbook/css/widget_defaults_self.tcss`: regenerate through normal CSS builder from merged Python/source styles; no manual generated CSS resolution.

Only these six resolved conflict files were staged. The existing staged task patch was left in place. No commit/push/PR, external call, credential read, or original-worktree mutation performed.

## Overlap checks

Nine Python owners changed on both task and current dev: local_tool_provider, console_agent_bridge, console_chat_controller, console_chat_store, console_fleet_wake, console_turn_context, chat_screen, registry_service, config.

- Current dev changed no AgentRuns DB owner since the task base; its schema remains 24, and the task additive body-free boundary migration is version 25. Existing goal fields/migrations remain present. No renumbering or schema redesign required.
- Shared LocalToolProvider retains current dev project-index capability/read-floor logic; review-source authorization only accepts existing native read tools and rechecks binding plus deny floor.
- Current dev automatic-primary worker drain remains after wake delivery settlement; task review-boundary eligibility is composed into that existing owner and does not replace its capacity/slot lifetime.
- Store cancellation cleanup order, durable activity read parameter, session change-review root strings/stamps, workspace index notifications/fences, and config credential/write-lock behavior remain current-dev-owned code; task additions are separate seams.
- ChatScreen has one `u` binding and a matching report action. No duplicate direct methods/functions in six overlapping large owners (valid property getter/setter pairs excluded).
- Both conflict Python files compile. Explicit `PYTHONPATH` plus `tldw_chatbook.__file__` asserts imports resolve to delivery tree, while interpreter/dependencies are borrowed from preserved original worktree `.venv`.
- Normal `build_css.py` completed. Two additional generated siblings remain unstaged for root delivery staging: `css/widget_defaults_scoped.tcss` (code-review modal dedent under current builder) and `css/screen_modal_file_notes_history.tcss` (new control-action token copied into generated split). These are expected deterministic builder output.
- `git diff --cached --check` passed.

## Targeted verification

Main targeted command (97 nodes), using the original `.venv/bin/python`, delivery `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, and `pytest -q -p no:cacheprovider`:

- `Tests/Agents/test_code_review_boundary_ledger.py`
- `Tests/Chat/test_console_code_review_continuation.py`
- `Tests/Chat/test_console_agent_bridge.py::test_builtin_child_binds_its_own_native_continuation_under_custom_parent`
- `Tests/Chat/test_console_agent_bridge.py::test_primary_custom_endpoint_keeps_identity_with_selectable_execution_key`
- `Tests/DB/test_agent_runs_db.py::test_schema_version_constant_agrees_with_the_version_table`
- `Tests/UI/test_design_token_governance.py`
- `Tests/UI/test_component_pattern_governance.py`

**PASS: 97 passed in 255.67s (0:04:15), exit 0.** Raw command/output: `integration-tests.log` beside this report. The two exact current-dev bridge Stop/stall controls separately passed (`2 passed in 2.17s`, exit 0), raw log `integration-dev-stop-controls.log`. Both test processes emitted pytest end-of-run cleanup warnings for pre-existing temporary `garbage-*` directories; no product/test failure and no cleanup attempted. No broader feature/provider cohort or full suite.

All six conflict resolutions are staged; `git ls-files -u` is empty. No source edits occurred after the freeze notice. Root is handling the task-local ADR renumber and final generated-sibling staging.

## Scope/risks

No demonstrated API/schema blocker from bounded overlap inspection. Documentation integration blocker: current dev already owns `backlog/decisions/240-mcp-destination-retirement-tools-hub-in-settings.md`, while this patch adds `240-primary-agent-multi-model-code-review.md`; numeric ADR-240 references are ambiguous and require task-local renumbering/link correction before delivery. Root owns documentation/task edits. Final main targeted run passed, 97/97 nodes; separate current-dev Stop controls passed, 2/2. AC24 remains open; no live provider qualification attempted or implied. The task's source-only review boundary, role restrictions, finite accounting and default two-child ceiling remain intact.
