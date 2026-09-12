# Task 1 report: TASK-15666

Status: DONE

## Changes

- Removed `_prune_settled_fleet_survivors(conversation_id)` from `ConsoleAgentBridge.fleet_snapshot()` only. Service lookup and live survivor handle filtering are unchanged.
- Documented that `fleet_snapshot()` is observational, that existing lifecycle/action paths release settled retained owners, and that terminal coordinator handles remain until next-turn pruning.
- Removed the stale controller comment that described the navigation count as prune-on-read.
- Extended the real bridge/controller survivor test to capture and compare retained owners and coordinator handles across live and terminal busy-count reads. Moved the existing survivor owner-cleanup assertion after `live_snapshot()` and verified terminal handles remain retained for next-turn pruning.
- Added TASK-15666 implementation notes. Acceptance criteria remain unchecked and status remains In Progress pending root-owned independent review and final completion.

## Red evidence

Command:

```sh
.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py -q -k 'busy_fleet_session_count_sees_a_session_whose_only_work_is_a_survivor'
```

Result before production changes: exit 1; 1 failed, 292 deselected, 1 warning in 1.17s. The intended terminal-read assertion failed because `_retained_fleet_owners(session.id)` changed from one retained `AgentService` to `[]` after `busy_fleet_session_count()`. Provider setup and the live count succeeded.

## Green evidence

Exact-test command:

```sh
.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py -q -k 'busy_fleet_session_count_sees_a_session_whose_only_work_is_a_survivor'
```

Result after production change: exit 0; 1 passed, 292 deselected, 1 warning in 0.67s.

Targeted-selection command:

```sh
.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py -q -k 'busy_fleet_session_count or survivor or finished_childs_row or fleet_coordinator_factory'
```

Result: exit 0; 10 passed, 283 deselected, 1 warning in 1.85s. Pytest also emitted pre-existing dependency and temporary-directory cleanup warnings.

## Lint and formatting baseline comparison

Current-file Ruff command:

```sh
.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python -m ruff check --statistics tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_agent_bridge.py
```

Result: exit 1 with 235 existing findings; the leading categories were BLE001 (88), UP037 (81), and I001 (14). The same command against the three `HEAD` blobs reported 237 findings, including BLE001 (88), UP037 (81), and I001 (18). A JSON Ruff check filtered to changed lines reported `changed-line violations: 0`.

Formatting command:

```sh
.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python -m ruff format --check tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_agent_bridge.py
```

Result: exit 1, `3 files would be reformatted`. The same command against the three `HEAD` blobs also reported `3 files would be reformatted`; no whole-file reformat was applied because the brief forbids rewriting pre-existing large-file debt.

Whitespace command:

```sh
git diff --check
```

Result: exit 0 with no output.

## Self-review

- Scope is limited to the brief's source, tests, task notes, and this report.
- The behavior change is one removed cleanup call. No dependency, runtime abstraction, setting, guard threshold, ownership, cancellation, conservative cap, or between-turn pruning logic changed.
- Mutation check: restoring the removed cleanup call makes the terminal owner-preservation assertion fail. Removing existing `live_snapshot()` cleanup would make the post-live-snapshot owner assertion fail. Pruning terminal coordinator handles during either read would make the handle equality assertions fail.
- The test uses the real bridge, controller, service, coordinator, and gated provider fake; it does not mock the method under test.

## Commit

Task-scoped commit containing this report; the immutable hash is recorded in
the final handoff because a commit cannot embed its own hash.

## Concerns

- Independent review is root-owned and has not yet run, so TASK-15666 remains In Progress with acceptance criteria unchecked.
- Ruff and formatter failures are pre-existing whole-file baseline debt and did not worsen on changed lines.
