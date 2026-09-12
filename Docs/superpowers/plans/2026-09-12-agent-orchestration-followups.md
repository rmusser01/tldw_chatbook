# Agent orchestration follow-up reliability plan

> **For agentic workers:** Use subagent-driven-development to implement this plan task by task.

**Goal:** Make navigation's busy-session read observational and record the remaining orchestration work against merged dev.

**Architecture:** Keep the conversation-owned coordinator and retained AgentService ownership. Remove retained-owner cleanup from the read-only fleet snapshot; the existing rail, cancellation, and lifecycle paths keep their cleanup behavior. Terminal coordinator handles remain pruned at turn start.

**Tech Stack:** Python, Textual controller, threading, SQLite, pytest.

**Spec:** `backlog/tasks/task-15666 - busy_fleet_session_count-prunes-the-fleet-as-a-side-effect-of-a-read.md` (acceptance criteria), `backlog/decisions/129-fleet-mailbox-and-wake-reliability.md` (existing lifecycle contract).

ADR required: no
ADR path: backlog/decisions/129-fleet-mailbox-and-wake-reliability.md
Reason: restore the documented read-only interface; no new ownership, storage, pruning, or execution policy.

## Global Constraints

- Work only in the existing isolated `agent-orchestration-pr` worktree on `codex/agent-orchestration-followups`; preserve the dirty main checkout.
- Preserve conversation ownership, cancellation authority, conservative execution caps, and between-turn coordinator pruning.
- Use targeted tests only; do not run a full suite or change architecture, import, or CSS guard thresholds.
- Do not add dependencies, new runtime abstractions, or new settings.
- Use the isolated interpreter `.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python`; the shared interpreter can import the dirty main checkout.

### Task 1: Make fleet snapshots observational (TASK-15666)

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`fleet_snapshot`, `_prune_settled_fleet_survivors` documentation).
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (stale `fleet_has_unsettled_children` pruning comment only).
- Test: `Tests/Chat/test_console_agent_bridge.py` (existing survivor/count/cleanup tests).
- Update: the TASK-15666 file above (implementation notes and AC after verified review).

**Interfaces:**
- Consumes `ConsoleChatController.busy_fleet_session_count() -> int`, which calls `fleet_teardown_split()` and the real bridge's `fleet_snapshot(conversation_id) -> list[FleetHandle]`.
- Produces the same counts and snapshots without changing coordinator handles or retained owner membership.
- Existing `live_snapshot`, cancellation, and lifecycle cleanup continue dropping settled owners. `_conversation_fleet_coordinator` continues pruning terminal handles at the next turn start.

- [ ] Extend `test_busy_fleet_session_count_sees_a_session_whose_only_work_is_a_survivor` to capture the real coordinator snapshot and retained owner list before reading the count while the child is live and after releasing its gate and joining it. Assert the count is respectively 1 and 0, and both snapshots remain unchanged. Core assertions:

```python
owners_before = bridge._retained_fleet_owners(session.id)
handles_before = bridge._conversation_fleet_handles(session.id)
assert owners_before
assert controller.busy_fleet_session_count() == expected_count
assert bridge._retained_fleet_owners(session.id) == owners_before
assert bridge._conversation_fleet_handles(session.id) == handles_before
```

  After the terminal read, call `live_snapshot` and assert the owner is released while the terminal handle remains until the next turn. Reuse real bridge/service/coordinator and the existing gated provider fake; do not mock the method under test.
- [ ] Run that exact test before production changes. The intended failure is retained owners disappearing on the terminal busy-count read, not a provider/setup error.
- [ ] Remove the `_prune_settled_fleet_survivors(conversation_id)` call from `fleet_snapshot` only. Keep the service lookup and survivor handle filtering unchanged. Update its and the cleanup helper's documentation to describe the current lifecycle. Remove the controller comment saying the navigation count prunes coordinator state.
- [ ] Update `test_a_survivor_is_visible_and_stoppable_after_its_turn_returns` so its retained-owner cleanup assertion follows `live_snapshot`, the existing cleanup path. The read-only snapshot still returns an empty live fleet after settlement.
- [ ] Run the existing targeted selection:

```sh
.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py -q -k 'busy_fleet_session_count or survivor or finished_childs_row or fleet_coordinator_factory'
```

- [ ] Run changed-line Ruff/lint and formatting checks without reformatting pre-existing large-file debt; record exact commands, failures, and baseline comparisons. Run `git diff --check`.
- [ ] Record implementation notes, red/green evidence, and any limitations. Commit only task-scoped files. Root arranges independent task review and whole-branch review, then sets Done via Backlog CLI once all criteria and checks are satisfied.

## Revalidation and durable inventory

In parallel with this bounded fix, root records the remaining workstream in `backlog/docs/agent-orchestration-followups-2026-09-12.md`. TASK-13215 is independently revalidated against the current interrupt host. Investigation does not authorize speculative code changes or closing an older ticket without evidence. Newly confirmed fixes receive a task plan before implementation.

Preflight: no open PR or remote branch names matched TASK-15666/TASK-13215; current source changes are included in merged PR #2631. Open PR #2427 is broad test cleanup and may overlap older harness tickets, so those require file-level comparison before repair. These checks were made against `origin/dev` at `8ab21ecaf3` on 2026-09-12 UTC.
