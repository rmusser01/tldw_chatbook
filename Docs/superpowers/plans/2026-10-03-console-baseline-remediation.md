# Console baseline remediation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every recorded baseline test failure before the authorized Console chat-start PR.

**Architecture:** Keep existing production contracts intact. Repair stale test doubles at their dependency boundaries and verify actual token admission and fork publication behavior. The original chat-start implementation and its review records remain the foundation.

**Tech Stack:** Python 3.12, pytest, real SQLite, Textual 8, Ruff.

**Spec:** `backlog/tasks/task-34215.1 - Clear-recorded-Console-baseline-test-failures-before-PR.md`; existing feature authority is `Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md` and ADR-211.

ADR required: no
ADR path: N/A
Reason: Repairing test doubles preserves runtime, database, ownership and budget policy. ADR-211 continues to govern the unchanged feature.

## Global Constraints

- Repair every one of the 21 recorded failing nodes listed below; run all seven complete owner modules together.
- Preserve observable outcome, projected-request, token-budget, lineage and publication assertions. Do not skip, xfail, broaden warning filters, raise FD limits or relax production guards.
- Changes belong in the managed worktree `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`; retain the original checkout and user files.
- Reuse `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`. Run targeted owner checks only. Worktree writes require scoped `exec_command require_escalated` because its sandbox root was not extended.
- The existing `fork_chat` contract and automatic allowance/receipt/provenance invariants remain unchanged.
- No dependencies, settings, migrations or production behavior changes are planned. Investigate any unexpected owner failure before proposing a production change; update task scope before implementing such a change.
- Record warnings accurately; passing tests are not a claim of general resource cleanup.
- User authorized push and PR, plus baseline fixes. PR destination is pending because the original local base is not on GitHub; do not publish until that choice is resolved.

---

### Task 1: Restore the baseline test contracts and verify all recorded owners

**Files:**
- Modify: `Tests/Chat/test_console_agent_project_instructions.py`
- Modify: `Tests/Chat/test_console_chat_fork.py`
- Controller-owned: the task, this plan and final QA summary/evidence.

**Interfaces:**
- Consumes: `_count_model_messages(messages, model, provider, *, reasoning_replay=None, tokenizer_model=None) -> int` and `ConsoleChatStore._persist_active_leaf(session_id, message_id, *, content_safe_diagnostic=False) -> bool`.
- Produces: passing real token-admission/projection and fork-publication tests under those existing interfaces; no new production API.

- [x] **Step 1: Read the task, preserved failure traces and current dependency owners.** Confirm both token stubs reject the existing keyword and the fork blocking stub loses the persistence return. The tests must continue catching incorrect parent/child instruction admission and publication during mutation.

- [x] **Step 2: Reproduce every recorded failing node before edits.** Write `baseline-nodes.json` from the inventory below into this plan's scratch directory. Run using the interpreter and a unique basetemp, recording exact command and output:

```python
import json, subprocess, os
from pathlib import Path
scratch = Path('.superpowers/sdd/2026-10-03-console-baseline-remediation')
nodes = json.loads((scratch / 'baseline-nodes.json').read_text())
command = ['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python', '-m', 'pytest', *nodes, '-q', '--tb=short', '--basetemp', str(scratch / 'pytest-recorded-red')]
environment = dict(os.environ, TLDW_TEST_GC_EVERY='1')
with (scratch / 'recorded-red.log').open('w') as output:
    result = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
print(result.returncode)
```

- [x] **Step 3: Repair only confirmed fixture mismatches.** Keep the two return branches and sentinel-row assertions intact. Accept the current explicit keyword parameters in the controlled count stubs:

```python
lambda messages, *_args, reasoning_replay=None, tokenizer_model=None: (
    95 if 'sub-agent' in str(messages[0].get('content', '')).lower() else 10
)
```

The complementary primary-omission test retains its existing 10/95 branch. For fork publication, preserve the genuine persistence result after the held barrier:

```python
persist_leaf = store._persist_active_leaf

def blocking_leaf(session_id, message_id):
    entered.set()
    assert release.wait(2)
    return persist_leaf(session_id, message_id)
```

- [x] **Step 4: Run all 21 nodes again, then all seven complete owners in one targeted command.** Use the same subprocess pattern with `recorded-green.log` and `owners-green.log`, separate basetemps, and the full module list below. Investigate any other recorded or owner failure through root cause and preserve its RED/GREEN evidence.

- [x] **Step 5: Verify scoped static checks.** Before changes, snapshot each amended old file against the task's recorded BASE using `scripts/terminal_qualification/format_ratchet.py snapshot`. Run Ruff `check --select E9,F63,F7,F82`, its exact changed-file formatter ratchet, and `git diff --check`. Verify ratchets against committed HEAD after the commit.

- [x] **Step 6: Self-review, report and commit only owned fixes.** The report must contain exact commands, output summaries and log paths; name the production mutations the tests catch. No helpers/reviewer agents, full sweep, merge, push or PR. Controller dispatches independent review and owns publication/QA/task closure.

## Recorded baseline inventory

The four source logs are under `Docs/superpowers/qa/2026-10-02-console-chat-starts/verification-logs/`: `baseline-task-1.log`, `task2-baseline.log`, `task2-fix1-budget-baseline.log`, and `final-fix-baseline.log`. Logs are historical evidence and are not rewritten.

```json
[
  "Tests/Chat/test_console_prompt_queue_coordinator.py::test_close_tombstones_before_cancel_and_never_starts_next_prompt",
  "Tests/UI/test_console_runtime_ownership.py::test_runtime_owned_custody_tracks_only_lifetime_handles",
  "Tests/UI/test_console_runtime_ownership.py::test_runtime_tombstones_before_shutdown_and_disposes_via_to_thread",
  "Tests/UI/test_console_runtime_ownership.py::test_persistent_attach_sync_failure_has_bounded_backoff_and_resume_retry",
  "Tests/UI/test_console_runtime_ownership.py::test_reconciled_view_keeps_each_live_poll_reason_and_one_timer[wake]",
  "Tests/UI/test_console_runtime_ownership.py::test_opening_console_during_a_headless_delivery_arms_the_poll",
  "Tests/UI/test_console_launch_wake.py::test_a_launch_into_console_delivers_without_stealing_the_active_tab",
  "Tests/UI/test_console_launch_wake.py::test_a_launch_with_no_marks_constructs_nothing_and_reads_once",
  "Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_invalid_ownership[bad_role]",
  "Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_invalid_ownership[cross_conversation]",
  "Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_considers_only_checkpoint_owners_on_the_selected_active_lineage",
  "Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_malformed_or_mismatched_checkpoint_identity[assistant_message_id]",
  "Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_state_cas_requires_every_expected_owner_predicate[assistant_state]",
  "Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[terminal_content]",
  "Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[sync_intent]",
  "Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[checkpoint_delete]",
  "Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_is_atomic_and_returns_committed_proof",
  "Tests/Chat/test_console_agent_project_instructions.py::test_child_chain_uses_its_own_exact_first_request_budget",
  "Tests/Chat/test_console_agent_project_instructions.py::test_primary_token_omission_is_delivery_local_when_child_admits",
  "Tests/Chat/test_console_chat_fork.py::test_configuration_and_leaf_writers_block_fork_through_live_publication[active_leaf]",
  "Tests/Agents/test_agent_chat_create_tools.py::test_new_chat_schema_shape"
]
```

## Complete affected owner modules

- `Tests/Agents/test_agent_chat_create_tools.py`
- `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`
- `Tests/Chat/test_console_agent_project_instructions.py`
- `Tests/Chat/test_console_chat_fork.py`
- `Tests/Chat/test_console_prompt_queue_coordinator.py`
- `Tests/UI/test_console_launch_wake.py`
- `Tests/UI/test_console_runtime_ownership.py`
