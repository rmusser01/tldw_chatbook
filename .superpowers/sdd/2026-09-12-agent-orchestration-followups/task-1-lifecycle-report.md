# TASK-15666 headless cleanup implementation report

Status: DONE

## Changes

- Added settled-owner pruning at the beginning of `_conversation_fleet_coordinator`, before settings lookup and the fleet-disabled early return, and outside the admission lock.
- Left `fleet_snapshot()` observational and left `coordinator.prune_terminal()` at its existing later point in next-turn setup.
- Updated retained-owner comments to describe next-turn and lifecycle/action cleanup without claiming immediate release or a settled-owner bound derived solely from live children.
- Added real bridge/service/coordinator tests covering sequential headless cleanup, preservation and cancellation of the current live owner, and preservation of a live survivor across the fleet-disabled early return.
- Appended implementation evidence to TASK-15666. Status and acceptance-criteria closeout remain root-owned pending independent review.

## Red evidence

Command, run before production changes:

```sh
.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py -q -k 'headless_next_turn_prunes_settled_owners_but_keeps_live_owner or fleet_disabled_next_turn_keeps_a_live_survivor_owner'
```

Result: exit 1; 1 failed, 1 passed, 293 deselected, 1 warning in 1.61s. The headless regression expected one retained live owner after turn two but observed two, proving the settled turn-one owner survived the turn boundary. The fleet-disabled live-owner control passed before the source change.

## Green evidence

The same two-test command after the production change exited 0 with 2 passed, 293 deselected, 1 warning in 1.19s.

Targeted amended selection:

```sh
.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py -q -k 'busy_fleet_session_count or survivor or finished_childs_row or fleet_coordinator_factory or headless_next_turn_prunes_settled_owners_but_keeps_live_owner'
```

Result: exit 0; 12 passed, 283 deselected, 1 warning in 2.67s. The known Requests dependency warning and pytest temporary-directory cleanup warnings remained.

## Static evidence

```sh
.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python -m ruff check --statistics tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_agent_bridge.py
```

Result: exit 1 with 47 whole-file findings. Baseline `405dfe2456`, checked with its `pyproject.toml`, had the same 47 findings. A JSON Ruff result filtered to changed lines reported `changed-line violations: 0`.

```sh
.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python -m ruff format --check tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_agent_bridge.py
```

Result: exit 1 with both files requiring whole-file formatting. The same two files required formatting at baseline `405dfe2456`. Ruff's proposed changes outside the existing debt did not include the task's final changed lines.

```sh
git diff --check
```

Result: exit 0 with no output.

## Self-review

- The cleanup call is before the fleet-disabled return, ensuring a headless next turn always reaches it.
- The helper filters owners by `live_subagent_handles()`, so live cancellation ownership survives. The test cancels through the retained owner to verify the capability, rather than checking membership alone.
- The helper uses the existing survivor lock and is called before the separate admission lock, preserving lock separation.
- No settlement callback, new lock, dependency, setting, runtime abstraction, cap, snapshot behavior, or coordinator-handle timing changed.
- The original pre-production red run proves removing the cleanup reproduces the two-owner failure. The executed placement mutation below proves moving cleanup after the kill-switch return fails the disabled-path settled-owner assertion. Unconditional clearing was assessed by inspection only and was not executed as a mutation.

## Commit

Task-scoped commit; its hash is reported in the final handoff.

## Remaining concerns

- Independent scoped review has not yet run, so TASK-15666 remains In Progress and AC #5 remains unchecked.
- Whole-file Ruff/format failures are existing large-file debt; the Ruff count matches baseline and changed-line filtering is clean.

## Review corrections and executed placement mutation

- Added local failure-safe cleanup for every gated stage. Each `finally` releases all relevant gates, performs bounded joins, and asserts that the observed fleet threads terminated.
- Strengthened the fleet-disabled test to create one settled retained owner beside one live retained owner. The disabled third turn must release only the settled owner, retain the live owner, and cancel its real child through that owner.

Temporary mutation applied for verification: moved
`self._prune_settled_fleet_survivors(conversation_id)` from method entry to
immediately after the `if max_live <= 1: return None` branch. Production source
was restored immediately after the focused run.

Command:

```sh
.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py -q -k 'fleet_disabled_next_turn_keeps_a_live_survivor_owner'
```

Mutation result: exit 1; 1 failed, 294 deselected, 1 warning in 0.82s. The intended assertion failed because retained owners were `[settled_owner, live_owner]` instead of `[live_owner]`. The test's `finally` released both gates and its bounded termination assertion completed without masking the intended failure.

After restoring production placement, the final fresh 12-case targeted selection exited 0 with 12 passed, 283 deselected, 1 warning in 2.94s. Changed-line Ruff again reported zero findings and `git diff --check` passed.
