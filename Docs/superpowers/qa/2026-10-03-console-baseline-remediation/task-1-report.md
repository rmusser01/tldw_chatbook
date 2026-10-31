# Task 1 remediation report

Status: scoped implementation and targeted verification complete; independent review and controller-owned closure/publication remain outstanding.

## Scope and root causes

BASE: `bc2be106f1cc0409ef283c60c6778ff12c972d7a` on `codex/console-chat-starts`.

The 21 recorded nodes reproduced 3 failures and 18 passes before edits. The remaining failures were stale test doubles, requiring no production changes:

- In `test_child_chain_uses_its_own_exact_first_request_budget` and `test_primary_token_omission_is_delivery_local_when_child_admits`, the count lambdas accepted positional arguments only. The existing `_count_model_messages` interface and its real request-building/budget consumers pass `reasoning_replay` as a keyword (and support `tokenizer_model`). The doubles therefore raised `TypeError` instead of returning their controlled 10/95 counts, and the agent outcomes were `error` rather than `done`.
- The active-leaf case of `test_configuration_and_leaf_writers_block_fork_through_live_publication` held the barrier but implicitly returned `None`. Real `set_active_leaf` checks the persistence boolean and raised `RuntimeError('Conversation cursor change was refused.')`, which the existing worker-failure assertion reported.
- The other 18 historical failure nodes already passed on the current BASE and required no additional changes.

## Owned changes

- `Tests/Chat/test_console_agent_project_instructions.py`: both count doubles explicitly accept the current keyword parameters. Their complementary 95/10 and 10/95 branches, actual request sentinel checks and delivery-local omission assertions are unchanged.
- `Tests/Chat/test_console_chat_fork.py`: capture the genuine `_persist_active_leaf` bound method before installing the barrier, pass through the actual session/message IDs after release, and return its boolean result. Existing barrier, fork-ineligibility, worker-completion, failure and transition-release assertions remain intact.

No production files, dependencies, UI, migration, warning filters, resource thresholds, skip/xfail markers, or controller-owned task/plan/QA files were changed. ADR required: no; this test-only repair preserves existing contracts and task ADR-211 context.

## Verification commands and evidence

Every command below ran from `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`, using the existing source-checkout virtualenv interpreter. Test commands used `TLDW_TEST_GC_EVERY=1` as required by the supplied brief. Each test run used its own basetemp. The command JSON files preserve literal argv and environment; historical QA logs were read and remain unchanged.

### format-snapshot

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/terminal_qualification/format_ratchet.py snapshot --base bc2be106f1cc0409ef283c60c6778ff12c972d7a --output .superpowers/sdd/2026-10-03-console-baseline-remediation/format-baseline.json --path Tests/Chat/test_console_agent_project_instructions.py --path Tests/Chat/test_console_chat_fork.py
```

Exit 0. Both baseline files have zero inherited formatter debt; Ruff 0.16.6.

Output: `.superpowers/sdd/2026-10-03-console-baseline-remediation/format-snapshot.log`. Exact command: `.superpowers/sdd/2026-10-03-console-baseline-remediation/format-snapshot-command.json`.

### recorded-red

```sh
TLDW_TEST_GC_EVERY=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_prompt_queue_coordinator.py::test_close_tombstones_before_cancel_and_never_starts_next_prompt Tests/UI/test_console_runtime_ownership.py::test_runtime_owned_custody_tracks_only_lifetime_handles Tests/UI/test_console_runtime_ownership.py::test_runtime_tombstones_before_shutdown_and_disposes_via_to_thread Tests/UI/test_console_runtime_ownership.py::test_persistent_attach_sync_failure_has_bounded_backoff_and_resume_retry 'Tests/UI/test_console_runtime_ownership.py::test_reconciled_view_keeps_each_live_poll_reason_and_one_timer[wake]' Tests/UI/test_console_runtime_ownership.py::test_opening_console_during_a_headless_delivery_arms_the_poll Tests/UI/test_console_launch_wake.py::test_a_launch_into_console_delivers_without_stealing_the_active_tab Tests/UI/test_console_launch_wake.py::test_a_launch_with_no_marks_constructs_nothing_and_reads_once 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_invalid_ownership[bad_role]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_invalid_ownership[cross_conversation]' Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_considers_only_checkpoint_owners_on_the_selected_active_lineage 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_malformed_or_mismatched_checkpoint_identity[assistant_message_id]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_state_cas_requires_every_expected_owner_predicate[assistant_state]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[terminal_content]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[sync_intent]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[checkpoint_delete]' Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_is_atomic_and_returns_committed_proof Tests/Chat/test_console_agent_project_instructions.py::test_child_chain_uses_its_own_exact_first_request_budget Tests/Chat/test_console_agent_project_instructions.py::test_primary_token_omission_is_delivery_local_when_child_admits 'Tests/Chat/test_console_chat_fork.py::test_configuration_and_leaf_writers_block_fork_through_live_publication[active_leaf]' Tests/Agents/test_agent_chat_create_tools.py::test_new_chat_schema_shape -q --tb=short --basetemp .superpowers/sdd/2026-10-03-console-baseline-remediation/pytest-recorded-red
```

Exit 1: 3 failed, 18 passed in 29.62s. No warning summary.

Output: `.superpowers/sdd/2026-10-03-console-baseline-remediation/recorded-red.log`. Exact command: `.superpowers/sdd/2026-10-03-console-baseline-remediation/recorded-red-command.json`.

### recorded-green

```sh
TLDW_TEST_GC_EVERY=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_prompt_queue_coordinator.py::test_close_tombstones_before_cancel_and_never_starts_next_prompt Tests/UI/test_console_runtime_ownership.py::test_runtime_owned_custody_tracks_only_lifetime_handles Tests/UI/test_console_runtime_ownership.py::test_runtime_tombstones_before_shutdown_and_disposes_via_to_thread Tests/UI/test_console_runtime_ownership.py::test_persistent_attach_sync_failure_has_bounded_backoff_and_resume_retry 'Tests/UI/test_console_runtime_ownership.py::test_reconciled_view_keeps_each_live_poll_reason_and_one_timer[wake]' Tests/UI/test_console_runtime_ownership.py::test_opening_console_during_a_headless_delivery_arms_the_poll Tests/UI/test_console_launch_wake.py::test_a_launch_into_console_delivers_without_stealing_the_active_tab Tests/UI/test_console_launch_wake.py::test_a_launch_with_no_marks_constructs_nothing_and_reads_once 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_invalid_ownership[bad_role]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_invalid_ownership[cross_conversation]' Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_considers_only_checkpoint_owners_on_the_selected_active_lineage 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_malformed_or_mismatched_checkpoint_identity[assistant_message_id]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_state_cas_requires_every_expected_owner_predicate[assistant_state]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[terminal_content]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[sync_intent]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[checkpoint_delete]' Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_is_atomic_and_returns_committed_proof Tests/Chat/test_console_agent_project_instructions.py::test_child_chain_uses_its_own_exact_first_request_budget Tests/Chat/test_console_agent_project_instructions.py::test_primary_token_omission_is_delivery_local_when_child_admits 'Tests/Chat/test_console_chat_fork.py::test_configuration_and_leaf_writers_block_fork_through_live_publication[active_leaf]' Tests/Agents/test_agent_chat_create_tools.py::test_new_chat_schema_shape -q --tb=short --basetemp .superpowers/sdd/2026-10-03-console-baseline-remediation/pytest-recorded-green
```

Exit 0: 21 passed in 32.18s. No warning summary.

Output: `.superpowers/sdd/2026-10-03-console-baseline-remediation/recorded-green.log`. Exact command: `.superpowers/sdd/2026-10-03-console-baseline-remediation/recorded-green-command.json`.

### owners-green

```sh
TLDW_TEST_GC_EVERY=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_agent_chat_create_tools.py Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py Tests/Chat/test_console_agent_project_instructions.py Tests/Chat/test_console_chat_fork.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/UI/test_console_launch_wake.py Tests/UI/test_console_runtime_ownership.py -q --tb=short --basetemp .superpowers/sdd/2026-10-03-console-baseline-remediation/pytest-owners-green
```

Exit 0: 443 passed, 7 warnings in 348.15s (0:05:48), all seven complete affected modules together. No skips/xfails or failures reported.

Output: `.superpowers/sdd/2026-10-03-console-baseline-remediation/owners-green.log`. Exact command: `.superpowers/sdd/2026-10-03-console-baseline-remediation/owners-green-command.json`.

### ruff-worktree

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check --select E9,F63,F7,F82 Tests/Chat/test_console_agent_project_instructions.py Tests/Chat/test_console_chat_fork.py
```

Exit 0: All checks passed.

Output: `.superpowers/sdd/2026-10-03-console-baseline-remediation/ruff-worktree.log`. Exact command: `.superpowers/sdd/2026-10-03-console-baseline-remediation/ruff-worktree-command.json`.

### ratchet-worktree

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-baseline-remediation/format-baseline.json
```

Exit 0: no formatter debt or overlap introduced.

Output: `.superpowers/sdd/2026-10-03-console-baseline-remediation/ratchet-worktree.log`. Exact command: `.superpowers/sdd/2026-10-03-console-baseline-remediation/ratchet-worktree-command.json`.

### whitespace-worktree

```sh
git diff --check
```

Exit 0: no whitespace errors.

Output: `.superpowers/sdd/2026-10-03-console-baseline-remediation/whitespace-worktree.log`. Exact command: `.superpowers/sdd/2026-10-03-console-baseline-remediation/whitespace-worktree-command.json`.

## Mutation-based self-review

These are mental production mutations mapped to existing real-behavior assertions; no production mutation was applied:

- Reuse the parent instruction budget for the child, always admit automatic rows, or always omit them: the two complementary tests assert the actual outbound sentinel-row counts `[1, 0, 1]` and `[0, 1, 0]` across parent/child/parent requests. These literals are independent of the implementation's admission logic.
- Make primary omission global, prevent the child from independently admitting, or lose the omission reason: the second test requires an empty `global_outcomes`, the literal primary delivery code `omitted_token_budget`, a child sentinel row and empty parent sentinel rows.
- Fail to complete the real parent/child run: both require `RUN_DONE`; the scripted provider records requests made by the actual agent service, rather than asserting that a mock counter exists.
- Allow fork admission while the held persistence mutation is live, or release its ownership guard before persistence returns: the threaded test requires `fork_eligibility(...).eligible is False` while the barrier is held.
- Turn successful cursor persistence into refusal or lose the persistence return: the real mutation must finish with `failures == []`. The preserved RED run demonstrated this exact break when the fixture returned `None`.
- Leak the transition or leave the writer blocked after release: the same test requires the thread to finish and the session's transition entry to be absent.

The count doubles accept only the current named keywords rather than arbitrary `**kwargs`; unrelated unexpected keyword changes remain visible. The barrier delegates to the genuine persistence method and keeps refusal semantics real. Final diff self-review found exactly the intended two keyword-signature repairs and one boolean-preserving delegation, with no assertion changes.

## Warnings, concerns and limits

Combined-owner warnings were preserved without suppression: six `<unknown>` `SyntaxWarning` entries for invalid escape sequence `\ ` in the AST-reading construction-site test, and the session sentinel `UserWarning` from `Tests/conftest.py:529`: descriptors grew by 751 (start=12, end=763, limit=200), despite `TLDW_TEST_GC_EVERY=1`. No threshold or warning-filter change was made. The 21-node RED and GREEN logs had no warning summary. The owner run completed normally; its slowest test took 31.84s, with no timeout or stall. This test repair does not remediate the remaining descriptor-growth warning or establish repository-wide SQLite/descriptor cleanup. Committed-HEAD static evidence is added below. No full repository sweep was requested or run. Independent review, QA/task updates, PR destination choice and publication remain controller-owned.

## Commit and committed-HEAD checks

Implementation commit: `1c1a7e3566a01b2d0b963673272eae562025ef25`. Only the two owned test files were committed (5 insertions, 3 deletions). SDD report and raw evidence remain in the task scratch directory for controller packaging. No Backlog status change was made.

Commit command: `git -c gc.auto=0 commit -m 'test: restore Console budget and fork fixture contracts'`.

### head-source-match

```sh
git diff --exit-code HEAD -- Tests/Chat/test_console_agent_project_instructions.py Tests/Chat/test_console_chat_fork.py
```

Exit 0. Output: `.superpowers/sdd/2026-10-03-console-baseline-remediation/head-source-match.log`. Exact command: `.superpowers/sdd/2026-10-03-console-baseline-remediation/head-source-match-command.json`.

### ruff-head

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check --select E9,F63,F7,F82 Tests/Chat/test_console_agent_project_instructions.py Tests/Chat/test_console_chat_fork.py
```

Exit 0. Output: `.superpowers/sdd/2026-10-03-console-baseline-remediation/ruff-head.log`. Exact command: `.superpowers/sdd/2026-10-03-console-baseline-remediation/ruff-head-command.json`.

### ratchet-head

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-baseline-remediation/format-baseline.json --head 1c1a7e3566a01b2d0b963673272eae562025ef25
```

Exit 0. Output: `.superpowers/sdd/2026-10-03-console-baseline-remediation/ratchet-head.log`. Exact command: `.superpowers/sdd/2026-10-03-console-baseline-remediation/ratchet-head-command.json`.

### whitespace-head

```sh
git diff --check bc2be106f1cc0409ef283c60c6778ff12c972d7a 1c1a7e3566a01b2d0b963673272eae562025ef25 -- Tests/Chat/test_console_agent_project_instructions.py Tests/Chat/test_console_chat_fork.py
```

Exit 0. Output: `.superpowers/sdd/2026-10-03-console-baseline-remediation/whitespace-head.log`. Exact command: `.superpowers/sdd/2026-10-03-console-baseline-remediation/whitespace-head-command.json`.

The committed-source equality command proves the scoped Ruff invocation checks the same bytes as committed HEAD. The ratchet explicitly measured the commit against the immutable BASE snapshot; both baseline and final files have zero formatter debt. Final `git status --short` was empty. No test code changed after the recorded GREEN and combined owner runs, so duplicate test runs were not performed.
