---
id: TASK-42513
title: Thirty-two Console trace tests are red on dev in five files no PR lane runs
status: To Do
assignee: []
created_date: '2026-11-24 14:29'
labels:
  - console
  - trace-capture
  - tests
  - ci-lanes
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Running `Tests/Chat/test_console_trace_*.py` at `-n 4` on dev `578222f272` gives 34 failures, spread over six files: `sidecar_replay` (21), `sidecar_redaction` (5), `proprietary_settlement` (4), `current_turn_transforms` (2), `graph_gc` (1) and `project_turn_service` (1). No PR lane runs any of the 34 nodes: five of the files are on no lane, and the only `graph_gc` node on a lane is a different one (`test_stale_mark_never_sweeps_after_concurrent_reachability_change`, console-p0-gate). So PR CI stays green while these stay red, and the trace behaviour they guard (private-history replay, redaction, proprietary settlement, GC privacy, project-turn validation) has no working regression check.

TASK-32032 already tracks the two `current_turn_transforms` failures (`test_next_send_after_failed_transformed_call[error]` and `[stopped]`; AC5 reopened 2026-11-01). This task covers the other 32.

Evidence (local macOS only, not yet seen on Linux CI, because no lane runs them): a detached dev `578222f272` worktree, through the round's scratch-profile wrapper (isolated `HOME`/XDG/`TLDW_CONFIG_PATH`, no keyring, no provider), 2026-11-24, load 5-6:

    run.sh <dev wt> <label> pytest -n 4 -q -p no:cov Tests/Chat/test_console_trace_sidecar_replay.py Tests/Chat/test_console_trace_sidecar_redaction.py Tests/Chat/test_console_trace_proprietary_settlement.py Tests/Chat/test_console_trace_current_turn_transforms.py Tests/Chat/test_console_trace_graph_gc.py Tests/Chat/test_console_trace_project_turn_service.py --tb=short

gives `34 failed, 79 passed`, the same 34 ids each time. Run alone (only the 34 ids, at `-n 4` and serially), all 34 still fail, so they do not depend on order.

Failure classes, by first `E` line:

1. **Test harness: the config sandbox trap (30 tests: all 21 `sidecar_replay`, all 5 `sidecar_redaction`, all 4 `proprietary_settlement`).** The send is refused with `visible_copy='Hooks unavailable; review or disable hooks before sending.'` (or a follow-on `assert (False)` / `IndexError` on the missing call). A temporary traceback in `hook_admission_reason` (`Chat/console_automatic_submission.py`) shows the real error: `read_hooks_config_snapshot()` -> `_config_write_lock` -> `raw_participants._register_raw_participant` raises `RecoveryRequired: raw_source_selection_changed`, which the hook gate turns into "Hooks unavailable". A temporary probe at that raise recorded exactly these 30 tests (setup and call) and none of the other four. These tests are sandboxed (no `bootstrap_profile` marker) but read config in-process on the hook-admission path. Per `lessons-testing-evidence.md` ("A Tests/UI suite that errors at setup locally can still run, under the bootstrap profile"), that is the per-test config sandbox meeting a config module registered for the collection-time profile: the tests would fail in any lane that ran them plain, not only on this machine.
   - With a throwaway `-p` plugin that marks every item `bootstrap_profile` (deleted after), 26 of these 30 pass. Behind the trap, 4 `sidecar_replay` cases still fail: `test_capture_preserves_selected_continuation_and_successor[True-False]`, `[True-True]` and `test_capture_preserves_selected_displayable_thinking[True-replay_harness0]`, `[True-replay_harness1]`, each `assert 'alias request' == 'expanded request'` (lines 244 and 282). Four other cases error at teardown under the probe with `ValueError: Resolve pending dispatch before changing conversation branches.` Whether those 8 are product drift or an artefact of the probe's shared profile is not established.
2. **Test drift: owner decision of 2026-11-06 (1 test, `project_turn_service`).** `test_project_turn_preparation_validates_current_context_and_prior_anchor[wrong_policy]`: `DID NOT RAISE <class 'ValueError'>` (line 132). The scenario changes only the project-instruction descriptor's credential filter version (`credentials-v2`). Since TASK-39123, `trace_policies_agree` (`Chat/console_trace_redaction.py`) treats two readable credential versions as agreeing ("a turn stored under an older version continues under the current one", owner decision 2026-11-06), so the service correctly accepts it. The scenario needs a policy that still disagrees (for example a PII setting).
3. **Probably test drift: GC keeps a live message's revision (1 test, `graph_gc`).** `test_reopen_has_no_trace_payload_in_any_durable_owner`: `assert False` (line 888). A temporary count shows the only non-empty table after GC is `console_trace_semantic_revisions`, one row: `creation_reason='message_create'`, `live_message_id` set, no content column. GC has seeded live-message revisions as metadata roots since `3e618d7119` (2026-10-15, `console_trace_maintenance.py`, `WHERE live_message_id IS NOT NULL`), so the row is kept by design while its message lives. TASK-40577 and TASK-41310 recorded this node as already red on dev; neither tracks it.

Not covered here: the two `current_turn_transforms` failures (TASK-32032).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every one of the 32 named nodes passes on dev, or is changed with a stated reason when the product's current behaviour is the intended one (for example the 2026-11-06 credential-version ruling); no assertion is weakened without that reason
- [ ] #2 The 30 sandbox-trap tests no longer read config in-process under the per-test sandbox, or carry the bootstrap_profile marker, and the four 'alias request' cases and four teardown errors that appear behind the trap are diagnosed as product or test drift, with evidence
- [ ] #3 The five files holding the 32 run on a PR lane, timed with that lane's flags, so a new red in them fails PR CI; `graph_gc`'s node already in console-p0-gate is folded into the whole-file entry so no node runs twice, and `current_turn_transforms` joins when TASK-32032 closes (or runs with its two nodes deselected and named)
- [ ] #4 The fix is verified on Linux (CI or the Docker recipe) as well as macOS, since this task's evidence is macOS only
<!-- AC:END -->
