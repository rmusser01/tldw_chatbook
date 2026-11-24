---
id: TASK-42513
title: Thirty-two Console trace tests are red on dev in five files no PR lane runs
status: Done
assignee:
  - '@claude'
created_date: '2026-11-24 14:29'
updated_date: '2026-11-24 23:08'
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
- [x] #1 Every one of the 32 named nodes passes on dev, or is changed with a stated reason when the product's current behaviour is the intended one (for example the 2026-11-06 credential-version ruling); no assertion is weakened without that reason
- [x] #2 The 30 sandbox-trap tests no longer read config in-process under the per-test sandbox, or carry the bootstrap_profile marker, and the four 'alias request' cases and four teardown errors that appear behind the trap are diagnosed as product or test drift, with evidence
- [x] #3 The five files holding the 32 run on a PR lane, timed with that lane's flags, so a new red in them fails PR CI; `graph_gc`'s node already in console-p0-gate is folded into the whole-file entry so no node runs twice, and `current_turn_transforms` joins when TASK-32032 closes (or runs with its two nodes deselected and named)
- [x] #4 The fix is verified on Linux (CI or the Docker recipe) as well as macOS, since this task's evidence is macOS only
<!-- AC:END -->

## Implementation Plan

1. Sandbox trap (30): `sidecar_replay`, `sidecar_redaction` and `proprietary_settlement` send through `hook_admission_reason`, which reads `[hooks]` in-process (`read_hooks_config_snapshot`). Give the three files a module `pytestmark = pytest.mark.bootstrap_profile` with the reason beside it, as `test_console_first_send_atomicity.py` (whose `_controller` they reuse) already does.
2. Behind the trap: the 4 "alias request" cases call `_chat_dictionary_applier` with the frozen prompt-transform inputs (3 args); the harness's 2-arg lambda raises and `_apply_chat_dictionaries` swallows it. Test drift: give the double the current signature (as the sibling transform files do).
3. Behind the trap: the 4 teardown errors are `controller.shutdown()` refusing with the pending-dispatch ValueError after a deliberately refused capture (accepted turn, PAUSED, durable dispatch checkpoint). TASK-42155 decided that refusal is the product's; make the harness teardown tolerate exactly that text, as `shut_down_console_controllers` does.
4. `project_turn_service[wrong_policy]`: TASK-39123 (owner decision 2026-11-06) made readable credential versions agree. Keep the "wrong policy refuses" assertion with a PII disagreement, and pin the credential-version case as accepted.
5. `graph_gc` reopen privacy: ADR-337's 2026-10-13 amendment (3e618d7119) keeps live-message canonical revisions as metadata roots. Keep "no payload in any table" and assert the semantic-revision table keeps only the live source message's metadata locator.
6. Lane the five files in a non-P0 step (not the admission-sensitive Console P0 step), moving `graph_gc`'s P0 node into the whole-file entry and updating the Tests/CI pins; leave `current_turn_transforms` alone (TASK-32032). Time with the lane's flags; check config reads with the RecoveryRequired recorder.
7. Verify on Linux (Docker, CI's install line) as well as macOS.

## Implementation Notes

All 32 nodes pass, test-only fixes; no product code changed. Each node class, its cause and evidence:

- **Sandbox trap (30: sidecar_replay 21, sidecar_redaction 5, proprietary_settlement 4).** Every send passes `hook_admission_reason` (`Chat/console_automatic_submission.py`), which reads `[hooks]` in-process through `read_hooks_config_snapshot` (on the send path since aed1b13501, 2026-09-27). Under the per-test config sandbox that read raises `RecoveryRequired(raw_source_selection_changed)` and the send is refused as "Hooks unavailable". The fix is a module `pytestmark = pytest.mark.bootstrap_profile` in each of the three files, with the reason beside it. This matches `test_console_first_send_atomicity.py`, whose `_controller` they reuse. With the mark, a `RecoveryRequired.__init__` recorder counts 0 constructions over the five files serially (74 passed).
- **Behind the trap, 4 "alias request" cases: test drift.** `_apply_chat_dictionaries` calls the applier with the turn's frozen prompt-transform inputs as a third argument. Every turn has frozen those inputs since 3e618d7119 (2026-10-15). A temporary probe showed the call args were `(conversation_id, 'alias request', mappingproxy({...}))`. The harness's two-argument lambda raised `TypeError`, the method's broad `except` returned the payload unchanged, and the send went out untransformed. The double now takes the third argument, as `test_console_trace_current_turn_transforms.py` and `test_console_trace_transform_continuations.py` already did. The assertion is unchanged.
- **Behind the trap, 4 teardown errors: test drift, decision TASK-42155.** The four cases (`rejects_unsaved_thinking_under_saved_owner` x2, `rejects_private_sidecar_value...`, `credential_filter_equivalence...`) refuse capture after acceptance on purpose. A probe showed each left the turn PAUSED (`blocker=TRACE_CALL`, "Retry, Send without capture, or Cancel.") with an `accepted` row in `console_dispatch_checkpoints`. `controller.shutdown()`, added to the harness teardown in 27795ce516 (TASK-35584), then refuses to delete the paused turn with `ValueError: Resolve pending dispatch before changing conversation branches.` TASK-42155 decided that refusal is the product's, and `Tests/Chat/conftest.py::shut_down_console_controllers` tolerates exactly that text as a warning. The harness now does the same (exact text only; gateway and database retirement still run).
- **`project_turn_service[wrong_policy]`: test drift, decision TASK-39123 (owner decision 2026-11-06, 8173e2aa8a).** `trace_policies_agree` treats two readable credential versions as agreeing, so a `credentials-v2` descriptor is now accepted. The "wrong policy refuses" assertion is kept with a policy that still disagrees: PII redaction on, with the built-in ruleset. A new `readable_credential_version` case pins the decision: the v2 descriptor is admitted, like `valid`.
- **`graph_gc::test_reopen_has_no_trace_payload_in_any_durable_owner`: test drift, ADR-337's current-owner amendment (2026-10-13, 3e618d7119).** GC keeps a live message's canonical semantic revision as a metadata root (`WHERE live_message_id IS NOT NULL`). The table has no content column. The test keeps "every payload table is empty" for the other 13 tables. For `console_trace_semantic_revisions` it now asserts that exactly one row is left: the live source message's `message_create` locator.

Lane (AC#3): the five files run in dreams-tracking's riders step. `graph_gc` and `project_turn_service` read no config (recorder), so they run before `Tests/Tool_Packs`. The three bootstrap_profile files run after it. `graph_gc`'s `test_stale_mark_never_sweeps_after_concurrent_reachability_change` left console-p0-gate's admission-sensitive step (nothing was added there), so it runs once, inside its whole file. Updated with it: `_PR3023_CRITICAL_TARGETS` (Tests/CI/test_derived_artifacts_workflow.py), the admission step list (Tests/CI/test_response_rules_critical_gate.py) and `DREAMS_TRACKING_RIDERS` (Tests/CI/test_ci_queue_pressure_contract.py). `current_turn_transforms` is left alone for TASK-32032. Timings, serially with the step's flags at load 14-19: graph GC 10.2 s (21 tests), project turn 13.3 s (22), sidecar replay 25.2 s (22), sidecar redaction 6.8 s (5), proprietary settlement 6.8 s (4). The whole riders step (42 targets, with TASK-42517's three) took 1084.2 s locally at load 9-20, against 821.9 s for 34 targets at load 8 before. On Ubuntu 24.04 it took 669 s (the eight files 110 s serially on Linux), about +1.8 min on a job that ran about 9 of its 20. The step's only failures there were `test_finite_db_retirement.py`'s 18 drain checks, which fail identically at base in that container. The admission-sensitive Console P0 step, one node fewer, passed 528 tests in 1623.5 s locally at load 10-20.

Red/green: base 75bac413f5 `-n 4` gives the 32 red (plus the two TASK-32032 nodes). With a throwaway bootstrap_profile plugin, base gives 4 alias failures and 4 teardown errors. The branch gives 74 passed (`-n 4`, and serially under the recorder). Linux (AC#4: python:3.12-bookworm container, CI's install line, serial, lane flags): base fails all 32 (the 30 with "Hooks unavailable"); the branch passes all 74 in the five files. The whole riders step on Ubuntu 24.04 (Python 3.12.3, SQLite 3.45.1, the version CI's ubuntu-latest has) also passed all 127 new tests. Soak: the five files with TASK-42517's three, plus the v87 and restore files, 20 runs at `-n 4` and load 14-22, gave 20/20 green with 143 passed each.

Files: Tests/Chat/test_console_trace_{sidecar_replay,sidecar_redaction,proprietary_settlement,graph_gc,project_turn_service}.py, .github/workflows/derived-artifacts.yml, Tests/CI/test_{derived_artifacts_workflow,response_rules_critical_gate,ci_queue_pressure_contract}.py.
