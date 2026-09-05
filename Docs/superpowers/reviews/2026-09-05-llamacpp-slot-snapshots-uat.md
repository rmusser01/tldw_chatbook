# Manual llama.cpp snapshots — live UAT, 2026-09-05

## PR integration verification

The snapshot-only commits were replayed onto `origin/dev` at `93388ba69b`, on
`codex/llamacpp-slot-snapshots-pr`. The original tested feature branch is preserved.
Integration retains dev's lazy pane population, settings focus ownership and
recovery fencing, and all app-owned shutdown lifecycles. Snapshot refresh now runs
at completed pane activation. Generated CSS and diagnostic inventory were rebuilt;
the inventory matches exactly with only the two snapshot-store sinks added.

Fresh normal Models UAT on the integrated branch: **1 passed, 1 warning, 321.58s**.
The same b10816/Gemma/projector assets and normal keyboard/confirmation path were
used, without diagnostic plugins or modal-refresh suppression. All six cache
counters exactly match the table below. Default retention, lowered retention,
cancelled Delete, and selected-record Delete passed. The rendered final frame
shows Idle, one matching snapshot, keep count 2, and the server still running
before owned-child cleanup. Evidence: `pr-normal.xml`, `pr-normal.log`, stage SVGs
under `pr-normal/`, and `pr-final.png` in the same scratch evidence directory.

Snapshot modules, widget and test lint/format checks pass (16 files); the generated
diagnostic inventory check reports no drift. The full repository suite was not run.

The 13-file integration run finished **579 passed, 5 failed, 1 deselected** in
484.04s (`pr-targeted.log`). Follow-up isolated and corrected two test assumptions:
the snapshot shutdown double now includes dev's additional lifecycle methods;
the Remote memory/recompose test waits for the actual mounted Models window,
not merely its presence during composition. The multiprocessing test passed with
`TMPDIR=/private/tmp` (the initial manager socket exceeded macOS AF_UNIX's path
limit). The navigation timing assertion passed in isolation; its initial 0.520s
gap during concurrent live UAT is retained, not treated as a proven root cause.
Final combined recheck: **39 passed, 1 warning, 7.89s** (`pr-final-recheck.log`).

The remaining Notes runtime-start test also fails on an untouched archive of
`origin/dev` at `93388ba69b`: its Library screen has not mounted before the test's
deadline. The initial integrated run instead observed extra refresh callbacks;
that differing failure is retained in `pr-targeted.log`. Baseline comparison is
recorded in `pr-dev-baseline.log`. This is not an all-green integration claim.
The existing large Models adoption test file has the same nine Ruff findings on
dev and this branch; these unrelated findings were not reformatted. The larger
run also reported file-descriptor growth and splash SyntaxWarnings. No warning
was suppressed. Final process inspection found no remaining llama-server children.

## Current verdict: PASS — normal Models live UAT after the race fix

The final normal-UI run passed in **274.86 seconds**, with no diagnostic plugin
and no modal-refresh suppression. Actual b10816/Gemma 4 text and image
save/restart/restore, newest-10 retention, lowering the limit to 2 with pruning
only after the next Save, cancelled Delete and selected-record Delete all passed.
All six owned server children were reaped. AC1 and AC5 now have live evidence.

Final counters exactly match the table below: restored text 22/23 reused,
restored same image 105/106, and a changed image only 19/106 (the native text
prefix). The final frame was rendered and inspected: Idle, matching remaining
snapshot, keep count 2, and a server left running by Delete before final cleanup.
This remains the scoped mounted-Models UAT described below, not a test of the
Console send adapter or every model/runtime/platform.

Evidence: `/private/tmp/chatbook-snapshot-uat.OT7gEe/fixed-normal-final.xml`,
`fixed-normal-final.log`, per-stage SVGs under `fixed-normal-final/`, and rendered
`fixed-final.png`. The command explicitly unsets the diagnostic control and
PYTHONPATH, loads no diagnostic plugin, and uses the documented existing assets.

### Remediation and regression evidence

`_refresh_generation` now retains the last completed readiness observation while
a new probe is pending. Failed probes still clear readiness; invalidated claims
still fail admission; every operation still performs its own fresh probe.
Six barrier tests cover Save/Restore during pending-success, failed, and
invalidated background observations. Before the fix the two success cases failed;
after the fix all six passed. Removing the actual-failure invalidation in a
temporary mutation made both failure cases detect unwanted dispatch; the guard
was restored and all 36 service tests passed. Independent scoped review found no
production issues; its test-settlement finding was corrected and re-reviewed closed.

Targeted service/live-helper/Models/F9 verification: **95 passed, 1 deselected**,
81.22s. After the test-settlement and keyboard-harness refinements, the affected
service/live-helper files passed **64 tests, 1 deselected**, 2.06s. The earlier F9
case passed in the 95-test run; its previous failure is retained below, not erased
or attributed to a proven cause. Ruff lint/format and whitespace checks pass.
The existing RequestsDependencyWarning remains. No full repository sweep ran.

The first post-fix live replay passed all text/image controls but then exposed a
harness-only rapid-keypress issue: Textual ignores Enter during the Button's
0.2-second active feedback. Only one of the repeated Saves was admitted, and the
test waited for an operation it had never started (870.68s total failed run,
`fixed-normal.xml`). The harness now waits for the real button's active effect to
end and separately bounds admission at ten seconds. It does not disable feedback
or bypass the keyboard handler. The subsequent complete final replay passed.

No new ADR, config schema or dependency was needed. ADR-119 applies. Changes remain
in the feature worktree; this verification did not merge, push or run broad
integration/whole-repository gates. The following sections preserve the original
failed-UAT investigation and diagnostic controls as historical evidence.

## Initial verdict: FAIL — confirmed Restore races readiness refresh

Execution used the existing local b10816 bundle, Gemma 4 GGUF and its adjacent
vision projector. The previous statement that these assets were missing was
incorrect. No download, production config change, model edit, merge or push was
needed. Production code was unchanged during that initial UAT turn.

The real mounted Models screen launched and stopped actual llama-server children,
saved actual binary snapshots, displayed timestamped catalog entries and opened
Restore confirmation. Three runs with the normal confirmation/reentry behavior
failed at the first confirmed Restore after restart with `launch_unavailable`.
At that point AC1 and AC5 remained open; diagnostic controls could not close them.

## Confirmed defect

Returning from the confirmation dialog calls
`LlamaCppSnapshotManager._screen_reentered()`, which schedules `service.refresh()`.
The admitted Restore performs its own readiness check and stages its source.
The additional refresh starts during staging and temporarily sets
`generation.ready = False`. Restore's next `_eligible()` rejects the still-valid
launch, before the Restore POST. The additional refresh then finishes successfully,
leaving the visible contradiction **Idle · launch unavailable** while the server
is running and the snapshot reports a matching configuration.

The trace around the failure (monotonic seconds; no private payloads):

```text
1865648.526746 refresh-end valid=True ready=True operation=True status=staging_and_verifying
1865648.534850 eligible valid=True ready=True operation=True status=staging_and_verifying
1865648.536849 refresh-start valid=True ready=True operation=True status=staging_and_verifying
1865648.538852 eligible valid=True ready=False operation=True status=staging_and_verifying
1865648.539269 refresh-end valid=True ready=True operation=True status=staging_and_verifying message=launch_unavailable
```

Inspection locations: `Widgets/llamacpp_snapshot_manager.py` screen-reentry handler;
`LLM_Management/snapshot_service.py` `_refresh_generation`, `_eligible`, `_operate`.
The fix must coordinate background refresh with admitted operations without
weakening rejection after an actual failed readiness probe or invalidated launch.
No production fix was applied as part of this UAT request.

## Diagnostic control: actual text and image cache reuse

A scratch pytest plugin temporarily suppressed **only** the manager's
`_screen_reentered()` callback. Service, client, filesystem storage, launch
admission, save/restore requests, and model inference were real and unchanged.
This causal control passed the restart/reuse checks. It is explicitly **not a
passing UAT of the normal UI** and is not included in production or committed tests.

| Request | Reused tokens (`cache_n`) | Newly processed (`prompt_n`) | Total |
| --- | ---: | ---: | ---: |
| Cold text | 0 | 23 | 23 |
| Restored same text | 22 | 1 | 23 |
| Cold image A | 0 | 106 | 106 |
| Native in-memory image A → B | 19 | 87 | 106 |
| Restored image A → A | 105 | 1 | 106 |
| Independently restored image A → B | 19 | 87 | 106 |

Same-image restore reused beyond the text prefix; the different image did not.
The image snapshot contained 106 tokens and 23,883,576 bytes; its filename was
automatically timestamped. Two synthetic 64×64 checkerboard PNGs with different
pixel content/byte hashes avoided using private images. Requests used ordinary
`/v1/chat/completions`, no `id_slot`, temperature 0, seed 1, max_tokens 1, and
cache_prompt true. This tests snapshot integration, not the Console send adapter
or persistence of real user conversations/tools/approvals.

## Runtime identity and scope

- llama-server: b10816, commit `427291b5b`, existing macOS ARM64 release bundle.
- Executable SHA-256: `d707b6db4c1397a7383176fba12d339e5b33c7513669d74c8fbc2a76f6979a72`.
- Gemma 4 26B A4B Q4_K_M SHA-256: `acae52237b2abba49223a346ff8154fa15489f103676e6d10107cfa099720e38`.
- BF16 projector SHA-256: `b3ee6c97d5a5bb1ae9eb93bf14c1d1b51a0179a45ac1076b195931814c759e1e`.
- CPU, one slot, context 8192, flash attention off, fit off, explicit
  `--device none --n-gpu-layers 0 --mmproj-device none --no-mmproj-offload`,
  `--swa-full --cache-ram 0 --no-warmup`.
- The first attempt exposed a harness/documentation error: `--mmproj-device CPU`
  is rejected by this binary with `invalid device: CPU`. Corrected to the binary's
  documented CPU-only value `none`; no product behavior was changed.
- Headless Textual Pilot drives the production Models hierarchy and controls.
  The shared app factory isolates/stubs unrelated app services; Ollama discovery
  is suppressed. Snapshot service/store/client/lifecycle are not mocked.
- All config, databases and generated snapshot data are disposable; only numeric
  loopback requests to owned children are made. No user runtime is adopted/stopped.

## Runs and retained evidence

Local evidence directory: `/private/tmp/chatbook-snapshot-uat.OT7gEe`.

| Run | Result |
| --- | --- |
| `live.xml` / `live.log` | Failed startup, invalid CPU projector argument; 27.29s |
| `live-round2.xml` / log | Failed first confirmed Restore; 48.64s |
| `live-round3.xml` / log | Same failure with safe reason and screenshots; 52.93s |
| `unmodified-trace.xml` / log | Same failure, traced production behavior with no suppression; 59.05s |
| `diagnostic-control.xml` / log | Diagnostic-only text/image controls: 1 passed; 269.89s |
| `retention-control.xml` / log | Diagnostic-only full replay plus real retention/Delete: 1 passed; 270.32s |

The normal and diagnostic runs use the documented live gate/environment variables.
The traced runs additionally load the scratch `uat_diagnostics.py` pytest plugin.
Only control runs set `SNAPSHOT_UAT_MODAL_REFRESH_CONTROL=1`; normal UAT does not.
Distinct `--basetemp` directories preserve evidence against unrelated concurrent
pytest retention cleanup. JUnit uses `-o junit_family=xunit1` for record_property.

`round3/test_models_live_persistence_a0/restore-2.svg` and rendered
`restore-failure.png` were visually inspected: running server, matching saved
configuration, and **Idle · launch unavailable** are visible together.

Supplementary targeted checks (not live UAT): 41 passed, 1 failed, 1 deselected.
The existing F9 combined-provider-save/whitespace-normalization case failed its
visible-field assertion, then passed unchanged when rerun alone (1 passed, 4.08s).
That rerun does not erase the initial failure or establish its cause. The existing
RequestsDependencyWarning remains; no dependency changes or warning suppression.

## UAT artifacts and follow-up

The harness now retains per-stage SVGs, safe status/error codes and token counters,
and includes real-file default-retention, lowered-limit and confirmed/cancelled
Delete checks. The normal UAT stops at the Restore defect before these later
scenarios. The additional diagnostic control passed all of the following:

- Eleven further real Saves retain exactly ten records and remove the prior
  oldest records; all names are generated, not supplied by the operator.
- Models Details & preferences Apply persists a keep count of 2. Merely changing
  that setting leaves ten existing records intact. The next completed Save leaves
  two matching records, also confirmed in the rendered frame.
- Escape from Delete confirmation preserves the selected record. Confirming Delete
  removes only the selected record and its binary, leaving the other record and
  the active server intact.
- All six owned children in each passing diagnostic run were stopped and reaped
  by the test's finally block. A final process inspection found only an unrelated
  task's runtime on port 18485; it was not touched.

Only disposable synthetic UAT snapshots were pruned/deleted. Remaining test files,
logs, SVGs and JUnit evidence are retained in the scratch directory above.
`retention-lowered.png` was rendered and visually inspected alongside the failure
frame. This diagnostic retention pass does not override the normal Restore failure.

The changed harness passes Ruff lint/format and whitespace checks. A supplementary
F9 timing-sensitive failure remains recorded above rather than being relabeled as
a wholly green targeted run. No full repository suite was run.

Existing ADR-119 governs this work; no new architecture or dependency decision.
The initial failed UAT required the reviewed fix and normal-UI replay recorded above.
