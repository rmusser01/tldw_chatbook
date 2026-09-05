# Manual llama.cpp snapshots — live UAT, 2026-09-05

## Final closeout — merged

[PR #2419](https://github.com/rmusser01/tldw_chatbook/pull/2419) merged into dev
on 2026-09-05 at 19:44 UTC as `ec55a0ed3b4d5dc4b772254ba832dad030c566d3`.
Final reviewed head was `f416ba31197ba6e0e0e876c9607df4d2660dbe38`;
required CI passed and all seven review threads were resolved.
The final Queue-edit dev integration passed 69 targeted checks and all six
derived-artifact checks. The most recent live UAT remains the explicitly
identified Scheduling-base run below; it is not relabeled as final-head UAT.
Later integrations did not change snapshot production behavior.

The manual manager is complete. Follow-up work is filed, unassigned, and deferred:

- [TASK-31738: automatic per-conversation reuse](../../../backlog/tasks/task-31738%20-%20llama.cpp-opt-in-automatic-per-conversation-prompt-cache-reuse.md).
- [TASK-31739: real-server audio qualification](../../../backlog/tasks/task-31739%20-%20llama.cpp-real-server-audio-snapshot-reuse-qualification.md).
- [TASK-31740: Windows private storage](../../../backlog/tasks/task-31740%20-%20llama.cpp-Windows-private-snapshot-storage-support.md).

The remaining sections preserve the execution history. Their pending merge,
missing-evidence, and pause statements describe those earlier checkpoints,
not the current status. Audio reuse and Windows private storage are not claimed.

## Subsequent dev integration: Scheduling and Meetings

After the approved repair, dev advanced twice while CI ran. Scheduling
`da2fbdbc2` rebased cleanly; **279 targeted Scheduling/snapshot service/UI tests**
passed (232.17s), plus **18 boot/import guards** and all six artifact checks.
Exact head `eaa30045c03eab84099edb972601e1f0f3f9c31a` passed a new real Models
b10816/Gemma vision UAT in **271.62s**: text22/23, sameimage105/106,
changedimage19/106, retention/Delete. Final capture was visually checked and
test-owned children exited. Another validation server using the shared binary
was identified by its different parent/runtime path and left untouched.

Meetings `c14dadd77080be929d47e3acef41be790ee5d8d1` then required one diagnostic
summary conflict resolution: preserve 12 sink files and Meetings' 30 classified
calls. Source-derived verification passes all six checks. Snapshot production
code and Library/Settings repair remain unchanged. All **18 boot/import guards**
pass: **495 modules / 366623 LOC**, Library113319, UI-ready972. Existing tightened
limits are unchanged; the new Meetings route is included in the complete census.

Targeted Meetings/snapshot integration initially passed117 and failed3: shutdown
test doubles lacked `_meeting_session_owner`, newly initialized by the real app.
The shared fixture now mirrors the real unused-owner value (`None`); no production
logic or shutdown assertions changed. Complete affected service/Meetings recheck:
**87 passed**, 4.27s; lint/format/whitespace pass. Models/F9 tests were among the
117 passing checks and were not modified. The live result above is explicitly
pre-Meetings, not relabeled as an exact Meetings-base execution.

Evidence: `paydown-schedules-rebase-{guards,ui,preflight}.log`,
`paydown-rebase-live.{log,xml}`, `paydown-rebase-live-final.png`,
`paydown-meetings-rebase-{guards,ui,preflight}.log`, and
`paydown-meetings-fixture-green.log` in the task scratch directory below.
Prior-head required CI passed; current-head CI/Qodo remain merge gates.

## Owner-approved pre-import repair — ready for current-head CI

The owner approved the inherited budget repair, superseding the pause below.
Latest base is dev `2c9c144181b942af2d29d16b9eb2681d7f5a7212`; its delta from
`22006e84d` is backlog-only. Library's six runtime controllers and note-import
helpers now load at construction/use, while Settings defers RAG adapter and
Tool Pack services/modals. Existing named RAG patch seams and eager Textual
event classes are preserved. No package, service, ownership or routing contract
changed; ADR-097 governs the repair.

The unchanged complete census falls from **547 / 422544** to **490 modules /
363740 LOC**, with Library **113319 LOC**. No limits rise. ADR-097 tightening
banks the savings: whole-pass LOC **380000 → 378740**, per-route LOC
**145000 → 123319**, module cap remains **500**. The canonical snapshot tool
refreshes only the now-passing pre-import snapshot.

Verification:

- New Library and Settings cold-process regressions observed genuine RED before
  edits. Combined census/closure repeat passes all five tests.
- Library mounted first-use selection: 122 passed / 3 source-inspection failures
  during an import-formatting edit; the unchanged-source architecture repeat
  passes all 17. This reproduces the already-documented `inspect`/`linecache`
  lesson, not a behavior failure. Final complete unchanged-source repeat:
  **125 passed**, 94.36s.
- Five complete affected Settings files: **299 passed**, 140.95s. Independent
  Settings review additionally ran 33 closure/Tool Profile checks, all passed.
- Other boot/closure guards: **13 passed**; import weight **641/660**,
  UI-ready **972/972**, CSS **785755/804000 bytes**.
- Independent Library and Settings reviews: no actionable findings. New files
  lint/format clean; no added diagnostics against existing broad-file baselines.
  All six derived-artifact checks pass. No full repository sweep was run.
- Real mounted Models UAT after runtime edits: **1 passed**, 267.44s, using the
  existing b10816/Gemma 4/adjacent vision projector. Text reuse **22/23**, same
  image **105/106**, changed image **19/106**, matching the native text-prefix
  control. Newest-10/lowered-after-save retention and cancel/confirmed Delete
  pass. The final capture was visually checked; no UAT-owned server child remains.

Evidence directory: `/private/tmp/chatbook-snapshot-uat.OT7gEe`. Logs:
`library-closure-red.log`, `library-paydown-green.log`, `library-wiring-repeat.log`,
`library-paydown-final.log`, `paydown-boot-guards.log`, `paydown-banked-guards.log`,
`paydown-preflight.log`, `paydown-live.log`, and `paydown-live.xml`.
Settings's 299-test tool-output receipt is explicitly labeled at
`/private/tmp/settings-preimport-299-tests-transcript.log`; fresh scoped static
evidence is `/private/tmp/settings-preimport-final-static-checks.log`.
Final image: `paydown-live-final.png`. Existing Requests dependency mismatch and
the Settings combined-run descriptor-growth warning remain visible, unsuppressed.
GitHub review/check settlement and the requested merge are the remaining steps.

## Historical pause: inherited whole-registry pre-import breach

Latest dev `22006e84d` (Library media/focus) rebased with only a lessons-document
conflict; both additions were preserved. Integrated startup/Console/Models/F9
verification passed 71 tests but exposed the broader pre-import guard:
549 modules / 422600 LOC against limits 500 / 380000. An untouched archive of
the exact dev measured 547 / 422128, including Library at 147363 LOC against
its 145000 single-route cap. This is not a passing baseline.

Removed this PR's two added modules by deferring F9 snapshot-preference imports
to save/revert/provider rendering. A fresh whole-route closure regression failed
before and passes after; 20 closure/F9/UI-ready tests pass, independent review is
clear, and no new baseline-relative lint findings remain. Final whole-pass census
is 547 / 422544: module parity with dev, not budget compliance (the existing screen
files include additional feature code). No limit or pinned snapshot was raised.
Under ADR-097, merge is paused pending owner direction for the unrelated multi-route
payload reduction. Task remains In Progress despite completed feature criteria.

## Latest-dev reuse integration

Rebased through dev `5f12507c1` (Library reuse) and `64ce47a04` (Console reuse).
Library integration passed 106 focused tests and two changed lifecycle tests.
A broader selection was interrupted at 173 passed / 5 failed; all five failures
reproduced on untouched `5f12507c1` (three stale Notes-row selector assertions,
two incomplete `active_authority` test doubles). No unrelated Library fixes were
added. Console reuse integration passed 131 targeted tests. Normal real b10816/
Gemma text/vision UAT passed on both rebases (252.25s and 252.67s respectively),
with text 22/23, same image 105/106, changed image 19/106, retention/Delete, and
owned-child cleanup. The last run predates the suspension guard below.

Independent integration review then found Environment collectors could still run
after a reusable Console was covered: suspension preserves `display`. A mounted
regression reproduced four collector dispatches while covered. The shared rail
accessor now requires `self.app.screen is self`, also fencing deferred network
dispatch, and resume refreshes an existing owner after workspace reconciliation.
The timer remains a cheap no-op while covered; never-opened Inspect stays cold.
Textual `is_current` was unsuitable because it includes background screens; that
initial harness-assumption failure is retained alongside the genuine RED result.
Final Environment/controller/reuse/suspend/census verification: **55 passed**,
61.78s; independent re-review clear; baseline-relative lint and whitespace pass.
Existing ADR-097 applies. Current-head CI remains the final merge gate.

## Approved Console startup remediation

The owner approved fixing the inherited Console regression in PR #2419.
Environment now initializes on first Inspect use, retains one controller after
that, and keeps never-opened focus/fleet/poll callbacks cold. Initial closed-rail
composition uses empty placeholders; first-open paints even without a workspace
(where no worker landing would otherwise occur). Reopen/recompose reuse the
captured snapshot. Rail IDs live with rail state, with compatibility exports from
the Environment module. Existing controller dispatch, TTL and failure policy are
unchanged, including callbacks to an already-created owner while hidden.

The original census was observed RED at 976/972. Final census is **972/972**,
with all four Environment implementation modules absent. New mounted coverage
exercises closed focus/poll/fleet, first-open no-workspace rendering and owner
reuse on reopen. Verification: **226 passed, 2 failed** in the initial broader
Environment/rail/fleet/workspace run; the two callback compatibility failures
were corrected and all **24 wiring tests pass** (35.80s). The affected three-case
recheck passes; boot plus snapshot-service checks are **59 passed** (32.47s),
and the final projection/census repeat is **32 passed** (11.74s).

Evidence: `pr-env-lazy-red.log`, `pr-env-targeted.log`,
`pr-env-callback-recheck.log`, `pr-env-wiring-final.log`,
`pr-env-final-boot.log`, `pr-env-exports-final.log` in the same scratch directory.
Independent read-only review and its final re-review report no actionable
findings. Generated CSS/inventory and whitespace checks pass; lint findings on
the six touched Python files match their pre-change baselines (203 total).
The two baseline-clean files pass Ruff lint; existing broader formatting debt
was not rewritten. No assertion, budget, network policy or dependency was relaxed.

The next dev advance, `e49a7a16d32053434053895ba3559b970ec06289`, contains
only Buddy UAT documentation/assets, with no production/test-source changes.
The earlier paused-merge statements below are historical and superseded by this
approved fix and subsequent GitHub check settlement.

## Latest dev rebase and boot integration

`dev` advanced again to `7e904737c787886c983c6c3312f0f9ca67c43453`
(Console Environment redesign). The second rebase preserved its AppFocus
forwarding and Console changes; only the diagnostic inventory aggregate
conflicted. Rebuilt that artifact from the combined sources, retaining dev's
new owner/call rows and the snapshot-store sinks. Earlier integration evidence
below is retained with its exact base.

Latest-base verification: **65 passed, 1 failed**, 50.74s
(`pr-environment-rebase.log`). The failure is the UI-ready census, now 976/972;
all snapshot service, mounted Models/F9 and other boot checks pass. An untouched
archive of dev `7e904737c787886c983c6c3312f0f9ca67c43453` reproduces exactly
976/972 (**3 passed, 1 failed**, 9.22s; `pr-latest-dev-census.log`). The four
additional modules are Console environment state/UI and workspace status/git
support; no snapshot modules are resident. This is an inherited base regression,
not an all-green integration result. The requested remote merge is paused for
the owner's choice whether to include that unrelated Console startup fix.

Rebased onto `dev` at `e990738b2812876c2593b91f62d0b2c5b2e3b69d`
(Chunking Lab integration), without code conflicts. Range-diff preserves the
snapshot changes and dev's additional app lifecycles. The merged diagnostic
inventory needed only its aggregate sink-file count regenerated from 11 to 12;
the two snapshot-store rows are unchanged and no diagnostic bodies were added.

Post-rebase combined run: **460 passed, 3 failed, 1 deselected**, 117.09s. All
three failures were denied socket binds in the sandbox, including the process
manager; rerunning those three with local-socket permissions passed in 1.41s.
The 463 checks include mounted Models first-use/layout checks. Evidence:
`pr-post-rebase-targeted.log` and `pr-post-rebase-sockets.log`.

The expanded boot check exposed three snapshot bare-type CSS rules exceeding
the parsed selector ratchet (277 > 274). Re-keyed these to existing snapshot
IDs/classes, also narrowed the table selectors, and rebuilt the CSS bundle.
All **18 boot checks pass** in 27.23s: UI-ready modules 972/972 and boot CSS
785,185/804,000 bytes. No budget was raised. Initial and corrected evidence:
`pr-post-rebase-boot.log`, `pr-post-rebase-boot-fixed.log`. CSS reproduction,
diagnostic inventory reproduction and whitespace checks pass. Existing dependency,
deprecation and multiprocessing-resource warnings remain recorded in the logs.

The first CSS UI rerun was **34 passed, 2 failed**: reduced selector specificity
let launcher disabled borders clip Restore, and removing the broad Static rule
lost inherited Checkbox/CollapsibleTitle styling at 80 columns. Retained ancestor
specificity for the exact button IDs and explicitly included the checkbox IDs and
collapsible title types. The button checks then passed; the title-wrap correction
restored the 80-column F9 check (**1 passed**, 4.78s). Failed runs and the computed
checkbox-style diagnostic are retained as `pr-post-rebase-css-ui.log`,
`pr-css-cascade-recheck.log` and `pr-css-checkbox-diagnostic.log`.

Final combined layout/boot check: **25 passed, 6 warnings**, 48.83s
(`pr-final-css-boot.log`), covering both terminal sizes, normal Models first use,
launcher primary controls and all 18 boot checks. CSS regeneration and whitespace
checks pass. The live UAT below predates this final selector-only paydown; no
model/runtime or snapshot service behavior changed in that paydown.

## PR #2419 review remediation

Qodo's six validation/style findings are addressed: configured key-file paths
use the shared validator without resolving away symlinks; the credential byte
limit is named; launch argv/environment/ID pass a strict Pydantic boundary while
preserving the public Mapping contract; both keep-count UI surfaces use shared
bounded-integer validation; retention wording has its public docstring; and
snapshot-service local imports form one contiguous group.

The seventh finding, claiming socket errors fall through to launch, is not
reproducible: only ConnectionRefusedError is caught. Timeout and other OSError
exceptions already propagate to the worker before directory creation or spawn.
Three socket-level worker regressions pass. A runtime-only mutation catching all
OSError makes both ambiguous-error rejection cases fail; the refusal control passes.
No production network handling was weakened or changed.

CI also caught eager snapshot composition adding nine UI-ready modules (981 >
972). The app now composes the owner on first use, schedules default storage in
the running app loop, and drains only an existing owner on shutdown. Pre-run
access, explicit storage initialization and pending setup are covered. The actual
startup guard passed twice at **972/972**, with no snapshot modules resident and
no budget increase (ADR-097). A direct helper invocation outside the pytest
harness measured 973; that is not substituted for the actual guard's evidence.

Final combined verification: **460 passed, 1 deselected**, 114.68s, plus **3 passed**
mounted Models first-load/layout checks, 10.11s. The fresh normal live UAT passed
in **258.67s**, with the same text/image counters, real retention and Delete
results, and normal confirmation callbacks. Evidence in the same scratch directory:
`pr-qodo-targeted.log`, `pr-qodo-models-mount.log`, `pr-qodo-live.log`,
`pr-qodo-live.xml` and per-stage SVGs under `pr-qodo-live/`.

The shared validation regressions were observed RED before correction, including
non-string launch coercion, disallowed key filenames, boolean keep-count coercion,
and Mapping compatibility. The lazy composition regression also failed before its
fix. Eight scoped files pass Ruff lint and format; large existing files retain
their independently compared baseline lint debt. CSS reproduction, diagnostic
inventory and whitespace checks pass. Existing dependency warnings remain. No
full repository sweep, dependency change, or boot-budget exception was used.

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
