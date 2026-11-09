# TASK-39147 end-to-end verification and UAT — 2026-11-07

The complete native workflow passed functional end-to-end verification, including
backup, isolated restore and rebackup. Five mounted acceptance cases passed.
The prompt-editor clipping found at 80×24 is corrected by the follow-up below.
Original timing, mounted-test and broader CSS qualifications remain recorded;
these results are not a certification that all bugs are absent.
The selected Linux CI contract subsequently passed overall on exact `2d4241b`
([run42509127239](https://github.com/rmusser01/tldw_chatbook/actions/runs/42509127239));
see the final closeout section for the source boundary and retained limits.

## Initial UAT scope and environment

- Current staged TASK-39147 production source, HEAD
  `e08627ac38392fb890984169376b1c242c3a2f3e`; initial verification made no production edits.
  The subsequent prompt-editor correction is recorded separately below.
- Private profiles, real SQLite, native AgentService/ExecutionOwner lifecycle,
  real filesystem/shell tools and complete-profile backup/restore. Only provider
  responses and deliberate delivery failures are scripted. Network and real-user
  profile guards remain installed.
- Mounted real ChatScreen and workflow/prompt/confirmation widgets with production
  CSS, driven by Textual Pilot. This is mounted automated UAT, not a human-driven
  terminal or external-provider certification.
- Observed host load was approximately 49–56 on 18 logical CPUs. This contextualizes
  timing; it does not establish the cause of every failed deadline.

## Passing evidence

| Scenario | Result and evidence |
| --- | --- |
| Native correction, independent review, final acceptance, Stop, backup, isolated restore and rebackup | **1 passed in 217.49s**, `/tmp/pr39147-uat-native-functional.txt` |
| Same/future prompt saves through Apply during running planning | Passed, `/tmp/pr39147-uat-mounted-final/test_mounted_acceptance_prompt0/uat.json` |
| Same/future prompt saves through Apply during Stop | Passed, `/tmp/pr39147-uat-mounted-final/test_mounted_acceptance_prompt1/uat.json` |
| Lost closing-review response, known accounting, fresh visible retry and result removal | Passed, `/tmp/pr39147-uat-mounted-final/test_mounted_acceptance_closin0/uat.json` |
| Lost closing-review response, unknown accounting, fresh visible retry and result removal at 80×24 | Passed, `/tmp/pr39147-uat-mounted-final/test_mounted_acceptance_closin1/uat.json` |
| The four mounted acceptance cases together | **4 passed in 163.59s**, `/tmp/pr39147-uat-mounted-final.txt` |
| Mounted two-phase plan approval, next-phase approval, each saved checkpoint and final acceptance | **1 passed in 56.37s**, `/tmp/pr39147-uat-phase-controls.txt` |
| Original workflow-command/setup UAT | Passed in the original three-test batch, `/tmp/pr39147-uat-existing-controls.txt` |

The native journey retained the actual failed verifier (exit 7), then recorded a
fresh passing verifier (exit 0, complete output `verified`, different input
manifest). Five native children produced preserved handoffs; accounting recorded
15 model calls and five child launches. User edits, Git HEAD and index survived.
The restored profile reproduced the original durable checkpoints/results/windows,
started no native executions, and produced a second coherent backup. See
`/tmp/pr39147-uat-native-functional/test_native_correction_keeps_f0/original/native-journey.json`
and the sibling `wave-rebackup.log` marker
`WAVE_RESTORED_INERT_REQUIRED_EVIDENCE_REBACKED_UP`.

The prompt cases checked the actual admitted provider payload, persisted future
settings, returned lifeline/thread, released owner/intake/capacity and normal
shutdown. Closing cases used real confirmation clicks, lost a response only after
the real closing transaction committed, retried at the displayed fresh revision,
and preserved charges, plans and the immutable marker after removal. No recovery
fence or native execution remained at successful shutdown.

## Timing qualification

The unmodified complete test failed its **130s overall process watchdog** after
workflow completion and Stop, as backup began (`/tmp/pr39147-uat-native.txt`,
1 failed in 130.95s). A diagnostic run allowing 240s overall but retaining the
original five-second planning and 35-second phase waits failed the correction wait,
then reached its expected state about five seconds later
(`/tmp/pr39147-uat-native-budget.txt`, 1 failed in 83.05s).

The functional pass used `/tmp/pr39147-uat-native-functional.py`: 240s overall and
60s test-driver status waits. All original functional/resource/storage assertions
and production budgets remained intact; backup/restore watchdogs were unchanged.
This certifies functional completion under those bounds. It **does not turn the
original timing tests green or certify latency**. The test suite and production
deadlines were not edited.

## Initial UAT finding — corrected below

**P2: system-prompt editor action row is clipped at 80×24.** Open the Console's
system-prompt editor at that terminal size. The title, editor and Library-name row
render, but Apply/Clear/Cancel are outside the visible dialog and cannot be clicked.
The fixed textarea minimum plus the other rows exceeds the dialog's capped height.
The source widget and stylesheet are unchanged from HEAD, so this is outside the
TASK-39147 changes at that point. The authorized correction below now resolves it.

Relevant source: `tldw_chatbook/css/features/_console_panels.tcss:430–466` and
`tldw_chatbook/Widgets/Console/console_system_prompt_modal.py`.
The same editor's Apply controls passed at 140×42. Workflow recovery/removal
controls passed at 80×24; that does not qualify the separate prompt editor.

![Clipped system-prompt action row at 80×24](2026-11-07-task-39147-uat/narrow-prompt-editor.svg)

## Failed and corrected probe setup

The original existing mounted approval tests stopped before workflow actions: one
failed a chat-tab physical-click assertion; the two-phase case timed out waiting
for setup after Send. The batch was stopped at two failures: **1 passed, 2 failed
in 105.77s**. These tests are not reported green.

The extra acceptance probe initially used a wrapper named differently from the
supported `resolve_recovery` operation, so service admission refused before the
intended injection. A second run also waited for the active tab to remain the
captured chat, although the captured prompt had saved and the host's late startup
selected a blank chat. The unknown-accounting probe initially released its provider
before the actual Stop cancellation was delivered. Final probes preserve the public
operation name, wait for the captured chat's payload revision, select its real tab
before each edit, and wait for actual cancellation before releasing the provider.
Earlier failed packets remain separate; they are not workflow regression evidence.
Probe source: `/tmp/pr39147-uat-probe.py`; final command/output:

```sh
.venv/bin/python -m pytest /tmp/pr39147-uat-probe.py \
  -o 'markers=bootstrap_profile: private acceptance profile' \
  -q --timeout=240 --basetemp=/tmp/pr39147-uat-mounted-final
```

The final four-case run predates the additional phase-controls parameter; its
collection and all four results are retained in the log. The additional case ran
separately with `-k phase-controls --basetemp=/tmp/pr39147-uat-phase-controls` and
passed in 56.37s. It started a real planner, then used physical Approve/Accept
clicks for both phases, selected each actual saved checkpoint, verified exactly
two native children/three provider calls and the four durable decision records,
and closed normally. It does not relabel the original pre-workflow test failures.

## Limits and preservation

No full-suite run, remote provider, other operating system or unrelated archive
lineage was certified. The earlier 71 targeted regressions remain separate evidence
on the same unchanged production source. Passing tested scenarios does not prove
the absence of all bugs.

The initial verification-only follow-up staged five TASK-39147 documents/evidence
paths. The starting snapshot is `/tmp/pr39147-uat-start.json`; final exact-path staging,
unrelated file/index preservation and unchanged HEAD are audited in
`/tmp/pr39147-uat-final-preservation.json`. No commit, push, reset, stash or merge.


## Prompt-editor correction and rerun

The dialog now uses the existing native scrolling/docked-action pattern. The
frame scrolls within its existing size cap, the action row reserves its own space,
and Library feedback scrolls into view after its wrapped height is laid out.
Widget ids, action results and the existing
visual tokens are retained. The generated app bundle contains the matching dock
rule, rebuilt from its source module.

- The new painted-bounds regression failed before correction: **1 failed in
  11.41s**, `/tmp/pr39147-prompt-layout-red.txt` (Apply was absent from the
  compositor's visible widgets at 80×24).
- Initial corrected-layout regressions: **7 passed in 111.83s**,
  `/tmp/pr39147-prompt-layout-green-isolated.txt`. These check actual painted labels
  and physical Apply/Clear/Cancel clicks at 80×24 and 140×42, resize round trips,
  retained long prompt text, wrapped Library feedback, and Tab/Enter focus.
- Existing Apply/Clear/Cancel persistence and Library save/duplicate/missing-name
  controls: **6 passed, 20 deselected in 83.43s**,
  `/tmp/pr39147-prompt-layout-existing-modal.txt`. The same six controls passed
  again against the final scroll-timing code: **6 passed, 20 deselected in
  213.53s**, `/tmp/pr39147-prompt-layout-existing-final.txt`.
- Mounted Console UAT rerun: **4 passed in 379.34s**,
  `/tmp/pr39147-prompt-layout-uat-final.txt`. Both terminal sizes exercised same/
  future prompt saves during running planning and Stop, actual admitted provider
  input, recovery review, physical Cancel/Clear, persisted state and normal native
  shutdown. No UI or process deadlines were relaxed for this rerun.
- Final-code narrow UAT rerun after the feedback scroll-timing correction:
  **2 passed, 2 deselected in 254.35s**,
  `/tmp/pr39147-prompt-layout-uat-source-final.txt`. Both running and Stop/recovery
  cases physically applied prompts, canceled drafts, cleared persisted prompts,
  retained admitted provider input and completed actual native cleanup/shutdown.
  This overlaps the initial four-case UAT rather than adding new scenarios. Wide
  geometry is unchanged by the final timing-only adjustment and remains covered
  by the final seven-case regression file. No deadlines were changed.
- New test lint and both touched Python files' formatting pass. The modal retains
  four inherited lint findings and introduces none. The design detector returned
  no findings. The prompt-specific bundle reproduces exactly from its modular
  sources (`/tmp/pr39147-prompt-layout-own-bundle-check.txt`). Final static
  evidence is `/tmp/pr39147-prompt-layout-static-final.json` and
  `/tmp/pr39147-prompt-layout-lint-format-final.txt`.

Fresh before/after evidence retains the original clipped image above and the
corrected views below. The two-size capture passed in **80.12s**
(`/tmp/pr39147-prompt-layout-capture-cells.txt`); it predates only the final
Library-feedback scroll-timing correction, which does not change these initial
editor views. Capture uses the mounted Console's painted compositor strips
at each size. The raw SVG exporter sometimes produced title-only frames in this
two-App harness; those empty captures are not visual evidence of the fix.

![Corrected prompt editor at 80×24](2026-11-07-task-39147-uat/narrow-prompt-editor-fixed.svg)

![Corrected prompt editor at 140×42](2026-11-07-task-39147-uat/wide-prompt-editor-fixed.svg)

A stricter follow-up asserted the entire wrapped feedback text in painted cells.
It exposed the last line beneath the docked footer (**1 failed in 40.92s**,
`/tmp/pr39147-prompt-layout-feedback-paint.txt`): the original scroll request used
Static's stale one-line geometry immediately after `update`. The private diagnostic
recorded scroll offset 7 versus maximum 9, and the three-line feedback intersecting
the action row (`/tmp/pr39147-prompt-layout-feedback-geometry.txt`). Deferring
`scroll_visible` until after layout refresh corrects the measurement. The complete
new regression file, now requiring all wrapped feedback text to be painted,
passed **7 tests in 204.41s** against the final code
(`/tmp/pr39147-prompt-layout-green-final.txt`). An initial
external diagnostic could not collect before private-profile admission and ran
no tests; the repository's private-profile diagnostic reproduced the painting
failure. No real-profile recovery action was taken.

### Broader checks and probe qualifications

The wider CSS batch had **16 passed, 3 failed, 20 deselected in 502.71s**
(`/tmp/pr39147-prompt-layout-controls.txt`), stopping at three failures:

1. Full generated-sheet sync reports unrelated staged portrait declarations.
   A full build also produced portrait rules using unavailable tokens, causing
   **7 parsing failures in 93.16s** before layout could be exercised
   (`/tmp/pr39147-prompt-layout-green.txt`). Those two unrelated generated widget
   sheets were restored to their exact starting index bytes and are not staged by
   this fix. The prompt-specific app bundle is independently reproducible.
2. The canonical-class scan exceeded its 90s test deadline; it remains unqualified.
3. The dimension floor reports existing numeric values in `_agentic_terminal`,
   `_settings_splash_theme`, `_wizards` and `_workflows`. All four files match their
   starting index bytes; this correction adds no numeric style values.

The first expanded UAT batch had one process-watchdog failure and one successful
80×24 Stop case before interruption. A diagnostic run showed the host's startup
blank tab had changed the active target before initial Apply. Final UAT waits for
that tab to finish mounting, then selects the actual fixture chat. The failed
packets remain `/tmp/pr39147-prompt-layout-uat.txt` and
`/tmp/pr39147-prompt-layout-uat-debug.txt`; they are not relabeled as passes.

Final narrow command (same private runner and deadlines):

```sh
.venv/bin/python -m pytest /tmp/pr39147-prompt-layout-uat.py \
  -o 'markers=bootstrap_profile: private acceptance profile' \
  -k 'prompt-running-80 or prompt-stopping-80' \
  -q --timeout=240 --maxfail=1 \
  --basetemp=/tmp/pr39147-prompt-layout-uat-source-final
```

The preceding full native backup/restore journey was not repeated for this scoped
layout correction. No full suite, external provider or other-platform run. Exact
fix staging and preservation are audited in
`/tmp/pr39147-prompt-layout-final-preservation.json`, against
`/tmp/pr39147-prompt-layout-start.json`. Unrelated staging and files remain intact.


## CI follow-up — 2026-11-08 (in progress)

Published task commit: `3285573d77da4e646faa9ed8f974d4b42f99fea5` on
`codex/task-39147-native-phase-rebuild`, parent `e08627a`.
The [initial Derived Artifacts run](https://github.com/rmusser01/tldw_chatbook/actions/runs/42426760674)
has 12 failed and 3 successful test jobs. Its final aggregate job was still queued
when these failures were diagnosed. No success or bug-free claim follows from the
earlier local UAT. Integration onto newer remote dev remains outside this work.

Repairs are built as Git blobs through `/tmp/pr39147-ci-repair-state.json` and a
private index. Local verification loads the candidate's Python and relevant
package resources from Git objects in memory, including private pytest children;
it does not replace checkout files or create a worktree. Local handles, adapters
and assertions remain real except the tests' existing provider doubles. This
harness does not replace actual Linux CI, filesystem-membership checks or isolated
`python -I` execution; those remain qualified by the pushed run.

Observed verification before publication of the repair batch:

- Initial representatives: four migration variants and one standalone-controller
  send failed. Two saved-hook controls also failed. Full baseline controls
  reproduced the migration/hook defects, so they are inherited native-foundation
  compatibility failures rather than claimed new task regressions.
- Core repair candidate `853640e`: **21 passed in 60.25s**, covering the four
  migration variants, controller send, saved hooks and publication, strict custody
  fields, workflow atomic-save/invalid-value controls and app/Settings size pins.
  Packet: `/tmp/pr39147-ci-core-targeted-green.txt`.
- Broader targeted Console controls: **61 passed, 2 failed in 235.98s**.
  Packet: `/tmp/pr39147-ci-secondary-targeted.txt`. Both failures were retained:
  real completed-tool response-rule repair and the external Stop follow-up.
- Binding the actual replacement bridges through runtime APIs makes the completed
  tool case, real-agent boundary and unchanged-target native acceptance pass:
  **3 passed, 1 failed in 100.27s**. The remaining prepared-start/manual-send case
  hits the mounted configuration refusal, not a discarded assertion.
  Packet: `/tmp/pr39147-ci-bound-fixtures-targeted.txt`.
- Supplying the lifetime stub's actual store: **1 passed in 23.21s**;
  `/tmp/pr39147-ci-custody-store-targeted.txt`.
- The new ordinary-chat Stop case fails before the maximum fix:
  **1 failed in 15.17s**; `/tmp/pr39147-ci-host-hook-red.txt`. A trace of the
  external Stop control reports `ordinary maximum does not cover this effect` at
  event prehold. The repair includes the installed engine's enabled definitions
  before admission; it retains the existing opaque claim and physical-drain check.

Final scoped checks of the repair batch:

- Expanded original Stop physical-drain controls (agent runtime on/off,
  onboarded/fresh) plus the actual external-Stop/native-rule follow-up:
  **5 passed in 48.02s**; `/tmp/pr39147-ci-host-hook-green.txt`, source tree
  `37e9ae83e420ef9e7c06208c4de96d3dc948ca24`.
- The originally failing mounted Settings streaming-save case:
  **1 passed in 62.54s**; `/tmp/pr39147-ci-settings-render-green.txt`, source tree
  `52ccdd59bd3298cc9bb5e18bbecb9d37ff986adb` (same product sources plus evidence).

Exact narrow commands use the Git-source harness environment described above:

```sh
.venv/bin/python -m pytest \
  Tests/Chat/test_console_ordinary_tree.py::test_original_stop_hook_holds_same_ordinary_tree \
  Tests/Chat/test_response_rules_external_hook.py::test_real_external_stop_and_native_violation_commit_one_followup -q
.venv/bin/python -m pytest \
  'Tests/UI/test_settings_console_fallback_rows.py::test_choosing_streaming_on_saves_true_over_a_saved_legacy_or_unset_value[saved-off]' -q
```

Remaining diagnosis: the mounted preparation refusal also reproduces with full
Git sources from `e08627a` (**1 failed in 19.78s**;
`/tmp/pr39147-ci-mounted-baseline.txt`). The synthetic app factory selects MCP
files outside the configured profile, leaving the actual source unbound; the
stock capture correctly refuses it. Memory-backed fixture sources also do not
qualify for that stock route. Production refusal and actual source/recovery
checks have not been weakened to accommodate these fixtures. Remaining fixtures
and any task-related compatibility defects still need resolution and remote CI.

Ruff comparison over every changed Python file introduces **zero new findings**;
inherited lint debt remains explicit. The app shrinks to 5708 lines (ratchet
reduced from 5710), and Settings fits the unchanged 35818-line budget. No assertion,
timeout or budget is relaxed. Preservation auditing compares the complete real
index, dev HEAD/status and hashes/links/missing states of all 374 starting dirty
and untracked paths; `/tmp/pr39147-ci-repair-preservation.json` is the current
packet. No full-suite local sweep, external provider or other-platform UAT was run.


## Second CI repair batch — 2026-11-08 (CI still required)

The [run on ab472e8](https://github.com/rmusser01/tldw_chatbook/actions/runs/42441233109)
completed all 15 test jobs: **11 failed, 4 passed**. The failures are retained in
`/tmp/pr39147-ci-job-<job-id>.log`. Its aggregate stayed queued after the tests
finished; normal then force cancellation applied only to that failed task run.
No unrelated workflow was cancelled. The initial run is also cancelled, not green.

Actual job logs identify the remaining mounted fixture failures: Console P0
blocked-send and Retry (`128610694933`), prepared-start/manual Send
(`128610695192`), hidden-field Settings (`128610695208`), Stop controls at all four
sizes and key/palette/slash routes, visible-turn attention, Resend and Send
acknowledgement, and pending-interrupt/no-controller cases. Baseline reproduction
of the source refusal remains inherited-failure evidence; it does not justify
bypassing native preparation.

The fixture correction selects the configured private profile, gives ordinary
sends actual file-backed storage and explicitly selects their existing gateway-only
mode. Factory ownership tracks the exact file database, closes its real connection
and leaves replacement fields caller-owned. Native-start setup drains its original
scratch manager before borrowing the runtime's and captures configuration through
the mounted asynchronous producer. Direct-admission fixtures perform native
precapture; the pending-interrupt seed uses the controller's capture API. The
blocked-send helper awaits actual turn retirement instead of watching only the
legacy acceptance entry. The button observer watches the common visible request,
which precedes both the new native receiver and legacy dispatch. The no-controller
case explicitly removes/restores its passive store through the real runtime API.
Assertions and deadlines are retained.

With those real sources, ordinary agent-runtime-off Send exposed a product defect:
legitimate cold permission publication changed the captured MCP source, but its
receipt was retained only when an agent source existed. The native publisher now
hands the same exact service/store/candidate/binding/path receipt to the ordinary
owner under its live opaque claim. Consumption retains actual resource, configured
path/binding and unchanged-other-source checks, then clears the receipt. It no
longer attests permission-store function/code identity. Replaced sources/paths and
missing original task context still refuse before effects. A separate sequential
fixture failure was traced to a stale imported config-module alias; the snapshot
fixture now rebases that alias to the actual newly installed module along with its
existing callable aliases. The config-installation guard is unchanged.

Retained RED and scoped GREEN packets (cohorts overlap):

- Profile/database ownership regression: **1 failed in 7.22s**, tree
  `3d9810b78178e9be6ecdb235a8b48cc807658f3d`,
  `/tmp/pr39147-ci-round2-profile-fixture-red.txt`; corrected ownership plus the
  existing exact-original/replacement cleanup control: **2 passed in 11.79s**,
  tree `7dff96d8bb6151947a07f2a9a6129e24b5444760`,
  `/tmp/pr39147-ci-round2-profile-fixture-green.txt`.
- Agent-runtime-off/on cold permission receipt: **2 failed in 9.85s**, tree
  `eed4069574ca3c96b7520a56e982aa14f753dc7a`,
  `/tmp/pr39147-ci-round2-cold-mcp-red.txt`. A first corrected packet had **7 passed,
  1 failed in 27.05s**; sequential stale config-module setup caused the remaining
  failure, while the agent-on case alone passed. Those packets remain qualified.
- Final MCP/publication controls: **15 passed in 39.01s**, tree
  `a4dc048c5f3e06871594e63e04f748871b76d77d`,
  `/tmp/pr39147-ci-round2-mcp-full-controls.txt`. Both runtime modes pass together;
  controls retain missing-context, replaced receiver/path/factory/manifest,
  unqualified cold-source, replaced-controller, fresh switch and native dependency
  retarget refusals. No diagnostic plugin or bypass environment is active.
- Final mounted first-send/Retry, native-start/manual Send and Anthropic hidden
  Settings submission: **4 passed in 79.74s**,
  `/tmp/pr39147-ci-round2-final-native-controls.txt`.
- Send acknowledgement ordering at 80×24 and visible attention from Home:
  **2 passed in 32.71s**, `/tmp/pr39147-ci-round2-send-attention-controls.txt`.
- The old button observer missed native Send and failed its positive assertion:
  **1 failed in 33.15s**, `/tmp/pr39147-ci-round2-final-ui-controls.txt`.
  The corrected observer and complete pending-interrupt projection journey pass
  **2 cases in 125.55s**, `/tmp/pr39147-ci-round2-final-fixtures-green.txt`.
- Focused Dictate and focused Stop retain their no-unintended-Send assertions:
  **2 passed in 58.31s**, `/tmp/pr39147-ci-round2-final-send-observers.txt`,
  tree `2678e5cff2298af0ade850f72214c016799188cb`.
- Representative held Stop at 80×24, three sequential visible replies, keyboard
  Resend and no-controller Stop: **4 passed in 77.19s**,
  `/tmp/pr39147-ci-round2-ordinary-ui-green.txt`, preceding only final fixture
  alias/observer refinements. The earlier two ordinary failures in
  `/tmp/pr39147-ci-round2-ui-other-controls.txt` are not called passing.
- Late first-batch cleanup/recovery packet: **29 passed, 1 failed in 910.71s**,
  `/tmp/pr39147-ci-native-and-settings-targeted.txt`. Its 28 native cleanup/recovery
  controls and Settings case passed; its remaining manual-send fixture refusal is
  corrected and passes in the final four-case mounted packet above.

The final mounted native and projection packets use product/test tree
`2678e5cff2298af0ade850f72214c016799188cb`; subsequent task, architecture and QA
edits plus line-formatting adjustments do not change executable behavior. Commands run with
`PYTHONPATH=/tmp/pr39147-ci-harness`, `PYTEST_PLUGINS=pr39147_ci_overlay`, and
`PR39147_CI_TREE=<recorded tree>`; the in-memory loader covers private pytest
children. This remains local targeted evidence, not actual Linux CI or a full
suite. All changed Python files add **zero Ruff findings** compared with the
published task source; inherited debt is retained. `git diff --check` passes.
No size budget, timeout, assertion or test selection is relaxed in the branch.

Preservation: `/tmp/pr39147-ci-fix-start.json` still covers the original 374
states. Another writer added unrelated untracked `300` between heartbeats; the
extended `/tmp/pr39147-ci-fix-round2-start.json` and current audit cover 375 states,
plus exact real index entries/status and original dev HEAD. No real checkout file
was edited, and no new worktree or source export was created. AC #8 and task status
remain open pending a successful exact-head workflow. No full local suite,
external-provider call, other-platform UAT or newer-dev integration is certified.


## Third CI repair batch — concurrent read and streaming Stop

The [run on 3a2f2a4](https://github.com/rmusser01/tldw_chatbook/actions/runs/42450072567)
passes the admission-sensitive gate and previously failing Settings/attention/Stop
lanes, but UI lane 4 reports **170 passed, 1 failed in 851.12s**. Its remaining
failure is the focused Space-on-Stop case at 160×45: the button receives the key
and the provider submit reports stopped, but the test never observes the required
`Response stopped by user.` system row. Actual job log:
`/tmp/pr39147-ci-job-128637182697.log`. Other lanes were still running during this
repair; the workflow is not green and no active test job was cancelled.

That exact case passes on unchanged pushed code in isolation: **1 passed in
69.09s**, `/tmp/pr39147-ci-round3-space-stop-red.txt` (the exploratory filename is
not a claim that the test failed). A separate instrumented diagnosis passes
**1 case in 29.25s**, `/tmp/pr39147-ci-round3-space-stop-diagnostic.txt`; its real
controller snapshot has no concurrent preparation read and records the user Stop
normally. `/tmp/pr39147-ci-round3-stop-diagnostic.jsonl` retains the diagnosis.
Those isolated passes do not explain away or relabel the actual CI failure.

Source analysis finds an early return after cancelling current preparation reads.
It can consume Stop while an already streaming response still needs terminal
settlement. The new test parks an actual finite native-read worker concurrently
with a real persisted streaming response. App-scoped and same-session variants
fail before correction; the unrelated-session control passes: **2 failed, 1 passed
in 22.63s**, `/tmp/pr39147-ci-round3-concurrent-read-stop-red.txt`, tree
`f177d837fcb32c0cd871d99d7f7b1272699e7693` (unchanged product source).

The repair replaces that early success return with a retained cancellation result
and continues through existing native commit/stream Stop logic. It retains original
read attribution/cancellation, exact native settlement and all existing source,
permission, revision and recovery checks. It does not restore implementation
attestation. Repeated Stop records one explicit marker. The original finite-read
record remains unretired while its worker is blocked and retires only after
physical return; an unrelated-session read is not cancelled.

Verification on tree `10f668eeb6ace9a255955d118197c08b7126f94f`:

- New three-way concurrent-read regression, existing six direct/agent/sync
  dispatch-CAS settlement cases and the two repeated-cancellation read ownership
  controls: **11 passed in 65.84s**,
  `/tmp/pr39147-ci-round3-stop-read-controls.txt`.
- Original focused Stop Enter at 80×24 and Space at 160×45:
  **2 passed in 53.50s**, `/tmp/pr39147-ci-round3-mounted-stop-controls.txt`.
  The original assertions and five-second Stop wait remain unchanged.
- Ruff comparison across all branch-changed Python files adds **zero findings**;
  `/tmp/pr39147-ci-round3-lint.txt`. The new test formats cleanly and subsequent
  formatting preserves its AST. Existing formatter/lint debt is retained.

Commands use the existing Git-source harness with `PYTEST_PLUGINS=pr39147_ci_overlay`
and the recorded tree, without diagnostic plugins in verification. New regression
node: `Tests/Chat/test_console_durable_commit_offload.py::test_stop_with_concurrent_preparation_read_still_finalizes_live_stream`.
No assertion, required CI check, deadline or size budget is relaxed. The original
CI failure remains unqualified until the new pushed head is tested. No full local
suite, external-provider or other-platform run; newer-dev integration and merging
remain outside scope. The complete real index, dev HEAD and 375 preserved path
states remain unchanged. Existing ADR356 covers this direct Stop/lifetime fix.


## Fourth CI follow-up — native fixture custody and poll evidence

All 15 test jobs on [3a2f2a4](https://github.com/rmusser01/tldw_chatbook/actions/runs/42450072567)
finished: **12 passed, 3 failed**. The only queued job afterward was the redundant
aggregate. Normal then force cancellation stopped only that old failed task run;
its completed test evidence is retained. The Stop repair is published separately
as `591fcd64b1ec073dcacfc70fcb73e3cc54572ea4` and has not yet received a CI verdict.

Besides the Space case already documented, the actual remaining logs show:

- UI lane 1: **239 passed, 1 failed in 826.31s**,
  `/tmp/pr39147-ci-job-128637182834.log`. Resend's incremental content is already
  in the store, but the UI assertion after the existing 0.65-second observation
  still sees Preparing. No timeout or content assertion is changed to hide this.
- UI lane 10: **157 passed, 1 failed in 778.01s**,
  `/tmp/pr39147-ci-job-128637182863.log`. The interrupt fixture's direct controller
  seed reaches cold skill capture without a matching runtime request record and
  correctly refuses with `Original skill preparation custody changed; send again.`

The interrupt fixture now follows the existing mounted producer: native
precapture, `_admit_console_turn_to_runtime` and `runtime.wait_for_turn`. It retains
the actual request/task custody required by cold skill preparation. The production
custody and selected-source checks are unchanged. Warm-catalog local passes of the
old direct seed did not certify the cold route.

The Resend test is strengthened to assert that the exact existing timer survives
its held admission callback. It adds actual run state, timer presence, sync flags
and in-flight/custody counts to failure output. It retains its original content
assertion and 0.65-second observation. No production Resend change or latency fix
is claimed. An unchanged-head local control passes that Resend case in 19.31s;
its companion interrupt test fails with a **different** Textual message-pump
`WaitForScreenTimeout`: **1 passed, 1 failed in 118.69s**,
`/tmp/pr39147-ci-round4-mounted-controls.txt`. That is retained as a failure rather
than relabeled as reproduction of the CI custody error.

Final candidate tree `9d0f7ba3ce6912d656707ede2657da35239414be`:

- Strengthened Resend and the complete six-journey interrupt projection fixture:
  **2 passed in 193.17s**, `/tmp/pr39147-ci-round4-fixture-controls.txt`.
  Individual calls: Resend 26.39s, interrupt journeys 159.73s. Original waits,
  process/UI limits, assertions and real cleanup remain in place.
- Supported native skill capture, prepared-skills-without-selected-source refusal
  and unknown cold-source refusal: **3 passed in 5.03s**,
  `/tmp/pr39147-ci-round4-skill-source-controls.txt`.
- Ruff comparison over all branch-changed Python files has **zero new findings**,
  `/tmp/pr39147-ci-round4-lint.txt`. Changed lines format cleanly; the pre-existing
  unrelated assertion formatting in the interrupt file is retained. Diff checks
  and exact preservation auditing pass.

The controller's broad size pin already exceeds its ceiling on the baseline
(29648 lines versus the 29184-line pin). The Stop repair measures 29598 lines and
changes no pin; this inherited whole-module debt is outside the selected CI size
checks. It is not called a passing full-suite result or repaired by raising a
budget. No full local suite, external-provider or other-platform UAT was run.

This batch changes fixture tests, task notes and QA evidence only. Existing ADR356
applies; no new architecture decision or production authority change. Dev HEAD,
full real index/status and all 375 recorded dirty/untracked states remain intact.
The next workflow must verify the actual new pushed head; old-head successes do
not close AC #8 or certify the absence of intermittent bugs.


## Fifth CI follow-up — aggregate attribution and task-only repair

[Run 42456510388](https://github.com/rmusser01/tldw_chatbook/actions/runs/42456510388)
on `d5d10f866bff7c85d409953dc48ea99149d5de7f` passed **all 15 test jobs**, including
UI lanes 1, 4 and 10. The workflow is nevertheless **not green**: aggregate job
`128681770614` failed its diagnostic inventory, index-plan census, Textual worker
and markup checks. Its actual log is `/tmp/pr39147-ci-job-128681770614.log`.
The unrelated merge-queue tick was not cancelled.

Immutable baseline `e08627ac38392fb890984169376b1c242c3a2f3e` reproduces all four
aggregate failure classes with the unmodified checkers. Git-source baseline
packet: `/tmp/pr39147-ci-round5-baseline-controls.txt`; per-source attribution:
`/tmp/pr39147-ci-round5-static-attribution.txt`. Both use in-memory Git blobs,
not the dirty checkout as the reference. This is local static evidence, not a
second baseline Linux workflow run.

- All three missing workflow indexes already exist in baseline schema/migrations.
  No index definition was added by TASK-39147.
- All seven flagged workflow UI functions are already absent from the baseline
  DOM census. Setup's button handler grows from six flagged lookups to seven in
  this task; the other counts are unchanged.
- Console Undo's three markup notifications and its entire `chat_screen.py` are
  byte-identical to baseline.
- All nine stale diagnostic owner rows are already stale on baseline.
  TASK-39147 adds one constant-only warning, `could not persist native child
  failure`, in AgentService, with no interpolated user input or new disk sink.

The task-owned correction makes worker-refusal presentation query surviving
notice widgets rather than requiring a removed widget. Existing retry state,
submitted objective and callable worker admission remain checked. The native-child
owner inventory row is updated after statement review; the other inherited rows
are deliberately retained. No census allowance, assertion, wait, resource fence,
revision check or recovery protection is weakened.

Targeted regression on tree `6cc6cdc64cdc010e777d9d29dceb7dd8e866f4d1`:
**1 failed, 1 passed in 3.73s**. The removed-notice case raises `NoMatches` from
the original error handler. Corrected executable tree
`3d1611e6b22d40744e438d75e2fec406c6324d6f`: **2 passed in 3.74s**,
`/tmp/pr39147-ci-round5-worker-green.txt`. RED packet:
`/tmp/pr39147-ci-round5-worker-red-verified-source.txt`. Two preliminary harness
runs accidentally reused a cached single-case test module; their one-case passes
are not candidate evidence. The rebuilt harness disables cached pytest bytecode
reads and confirms both parameterized nodes execute from the Git source.

`/tmp/pr39147-ci-round5-verification.txt` records zero new Ruff findings in both
changed Python files, clean formatting, `git diff --check`, and restoration of
Setup's flagged lookup count to the baseline six. This resolves the task delta,
not the inherited worker census failure. No full local sweep, external provider,
other-platform UAT, newer-dev integration or merge is certified.

Earlier temporary snapshots/harness files disappeared between wakes. Per the
preservation requirement, scope was reconstructed from the current branch's 96-path
diff against baseline, and fresh evidence was captured **before editing**:
`/tmp/pr39147-ci-round5-preservation-start.json`. It covers dev HEAD e08627a,
the complete real index/status and all 375 current dirty/untracked path states.
The new audit proves those states unchanged since that capture; it cannot recreate
missing earlier snapshot bytes. Only private Git objects/index were edited.

The user was asked whether the minimal inherited baseline repairs may be included,
because the standing authorization excludes unrelated file changes. That scope
answer remains pending. AC #8 stays open. A new full selected CI dispatch is held
until those known aggregate blockers can be addressed, avoiding a knowingly
redundant run; no active run or unrelated automation is changed.

## Sixth CI follow-up: approved minimal inherited gates (2026-11-08)

The user explicitly answered **“Include the minimal baseline CI repairs.”** The
private index advanced from published `6aaaeb33ab`, preserving the real checkout.
The task plan and AC #20 were recorded before implementation. No new ADR is needed
for existing view contracts, literal rendering or evidence for unchanged indexes;
ADR356 and the UI conventions remain applicable.

The previous summary omitted four **only-in-rebuild** diagnostic owners already
listed in aggregate log 128681770614: `Chat/console_workflow_preparation.py` (1),
`Chat/console_workflow_runs.py` (8), `UI/Console_Modules/workflows.py` (1), and
`Workspaces/worktree_activity.py` (1). All eleven statements and their context are
now reviewed: fixed native-return/observer diagnostics and allowlisted view error
metadata. The eight stale owner rows were compared with source `c5ceca1247`, whose
per-file counts/digests were verified to match the pin. Review separates formatting
and reindentation from fixed-string changes, hook attribution/type-only additions
and existing exception-capture forms. No logging source, sink topology or privacy
metadata changes here. Packets: `/tmp/pr39147-ci-round6-diagnostic-statements.txt`
and `/tmp/pr39147-ci-round6-diagnostic-review.json`.

Mounted partial-DOM regressions hold actual finite reads, remove controls and then
release them. Both history/checkpoint success and refusal paths, review refusal,
planning completion and three queued-button paths reproduce `NoMatches`; three
Undo cases prove bracketed external text was interpreted as markup. Isolated RED
on tree `faf92ff9c741f2b87c1d02ba1aaf07272609d6ec`: **12 failed in 8.35s**
(`/tmp/pr39147-ci-round6-dom-undo-red-isolated.txt`). Earlier fixture attempts had
automatic mount/resume workers and are discarded as clean regression evidence.
NoMatches handling now contains presentation teardown, retains existing finally
cleanup and clears stale checkpoint review. Worker/markup census counts and the
checkers themselves are unchanged. The success-count notification is literal too.

The real current AgentRuns schema is initialized in SQLite `:memory:` and populated
through native workflow/phase/task APIs. With `sqlite_stat1` explicitly absent,
`EXPLAIN QUERY PLAN` selects `idx_workflow_conversation` for the production history
page (no temp sort), `idx_workflow_open_interval` for the open-interval lookup, and
covering `idx_workflow_task_phase` for phase claims. Each dropped-index control
becomes a scan; history also needs a temp sort. No schema/index/DDL change and no
pre-convention exemption. Three plan cases pass in **0.74s**; see
`/tmp/pr39147-ci-round6-index-plans.txt` and `Tests/DB/test_workflow_runs.py`.

Final executable tree `7867d352de1d7937b0ad32892ed6a47232cc0925`:

- **39 passed in 10.48s**: complete workflow setup-worker/DOM-loss and workflow
  storage files, including all three plan pins (`round6-bootstrap-green.txt`).
- **12 passed, 3 deselected in 9.37s**: nonmounted Undo logic and literal text
  (`round6-undo-unit-green.txt`). This is an explicit targeted cohort.
- All **four unchanged static checkers return zero**: 691 diagnostic owners/16
  sinks; 327 declared indexes and census rows/111 plan pins; no new worker contract
  violations; no new markup violations (`round6-static-final.txt`).
- All six changed Python files add zero Ruff findings. Five complete modules format
  cleanly; all four changed Undo notification ranges format cleanly while inherited
  whole-chat-screen format debt remains. Diff check passes (`round6-verification.txt`).

The broader local attempt had **51 passed, 3 failed in 14.87s**. The three mounted
Undo fixtures refuse `raw_source_selection_changed` during app construction, before
Undo behavior. Separate pre-repair `6aaaeb33ab` controls reproduce all three
(**3 failed in 7.52s**); separate candidate Undo-only execution gives **12 passed,
3 failed in 8.98s**. Packets: `/tmp/pr39147-ci-round6-targeted.txt`,
`/tmp/pr39147-ci-round6-undo-baseline.txt`, and
`/tmp/pr39147-ci-round6-undo-isolated.txt`. These files are outside the selected CI
census; no extra baseline fixture repair or passing mounted-Undo claim is made.
The focused cohorts do not erase those failures.

Self-review traced the workflow callers and existing generation/revision checks,
kept read joining and worker admission callable, reviewed the substantive diff
independently of indentation, and confirmed no changed assertion budget, scanner,
worker/markup allowance, migration or native authority/resource fence. Subsequent
candidate changes are evidence Markdown only. The complete real index/status,
HEAD `e08627a` and all 375 preserved states audit unchanged. This source-overlay
harness is partial local evidence, not full-filesystem/isolated-Python or Linux CI.
No full local suite, external provider, other-platform UAT, merge or newer-dev
integration is certified. Overall CI on the next published head is still pending;
TASK-39147 remains In Progress with AC #8 open.

## Seventh CI follow-up: supported runner for the immutable build (2026-11-08)

[Run42493042973](https://github.com/rmusser01/tldw_chatbook/actions/runs/42493042973)
tests exact `1a921835b6c26be0512482696538734577c5846f`. All15 test jobs succeed;
aggregate128785469019 confirms diagnostic inventory, index plans, worker and
markup gates now pass in Linux CI. The aggregate fails earlier during
`actions/setup-python@v5`: **3.12.11 x64 was not found for Ubuntu26.04**. The
subsequent Mermaid check says to use its pinned Python because setup failed.
Actual packet: `/tmp/pr39147-ci-job-128785469019.log`. The separate merge tick
128791336060 remains queued and is not cancelled. Overall run success is absent.

The [official action manifest](https://api.github.com/repos/actions/python-versions/git/blobs/25851576302b86c37ffa01ee654d024d6d06b6de)
(blob identity, not a commit URL) provides Python3.12.11 for Linux x64 Ubuntu22.04
and24.04, with no26.04 build. Recorded source and exact download entries:
`/tmp/pr39147-ci-round7-python-support.json`. This time the stable version pin
outlived the moving runner label. Pin only `derived-artifacts.runs-on` to
`ubuntu-24.04`; keep the existing `3.12.11` build pin, artifact bytes/manifests,
checker assertions, all test selections and budgets unchanged. ADR124 applies;
no new architectural contract or ADR is introduced.

The existing Python-pin workflow regression adds the supported runner constraint.
RED tree `4d4f92eb4fa820c961ad0c4a80150b9f79db30c0`: **1 failed in0.81s**, solely
`ubuntu-latest != ubuntu-24.04` (`/tmp/pr39147-ci-round7-runner-red.txt`). GREEN
executable tree `9f6394c7085310d3acb49bc27011a9a9d5a71d2e`: **43 passed in5.17s**,
complete derived-artifacts and dispatch-safe workflow test modules, including
existing negative gate/membership controls (`/tmp/pr39147-ci-round7-workflow-green.txt`).
Parsed YAML comparison proves the only semantic change is that one runner label.
All steps, other jobs, Python pin and budgets compare equal. Zero new Ruff findings
and diff checks pass (`/tmp/pr39147-ci-round7-verification.txt`); existing full
CI-test-module formatting debt is retained. Self-review confirms no runtime,
asset, checker, production source or authority changes. Subsequent changes are
Markdown only. These are local configuration/manifest checks; actual supported
Ubuntu runner execution remains pending on the next pushed head. No full local
suite or new live UAT. Prior three mounted Undo profile-setup refusals and other
qualification limits remain recorded above. Real dev HEAD, full index/status and
all375 preserved states remain unchanged. Hold the next dispatch until the old
run including its merge tick completes; never cancel that tick or another run.


## Eighth CI follow-up: passive recovery under a configuration rebuild (2026-11-08)

[Run42501214598](https://github.com/rmusser01/tldw_chatbook/actions/runs/42501214598)
on exact6350ea97c109c074bd1d179f85803aff74236ecc finishes fourteen test jobs green
and UI10 red. Actual job128794172577 has157 passes and one failure in726.33s:
`test_rule_phase_events_project_only_the_mounted_current_session[rebuild]`.
The watchdog captures transcript sync calling passive trace recovery, then
`_ensure_console_chat_controller`, provider selection, `load_settings` and the held
configuration lock. Log: `/tmp/pr39147-ci-job-128794172577.log`. The passive method's
body is identical on e08627a and6350 (attribution packet
`/tmp/pr39147-ci-round8-attribution.json`); the user-approved minimal inherited
scope covers this repair. Its unmodified isolated case passes locally in20.01s,
which is retained as an intermittent reproduction, not evidence of a fix.

The existing response-rule test now explicitly refreshes the real mounted recovery
region while the configuration lock is held. It retains the1s Stop wait,2s watchdog,
current-session/other-session assertions and physical cleanup/draft checks. Existing
live callback tests publish the installed controller and test both provider display
readers and no-controller projection. The corrected test-only tree
`1b8e8cd4dd771ab98cdc97cf7ca3e3eddb239c29` gives **7 failed in24.62s**, solely
expected UI blocking/initializer calls (`/tmp/pr39147-ci-round8-region-red.txt`).
An earlier read-only-property fixture assignment error is not clean regression
evidence. Passive trace projection now uses its already injected live getter;
both provider display builders read the installed screen controller once. None
produces inert recovery display. Recovery action initialization/dispatch, actual
preparation/hold/message reads and permission/revision/resource fences are unchanged.
ADR220 applies without a new ADR or abstraction.

Executable tree `a53cc1078a23f12f043b42ca636550aca25e5fb9`:

- **9 passed in27.60s**: full four-variant held/no-controller response-rule function
  and five live recovery callback tests (`round8-recovery-green.txt`).
- Separate trace/provider UI recovery controls: **29 passed,1 failed in19.21s**
  (`round8-recovery-controls.txt`). The mounted real-screen provider fixture refuses
  `raw_source_selection_changed` during app construction. Exact unchanged6350
  reproduces that refusal (**1 failed in9.00s**, `round8-provider-baseline.txt`).
  No mounted-provider pass or additional baseline fixture repair is claimed.
- Formatting-only test lines yield tree `051adf8e2b7e363a1521d979fd8b8aae5b580621`,
  with equal test AST and byte-identical production sources. Every changed Python
  range formats cleanly with zero new Ruff findings; inherited whole-module format
  debt remains. Logger AST, recovery action callbacks, initializer and full workflow
  compare equal (`/tmp/pr39147-ci-round8-verification.txt`); diff check passes.

Self-review traces the full recovery refresh and all three passive readers; no
assertion, check, inventory allowance, Python pin or timeout was relaxed. The real
e08627a checkout, full index/status and375 preserved path states audit unchanged.
Prior mounted Undo failures remain qualified separately. Partial source overlays
are not actual Linux or full-filesystem/isolated-Python qualification. No full local
sweep, new external-provider/other-platform UAT, newer-dev integration, PR or merge.
Subsequent candidate changes are Markdown only. AC22 is checked; task In Progress
and AC8 open. The existing aggregate and separate queue tick must finish before
another dispatch; they are not cancelled. Actual overall success on the next
published head remains required.


## Final green CI and task closeout — 2026-11-08

[Derived Artifacts run42509127239](https://github.com/rmusser01/tldw_chatbook/actions/runs/42509127239)
completed **SUCCESS** on exact [2d4241b78e51bb59f3808fa805bfca0f51ac4e2f](https://github.com/rmusser01/tldw_chatbook/commit/2d4241b78e51bb59f3808fa805bfca0f51ac4e2f),
tree `78c789cf0d79d191f6f8e9508df79aaf61c9f462`. The final result was verified
with `gh run view` including jobs, and the remote task ref was independently
confirmed at that same commit. All **17 jobs succeeded**:

- All15 selected test jobs: both PR lanes, Console P0, Dreams tracking/boundaries,
  and all10 UI shards. The formerly failing UI10 job128818142718 succeeds.
- Aggregate [128832706229](https://github.com/rmusser01/tldw_chatbook/actions/runs/42509127239/job/128832706229)
  succeeds, including all static/artifact checks and unchanged pinned Python3.12.11
  on Ubuntu24.04. Generated sheets/Mermaid, profile/diagnostic/markup/UI censuses,
  task IDs/readability, table/index decisions, workers and UTC writers pass.
- Separate queue tick128834131516 succeeds; overall completion was awaited.

Only one dispatch was made on this source head, after old run42501214598 and its
queue tick finished. That old run's aggregate128810725154 failed only for known
UI10; its supported runner and artifact checks already passed. Earlier runs and
local failures retain their own exact-head qualification; none are relabeled green.
The final full run receipt and preservation packet were retained locally as
`/tmp/pr39147-ci-exact-head-green-run.json` and `/tmp/pr39147-ci-exact-head-green.json`.
The public run/job/commit links above are the durable result references.

The Backlog CLI now checks AC8 and sets TASK-39147 Done with all22 acceptance
criteria checked. The task's native lifecycle, cancellation, permissions, concurrency,
restart/storage, ordinary-chat, UI, lint/changed-range formatting, documentation/ADR
and self-review evidence is recorded in this report, the linked plan and task notes.
The original publication exclusions are superseded only by the user's explicit
2026-11-08 request for task/QA closure and PR setup; merge remains excluded.

This closure changes only the task and QA Markdown. All executable/build inputs
remain byte-identical to qualified2d4241b. Its run is exact-source evidence, not a
claim that the later documentation commit or PR merge result was tested by that
same run. New PR-head checks are separate. The completed monitor was deleted and
is not restarted. A draft PR against dev exposes the existing integration limit:
the inspected comparison contains94 inherited baseline commits before the9 task
commits,1057 paths since the merge-base, and3584 base-side commits. No newer-dev
integration, rebase, merge or auto-merge qualification is claimed.

Full real-index/status/HEAD preservation and all375 captured path states (including
another writer's untracked300) pass the final audit. This proves continuity from
the fresh round5 capture; earlier lost temporary bytes cannot be recreated. CLI
status editing uses temporary task metadata, then only the two Markdown blobs are
published via the existing private index/commit-tree/CAS and a normal task-branch
push; the actual checkout and index remain untouched.

The scoped Done result retains all earlier limits: no full local suite, new remote
provider or other-platform UAT, latency certification or unrelated archive lineage.
Three mounted Undo and one mounted provider local app-construction refusals are
reproduced on the respective unchanged parents; selected CI does not erase them.
Inherited whole-module formatting/lint/size debt remains. Earlier timing/setup/CSS
failures and their qualified source versions remain in the report above.

## PR #3480 current-dev rebase verification (2026-11-08)

Target dev: `36a21dc4d270940e3576b49487a71b37c78896f8`. The private replay retained 101 non-equivalent source commits and two plan/ADR commits; three equivalent patches were omitted. This is a new integration, separate from the standalone Linux result on `2d4241b78e51bb59f3808fa805bfca0f51ac4e2f`.

| Scope | Actual local evidence |
| --- | --- |
| Installed and legacy AgentRuns migration | 45 passed on executable tree `a52a36c3a959355865f617cbb69545293406dbaa`. Real SQLite shipped review v25, workflow v26–31, exact historical v25–30, cancellation after DDL return, mixed-catalog refusal and rollback. |
| Send hook ownership | 36 passed on `a52a36c3a959355865f617cbb69545293406dbaa` with directory fixtures and repository-root metadata restored. Warm reads, foreign/stale owner, final revocation, actual approved launch and retained plugin requirements. Diagnostic tracing identified the extra footprint read; no tracing was used for the pass. |
| Current review dispatch/check boundaries | Both review modules' 90 cases passed on `a52a36c3a959355865f617cbb69545293406dbaa` with normal directory fixtures. Three accompanying child-scope controls aborted before child behavior on the `raw_source_selection_changed` fixture boundary; this is not a passing whole-cohort claim. |
| Plugin references | Eight initial, injected-body, selector and resumed controls passed using real files under an admitted private workspace and the original resolver. The paused preparation uses a fresh pending plugin ID without replacing its frozen requirement ceiling or rereading the file. |
| Source builders and ceilings | On `b855d4004e65f4bf53413553dd18d1526511f889`, 201 size checks pass, 15 tracked CSS outputs equal builder bytes, and the remote worker builder equality test passes. Six size pins only decreased; the worker remains below the original pin. Configured Ruff introduces zero findings relative to the two source origins; 261 existing findings remain. Changed Python ranges were formatted with AST parity. |
| Workflow and static gates | Preflight runner regression failed on `e1748510635709ce1db6dfa8462c7fef9a62876c` and passed after relocating the supported Ubuntu runner. Parsed YAML matches current dev except the preflight runner. The immutable-source static run passes diagnostic inventory, index plans, worker/DOM, timestamp writers, markup, schema, ADR, backlog files and UI census. The three reported task-ID conflicts were resolved by moving only unmerged claimants, leaving their landed twins unchanged. |

Local source overlays supply immutable executable bytes and targeted resources. They do not establish full filesystem, bare-interpreter or Linux PR qualification. The three planning uncertainty assertions also fail on unchanged standalone `3184178ac0849a72546b0c1abb54b4a4664e2751`; production retirement logic is unchanged. The mixed-tree backup probe (restored child loaded old schema81 rather than the candidate's86) is excluded. A broad RawCLI/SSH/first-send local batch was stopped after recorded failures and cleanup timeouts, so no pass is claimed for that batch. A macOS owned-MCP fixture refused `root_boot_identity_unavailable` before its behavior. The actual Linux preflight will run the full Mermaid build and PR selector checks. No full local sweep, external provider or additional platform UAT.

Real checkout HEAD, whole index, status and 375 captured path states were independently reverified against the preservation capture. This evidence applies from that capture only. No checkout, worktree, stash, source export or unrelated ref change was used. Actual PR-head CI and the review/thread checks remain pending at publication.

### Follow-up after publishing the rebase

- The pinned Ruff 0.16.6 scan of all 2,983 production Python files on `7bdb179f5f80fc20b5e1a00fbfdc93ad10327d34` found zero F findings but 231 B905 and 46 ASYNC109 (ceilings 213 and 45). On executable tree `884650848c5fbf32f54a0c35547b9e18cea4d81c` both are at their existing ceilings and every other rule is within its baseline. The two intentional prefix pairs preserve partial-submission cleanup and finite-read-prefix semantics; every other changed pair has a matched length. No lint or size ceiling increased.
- Seven changed-source reference tests on the rebased head discarded the body but lost the original refusal copy. All seven pass on the prior standalone head and after restoring the copy; the final tree also passes three real managed Git reference variants. The separate zero-root control aborts on both rebased and unchanged standalone heads when it traps ordinary turn admission. Its assertion and the ordinary admission fence remain unchanged.
- Final executable tree: 11 focused finite-return, release and recovery cases and all 201 size checks pass. The local source overlay was corrected to leave installed dependency resources in the virtual environment alone; an earlier collection error is excluded. Four other native legacy/plugin hook cases fail on both the follow-up and unchanged rebased head; no passing combined cohort or repair is claimed for those.
- The nested backup retry pinned the child processes to the candidate. Its rebackup still refuses at the complete-inventory preview; the unchanged dev temporary-media restore control also hits a complete-inventory refusal. The unchanged standalone workflow restore passes in this bootstrap. This is not a passing rebackup claim. The real migration and bounded recovery results above are separate.

GitHub's automatic pull-request runs and current-head review remain the qualification for AC23.

### Windows checkout follow-up

[Windows managed-plugin job 128906761569](https://github.com/rmusser01/tldw_chatbook/actions/runs/42538088446/job/128906761569) failed on `7bab6d66181faf818dbb2b3adfde06b3d209149f` before setup or tests. The log reported `Filename too long` for the retained-temporary archive manifest in the historical `hook_native_final` QA receipts. The logged checkout root and the 229-character repository path yield a 262-character absolute path.

Managed plugin qualification and Eval recovery both perform a full Windows checkout before their tests. The first step on Windows now enables Git long paths, using the same setup as the GGUF import evidence workflow. Eval recovery's existing private clone and ownership setup follow the initial checkout as before. A parsed workflow comparison verified that removing each added Windows step restores the previous workflow exactly. The historical receipt, manifests, and ruling digests are byte-identical. Actual Windows Actions results are pending.

### Qualification and startup follow-up after Windows checkout

All three [Eval recovery jobs](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733203) passed on fd2872f9500b67d388566c77d20ff6dda4f790a0. All three [managed plugin jobs](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733161) reached the suite and failed the same three SQLite inventory checks. The unchanged dev baseline reproduced those checks for the detached import path. The macOS job also reported separate provisional initialization and owned post-compaction hook failures.

The workflow catalog replays use the existing memory-only recovery schema owner. Current and legacy agent-run schema hashes and all 118 catalog rows matched the preceding head. Detached import retains the owner check, validates the source and the normalized output, copies committed WAL contents through the private backup boundary and leaves the source files in place. Local immutable-source controls passed 61 SQLite inventory cases, 38 workflow migration cases and six focused detached import cases. The earlier broader batch hit local overlay limits for the portable package and the nested workspace revision test. The revision test passed with the exact private-profile child. The four portable cases have no passing local result.

The [performance job](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733128) reported four budget failures. Deferred capacity and dictionary imports brought the local app import census to 686 against the strict limit of 688. Screen preimport measured 556 modules and 426563 lines against limits of 558 and 427830. The original mounted CSS census passed after selecting workflow action buttons by class. All 15 derived stylesheets reproduced. The warm UI-ready census remains open, with 1060 modules against the limit of 1038 in the actual failed job. Budgets and test thresholds are unchanged.

Both [Linux Tool Recovery jobs](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733126) failed before the Stop command when the held prompt did not stream. The mounted local reproduction traced the refusal to a resolved MCP definition ceiling paired with the unresolved composition mode. Ordinary and workflow admission now mark the ceiling captured after consuming the existing publication and checking the original source. The mounted Stop test passed locally in 28.20 seconds with its existing assertions. No new CI qualification is claimed.

The executable tree cd1d9e8a482efe46396ad2e731012075c284d90e passed the pinned Ruff census, the exact Mermaid build and PR selector mutation controls. Targeted controls use in-memory Git sources on this host. Actual PR CI and review remain open.

#### MCP and macOS controls

Eleven focused composition tests passed on the MCP state fix. The separate 90-case MCP batch reported 57 passes and the same 33 failing node IDs on the published head and the candidate. Those local cases remain outside the passing claim.

Using the real Darwin boot identity, the [macOS managed-plugin failures](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733161/job/128911750826) reproduced on the published head in 20.11 seconds. The provisional initialization test now registers and approves a real legacy Stop in its temporary profile. It records the original prepared notification and checks the subprocess receipt. All five provisional cases passed on tree 1da20e90165c8a421f3ef8d86c3073f50d5395e0 in 40.88 seconds. The existing success, terminal, denial, disconnect, cancellation, cycle, and audit assertions remain.

The standalone post-compaction case still fails on the rebased PR. The identical test passed in 27.22 seconds on the exact dev base 36a21dc4d270940e3576b49487a71b37c78896f8 with the same immutable loader and real Darwin boot identity. Diagnostics found an unresolved pending MCP selection with an empty tool ceiling at prospective hook composition. The owned snapshot still contains the required hook and MCP component. The provider keeps the ceiling, so the registry offers no owned MCP entry and the required hook refuses. The committed memory survives. This diagnosis does not qualify a production repair.

#### Derived Artifacts admission and UI failures

The [admission-sensitive project-chat job](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733304/job/128924516310) failed fenced UAT on fd2872f. An unchanged local run failed at the same project-instruction disclosure. A separate diagnostic run traced the refusal to the same unresolved composition mode receiving a resolved MCP ceiling during ordinary admission. On executable tree 434329ad4a0f31f4faa85609a267608b31471af0, the four mounted variants and their support tests passed (43 cases in 154.48 seconds). The repository evidence checker accepted all four receipts. The missing-README control failed after its actual file read in 35.91 seconds and the negative evidence check passed. The separate synthetic report tests require a direct file path that this partial source overlay lacks. They have no local pass claim.

[UI Fast Lane 6](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733304/job/128924516547) failed three cases. The trailing paste case passed in the targeted local run. The held-Enter case failed in the combined two-case run after the first send completed. The same case passed in isolation on 7349dd1c in 25.02 seconds and on the exact dev base in 15.83 seconds. The combined failure remains open. The character picker failure reproduced at screen construction. Its fixture now attaches the same in-memory Console store to its runtime before construction. All thirteen character tests passed on 7349dd1c5af2dd51fc5741fd93442142cc6e0314 in 54.40 seconds, including the original character provenance assertions.

The [Console P0 job](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733304/job/128924516232) failed two cases involving the first system-prompt send and provenance retry. Their verification remains open.

#### Architecture review and further P0 attribution

The [architecture-review job](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733304/job/128924516335) failed six checks. The same six passed on exact dev and failed on exact fd2872f under the immutable local loader. The PR had two private participant imports and exceeded the Chat identity pins (69/56 code, 35/27 defaults, 55/43 keyword defaults). The Agents/Chat checks also identified upstream import edges and 46 private crossings.

The two callers now use the existing public `core_operation` symbol. The previous private name was an alias of this same callable. The participant guard module passed all three cases, and the original native child launch and unused admission cancellation cases both passed on cb12bd32873a910c009ef3fd4e919efcb8157912. No pin, allowlist or boundary test was changed. The five other architecture assertions remain open.

The first system-prompt case passed in isolation on exact dev and in a traced isolated candidate run. Combined candidate runs showed mixed results. The earlier configuration-save message was outside the failing case. The more focused trace captured a provenance retry refusal at the original skill-preparation custody check. Another traced combined run passed both cases. No P0 fix or clean combined qualification is claimed.

#### Additional UI failures and cold preparation attribution

[UI Fast Lane 5](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733304/job/128924516581) reported eight failures. The existing character fixture repair also passed both targeted roleplay workspace cases locally. The background-tab completion and paste-token cases passed together on c9cd79e82493ecc5bc2035ea0a74d577a593a205. The draft-from-left-tab case still failed in that three-case run. The bound-skill compaction forecast reproduced the capture-source refusal on the candidate and passed on exact dev.

The existing cold mounted ordinary-receipt test reproduced the exact skill custody refusal. Its queue-before-publication control passed. The mounted admission path copies the captured configuration while adding its independently issued archive-recovery proof. This leaves the deferred witness pointing at the original configuration. Automatic approval review rejected a proposed witness rewrite because it could weaken the exact custody check. No rewrite was applied. The cold mounted case remains open.

[UI Fast Lane 11](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733304/job/128924516672) reported a pre-reservation discard and a stop during tool composition. Both unchanged cases passed together on c9cd79e82493ecc5bc2035ea0a74d577a593a205 in 26.03 seconds. This is targeted local evidence.

#### Cold mounted request repair

The follow-up preserves the original captured configuration and witness through registration. The mounted UI carries its separately issued archive origin on the request. The original archive proof registry accepts and releases the same proof. Only the existing execution copy receives it. The existing deferred custody predicate, origin checks and publication paths are unchanged. This approach does not rewrite the witness.

On d2e3cb7060144b20fb2f23e723d9b7653cd1194a, the cold mounted, queue-before-publication and changed-trust controls passed in 57.33 seconds. The mounted test also asserts that registration retains the exact original configuration, the issued proof is the same in execution and it retires at completion. The first system-prompt send and provenance retry passed together in 36.10 seconds on that tree. Both stock received proof routes and the Explore Home visibility case passed in 40.96 seconds on 6133c121a9c44da7d81e67de066a47fb3219164d. The only intervening executable-tree change adds assertions to the cold mounted test. Three broader archive dispatch controls stopped at the same original agent storage source refusal on the candidate and its immediate baseline. Those cases have no local passing claim.

[UI Fast Lane 4](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733304/job/128924516621) failed five cases, including the Start Chatting visibility variant with the same custody error on the published head. It also reported two quit-privacy waits, an automatic compaction forecast refusal and a held non-append edit wait. The lane remains open.

The unchanged Start Chatting visibility case passed in 38.02 seconds on d2e3cb7060144b20fb2f23e723d9b7653cd1194a. The other UI lane 4 failures remain open.

#### Storage architecture follow-up

The [architecture guard job](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733304/job/128924516353) failed four checks on fd2872f. The exact private baseline 2bce12748fb59cad5eb80d43c61f3dd740471a8c reproduced both storage guard failures. The legacy catalog loop target was bound by Python at module scope but the guard did not recognize that form. The catalog builder now keeps its intermediate state local. The guard itself and the recovery consumers are unchanged.

The already staged SQLite owner repair replaces the four published memory connections with the existing installed-schema helper. The exact private census instead reported four additions, two removals and one count change. The inventory now records the private memory owner, the separate staged input and normalized output, and the two read-only workspace file opens. It removes the superseded direct staging connection and moves the memory connection row to its actual caller. Neither the inventory scanner nor the filesystem code changed in this follow-up.

On d1705d28e18f01d2844a06dbb21e6656cf82b03b, the two architecture modules passed 22 cases in 32.85 seconds and the workflow migration module passed 38 cases in 7.42 seconds. Hashes matched for all 88 current and 30 legacy catalog entries. Ruff 0.16.6 passed the changed module's fatal checks. Mermaid outputs and the 12-lane, 464-target selector checks reproduced with the seven mutation controls. The two ChatScreen size checks remain open at the existing 24617-line ceiling. This evidence is local.

[UI Fast Lane 10](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733304/job/128924516698) failed the held-run layout check before its first stream chunk and the cross-tab send refusal case while waiting for the other send. It passed 133 cases. The layout log shows a controller-submit validation error, and the exact failure cause still needs comparison against the staged admission repair.

#### Screen projection limits and UI lane attribution

The two exact architecture checks reproduced on 68af35b17351454fa4b9c7c72f654c4aa9c29938. The screen measured 24718 lines. Removing the already separate Undo All plan and per-root outcome logic from the screen exposed a second limit at 744 direct methods against 741. The final change uses the existing review selection module for the planner, revalidation and worker apply sequence, and existing Console wiring for the lazy workflow controller and parsed command dispatch. The screen still owns the Textual event, confirmation, currentness callback and result publication. Native custody, disk preflight and per-path reversion follow the original paths.

On 29315c5950d64646f00b9887a989896e1f0c49d4, the screen measured 24520 lines and 741 direct methods. Both complete architecture modules passed all thirteen cases. Twelve focused Undo All controls and the real mounted native-card case passed, including confirmation, the original file and provider, persisted revert history and worker retirement. Three older mounted fixtures report `raw_source_selection_changed` on both the private baseline and final candidate. The first workflow command run could not read a tracked CSS file through the partial child loader. The grammar check passed. Pinned Ruff fatal checks passed on the four edited Python modules. The original screen preimport control passed with 556 modules and 426564 lines against 558 and 427830.

The held-run layout case from UI lane 10 passed in its exact candidate private-profile child on d1705d28 in 22.19 seconds. The first local run had failed to collect the missing checkout test path. The cross-tab spoken Send case passed on exact dev in 14.86 seconds and reproduced the wait on the private candidate. Its real mounted source check refused legacy preparation after the fixture replaced `get_kill_switch` on the managed MCP service. The test never reached its held read. This is an attribution, and the cross-tab case still fails on the private candidate.

Further exact-head [UI results](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733304) reported failures in lanes 8, 9, 12, 13 and 17. Lane 8 included pending projection failures with the same mode and original skill-custody messages as earlier lanes. Other failures involve focused Stop, bound workspace consent, Send acknowledgements and held-draft cases. Lane 9 also reported two composition read scheduling failures. Lane 12 and lane 13 reported separate durable commit and Stop waits. Lane 17 reported accepted setup Stop and spoken Send waits. Shared symptoms still require exact candidate controls before any broader qualification.

The CSS-aware child loader reached actual workflow command dispatch on the final projection tree. Setup refused with “Add a ready local folder in F4 Settings → Workspaces,” then the existing modal wait expired in both candidate runs (27.05 and 27.11 seconds). The exact same child loader and tests passed both cases on 68af35b1 in 21.29 seconds. The projection candidate remains unpublished pending isolation of that regression.

All three lane 8 pending-projection cases passed on 29315c59 in 116.74 seconds with the exact private-profile child loader and existing deadlines. Additional published failures appeared in admission-sensitive lane 1 and UI lanes 3 and 14. The admission-sensitive job reported an unknown Console session and three cold capture source refusals. The UI jobs reported projection replay, quit and send waits, held Stop, and staged policy failures. Their logged cases still require attribution.

#### Mounted workflow registry alignment

Both isolated projection variants reproduced the same modal timeout. A lightweight diagnostic on the pre-extraction tree 68af35b1 and the projection tree 29315c59 showed that setup received the active Default workspace and returned zero ready bindings. The candidate trace followed the same store from the named chat to Default through Console resume reconciliation, which uses the registry's active workspace. The fixture created the named chat and folder but left Default active in the registry.

On 1e6f9090ebfd4dd8580885e45b10105a6f66b4be the mounted fixture sets the intended workspace active through the existing registry before mounting Console. The command module passed both cases in 21.38 seconds. The two existing navigation controls passed in 30.39 seconds, including recovery after restoring the original registry table and preserving a stale Default binding in the passive probe. Production code, assertions and timeouts are identical to 29315c59. This resolves the local workflow setup refusal without changing the workspace admission fence.

The published PR later reported UI lane 15 failures covering later Send layout, Stop during hook setup, three Send start timing cases, and held Enter. GitHub verification remains open.

The two UI lane 18 pending-display cases that failed with the original skill custody refusal passed on 1e6f9090 in 46.85 seconds through their unchanged private-profile children. The original owner, saved-generation, custom-reader and malformed-metadata assertions ran. The other lane 18 cases still need verification.

A temporary test held the real MCP permission read without replacing the source's switch method. The off-pump key delivery case passed. The spoken cross-tab case progressed past the held-read assertion but still failed while waiting for the named UI hook refusal. The stock route and the custom hook callback control both reached runtime submission before the hold, after the older UI hook gate had returned. This experiment remains outside the candidate pending an earlier test seam that exercises the gate the assertion names.

The two mounted draft tests now hold the actual native MCP permission read and call through to its original implementation. The other fixtures retain their existing routes. Both original draft assertions passed on af48390f5b6aa062f1bb5098b4277fdd99a61635 in 22.75 seconds, covering the sent notice after a non-append edit and its absence after a hidden draft was replaced. The cross-tab Send refusal still needs a hold while its named UI hook gate owns the attempt. Both changed test files passed the pinned fatal Ruff checks.

The cross-tab compatibility test holds the original launch callback while the UI hook gate still owns the spoken Send. The fixture calls through to real admission and keeps the original timeout, named refusal, single-send and preserved-draft assertions. The two Persona handoff cases use the existing in-memory recording store from screen construction onward. All three cases passed on 161fff615aa2d9ced165a10f5dd20b4fc1a613bf in 20.27 seconds. Both changed test modules passed pinned fatal Ruff checks. UI lane 19 also reported a held character render that never streamed and a mounted archive recovery case that refused an uninstalled MCP source. The latter still reproduces locally.

The lane 19 held-character rendering case passed on 2f9e3df7 in 32.50 seconds with the exact private-profile child and unchanged assertions. The earlier CSS child loader failed to collect the missing checkout path. Mounted archive recovery reproduced the CI refusal on 2f9e3df7, while its unchanged exact-dev control passed in 19.91 seconds. The archive regression remains open.

Both exact lane 20 failures passed locally on 161fff615aa2d9ced165a10f5dd20b4fc1a613bf with the original fixtures and deadlines. The Stop key case passed through its private-profile child in 41.65 seconds. The accepted postwrite Stop case passed in 29.92 seconds. The published lane had 179 passes and two failures.

The archive recovery trace followed the missing binding to the app factory's separate temporary data directory. The real MCP constructor only binds the active profile's canonical source. A scratch fixture using the isolated profile directory reached a later runtime refusal. The archive fixture change remains outside the candidate pending full behavior and cleanup verification.

All five exact UI lane 21 failures passed locally on 161fff61 in 121.40 seconds with the original fixtures and deadlines. That cohort included four Library outcome cases and the accepted setup prewrite Stop. Lane 22's consent revocation case reproduced its published folder-admission timeout on the same tree. The runtime's native initialization issuer captures its original initializer at app wiring. Both failing Environment fixtures replace it after wiring. A test-only probe now exercises the actual initializer and keeps the original admission, worker, revocation and lease assertions.

The archive fixture's installed permission store probe reached durable commit, then the missing local tools flag made composition exceed the empty local maximum. Making the existing default explicit moved the four mounted archive controls to a separate Library catalog refusal. Their synthetic snapshot also omits the actual hydrated session Library policy. A scratch test carries that real policy snapshot. The archive changes remain outside the candidate pending the original behavior and cleanup assertions.

Both failing Environment fixtures now retain the original Change Review root initializer installed by app wiring. Replacing it after the native issuer captured its owner prevented the folder from becoming ready. The two original mounted tests passed on f33d2607e826ab1272e86b66a4f4ad06fa0db922 in 130.57 seconds, covering real readiness, revocation, worker delivery and retired registry leases. Pinned fatal Ruff passed for both fixture modules. The production initializer, fences, assertions and deadlines are unchanged.

Lane 24's Anthropic hidden-fields case passed unchanged through the exact private-profile child on f33d2607 in 76.82 seconds.

The archive trace confirmed the original registered ArchiveRecoveryToolProvider was refused at the native ordinary maximum for search_archived_conversations. Scratch df062a89 passed the original mounted refresh and imported-result case in 44.02 seconds and the two wide-screen button and agent restore cases in 58.88 seconds. A four-case control batch passed the foreign cursor test and hit the same three original storage-source failures on both the scratch and its prior candidate. The tighter scratch also matches the proof's turn to the current same-store received claim. It is still under test. The archive changes remain outside the candidate.

Ordinary catalog capture now projects the two reserved archive recovery names only for the app's accepted user proof matching the current same-store received claim. Binding and dispatch require the original runtime issuer and a genuine provider for that same session and turn. The existing registry, resolved Library policy, confirmation and storage checks still run. The mounted archive fixture uses its isolated profile, original permission store and saved Library policy. Refresh and retired-database cleanup passed in 26.67 seconds and all four button and agent restore variants passed in 90.28 seconds on a043a3a3. The catalog binding and dispatch control, existing session close and foreign cursor controls passed in 12.76 seconds on e85bab25. Three separate native dispatch cases still refuse changed storage on the earlier archive scratch and its prior candidate. GitHub verification remains open.

UI lanes 25 and 26 reproduced dispatch refusal on 19aa2f16 before provider work. Both mounted traces reached the MCP stock source check at console_configuration_preparation.py:402. The shared test app factory had created the MCP store outside the active isolated profile. The exact dev controls passed for the first Response Rules variant (21.57 seconds) and the collapsed Inspector (37.57 seconds). On bb4707d3 the two fixture builders use the existing active-profile pattern. All four original Response Rules variants passed (24.67 and 49.32 seconds across two batches). The collapsed Inspector case passed through its original private-profile child in 42.60 seconds, including its open-rail positive control. The two held transport cleanup cases passed in 20.94 seconds. The original slash Stop and off-screen completion passed separately on 19aa2f16. Lane 29's unchanged idle composer Enter case passed on 19aa2f16 in 26.64 seconds. Fatal Ruff and Git diff checks passed on the two fixture modules. Standalone owned MCP compaction and the other CI gaps remain open.

Moving the existing worker join to Agents hook ownership removed the command executor's upstream Chat dependency. An AST comparison verified the original cancellation, worker wait, result and exception body. The exact architecture scans measured 38 to 37 upstream imports and 46 to 45 private crossings, with no additions. Both guards remain red. Four original native command process ownership, cancellation and storage settlement controls passed in 18.81 seconds on ab19c645 with real macOS qualification. The separate native preparation module had 23 passes and the same eight failed node IDs on ab19c645 and 224cf93a. Fatal Ruff and diff checks passed for the three modules.

The published automatic and bound-skill forecast cases reached remember_pending_mcp_capture before the original local skill trust was ready. The current private baseline reproduced the automatic failure and the isolated bound-skill failure. Both have passing exact dev controls. In the paired private baseline run the second case passed, so the isolated control is retained. The two fixtures now await the real async initializer and verify the original local trust publication. The complete forecast module passed 20 tests in 21.44 seconds on 44b7a61b, including its separate cold read-free forecast test. The isolated bound-skill test passed in 13.82 seconds and pinned fatal Ruff passed. The production capture contract, provider preflight and all existing assertions remain in place. This qualifies the fixtures' synchronous preflight and makes no claim about cold production regeneration. UI lane 30's other collapsed Inspector journey also passed in the already recorded profile-bound layout control.

UI lane 32 reported three uncovered Console workflow modals. The exact 9ef6a5f3 control reproduced the reverse coverage failure. On 6fdb3a6b the existing width ladder covers setup and status at cap 150 and history at cap 120. The four original registry and reverse coverage checks passed in 18.00 seconds. The real dialogs mounted at 149, 150 and 200 columns and then resized back to 120 columns with the exact shipped stylesheet. Setup and status measured 90, 127, 150 and 90 columns and history measured 76, 120, 120 and 76 columns. Both original workflow command cases passed in 24.58 seconds. The original CSS performance guard passed in 18.97 seconds, both targeted token guards passed in 9.16 seconds, and both exact selector preflight checks passed in 4.52 seconds. All 15 tracked CSS outputs reproduced from the repository builder.

Published lane 31 failed ten cases. On 6fdb3a6b all four unchanged Stop geometry cases passed in 145.94 seconds, all four durable quiet-gap cases passed in 27.96 seconds, and the unchanged attachment vision focus case passed. The exact original citation edit reproduced the lane's function has no attribute close failure. The test's worker double now records the original lazy sync callable and checks that stale and cancelled attempts schedule nothing. All 12 citation readiness cases passed on 00e527fa in 33.27 seconds, including the original SQLite version, citation ownership, held key and stale target assertions. Fatal Ruff and diff checks passed. No production edits or CI qualification.

Admission-sensitive lane 1 reported a mounted native-start KeyError and three readiness refusals. The selector and runtime shared the actual store. The mounted destination lacked effective settings. Using the adjacent fixture’s destination defaults exposed a separate cold custody refusal in the direct native-start coordinator. The mounted Send race now uses those defaults and verifies the original trust service publication before capturing. The six acceptance variants also verify that publication before their synchronous capture. All ten targeted cases passed in 97.66 seconds on d1681aa7, including the mounted manual Send race, six acceptance variants and three mount-failure cleanup controls. The cold native-start route still needs separate custody verification.

The required [Derived Artifacts aggregate](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733304/job/128971535994) failed on fd2872f9500b67d388566c77d20ff6dda4f790a0. It recorded a cancelled PR Fast Lane and failures in both admission-sensitive groups, UI Fast Lane, Console P0, Architecture guard contracts and architecture-review tests.

All four published Temporary parked-turn failures reproduced on ef9538ed in 111.87 seconds. The exact dev quit-count control passed in 14.31 seconds. On the private baseline the mounted Send refused the MCP source outside the active profile. The shared fixture uses its active isolated profile and the original source checks. All nine quit and tab-close tests passed on 0ac7bb10 in 68.10 seconds, including parking, title privacy, loss totals and the clean quit control.

The held non-append and left-tab draft failures both reproduced on ef9538ed in 81.57 seconds before their permission read hold was entered. The original fixture shadows the managed MCP service's source callback. The two fixtures now use the existing hold on the admitted native permission read and call its original implementation. Both original cases and the two earlier draft tests passed on 1f0bb091 in 62.39 seconds. All existing draft, provider-count and timing assertions ran. Pinned fatal Ruff and diff checks passed on both changed modules.

Both lane 2 attachment failures reproduced on ef9538ed in 35.36 seconds. The cold readiness fixture had replaced the actual async publication callback with a synchronous spy returning a coroutine, which Textual refused. The fixture now records the readiness publication worker group and calls the original scheduler with the original callback. The cold early-return case and checked-config projection passed on 91159a0e in 25.29 seconds. The disposed-runtime assertion still checks that neither retry nor publication was scheduled. Image-only Send remains open.

The exact image-only Send passed on dev in 12.85 seconds. On the private baseline the launch spy changed the captured adapter and selected the legacy route, which refused managed preparation before dispatch. The fixture now records the real runtime custody task and awaits it. All six attachment readiness cases passed together on 831eafb4 in 36.28 seconds. The exact staged bytes, one provider request and image data URL were verified by the original assertions. The fixture leaves the original launch and authorization owners in place. Pinned fatal Ruff and diff checks passed.

Both lane 16 first-send import tests reached provider entry on 2e1a946f and exposed additional UI-loop imports. The provider-direct Automatic guard still reserves exactly its two Library loads. The unchanged first-send warm-up payload control passed at 93/94 modules, so the new imports need to be reconciled within that budget. Both lane 9 MCP composition scheduling checks reproduced on 2e1a946f and all three scheduling cases passed on exact dev. No runtime changes have been made for these two issues.

Lane 9's extra kill-switch read came from the Tools chip's light-surface count after MCP composition. The runtime-bound switch reader identifies its actual app source. The count uses an already available, successful observation for that same service. The original live reader stays in use for custom bridges and for an absent, failed or synthetic no-demand observation. No invocation, permission or resource path consumes this display data.

On 275b6d51 the three original scheduling cases, one-read, fingerprint-refusal and admission-change controls, and the new mounted no-demand control passed in 37.61 seconds. Both Tools chip checks and the native plain-send tool set passed in 16.15 seconds. Both preimport controls passed in 15.35 seconds, including the unchanged 93/94 warm-up bound. On the prior private candidate the broader unchanged-catalog case failed because reuse performed the extra read. The first scratch advanced to that case's separate full-surface count expectation on a light profile. This case has no passing claim. Fatal Ruff, diff and checkout preservation checks passed.

The full-surface catalog count case passed on exact dev in 11.73 seconds. On the candidate the active light surface correctly excluded its five built-in MCP entries. The fixture now enables full tools through the original saved setting. The original identity, count, kill-switch and revocation assertions remain. The complete catalog module also exposed duplicate switch reads in two direct composition cases where the caller had supplied no run-local observation. Both cases passed on exact dev. Direct composition now retains its existing switch observation only when it publishes Inspector counts. On d467cc63 the complete catalog reuse and loop scheduling modules passed all 14 cases in 65.93 seconds. Both Tools chip checks and the native light tool set passed in 16.00 seconds. No provider authorization or resource path changed.

Standalone owned MCP PostCompact still reproduced on e9b40c07 under actual macOS qualification. The original pending capture witness and its exact configuration identity are present when hook composition begins, and the fixture already has a published permission store. The failure still retains committed memory. The original native publication and owned provider checks remain required before resolving the pending maximum.

### Standalone owned MCP PostCompact follow-up

The original warm and new file-backed cold fixtures both failed on 543e23f4 in 20.44 seconds with the required-hook failure and retained memory. The cold test uses the isolated profile's canonical MCP store and starts with a fresh service. Its file-backed chat database qualifies the original native capture route. The in-memory route refused native affinity and returned a pending configuration without a witness.

On 76d45ed7f278784bb041e41de402526b810d5c7a, all seven selected cases passed together in 47.17 seconds under actual macOS qualification. Warm and cold PostCompact both ran their reviewed tool and original memory and one-shot handoff assertions. Cold verified the first permission publication. Copied configuration and a substituted MCP service were refused before publication, approval or memory commit and released the real submission, preparation read and capacity. The standalone command and both no-hook scope controls passed.

The existing native hook reservation owns the finite MCP read. Its original witness and the guarded permission publication receipt validate the same source before the execution copy reaches hook composition. The plugin provider's reviewed mapping, definition intersection and invocation-time permissions still run. Pinned fatal Ruff and diff checks passed. The original preimport and warm-up payload limits passed at 556/558 modules, 426564/427830 lines and 93/94 modules. Warm UI remained 1065/1038 on both trees. The same five architecture-review assertions failed with identical reported values. This qualifies the local PostCompact scope. The published required aggregate is failed and AC23 remains open.

### First-send import follow-up

Both original mounted import guards failed on 412c1441 after reaching the provider and passed on d27264e5e701a2193ceb7055daf5cc31d05c80d5. Automatic loaded its exact two Library modules. The existing received dispatcher owns custody execution and first-send persistence. Hook preparation retains the same submission context and original hook issuer. The relocated workflow and agent readers use the same original registrations. Reference readers load only for the existing nonempty candidate branch. An absent writer needs no source read. Every actual writer still uses the original validator and recovery path. The POSIX process-group predicate and all original capability requirements are unchanged.

Twelve local macOS native hook controls passed, including the seven PostCompact and scope cases and five provisional initialization cases. Three original native reference delivery cases passed on the hook relocation before the final source relocation. Two atomic first-send persistence cases passed on the final tree. The two mounted flag cases and mounted prompt handoff fail during configuration setup on both 412c1441 and d27264e5. Three native writer and definition controls also fail on both the prior scratch and unchanged candidate. Two workflow planning controls failed both before and after their reader relocation. These tests still need passing end-to-end evidence.

The archive result refresh case failed in the combined cohort at the later Library scope assertion, after the native dispatch and imported-result checks. Its original isolated case passed on both trees. The exact warm census measured 1065/1038 before and 1066/1038 after. The only changed resident was the already documented timing-sensitive character search module. A traced repeat measured 1065/1038 on both trees and loaded the module after readiness through the unchanged Console character recents worker. The census remains over budget. The original boot import guard passed at 686 with a limit below 688. Both preimport payload checks passed at 556/558 modules, 426564/427830 lines and 93/94 first-send modules. The five architecture-review failures and required PR gate remain open. This evidence covers the selected local tests.

### Goal provider reference import follow-up

On c74da59c the warm census measured 1063/1038. The comparison removed only the goal coordinator and script runner from the resident set. The constructor's AST matches the original apart from loading its result type on use. Goal and workflow call sites keep the original comparisons. Both exact first-send import guards and both preimport payload limits passed. Original native goal dispatch and four goal refusal or cleanup controls passed. The moved-authorization test failed at the provider entry wait on both trees. The workflow endpoint journey timed out in the concurrent run and reported the same unresolved native shutdown as the unchanged candidate when rerun. Those two journeys still need passing evidence. Pinned fatal Ruff, diff and checkout preservation checks passed. The required PR check and warm budget remain failed.

### Workflow presentation startup import follow-up

The existing workflow view setting moved into resident Console wiring with the same saved setting, environment precedence and boolean interpretation. The disabled startup path leaves the workflow presentation module unloaded. The original warm census measured 1062/1038, one fewer than c74da59c. No other resident changed in that comparison. Both workflow command controls passed on both trees with exact CSS sources. The initial scratch mounted run ended at stylesheet loading because the local loader omitted the checked-in plugin sheet. Both registry recovery and passive Default binding controls then passed on 4517131f. Fourteen screen architecture checks and both preimport payload checks passed. The local warm census and required PR check remain over budget or failed, respectively.

### Native plugin hook worker import follow-up

On 113df9e4 the original warm census measured 1057/1038. The only five removed residents were the hooks_v2 package, matching, models, ownership and validation modules. The native issuer is still created with the original runtime and sources. The unchanged shared worker join is imported before capture and is reused for the original service future. AST comparison matches the previous executable statements apart from import placement. Original acquisition, check, join, cancellation and cleanup behavior remains.

The unqualified sandbox run failed during storage setup because the system boot identity could not be read. With real macOS boot identity the 12 selected PostCompact, scope and provisional initialization cases passed. Six direct preparation cases yielded the same four failing node IDs and two passes before and after the import move. The two passing cases cover cold resources with no plugin rows and a held original operation release. Both unchanged Send import guards and both preimport payload limits passed. Pinned fatal Ruff, diff and checkout preservation passed. The warm census and required PR check remain failed.

### Shared endpoint comparison ownership

On f1119a83 all 203 endpoint and contract tests passed, plus 55 endpoint-related session cases and three qualified native child launch and refusal cases. AST comparison verified that all seven moved display and comparison functions are identical. The unchanged upstream scan removes only the Agents native execution import of Chat console endpoints, from 36 to 35. Both Send and preimport controls passed. The warm census stayed at 1057/1038. The architecture boundary, warm budget and required PR check remain open.

### Undo refusal import follow-up

On dcace224 the unchanged refusal type lives beside its existing ChangeTrackingError base. Both import paths resolve the same class. The native issuer, original confirmation, active writer refusal, real restore and retirement all ran with unchanged checks. Eleven revert-engine cases and four qualified native cases, including the mounted Console card, passed. Both preimport limits passed. The warm census shed only the revert engine and measured 1056/1038. The published required gate remains failed.

### Workflow wave startup import follow-up

The original workflow module is now imported at its three existing execution and quiescence call sites. AST comparison preserves all checks, callbacks and method bodies apart from import placement. Exact warm startup on 488e6d2b measured 1053/1038, removing only workflow wave, workflow models and goal models relative to the prior private tree. Both original Send import guards and both preimport limits passed.

The original native overlap case passed using the isolated app's existing save to enable Full agent tools. The original duplicate-approval launch and returned-execution notification controls passed with the same configured tools, before and after the import move. The fixture-only and import variants had two other unchanged failures in wave close and the receiving handler test. The default-light route still refuses before child execution because the light path discards the configured fleet. Only the coordinator's import placement is retained. The test settings were confined to qualification trees. AC23 and the published required check remain open.

### RawCLI issuer startup import follow-up

The same NativeRawCliSource class is exposed at its original import path and created at the original Runtime startup site. Its original app binding, source, physical custody, storage and retirement checks are unchanged. The AST comparison differs only by import placement. The warm census on a56f8047 shed only the command custody module and measured 1052/1038. Both original Send import and both preimport guards passed.

The original parent-only loader reported containment_unavailable and an error row on both trees. A separate qualification launcher installed the same immutable Git source finder in the actual spawned worker and excluded rerunning pytest as the worker main script. No worker, subprocess admission, timing or cleanup code was substituted. The real local command reached its original done row with proven cleanup. The existing model relay, independent worker containment gate, source substitution refusal and disposal tests passed. The same two first-send RawCLI persistence cases remain failed on the unchanged current tree and scratch. These cases have no passing claim. The published required gate remains failed.

### First-request plan ownership

On 835aea5c the frozen ConsoleFirstRequestPlan retains its fields, defaults, and native exact-class check. Its existing public Console import resolves to the AgentService class. The original eight macOS native launch, authority replacement and unused reservation cases passed. The upstream scan removed only the native child import of the Console bridge, from 35 to 34 prohibited edges. Both trees still report five failed architecture assertions.

The manual and Automatic Send guards passed with the exact private-profile child loader. The earlier mixed run passed the manual and both payload checks, while its Automatic test stopped at collection because the runner lacked the exact child path. The warm census remains 1052/1038. The published required gate remains failed.

### Fleet receipt startup import follow-up

On 9342e334 fleet attention imports the original receipt service class at its only settlement use. The AST comparison changes only that import. The same direct and lazy class checks, SQLite connection ownership, durable receipt publication, fallback mark and fanout remain. Seven qualified tests covered direct and lazy receipts, borrowed connections, fallback marks, custom getters, memory ownership and a failed close that keeps the actual lease. Both Send guards and both preimport payloads passed. The warm census shed only console_activity_receipts and measured 1051/1038.

The larger targeted batch returned 37 passes and three failed mounted cases. On both trees those three reached raw_source_selection_changed in storage setup before the mounted assertions. These cases remain unqualified. The published required gate remains failed.

### Workflow context startup import follow-up

On f588a7e5 only the original AutomaticWorkContext import moved. The phase constructs the same class and its accounting path performs the same exact-type and original owner checks. Seventy-three targeted workflow launch, attempt and accounting tests passed under real macOS boot identity. Both original Send guards and both payload limits passed. Warm startup shed only automatic_work_runtime and measured 1050/1038. The published required gate remains failed.

### Shared RawCLI invocation ownership

On 4fa2c5b8 the existing raw executor holds one invocation ContextVar shared by both adapters. The model binding and current-call lookup keep their original bodies and original import aliases. Exact stock provider, runtime and executor qualification remains in the original capture. Original user custody and per-call policy are unchanged. Six macOS controls passed with exact sources in the actual spawned worker, including the direct terminal, model relay and physical cleanup. Both Send and both payload guards passed. The architecture scan removed only the raw shell provider's Console custody import, from 34 to 33. Warm startup remains 1050/1038. The two separate first-send RawCLI tests and the published required check remain open.

### Native hook submission context ownership

On aef9f9e3 the original submission ContextVar lives with the shared hook owner. The original binding, read, reset, source validation and worker bodies remain. The first import-only experiment lost an existing indirect import and failed seven installed-hook tests. The shared-owner change passed all twelve original native tests on its own and on the combined tree, including provisional initialization, PostCompact and scope refusals. Both original Send and both payload limits passed on the combined tree.

Warm startup measured 1049/1038 after shedding console_hook_preparation. The architecture scan removed only the native command import of Console plugin preparation, from 33 to 32 prohibited imports. Five architecture assertions and the published required gate remain failed.

### Native plugin hook issuer startup import follow-up

On b8561bff the original issuer class moves beside the existing preparation read bookkeeping. Runtime still constructs it at the original startup site and the previous import resolves to the same class. The moved class matches the original AST apart from importing its worker request at capture. Original source checks, request ownership and physical retirement remain. Twelve installed native hook cases and both Send and payload guards passed. Warm startup shed only console_plugin_hook_preparation and measured 1048/1038. The required PR check remains failed.

### Workflow phase and Undo issuer startup imports

On 0f5a12c3 the original workflow phase class loads at its existing exact-class check, with an otherwise identical AST. The original Undo issuer class moved beside Change Review ownership. Runtime binds the same class at the original site, and its old import still resolves to the same class. Original confirmation and restore worker bodies stay in Undo custody.

The combined tree passed all 19 original native workflow preparation tests, four Undo cases including actual mounted restore and both Send and preimport payload guards. The broader workflow comparison returned 22 passes, one failed exclusion case and ten setup or teardown errors with the same node IDs and original failing assertions on unchanged and changed trees. Those failures remain open. Warm startup shed console_workflow_runs and change_revert_ownership and measured 1046/1038. The published required check remains failed.

### Change Review issuer startup imports

On 6fcd3624 the four original attachment classes live in existing change tracking. Their ASTs match apart from imports in capture. Runtime uses the original attachment sites, and the old issuer import paths resolve to the same classes. Original initialization, snapshot and Undo worker classes and their source checks remain in custody.

The scoped native batch returned 62 passes and seven failures, while the unchanged comparator returned 63 passes and six failures. The extra opaque-claim test failed in isolation on both trees after its original claim was unavailable. The other six failed node IDs matched. These failures remain open. Both Send and preimport payload guards passed. Boot measured 686 modules with a ceiling of 688, and warm startup shed only change_review_ownership to reach 1045/1038. The published required check remains failed.

### Workflow storage source startup import

On 550af164 the defining source module captures the same class checked by workflow storage validation. The validator imports that pin at its original use, and its exact-type, source, resource and observation statements match the prior AST. The 19 native workflow preparation, refusal, dispatch and settlement cases passed in 88.83 seconds. Both original Send and both preimport payload guards passed.

The paired warm trace measured 1045/1038 on e5669c2c and 1044/1038 on 550af164. The only census difference was console_agent_sources. The untraced scratch measured 1045/1038 while the existing asynchronous character recents worker also loaded character_conversation_search. The startup and required PR gates remain open.

### Config producer record startup import

On 632d4509 the config owner defines the two original producer recorders at their existing definition-time capture sites. The recorder bodies match after normalizing the private names. The original config namespace, producer objects, original-record contents and consuming validators are preserved. The removed module had no other callers.

All 17 targeted deferred Collections and first-Send atomic persistence cases passed in 120.35 seconds. This included borrowed owner, cancellation, source and changed keyword-default refusals, interleaved settings reconciliation and precommit failure. Both exact Send and both preimport payload guards passed. Boot measured 685/688. Warm startup removed only config_source_records from the paired census at 1043/1038. The startup and published required checks remain open.

### Workflow task refusal startup import

On e28f97f2 the existing workflow refusal functions import AutomaticWorkRefused at use. Their ASTs match after excluding only the import. The same permission, current-source, path, dispatch, refusal and budget handling statements run. All 67 original task-scope cases passed, as did both Send and preimport payload guards. Boot measured 685/688.

The paired warm census removed only automatic_work_budget and measured 1042/1038. The warm startup and published required checks remain open.

### Hook submission origin and shared storage helper

On 7eda7fd8 native hooks import the original ConsoleSubmissionOrigin from existing agent models. Console re-exports the same AST-identical enum. The original bound task, controller, session and exact origin type are checked. All 69 installed macOS cases passed on the independent origin tree and all 12 scoped installed cases passed after porting it to the current candidate. The architecture scan removed only the native hook dependency on Console chat models, from 32 prohibited edges to 31. Five architecture assertions remain open.

Workflow uses the same owned_goal_storage helper from the existing activation owner. The previous Goal module re-exports the same callable. Original acquisition, borrowed connection and cleanup statements are unchanged. All 19 workflow native preparation cases and seven Goal/storage controls passed on the combined executable, including fresh CLI, fresh and borrowed mounted runs, and cancellation during physical work. The borrowed CLI control timed out in retain_unknown_return on both the independent change and unchanged comparator. It remains open.

Both Send and both preimport payload guards passed on the combined executable. Boot measured 687/688 after moving the origin enum into existing agent models. Warm startup removed only goal_storage_lifetime from the preceding warm census and measured 1041/1038. The required check on the published head remains failed.

### Draft capture races

On d7f760f8 the original spoken-Send and cleared-draft tests hold the managed permission read without replacing the source method. The initial native-hold probe reached the committed-spoken-Send hold and then missed the paint hold in isolation and in the mixed run. A live Console can dispatch without acknowledgement paint. The final fixture observes the actual dispatch while its original pending capture exists, releases the spoken Send to commit, and then calls the original Enter dispatcher. Both original single-send and draft assertions remain.

The final targeted run passed all six cases, covering both spoken-Send race failures, Ctrl+U and Backspace, and the existing held Enter and spoken draft controls. Only the fixture changed. The original time limits, production source checks and dispatch behavior are preserved. Earlier exact runs also passed all four targeted lane 14 and six targeted lane 18 controls on 00d4444d. The published required check remains failed.

### Held Enter after first-Send publication

On 5dabe42a the original lane 6 mixed cohort passed character seeding and trailing paste and reproduced the held-Enter failure after the first reply. The same exact base cohort passed. Holding the real settings worker before replacement reproduced the spoken Send refusal at the original publication fence. The observer released that writer and the normal settings transaction completed.

On 8017142c the held-Enter fixture waits for the actual first-Send task to finish within its original first-turn wait before issuing the spoken Send. Enter stays parked through that wait. The exact held-writer run and all five mixed lane and neighboring draft cases passed. The original draft and provider count assertions ran. The published required check remains failed.

### Exact loader follow-up for portable cache and UAT reports

On 4c829160 all four detached-cache variants passed in 15.87 seconds. The immutable loader uses an absolute Git executable for its own reads because the real portable child runs with a restricted PATH. The original portable environment, real cache getter, archive generation, refusal cases and retained-file assertions ran.

All 32 synthetic project-chat report and required-workflow checks passed in 1.25 seconds on the same tree. The immutable loader reads the actual checker blob in both the parent and the separate CLI process and keeps the original optimization flags and environment. This covers exact positive and negative report validation, including malformed reports under Python optimization. The earlier mounted and negative UAT receipts are separate evidence.

A private workflow import trial passed 86 native tests and five first-Send, payload and boot checks. The warm census remained 1041/1038 because catalog listing also loads task scope during bridge construction. Five architecture assertions remained open. The import trial stayed separate from the candidate. The published required check remains failed.

### Lane 8 tab round trip and acknowledgement

On 67b1d337 the two round-trip cases failed before the old MCP hold, and the acknowledgement paint case failed before provider validation. The managed runtime rejected fallback preparation after the round-trip fixture replaced the MCP source getter and after the paint fixture replaced the context builder. The separate acknowledgement layout and front-of-paste cases passed on the same tree.

On 913f2a9d the round-trip fixture uses the shared native permission hold and calls the original read. The paint fixture observes the actual received intake at function entry and restores the previous profiler. Both round trips, the left-tab draft control, the two original acknowledgement cases and both wider mounted paint cases passed. The original assertions cover the tab A send and tab B draft, acknowledgement and provider ordering, and paint continuity. Existing limits ran.

The four local architecture guard modules covering the earlier published failures passed all 35 cases on 67b1d337. The three SQLite inventory cases and the original CSS count guard also passed on that tree. This evidence covers those modules and cases. The separate architecture review and published required check remain failed.

### Shared SSH invocation ownership

The workspace executor owns the original SSH invocation ContextVar, lookup and binder. Console retains the original capture, checks and lifecycle and re-exports the same helpers. The workspace transport and agent dispatch callers use the executor's original objects. The native request, worker and publication checks retain their original bodies.

On 1cbfb0e2 the architecture scan dropped only the local dispatch and virtual CLI edges into Console SSH custody (31 to 29). The original five architecture assertions remain red, with 44 private crossings and identity counts of 69/56, 35/27 and 55/43. Warm UI remained at 1041/1038. Both original Send checks, both payload limits and boot passed.

The 11-case native macOS comparison gave 10 passes and one matching failure on d7f3f906 and 1cbfb0e2. The original handler replacement was refused before it could run or use SSH, and the existing test expects the SSH diagnostic where ordinary handler custody reported `ordinary local handler source changed`. Actual model and virtual reads, stderr return and the CAS edit passed. Three more native cases passed for shared master adoption and session retirement, along with six transport controls for the real bundle, retarget refusal, reuse and fallback. No test or budget changes were needed. The published PR was unchanged and AC23 remains open.

### Warm Console imports for compaction and Library search

The original compaction finish helper loads at the existing compaction execution point. All other code matches by AST. On a541162d the unchanged warm UI census dropped from 1041 to 1040. Ten native macOS cases covered installed and no-hook compaction, warm and cold MCP hooks, copied configuration, changed source and competing admission. All 171 context and failure tests passed.

On 5b7944c6 the existing Library search state owns the unchanged query sanitizer, query limit, source selection and modal result contract. Both modal modules keep their existing exports. Console opens the original modal on demand. The original warm UI census passes at 1038, with both first-Send controls, both payload limits and boot passing. Fifty-one Library cases passed, including actual modal opening, independent standing policy, draft preservation and source selection. Two real-Console render variants failed in storage setup with `raw_source_selection_changed` before they reached the search action. The same-loader run on unchanged 33afa215 reproduced both failures at the same fence. Traced and untraced warm runs both measured 1038. The separate architecture review still reports five failures. The published required check remains failed and AC23 stays open.

### Ordinary execution context ownership

On 00fc3228 the existing execution owner holds the single ordinary tree ContextVar and original lookup and worker binding. Console publishes and clears at the existing points and checks the same original tree on clear. Both caller APIs return the same objects and run the same owner activation and reset. No admission, policy, worktree, storage or cleanup check changed.

The original architecture scan removed eight Agents-to-Console imports, from 29 to 21. The 44 private crossings, the three identity counts and all five remaining architecture failures are unchanged. The warm census passes at 1038. Both original Send checks, both payload checks and boot passed.

Eight of 12 targeted ordinary cases passed. The three Change Review cases and the mounted pre-capture false variant failed at the same assertions on the unchanged candidate. Those cases remain unqualified. Four real SSH and virtual CLI controls, twelve native hook subprocess and settlement cases, and all 72 capacity and worktree tests passed. The earlier interface trial exposed a private import and is excluded. The published required check still fails and AC23 remains open.

### Model Raw CLI capture ownership

On e8fbb5d8 the original model capture runs beside the executor-owned invocation. It resolves the same original Console runtime type, provider and executor at the existing point. Console re-exports the same callable and owns direct user capture. The moved function body matches by AST after resolving its imports.

All six native cases passed using the exact Git sources in the real spawned worker. They cover the actual SQLite command row, model relay, replaced runtime refusal, disposal, admission return and physical containment. Both original Send checks, both payload checks, boot and all three raw CLI import closure tests passed. Warm UI passes at 1038. The exact architecture scan dropped one Agents-to-Console edge, from 21 to 20. It still reports 44 private crossings and the same five failed assertions. The published check remains failed and AC23 stays open.

### Borrowed Goal tool storage return

On 007c3653 the actual Goal tool worker enters Goal storage inside the admitted ordinary source. The earlier handle trace showed fresh AgentRunsDB and WorkspaceDB handles still owned by the enclosing Goal scope when the ordinary source tried to verify their return. The Goal authorization keeps its existing binding and admission check. Goal borrows the source's active handles during the tool call, then returns its scope before the source closes and verifies those handles. The original uncertainty path still retains the source and operation if physical return is unproven.

All eight original lifetime cases passed, including real borrowed and fresh CLI and mounted Goal workers, cancellation and unrelated borrowers. Twenty-seven original source return and capacity controls passed, including callback error, failed cleanup and replaced handle. Both Send guards, both payload guards and boot passed. The untraced warm UI census passed at 1038. The same five architecture failures, 20 prohibited import edges and 44 private crossings remain. The published required check still fails and AC23 remains open.

### Workflow preparation binding ownership

On ad10586d the existing Agents run context owns the one workflow preparation binding. Console still binds the same preparation for the original source scope and resets it on return. The planning reader checks the exact native reply, phase and live runtime registration at the reply owner. The existing Console reader entry points load the same owner on use, and the tool catalog file is byte-identical.

All 95 native preparation, task scope and physical return cases passed. The original native primary planning case passed. Both Send guards, both payload guards and boot passed. Boot loaded 687 modules and the untraced warm UI census measured 1038. The architecture scan has 16 prohibited imports, down from 20, with 44 private crossings and the same five failing assertions.

The unchanged dc54dbf7 and ad10586d fail the same four changed-receiver planning cases. The executor, reader and resolver cases time out. The hash case invokes the test's replaced host-class hash and writes the fixture marker during validation. The exact published head reproduces that hash failure as well. These are separate from the passing native preparation and primary planning controls. The published required check still fails and AC23 remains open.

### Bounded native planning first request

On 6492ca64 the first request keeps the bounded fs_list and fs_read entries only when the live native planning authorization owns the exact composed registry and allow-list, and those entries are already among its original local tools. The remaining light-surface selection, prompt, discovery and budget logic is unchanged. The actual provider, permission, binding, root, exclusion, accounting and physical return checks still govern dispatch.

The original tools case failed on the published head and 404a45a3 after plain-chat narrowing removed both file tools. On 6492ca64 it passed with both real tool results, file content in the second provider request and the original plan return. A read-only inspection of its isolated SQLite database verified a completed, drained planning phase and both settled token reservations. Native primary planning, direct forbidden write and optional service suppression passed. All 67 original workflow task scope cases, 19 ordinary light-tool cases, both Send guards, both payload guards and boot passed. Warm UI measured 1038. The same five architecture assertions and four changed-receiver planning cases still fail. The published required check still fails and AC23 remains open.

### Default light setting and native workflow waves

On 9cef66f4 the existing native fleet and inbox stay with the prepared execution child. The first request includes local tools from the original checked task registry, current finite preparation and captured local provider. Plain chat retains its existing light selection. The native request trace contains only fs_read and fs_edit for both approved tasks, with no runtime schemas and a nested-spawn budget of zero. Existing approval, permission, root, exclusion, task budget and return checks still apply.

The original two-child overlap ran with default settings and wrote both intended files. The isolated SQLite receipt shows two completed task attempts, two done native runs, one completed Change Review window and one snapshot reference. Native cancellation after baseline, capture-off and duplicate approval passed. Existing planning reads and failed native creation return passed. The two changed-handler cases still stop during fixture setup because they look for the receiving closure in the old wrapper. They fail at the same lookup on the fleet-only comparison tree. All 67 task scope cases, 19 ordinary light cases, both Send and payload guards, and boot passed. Warm UI measured 1038. The same five architecture assertions still fail. The published required check still fails and AC23 remains open.

### Native wave handler replacement controls

The two replacement cases failed before mutation because the fixture looked for the existing receiving closure in the provider wrapper. On a0b9e236 the fixture finds the same closure in local_tool_dispatch. Both original replacement cases and the real overlap passed. The post-permission probe recorded one failed receiving result with dispatch_started false. Both profiles retired normally and had no escaped file or changed-handler entry marker. The original default-light notification failure case also passed on 11fe233c and verified that delivery cannot resurrect returned work. Production, test assertions, deadlines and permissions are unchanged. Published CI and AC23 remain open.

### Shared agent source issuance

On 5e212554 the existing Agents run context owns the original agent source binding, issuer and workflow fallback reader. Console loads that owner at its original entrypoints and keeps the same controller and task checks, publication and activation order. Both hook families read the original source directly. The architecture scan removes those two upstream imports and reports the same five failing assertions with 14 upstream and 44 private crossings.

The native command and legacy cohort returned 36 passes and four legacy publication or commit failures. The same four legacy cases failed at their original opacity assertions on 643fb51a. The separate mounted writer cohorts each had one admission refusal pass and three failures before a provider payload. Both original workflow cases, six native MCP and source unwind cases, and all eight Goal storage lifetime cases passed. Both Send guards, both payload guards, boot at 687 and warm UI at 1038 passed. Published CI and AC23 remain open.

### Workflow root provenance

The workflow models own the existing persisted directory-chain fingerprint. The workflow service and returned-planning recovery use the same encoder. The service keeps its existing alias for test callers. The bytes, captured filesystem metadata, approval, worktree protection and recovery rules are unchanged. ADR356 governs the existing behavior and requires no amendment.

On 96628c11, 15 startup, historical protection and native planning cases and all six returned-planning recovery cases passed. The latter include journal failure, recording delivery, unreviewed shutdown and directory and symlink replacement. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scans remove one private crossing (44 to 43), with 14 upstream crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Shared Raw CLI runtime qualification

The existing Raw CLI executor checks the same exact runtime class for model shell capture and workflow verifiers. The verifier still requires its original catalog provider and app-installed runtime. Model capture still checks the platform, provider and executor. Existing approval, process and verifier evidence requirements remain in effect under ADR093 and ADR356.

On 61219ddf, all 15 scoped native Raw CLI and workflow verifier cases passed, including the original real spawned worker, model relay, replaced runtime refusal, bounded output and native checkpoints. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scan drops only the verifier service's direct Console import (14 to 13), with 43 private crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Agent recovery source observation

Console obtains the existing agent recovery source observation through the activation owner. The owner uses the same guard and task identity, returning the existing accepted tuple or observing the original selectors afresh. The activation guard stays private to these callers and all execution and recovery admission behavior is unchanged. ADR285 and ADR356 govern the existing ownership.

On 0f81aced, 54 storage, reference and recovery activation cases, 15 native hook cases, and six native planning, phase-wave and settlement cases passed. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scan drops six private imports (43 to 37), with 13 upstream crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Unused Textual callback record

The combined tree bb8ba736 removes the unused Textual callback record and its sole import. It changes no dispatch behavior. The real two-child workflow overlap and both received and voice reference controls passed. Both Send, payload and boot guards passed, and warm UI measured 1038. The unchanged scan now counts 67 code references against 56, with the other counts, boundaries and five failing assertions intact.

### Workflow runtime and hook preflight

Workflow callers use the named runtime and idle-hook checks from their existing task-scope owner. The same validation runs at the original call sites. ADR285 and ADR356 govern the existing ownership.

On 8839b56e, eight mounted controls passed, including the four hook custody cases, native two-task overlap, cancellation after baseline, duplicate approval and shutdown after initial commit. The original task scope, child launch and partial-attempt cohort had 99 passes and the same failed repair assertion on both the trial and unchanged ad32a04c. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scan removes four private crossings (37 to 33), with 13 upstream crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Catalog ownership of built-in source comparisons

The catalog owner compares its original built-in provider type and retained pure planning inventory. Ordinary composition and workflow preparation use the same checks at their existing sites. ADR285 and ADR356 govern this boundary.

On 2c658512, all four native planning controls passed, including actual bounded reads, refusal of direct writes and suppression of optional tools. Both trees returned 12 passes and the same three ordinary catalog assertion failures. Those cases remain open. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scan removes three private crossings (33 to 30), with 13 upstream crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Workflow operation and capacity checks

The execution capacity owner validates the original operation and capacity objects. The existing launch, wait and close callers validate at their original points. ADR285 and ADR356 govern this boundary.

On 8b284f2d, all 27 original child launch cases and six mounted overlap, refusal, cancellation, duplicate approval and shutdown cases passed. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scan removes one private crossing (30 to 29), with 13 upstream crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Typed workflow launch results

The existing AgentService result types provide the wait and launch return consumed by Console. The previous private names remain aliases for original callers. The original admission and physical return checks still govern those results under ADR285 and ADR356.

On fdd98e42, five native mounted cases passed, including real overlap, cancellation, failed creation return, duplicate approval and shutdown. The manual-gap child fails at the same assertion on both the trial and unchanged d1cbf5c2. The broader launch comparison matched 30 passes, one exclusion assertion failure and ten freshness setup or teardown errors. Those failures remain open. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scan removes three private crossings (29 to 26), with 13 upstream crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Catalog import of the native planning reader

Built-in construction calls the existing native planning reader directly from its run-context owner. The active binding, exact class and runtime registration checks are the same under ADR285 and ADR356.

On 65e9ba2e, four native planning cases and 71 task-scope and ordinary catalog cases passed. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1037. The unchanged architecture scan removes the catalog's upward Console import (13 to 12), with 26 private crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Native wave resource checks

The wave owner exposes the original resource validation to the native task binder. All live wave validations and their ordering are retained. Two calls with no wave had no effect and were removed. The original quiescence checks still verify physical return under ADR285 and ADR356.

On d7dd4e7d, 67 task scope and six native wave and coordinator cases passed, including real two-child overlap, cancellation after baseline, shutdown, tool return, changed receiver refusal and coordinator retirement. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1037. The original architecture scan reports one fewer private crossing, from 26 to 25. Twelve upstream crossings and the same five architecture failures remain. The published PR head still fails the required aggregate and AC23 remains open.

### Unused Library and provider gateway records

Removed two unused Conversation Library metadata records and two unused provider gateway planning records. The live class record, native Library construction registration, planning dispatch and provider loop cleanup remain in their existing owners under ADR356.

The Conversation Library comparison matched 91 passes and seven existing ordinary construction failures. The exact combined tree passed mounted grounded Send. The zero-match reload passed in isolation on both trees, and both mixed runs reproduced the same intermittent stale line after the persisted record was removed. The gateway comparison matched 58 passes and one existing OpenAI Stop transport assertion. All four native planning controls passed on the combined tree, including bounded tools, optional provider suppression and direct-write refusal. The original two Send, two payload and boot guards passed, with 687 boot and 1037 warm UI modules. The original architecture scan reports 64/56 code, 32/27 defaults and 49/43 keyword defaults, with 12 upstream and 25 private crossings. Five architecture assertions and published CI remain red. AC23 stays open.

### Unused Library policy coordinator record

Removed one unused capture tuple from the coordinator. Its actual native validation records, saved policy reads and fail-closed paths remain in place under ADR356.

The policy batch passed 22 cases on both trees. Its parent-only loader could not collect four private child nodes. With the exact child loader all four original project Library policy combinations passed on c896e5e4. Grounded Send and Send Once without Library also passed mounted. Both Send guards, both payload guards and boot passed, with 687 boot and 1037 warm UI modules. The original architecture scan reports 63/56 code, 32/27 defaults and 49/43 keyword defaults with 12 upstream and 25 private crossings. The same five architecture assertions and published CI remain red. AC23 stays open.

### Native agent storage ownership

The existing native agent source, writer validation and physical return checks now live together in the Agents storage owner. Console retains the original aliases. The original selected databases, same-tree check, exact handle return and uncertain cleanup custody remain under ADR285 and ADR356.

On a16c0925, 15 original Goal, physical unwind and native MCP worker cases and six mounted workflow cases passed. Both Send guards, both payload guards and boot passed, with 687 boot and 1037 warm UI modules. The same-loader comparison on the original storage trial matched 84 passes and two reference failures, three passes and 28 ordinary failures, and 38 passes and four scoped hook failures. The stalled negative-drain case was excluded from the scoped comparison and remains open. The architecture scan removes three upstream crossings, from 12 to 9. The same five assertions and the published required check still fail. AC23 stays open.

### Native legacy hook attachment

The existing legacy hook owner names the same source class used by the runtime and workflow preflight. Both test fixtures install their runtime on the fixture app before invoking the normal attachment path. The non-chat case still verifies empty metadata and leaves the chat store and agent bridge unset. Native hook permission, source, process and host return checks remain under ADR345.

On 8cbf1611, all 32 original legacy hook cases and four mounted workflow hook validations passed with real macOS qualification. Both original Send and payload guards and boot passed, with 687 boot and 1037 warm UI modules. The original architecture scan removes one private crossing, from 25 to 24, with nine upstream and the same five failed assertions. Published CI still fails and AC23 stays open.

### Direct Raw CLI storage admission

The existing activation owner exposes the fresh selected-path observation and the original retained Raw CLI storage scope. Console uses them at the existing admission and late receipt storage points. The same source tuple, task and thread admission, physical operation, SQLite handles and cleanup checks remain under ADR285 and ADR356.

On 323a1eb4, ten original native Raw CLI cases passed on macOS with the exact spawned sources. These include cold receipt storage, a real SQLite terminal row, canonical readback, source loss, held admission, host disposal, model relay and worker containment. Both Send and payload guards and boot passed. Boot measured 687 modules and warm UI measured 1037. The original architecture scan removes one private crossing, from 24 to 23. Nine upstream crossings and the same five failed assertions remain. Published CI still fails and AC23 stays open.

### First-Send configuration validation ownership

The installed config owner checks the records it issues for first-Send persistence. Console keeps the captured original checker and its existing current module, runtime, controller, store, quiescence and mutation CAS checks. ADR356 applies to this ownership correction.

On 2458dcdb, both trees completed the real post-Send configuration CAS on an isolated app profile. Replacing the captured checker kept the config bytes and identity intact without invoking the replacement. This directly qualifies publication. The separate survivor test still fails before reaching checker replacement, and the two mounted flag tests fail at the same configuration setup on both trees. All 41 Collections and atomic first-Send tests, both Send and payload guards, and boot passed. Boot measured 687 modules and warm UI measured 1037. The original architecture scan reports 62/56 code, 31/27 defaults and 48/43 keyword defaults, with nine upstream, 23 private crossings and five failed assertions. Published CI still fails and AC23 stays open.

### MCP catalog entry ownership

The MCP provider reads its original catalog table and entry at the existing ordinary bind. Console retains the collision view, definition ceiling and exact table and entry identity checks at dispatch. The unused Console storage alias is gone. The real storage return operation and its test-facing alias stay in place. ADR285 and ADR356 apply to this ownership correction.

On fa847dff and unchanged 512cc575, the same two provider and mounted composition probes passed. They observed the exact table and entry after stock composition, refused an absent entry and refused an entry from a composed subclass. The five original provider tests passed. Both native batches returned seven passes and the same two failures at the killed-MCP replacement assertion and the deferred-permission worker observation. Those cases remain unqualified. Both Send and payload guards and boot passed. Boot measured 687 modules and warm UI measured 1037. The original architecture scan still has five failures, nine upstream imports and 21 private crossings. Published CI still fails and AC23 stays open.

### Live dev reconciliation and local size checks

The private reconciliation incorporates dev b498bc3a. It retains dev's approval
observations, terminal status reporting, and workflow checks alongside native
retirement and custody. The workflow and remote artifact selection passed
287 cases and the approval selection passed 73 cases on the reconciled sources.

On executable tree 4d19509e, the installed config lifetime owner holds the
original recorders and first-Send validator. Config still publishes the
records from its own namespace. The Personas dictionary statistics import
still loads at use. The original config and Personas size limits pass at
8,956/8,983 and 16,828/16,829 lines. All 216 selected size and boot tests
passed, as did 41 Collections and atomic first-Send cases, both first-Send
import guards, both preimport payload guards, and warm UI at 1,037/1,038.
The exact diagnostic scanner reproduces the reconciled manifest byte for byte.

The exact trial and unchanged reconciliation both completed the direct
first-Send configuration CAS and refused a replaced checker before
invocation. Both dictionary suites returned 18 passes and the same 70
failures, including storage selection refusals in mounted fixtures. The
five architecture-review failures remain open. This is local evidence
against dev b498bc3a. AC23 awaits qualified PR-head CI and review.

On c493f95d, all four original mounted project-chat UAT variants passed
in 115.54 seconds with the real loopback provider and isolated profiles.
The original checker accepted all four complete receipts.

### Ordinary provider observations

The Local and Library providers own their original class comparisons.
Ordinary composition reads the original immutable Local effects and
uses the existing catalog reader for the ordered built-in gate table.

On 2c007c7f, all 201 size checks, 85 catalog cases, both mounted Library
cases, and the six original Send, payload, boot, and warm checks passed.
The original checker accepted complete receipts from all four mounted
project-chat variants. Direct controls matched the original objects and
refused subclasses. The diagnostic inventory reproduced byte for byte.
The original native SSH selection returned four passes and the same
three assertion failures on this tree and a0d1eb69. Four private boundary
crossings were removed. Five architecture assertions remain open.

### Native workflow preparation interface

Console uses the named native child preparation type and the existing
task preparation operation from their Agents owners. The storage check
still captures the same class object. Task preparation retains its
accepted phase, registry, scope, and lock checks.

On 2cae33c5, all 94 task scope and child launch cases, four mounted wave
cases, six finite native return cases, and six Send and startup checks
passed. The size pins and diagnostic inventory pass. Two private
crossings were removed. Five architecture assertions remain open.

### Virtual CLI source interface

The Virtual CLI provider owns the existing static callback observations.
Console reads the same provider and parser registry classes by their named
bindings. Native checks retain the original callback slots, catalog, parser,
executor, and resource return.

On 841d6455, all 127 provider, parser, and SSH composition cases and all
six Send and startup checks passed. The actual macOS native comparison
returned six passes on both trees, including physical SSH return and four
remote source replacement refusals. The same two local Virtual CLI cases
failed on both trees at the nonopaque maximum and callback invocation
assertions. The size pins and diagnostic inventory pass. Four private
crossings were removed. Five architecture assertions remain open.

### Remote dispatch adapter ownership

The remote executor owns the existing wire-to-text adapter and its original
execute and ping records. Console loads the same adapter when it builds the
remote executor and preserves the old import. The typed remote Git and URI
refusals and direct preparation checks remain at their existing boundaries.

The tightened tree b6e602c5 passed all 20 SSH composition cases and all 201
size checks. The controller pin now records the measured 28,728 lines.
Its identical production blobs passed all 12 native SSH cases and all six
Send and startup checks. The combined loopback and composition run matched
the unchanged tree at 59 passes and one read-count assertion failure.
The diagnostic inventory passes. Five architecture assertions remain open.

### Shared workspace SSH custody

The workspace executor owns model SSH capture and the existing request,
master, and session return checks beside their original context binding.
Console retains direct preparation issuance and its original instruction and
registry checks. Both paths share the same physical return operations.
The existing activation owner provides the shared callable source checks.

On 394f197a, all six Send and startup checks, 41 atomic first-Send and
Collections cases, and 17 exact remote bundle cases passed. All four
mounted project-chat cases passed and the receipt checker accepted them.
The native SSH comparison matched 13 passes and the two existing local
Virtual CLI failures. The three initial bare-child probe failures came
from the local test loader inserting the application into those children.
The exact isolated bundle run kept the original guard and used the original
bundle bytes. The size checks and diagnostic inventory pass. Five
architecture assertions remain open with eight upstream dependencies and
six private crossings.

### Workflow planning interfaces

Console uses the existing planning issuer and asks the Raw Shell provider
to compare its live schema with the original prospective schema. The
same registry, bound read scope, policy, and metadata eligibility checks
remain in their original order.

On af246da0, all 99 task scope and Raw Shell provider cases, six native
planning cases, and six Send and startup checks passed. The native
comparison also passed all six cases on the unchanged tree. The size
checks and diagnostic inventory pass. Three private crossings were
removed. Five architecture assertions remain open.

### Native return validation

The prepared reply and ordinary tree owners perform their existing type
and source checks before recovery recording and storage retention. The
phase owner exposes the original unused-acceptance validator. The original
read, error, owner, and physical return checks still run at the same
points before recording or releasing resources.

On fa258d00, six storage and settlement cases and ten planning recovery
cases passed. All ten recovery cases also passed on the unchanged tree.
The unchanged private import guard reports zero crossings, and size and
diagnostic inventory checks pass. Four architecture assertions remain
open. The first mixed import run observed a cold RAG profile read on the
UI loop during manual Send. Both isolated manual controls passed, and all
six original Send and startup checks passed on the repeated mixed run.

### Workspace consent readers

The Workspace consent owner checks the original registry methods at the
existing native hook preparation boundary. Console retains its original
controller, registry, project, and consent checks. The exact installed
reader tuple and registry namespace remain in the check.

On fbebacd9, all 69 installed hook cases, both mounted consent cases, and
all six Send and startup checks passed. The direct source controls refused
replaced methods, instance shadows, changed defaults, and a copied
namespace. All size checks and the diagnostic inventory pass. Four
architecture assertions remain open.

### Native skill publication and hook source records

The app skill owner records the candidate after the actual app slot
assignment using the original marker, caller, thread, task, and slot
checks. Console still checks the original winner and issues the receipt.
The controller hook reader uses the existing callable source record
while retaining its original class, descriptor, and namespace checks.

On 83b39ed9, four native receipt cases passed. The stock trust and
builder comparison matched eight passes and two failures on each tree.
The original mounted Enter and button tests both failed before the
configuration SQL observation and remain unqualified. On 9c9b5eb7,
all 12 installed hook controls, both mounted consent cases, and all six
Send and startup checks passed. All three identity caps, the private
import guard, size, and diagnostic inventory pass. Eight upstream
dependencies remain in the one failing architecture assertion.

### Ordinary execution tree ownership

Agents owns the original admitted execution tree, its exact source and
member checks, and physical return validation. Console retains ordinary
admission and publication. Workflow preparation reads the original local
provider and Raw CLI sources from their owners. The old Console test
import still resolves to the same receipt encoder.

On 84f2ab1e, native two-child overlap, bounded planning reads, all eight
Goal lifetime cases, and all six Send and startup checks passed. The
warm census reached 1038 of 1038, and boot used 687 modules against the
688 limit. The preceding caller update also passed 12 native receipt,
return, and Goal controls. All three identity caps, private import and
size checks passed, and the diagnostic inventory matched byte for byte.
Seven upstream dependencies remain in the one failing architecture
assertion. The earlier provider comparison had 133 passes and the same
two failures on both trees. The Library reservation count and reserved
skill dispatch cases remain unqualified.

On retained tree 5ecb8581, the four mounted project-chat UAT variants
passed, and the original checker accepted all four complete receipts.
The missing-README control failed after the real file read as expected,
and the negative receipt checker accepted the rejection. All seven
native planning and shell checks passed, including primary planning,
three prospective verifier cases and three original revocation cases.

The shared runtime producer module now owns the original finite
preparation read, worker binding, observers, and physical retirement.
Console keeps the original imports for callers and fixtures. Workflow
and the installed Collections initializer use the same read at its
defining module. The original runtime producer class and the moved
finite read body are unchanged.

On executable 10e04ea6, 32 finite read and producer cases, four native
workflow cases, both first-Send guards, both payload guards, and the
boot and warm census passed. Boot measured 686 against the strict 688
limit. Warm measured 1038 against 1038. All four mounted project-chat
cases passed and the original checker accepted every complete receipt.
The diagnostic inventory matched byte for byte. The architecture
batch passed 222 cases, with the same seven upstream dependencies
reported by its one failing assertion. The mixed Collections,
first-Send, and attachment batch matched the unchanged reconciled
tree with 66 passes and one image-only Send return assertion failure.
The image-only case remains open. AC23 remains open.

The image-only Send fixture awaits the original visible action from a
Textual worker. The worker receives the native custody outcome, which
the button pump hands off to the runtime. The original assertion,
staged bytes, provider payload, and turn custody checks are preserved.

On executable 5d32fb86 all 67 selected Collections, atomic first-Send,
and attachment readiness cases passed. Both spoken review outcomes
passed separately. Two observed isolated runs confirmed native
custody acceptance and a successful worker result. The reconciled
tree remains private with the upstream architecture assertion and
AC23 open.

On reconciled tree e1ea1597, all 99 selected architecture and SQLite
inventory controls passed, including the original ownership, projection,
participant naming, and raw connection census cases reported by CI.
All three original lane 6 failures passed locally. Nine admission
controls also passed, including manual Send racing a prepared native
start, all six target revision variants, and two mount cleanup cases.

The lane 15 cross-session Send still timed out at the second provider
gate while the same case passed on dev b498bc3a. The two neighboring
Send-start cases passed. Local observation saw two disjoint scratch
paths, each with an opaque footprint for pending MCP capture. The first
original lease yielded a claim while the second yielded none. The
resource and permission checks remain in place. The upstream dependency
assertion and AC23 remain open.

On the reconciled sources, all eight Response Rules and Inspector
layout cases and all four original Library outcome cases passed locally.
The mounted slash Stop and native archive result refresh passed. All
nine selected installed hook cases passed with the actual macOS boot
identity, including approved provisional initialization, warm and cold
PostCompact, and the changed-source and copied-capture refusals.
The exact published Linux and macOS job results remain separate.
AC23 remains open.

On reconciled tree 6dea1ee1, all four previously failing Perf Guard cases
passed locally, covering boot modules, warm UI modules, the CSS rule
count, and preimport LOC. All nine selected UI lane 8 and 22 cases also
passed, covering pending projection and failed-worker retirement, bound
workspace consent and revocation, Send acknowledgement, and both tab
round-trip draft cases. The published required check and the local
upstream dependency check still fail. AC23 remains open.

On reconciled tree 5cfc7f56, all 146 selected modal tier and citation
readiness cases passed locally. They included the original reverse tier
guard and the citation edit readiness failure reported by lanes 32 and
31. All nine selected Stop, failed-commit discard, and spoken Send
controls passed, including the original failures from lanes 7, 11,
17 and 20 and the adjacent Temporary setup Stop control. The published
required check still fails. Cross-session Send and the local upstream
dependency assertion remain open, along with AC23.

Ordinary Virtual CLI binding records the stock provider callback slots
using the existing descriptor-free observation. The receiving check
refuses a changed callback before review or invocation. The original
local replacement test failed on 52d6a6d6 by invoking the substitute
and passed on a8087cac. Eight native controls passed, including local
held-worker retirement and real SSH stderr return. The original stock
local nonopaque footprint case still failed on both trees. Both Send
checks, both payload limits, boot and warm UI limits passed on the
exact repeat. The first mixed run reproduced an intermittent RAG
import during manual Send. Both isolated manual controls passed.
The upstream dependency assertion, published required check and
AC23 remain open.

### Run-start projection replay fixture

The local run-start coalescing test encountered a full-sync replay already
queued by the original core-state path. Its test-driven tick deferred before
projection even though no sync was running. This occurred after provider
validation. The published lane 3 failure timed out at validation itself.

The fixture uses the adjacent readiness predicate to wait for the real replay
and pending display reads before each existing test-driven sync. It keeps
the original deadlines, held provider, clock, manual ticks and assertions.
The trace on c3ae8586 saw the original pending replay and verified both
manual ticks and the freshness assertions. On 7a4b10a9 all ten targeted lane
3, 4 and 5 tests passed in 95.95 seconds. No new ADR is required for this
test-only change. The published required check and AC23 remain open.

### Reconciled Stop, durable projection and Resend checks

On private tree `d3d630af`, all nine selected published lane 31 tests
passed in 115.60 seconds. They exercised four Stop geometries, three
durable-commit projection cases, vision-block focus and warm layout.
The original citation case was covered separately by the 146-case
modal tier and citation run.

All nine published lane 18 tests passed in 119.93 seconds with a scoped
exception observer. The mounted Resend click completed in place, and the
archive test exercised repeated restores and resource retirement. The
observer saw no mix-in snapshot exception. An earlier mixed run did
refuse the retry after the click, and that refusal remains unexplained.
The 212-case retry, SQLite and Resend comparison returned 209 passes
and the same three viewless attachment failures on the earlier private
tree and live dev. Those cases remain unqualified.

Twelve original Chat Start cases passed in 30.92 seconds. They cover
literal request receipts, manual withdrawal, Stop and mount cleanup.
Cold managed skill custody still needs verification. The published
required check failed and AC23 remains open.

Local evidence is in `/tmp/pr39147-live-dev-published-ui31-1506.txt`,
`/tmp/pr39147-live-dev-ui18-nine-mixin-observe-exact-1501.txt`,
`/tmp/pr39147-live-dev-mixin-retry-storage-controls-1449.txt`,
`/tmp/pr39147-live-dev-mixin-retry-storage-dev-1455.txt` and
`/tmp/pr39147-live-dev-chat-start-lifetime-controls-1506.txt`.

### Cold native Chat Start custody

A Chat Start with an original pending skill snapshot now uses the existing
received turn and custody task. The original coordinator authorization,
configuration and literal draft reach the same durable cutoff and receipts.
The managed task retires through the existing runtime without opening a
prompt chain. The direct warm path remains available under ADR219 and ADR285.

On local tree `5d69fc90`, the cold completion and source Stop regressions
and both warm mounted controls passed. The completion confirmed the
conversation and ledger receipts, literal request, provider output, one
generation charge and complete custody retirement. The source Stop control
withdrew while the actual preparation was held, preserving the original
draft with no durable turn or generation charge. On the same cold setup,
unchanged production refused before acceptance. The observed positive
confirmed the original conversation checkpoint and exact attempt before
terminal settlement retired the row.

The same production blobs passed 14 original Chat Start tests, six
Send/payload/startup guards and the exact diagnostic inventory. The
architecture check still reports the seven upstream dependencies. Published
CI and review remain open under AC23.

Local evidence is in
`/tmp/pr39147-live-dev-cold-start-stop-1527.txt`,
`/tmp/pr39147-live-dev-cold-start-stock-selector-baseline-1527.txt`,
`/tmp/pr39147-live-dev-cold-start-stock-selector-observe-1527.txt` and
`/tmp/pr39147-live-dev-cold-custody-budget-1516.txt`.

### Project-chat receipts after cold Chat Start repair

All four original mounted project-chat variants passed on private tree
`ff075b09` in 118.08 seconds. Each ran in its own profile and used the real
loopback provider. The original checker accepted all four complete receipts.
See `/tmp/pr39147-live-dev-cold-start-uat-1533.txt`, the matching JUnit file,
and `/tmp/pr39147-live-dev-cold-start-uat-check-1533.txt`.

Two of the three previously failing Send-start tests passed on the same tree.
They cover queued and received starts ending at their own provider request.
The cross-session case still failed waiting for the second provider gate,
consistent with the earlier opaque MCP footprint observation.
See `/tmp/pr39147-live-dev-lane15-retained-1533.txt`. The published check and
AC23 remain open.

### Current-dev rebase and migration review

Replayed the 106 published commits onto live dev `b498bc3a` and retained
the locally qualified integration tree. An independent review identified
two stale migration fixtures and a race between AgentRuns constructors.
The runtime now rechecks the legacy catalog and each workflow version under
the same SQLite writer lock as the original DDL. This implements ADR356's
existing additive migration with the same version allocation, SQL and
per-step durability.

The file-backed concurrent regressions reproduced the duplicate v26 audit
row and duplicate legacy review-boundary column before the repair. Both
passed after it, preserving the exact current catalog, original history
and foreign keys. All 75 targeted Goal, automatic work, workflow and
archive recovery migration tests passed. Independent re-review found no
critical or important issue in the migration follow-up.

The cross-session Send case still fails while waiting for the second
provider gate. The architecture review still reports seven upstream
workflow dependencies. AC23 and published CI and review remain open.

### Cross-session Send and native workflow dependencies

Owned Console preparation now captures stock MCP metadata and policy on the
original finite worker before ordinary admission. Disjoint Sends with complete
metadata can hold their own bounded scratch claims. Cold permission publication
respects an existing opaque owner. Custom, unresolved and changed definitions
still require the original admission, and the admitted path retains the downgrade
write and audit. The configuration witness allows independent log publication.

The original accepted phase, SQL fences and wave now live with native execution.
Console retains the actual preparation and coordination. Native planning
validates the phase's original producer at both existing return checks. Remaining
acceptance requires the original issued cell, active reader and physical owner.
All seven upstream edges are gone, and the original architecture guard,
private-import check, identity caps and size limits pass.

On source-identical executable `440e7362`, the three original Send-start cases,
237 architecture and workflow-policy checks, six startup/Send/payload guards,
and four mounted project-chat cases passed. The original checker accepted every
complete receipt. The test-only follow-up `a18eb53d` passed all 16 combined MCP
controls, including cold publication, live deny, source replacement, cancellation,
late log initialization, deferred downgrade and a held opaque owner. The native
workflow repair separately passed real overlap, cancellation, bounded reads,
returned planning, six settlement controls and the Git-backed correction journey.
Independent review confirmed the final fixes and the test-only delta. The larger
MCP comparison reproduced the same eight adjacent failures on the original
candidate, and those cases remain unqualified. PR-head CI remains open under AC23.
