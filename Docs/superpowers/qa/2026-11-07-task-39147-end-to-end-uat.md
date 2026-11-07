# TASK-39147 end-to-end verification and UAT — 2026-11-07

The complete native workflow passed functional end-to-end verification, including
backup, isolated restore and rebackup. Five mounted acceptance cases passed.
The prompt-editor clipping found at 80×24 is corrected by the follow-up below.
Original timing, mounted-test and broader CSS qualifications remain recorded;
these results are not a certification that all bugs are absent.

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
