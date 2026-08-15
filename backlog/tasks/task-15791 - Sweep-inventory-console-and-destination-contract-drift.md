---
id: TASK-15791
title: 'Sweep inventory: console and destination contract drift (~38 tests)'
status: Done
assignee:
  - '@codex'
created_date: ''
updated_date: '2026-08-14 19:57'
labels:
  - test-health
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the task-15211 full-suite sweep (`Docs/Design/2026-08-13-tests-ui-sweep-inventory.md`,
chunks 2-5, 15): console and destination-frame contract drift on dev:

- 6x `test_console_staged_evidence_strip.py` assert `"Sources: N staged"`;
  production's label lost the ` staged` suffix in `7dbbc401b` (2026-08-07,
  TASK-2154). Decide which copy is right BEFORE updating either side.
- 4x `test_console_shell_regions.py` size2 geometry; 2x rail width budget;
  chip-swap `_swap_console_session_character()` signature drift; 4x
  dictionary/world-info send-integration; tab-scope focus tour; citation
  cache; composer collapse; live-work handoff retry.
- ~9x `test_destination_visual_parity_correction.py` workbench contracts and
  4x `test_workbench_visual_snapshots.py` — snapshot refresh needs a human
  eye on the renders, not a blind re-record.
- 4x `test_personas_generation_wiring.py`; settings batch
  (`rag_profile_region` x3, `workspaces` x2, catalog toggle); singletons
  (speech x2, schedules, navigation, responsiveness x2, watchlists
  check-now, skills canvas, phase6 x2, recovery taxonomy).
- 3x `product_maturity_phase1` "condition not met within 10s" — suspected
  CONTENTION (four suites shared the machine); re-run alone before treating
  as real.

Also: chunk 12 recorded two teardown errors on the settings provider-Test
button pair — check whether `settings_screen`'s endpoint-probe worker can
outlive its test the way the LLM screen's Ollama probe did (fixed in #1596);
if so, the same worker-lifetime + harness-seam remedy applies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Sources-staged copy question is decided (product vs tests) with the 7dbbc401b context, then applied consistently
- [x] #2 The settings endpoint-probe teardown pair is attributed, and fixed with the #1596 pattern if it is the same class
- [x] #3 The maturity-phase1 timeouts are re-run without contention before being treated as failures
- [x] #4 Each remaining cluster is attributed to its causing commit; genuine breaks fixed, not absorbed
- [x] #5 The listed modules pass whole on dev
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build an attribution ledger from TASK-16220 and the completed TASK-16244–TASK-16265 follow-ups, and identify any original sweep row without current evidence.
2. Re-run the two original clickable Settings provider-Test paths and the four originally failing Product Maturity Phase 1 modules in isolation.
3. Run every named Console, destination snapshot/parity, Personas, Settings, and singleton module as bounded related batches; generate and inspect a temporary current-dev render instead of blindly refreshing snapshots.
4. Apply only root-cause fixes supported by a reproducible current-dev failure, using RED-to-GREEN coverage for any production change.
5. Run scoped static checks, update the task with a per-cluster causal ledger and exact evidence, and close only if every named module is green and the canonical filename is preserved.

Detailed plan: [2026-08-14 Console and destination sweep closeout](../../Docs/superpowers/plans/2026-08-14-console-destination-sweep-closeout.md)

ADR required: no
ADR path: N/A
Reason: this closes a test-health inventory using existing product and test boundaries; any newly discovered architectural issue must be split into its own task before implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes (batch 1)

Re-baselined every module on current dev first: 43 rows still live (24
console + 19 destination/personas/snapshots).

**Sources-staged copy: decided for the product.** 7dbbc401b (TASK-2154, the
owner-driven 24-findings remediation) deliberately shortened the chip to
"Sources: N" in the same pass that renamed "RAG:" to "Library search:"; the
truthful COUNT is the contract and survives. All 6 pins updated with
attribution, plus the blocked-reason copy that took the same rename.
`test_console_staged_evidence_strip.py`: 30/30.

**The size2 geometry cluster is a REAL layout regression, filed as
TASK-16220 (high)** rather than absorbed: at 120x30 with the Inspector rail
open, every workspace-grid fr resolves to fr x FULL-WIDTH (left 354, main
1534, right 472 on a 120 screen) -- bisected to 7dbbc401b; a minimal Textual
replica of the same structure lays out correctly, so the trigger is
something production's children add. Full probe evidence in the task.

Remaining clusters for batch 2: rail_width x3 + rail_sections/internals
(likely the same 16220 geometry family), send-integration x4, snapshots x4
(need a human eye), visual parity, personas x5, singletons.

## Implementation Notes (closeout)

Closed the frozen inventory against current `dev` without running a broad or
full suite. The completed atomic follow-ups supplied the first attribution
layer: TASK-16220 owns the Console geometry regression; TASK-16244–TASK-16249
own the Console integration/controller/rail rows; TASK-16250–TASK-16253 own the
destination CSS, viewport, Schedules, and focus rows; TASK-16254–TASK-16258 own
the Library/privacy harness rows; and TASK-16259–TASK-16265 own Personas,
product-maturity, Settings-probe/responsiveness, and Settings interaction rows.

Fresh attribution and fixes for the remaining live drift:

- `0b8e9e408` moved video-card derivation behind `screen._video`; the citation
  fixture now installs that controller seam instead of a retired screen method.
- `42851b309`/`a2f7b8498` changed the default model evidence, `1edcd3dc9`
  established compact Inspector priority, and `51acb2d646` expanded the
  Inspector rows. Snapshot assertions now distinguish requested rail state
  from effective width and verify offscreen mounted content directly.
- `7dbbc401b` narrowed Send/Stop to six cells. Textual's default two-cell line
  padding clipped `Send` to `Se`; a mounted RED proved the product defect, and
  both mutually exclusive buttons now use zero line padding.
- `c072f9c592` added the instance-owned Console chat store; the app-free
  responsiveness fixture now initializes it explicitly.
- `6385771b2` retired Watchlists `_delete_item`; the mutation inventory now
  audits the live `handle_delete_requested` boundary.
- Rapid Speech view pruning reproduced the repository's documented TASK-1960
  `SelectCurrent/#label` race. The Playground axes and Studio preference
  selects now reuse the existing `PruneSafeSelect`; three race nodes passed
  three consecutive runs, the whole STTS module passed 163 tests, and the
  prune-safe widget contract passed 14 tests.
- `99bc69829c`, `ae1a23fbad`, `e610fac691`, and `d9be16a583` expanded Library's
  reviewed thread-worker/event-loop exceptions. The recovery sentinel now
  counts decorators with AST (including multiline decorators) and pins the
  factual totals: 13 thread workers and six annotated worker-thread loops.
- The Phase-6 power replay now waits for Library's deferred Import-media row,
  matching the existing current-shell replay contract before pressing it.

Hypotheses and evidence:

- The two Settings provider-Test paths passed 2/2. TASK-16264's harness
  stub-and-await already owned their teardown; this was not the production
  worker-lifetime defect fixed in #1596.
- The four Phase-1 modules passed 34/34 alone, confirming the original
  10-second failures were shared-machine contention rather than product bugs.
- Current bounded runs: Console 481 passed; destination/snapshot/header 141
  passed; Personas plus Settings 159 passed; singleton inventory 459 passed
  with two STTS batch-contention failures that both passed immediately alone
  (and within the separate 163/163 STTS run); maturity/recovery 19 passed.
- A temporary 160-column SVG was rendered to PNG and inspected. Rails and
  controls did not overlap, Send rendered fully, and no stale/raw-error copy
  appeared. All temporary capture artifacts were removed.

Static closeout: Ruff lint passes all changed Python files; six previously
formatted changed files pass Ruff format. The other three files reproduce the
same formatter debt at `HEAD`, and their changed ranges are formatter-clean.
Compileall and `git diff --check` pass. Targeted MyPy reports only four existing
unchanged-line errors in `console_composer_bar.py`; the two Speech files and all
new production lines add no diagnostic.

ADR required: no. This used existing controller, worker-policy, and
`PruneSafeSelect` boundaries. No new lesson was added because the Textual prune
mechanism and its tested remedy were already documented in the repository.

## Implementation Notes (batch 2 -- close-out)

Re-verified every remaining cluster on current dev (2026-08-15); the
inventory resolved three ways:

**Resolved by dev's own motion** (verified green, no change): the seven
singletons (tab_scope, chip_actions, citation_sources, composer_collapse,
live_work_handoffs, rail_sections, fleet_discoverability), personas x5
(module 9/9), destination parity + workbench snapshots (132/132 -- the
"needs a human eye" re-record never became necessary), 3 of the 4
send-integration tests, and the whole TASK-16220 geometry family (fixed by
another session's PR #1654; shell_regions + rail_width green).

**Fixed here**: (1) the last send-integration red -- a stale `_Bridge`
double predating the batched `subagent_counts` browser poll (perf finding
A); (2) a REAL regression from TODAY's browser consolidation (520b1ec12,
PR #1661): `_current_console_browser_rows` gained a `run_worker` spawn
inside a state DERIVATION, which raises NoActiveAppError on the suite's
established bare-screen builder convention -- three internals tests
deterministic-red within hours of merge. The best-effort refresh is now
gated on `self._screen.is_running` (the mounted path always is; the guard
also stops minting a never-awaited coroutine). Browser tests 26/26 confirm
the mounted refresh still schedules.

**Trap recorded**: the first guard used `self.is_running` -- but `self` is
the plain ConsoleWorkspaceController, not the screen, so every MOUNTED call
raised AttributeError and the module collapsed 140 -> 20 passing. Running
the module WHOLE immediately after the one-line fix is what caught it.

**Attributed, no action**: the chunk-12 settings provider-Test teardown
pair was frozen-tree residue of #1596 (settings modules 174/174 with the
maturity modules); the maturity-phase1 "condition not met within 10s"
timeouts were contention from four concurrent suites, green alone.

Console internals module: 140/140.
