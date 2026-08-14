---
id: TASK-15791
title: 'Sweep inventory: console and destination contract drift (~38 tests)'
status: To Do
assignee: []
labels:
  - test-health
  - console
priority: medium
---

## Description

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

## Acceptance Criteria

- [x] The Sources-staged copy question is decided (product vs tests) with the 7dbbc401b context, then applied consistently
- [ ] The settings endpoint-probe teardown pair is attributed, and fixed with the #1596 pattern if it is the same class
- [ ] The maturity-phase1 timeouts are re-run without contention before being treated as failures
- [ ] Each remaining cluster is attributed to its causing commit; genuine breaks fixed, not absorbed
- [ ] The listed modules pass whole on dev

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
