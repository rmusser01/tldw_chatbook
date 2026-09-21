---
id: TASK-32832
title: Add skill eval sub-harness (layered skill scoring)
status: To Do
assignee: []
created_date: '2026-09-20 20:20'
labels:
  - evals
  - skills
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Score a skill as the subject under test, PluginEval-style: deterministic static analysis + LLM-as-judge rubrics + seeded description-only Monte Carlo simulation, blended into a versioned composite. Skills never execute during evaluation. Design spec: Docs/superpowers/specs/2026-09-20-skill-eval-design.md (verified against code).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Any local-store skill or skill directory (by path) is evaluable at quick/standard/deep depth from the Evals screen without the skill ever executing
- [x] #2 Quick depth produces a static-only composite ("Estimated" confidence)
- [x] #3 Standard depth adds judge rubrics + triggering F1 ("Assessed" confidence), ~16 LLM calls
- [x] #4 Deep depth adds 50 seeded description-only simulations with Wilson/bootstrap/Clopper–Pearson CIs ("Certified" confidence), ~67 LLM calls
- [x] #5 Reports carry subject provenance (sha256 digest, trust tier, source), seed, decoy-set digests, and methodology_version
- [x] #6 Results persist in existing EvalsDB tables (no new tables), appear as a run group in the library rail, and render in a dedicated skill-eval detail view
- [x] #7 Preflight via provider_readiness blocks bad runs; judge JSON failures retry once then renormalize; cancellation persists partial results
- [x] #8 Untrusted skill content (and decoy descriptions) is inert delimited data in all judge/sim prompts; judge responses must satisfy strict JSON schemas
- [x] #9 Static layer flags the seven adapted anti-patterns, including dead allowed_tools grants and name collisions against the composition-time exclusion set
- [x] #10 Unit (static analyzer fixtures, scoring math incl. CIs, judge JSON parsing), integration (fake chat callable across depths, cancel, degradation, storage round-trip), and UI wiring tests pass
<!-- AC:END -->

## Implementation Notes

Implemented as a self-contained sub-harness package per ADR-172 (the
`character_probe`/`word_bench` precedent — no classic-runner registry entry,
no schema migration), delivered across ten tasks: T1 models/digest,
T2 subject snapshots, T3 static analyzer, T4 scoring/blend/grades,
T5 simulation + pure-Python stats, T6 judge layer + inert-data prompts,
T7 depth runner + preflight + live progress, T8 EvalsDB persistence,
T9 view-model + launcher panel, T10 screen wiring/rail entry/detail view,
T11 docs + polish + hygiene.

Key deviations from the plan briefs, each ratified during execution
(controller rulings recorded in the task briefs' errata):

- `line_count` counts full SKILL.md content lines (front matter included),
  not body-only — the Task 2 brief was self-contradictory against its own
  pinned test.
- Depth checks in `build_report` use explicit ranks, not `>=` —
  `SkillEvalDepth` is a str-mixin Enum, so relational operators compare the
  string values lexicographically.
- The Task 5 brief's beta continued-fraction transcription was broken; the
  Lentz CF (with the symmetry-swap branch) was implemented per Numerical
  Recipes and scipy-verified to <1e-9; the brief's unsatisfiable CP test
  windows were replaced with true reference values.
- Simulation chat-exception cells are appended as failure cells (never
  dropped) so `failure_rate` and the activation denominator count them.
- A parseable-but-indeterminate judge selection reply is a FAILED cell
  ("indeterminate"), not a silently skipped one.
- Runner progress is live per-call (layer-internal cumulative counters
  forwarded as deltas), not layer-end bumps.
- `Select` NULL value guards in the launch panel (Textual `Select` can hold
  `Select.NULL` after option reloads).
- Runner exposes `judge_result`/`sim_result` attributes so the screen worker
  can persist artifacts the report itself does not carry.
- Storage writes run metrics with type `"custom"` (the EvalsDB metric-type
  vocabulary has no numeric-without-tolerance kind).
- The consolidated UI test harness now loads `css/screen_feature_evals.tcss`
  (production loads it via TAB_EVALS first-navigation) — this production-parity
  fix cleared 30 previously "pre-existing" failures across the evals suites.

Known deliberate simplifications: judge trigger-check selections run with no
decoys (isolated routing measurement; decoys measure competition in the sim
layer), and `store_skill_names` limits NAME_COLLISION scope to
builtin ∪ store-skill names (full composition-time exclusion set needs
registry composition, deferred).

Files: `tldw_chatbook/Evals/skill_eval/` (models, subject, static_analyzer,
scoring, simulation, judge, prompts, runner, storage),
`tldw_chatbook/UI/Evals/skill_eval_{panel,detail,launch}.py`, wiring in
`tldw_chatbook/UI/Screens/evals_screen.py` + `UI/Evals/library_rail.py` +
`UI/Evals/evals_state.py`; tests under `Tests/Evals/skill_eval/` (44 tests)
and `Tests/UI/test_evals_skill_eval_{panel,screen}.py`; docs in
`tldw_chatbook/Evals/README.md` and `Docs/User_Guide/lab.md`.
