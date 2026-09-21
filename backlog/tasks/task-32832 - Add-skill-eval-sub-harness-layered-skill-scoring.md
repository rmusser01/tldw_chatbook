---
id: TASK-32832
title: Add skill eval sub-harness (layered skill scoring)
status: Done
assignee: []
created_date: '2026-09-20 20:20'
updated_date: '2026-09-21 02:16'
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

<!-- SECTION:NOTES:BEGIN -->
Implemented via subagent-driven development: 11 plan tasks + final-review fix wave. All ACs verified; targeted sweep 167 green; ADR-172 accepted; lessons entry added.

PR review (Qodo) fixes — all 15 actionable findings from the automated PR
review, each with covering tests (engine: 75 green locally; UI tests added
but machine-local `RecoveryRequired: raw_source_selection_changed` backup
state blocks every Tests/UI run here, verified identical on pure
origin/dev — UI logic verified via standalone mounts of the same
harnesses):

- F1 cost estimates omitted retries: `runner.max_estimate_calls` (0/32/
  34+sim) shown as a parenthetical in the panel's cost line, the README
  cost table, and the user guide.
- F2 synthesis accepted any list: `_validate_synthesis` now requires
  exactly 10 prompts with exactly 5 `should_trigger=true` / 5 false.
- F3 hyphenated `allowed-tools` ignored: both front-matter spellings
  resolve (underscore wins); store subjects prefer the service-normalized
  `allowed_tools` response field.
- F4 undispatchable models listed: `skill_eval_targets()` filters to
  providers registered in `API_CALL_HANDLERS`; `make_skill_eval_chat`
  adds a launch problem for handler-less providers.
- F5 cancelled queued calls still executed: both layers re-check the
  cancel token after acquiring the semaphore.
- F6 inert-marker escape via crafted content: `prompts.sanitize_untrusted`
  neutralizes `<<<`/`>>>` in every untrusted field across all builders.
- F7 failed cells stalled progress: `_Caller.completed` counts every
  terminated cell (success/failure/cancelled).
- F8 sim-cell evidence dropped by save_artifact: worker maps
  prompt_index/repeat into `input` and activated/error into metrics.
- F9 worker overlap window: the running flag is set in the handler before
  `run_worker` (rolled back if dispatch raises).
- F10 boolean rating accepted as 1.0: bool and non-finite ratings are
  rejected.
- F11 floor division dropped sim cells: exact `deep_sim_total`
  distribution (base+1 for the first remainder prompts); distinct runner
  warning when deep produced no sim prompts.
- F12 single-page store listing: paginated drain with a hard 25-page/
  5000-skill cap.
- F14 save_report non-atomic: reordered to metrics -> overrides ->
  terminal status so a failure never leaves a silently-completed run.
- F15 raw user paths bypassed path_validation: directory subjects route
  through `validate_existing_absolute_directory` (rejection surfaces as
  SubjectError).
- F16 skill-eval benches could not be deleted: inspector composes the
  delete control for `skill_eval_bench`; the delete handler accepts the
  kind; the disabled-reason gate covers skill-eval runs in flight.
<!-- SECTION:NOTES:END -->
