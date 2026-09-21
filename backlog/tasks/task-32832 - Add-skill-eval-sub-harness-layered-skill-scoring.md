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
<!-- SECTION:NOTES:END -->
