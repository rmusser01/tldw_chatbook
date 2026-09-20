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
- [ ] #1 Any local-store skill or skill directory (by path) is evaluable at quick/standard/deep depth from the Evals screen without the skill ever executing
- [ ] #2 Quick depth produces a static-only composite ("Estimated" confidence)
- [ ] #3 Standard depth adds judge rubrics + triggering F1 ("Assessed" confidence), ~16 LLM calls
- [ ] #4 Deep depth adds 50 seeded description-only simulations with Wilson/bootstrap/Clopper–Pearson CIs ("Certified" confidence), ~67 LLM calls
- [ ] #5 Reports carry subject provenance (sha256 digest, trust tier, source), seed, decoy-set digests, and methodology_version
- [ ] #6 Results persist in existing EvalsDB tables (no new tables), appear as a run group in the library rail, and render in a dedicated skill-eval detail view
- [ ] #7 Preflight via provider_readiness blocks bad runs; judge JSON failures retry once then renormalize; cancellation persists partial results
- [ ] #8 Untrusted skill content (and decoy descriptions) is inert delimited data in all judge/sim prompts; judge responses must satisfy strict JSON schemas
- [ ] #9 Static layer flags the seven adapted anti-patterns, including dead allowed_tools grants and name collisions against the composition-time exclusion set
- [ ] #10 Unit (static analyzer fixtures, scoring math incl. CIs, judge JSON parsing), integration (fake chat callable across depths, cancel, degradation, storage round-trip), and UI wiring tests pass
<!-- AC:END -->
