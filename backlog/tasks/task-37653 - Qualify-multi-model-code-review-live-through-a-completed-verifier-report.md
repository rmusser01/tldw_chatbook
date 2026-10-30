---
id: TASK-37653
title: Qualify multi-model code review live through a completed verifier report
status: To Do
created_date: 2026-10-30 03:54
labels:
- agents
- code-review
priority: high
references:
- backlog/tasks/task-35582 - Implement-primary-agent-multi-model-code-review-recipe.md
documentation:
- Docs/superpowers/qa/2026-10-18-multi-model-code-review/macos-final60/report.md
- backlog/decisions/281-primary-agent-multi-model-code-review.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-35582 merged the multi-model code review recipe on offline and scripted evidence. Its bounded live Console/provider qualifications sealed and presented reviewer reports, but the fresh verifier never completed a live unified report: the last attempt (Docs/superpowers/qa/2026-10-18-multi-model-code-review/macos-final60/report.md) failed with primary review_refused report reads and a 'Provider continuation could not be persisted' error at the 600-second deadline, so the verifier never launched. This task carries TASK-35582's former live-qualification criterion so users can rely on the verification stage with real models.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The macos-final60 review_refused report reads and the continuation-persistence error are traced to an internal cause or reproduced offline, and any product defect found is fixed with a regression test.
- [ ] #2 A bounded real Console/provider qualification uses two reviewer destinations and a fresh verifier, including a pre-authorized request under the default two-child ceiling, seeded defect/false positive/duplicate, failed reviewer, truncation, source drift, small-context and storage-overflow controls; live-model evidence is distinguished from scripted controls.
- [ ] #3 The live run seals the verifier report and produces a unified report that expands every original finding exactly once.
- [ ] #4 The owner approves the live request/child budget before the run, and actual spend is recorded with the evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
