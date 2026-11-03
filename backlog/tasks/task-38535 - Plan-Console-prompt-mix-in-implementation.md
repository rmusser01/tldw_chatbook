---
id: TASK-38535
title: Plan Console prompt mix-in implementation
status: Done
created_date: 2026-11-01 19:29
dependencies:
- TASK-38534
assignee:
- '@codex'
labels:
- design
- prompts
- console
documentation:
- Docs/superpowers/specs/2026-11-01-prompt-mix-ins-design.md
- Docs/superpowers/plans/2026-11-01-prompt-mix-ins-implementation.md
- backlog/decisions/286-console-prompt-mix-ins.md
updated_date: 2026-11-01 19:43
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn the approved prompt mix-in design into an executable, source-anchored implementation plan with independently testable delivery slices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The implementation plan links the approved spec and ADR-286 and defines file ownership, exact cross-task interfaces, schema/portability choices and dependency order.
- [x] #2 Each implementation slice has an existing Backlog record with outcome-based criteria and no reference to an uncreated successor task.
- [x] #3 The plan contains concrete regression examples, targeted commands and expected results, integration and live verification gates, and coverage of every approved spec requirement.
- [x] #4 Self-review and scoped documentation checks pass, and the execution handoff leaves product implementation unstarted.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/286-console-prompt-mix-ins.md
Reason: Plans direct implementation of the user-approved design and existing ADR; adds no new product policy.

1. Map current artifact/source, database/transaction, received-turn/queue, prepared-request/cache, and Library/Workbench owners.
2. Define minimal shared interfaces and independently testable Backlog slices in dependency order.
3. Write the implementation plan with concrete test examples, exact file targets, failure rules, targeted commands and live qualification.
4. Self-review spec coverage, names/types, dependencies, local links and task metadata; record design approval and commit only planning artifacts.
5. Offer inline or subagent-driven execution without starting product changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wrote the source-anchored implementation plan at Docs/superpowers/plans/2026-11-01-prompt-mix-ins-implementation.md and created TASK-38536 through TASK-38544 as nine independently reviewable To Do slices with predecessor-only dependencies. The plan defines schema-v3 literal records, portable v2 mix-in envelopes, existing-source capability refusal, conversation-owned operational snapshots through existing transaction contributions, exact admission/queue capture, original Retry versus current regeneration, stable-history cache gates, and existing Library/Prompt Workbench UI.

ADR required: no new ADR; direct implementation of accepted ADR-286. Recorded the user approval in the spec/ADR/index and closed the separate design task. Included exact file owners, public interface contracts, concrete regressions, targeted commands/expected outcomes, migration/fork/backup/privacy ownership, isolated live UI and provider evidence requirements, and a full spec coverage map.

Self-review corrected transaction-writer narrowing, old Markdown parser flattening, temporary/fork identity ownership, message-delete retention, source UUID reads, current root pytest configuration and exact existing composer Prompts route. Checks passed: 25 local links, 11 Backlog records with predecessor-only dependencies, nine slices, 11 syntactically valid Python examples, task frontmatter/owned sections, placeholders and scoped whitespace. Product implementation, runtime/full-suite tests and live/provider/cache qualification remain unstarted. Execution mode is the next user choice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
