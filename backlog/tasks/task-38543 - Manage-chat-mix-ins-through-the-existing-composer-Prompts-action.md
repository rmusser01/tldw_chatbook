---
id: TASK-38543
title: Manage chat mix-ins through the existing composer Prompts action
status: Done
created_date: 2026-11-01 19:30
dependencies:
- TASK-38540
- TASK-38542
labels:
- prompts
- console
references:
- backlog/decisions/286-console-prompt-mix-ins.md
documentation:
- Docs/superpowers/specs/2026-11-01-prompt-mix-ins-design.md
- Docs/superpowers/plans/2026-11-01-prompt-mix-ins-implementation.md
- backlog/decisions/286-console-prompt-mix-ins.md
assignee:
- '@codex'
updated_date: 2026-11-02 01:58
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users visible per-chat toggles and deterministic ordering inside the existing Prompt Workbench rather than a separate manager.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The existing composer Prompts action opens the Workbench Mix-ins section and reports enabled count with truthful narrow-layout/accessibility behavior; no separate permanent Mix-ins button or toolbar row is added.
- [x] #2 Before/After lists support add, enable/disable, inspect/edit, move up/down and remove while preserving disabled position, library-sort independence and exact-origin chat ownership.
- [x] #3 Next send preview, updated/unsaved/unavailable feedback, historical instruction inspection and Retry with original mix-ins labels reflect runtime snapshots and retain keyboard/focus behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/286-console-prompt-mix-ins.md
Reason: direct implementation of approved ADR286 and existing runtime/source/editor contracts.
1. Add mounted Workbench interaction regressions and observe expected failures.
2. Add fenced Mix-ins section, reuse shared editor and service, integrate count and read-only/historical inspection.
3. Run targeted UI/inspection checks, production-style captures at 160x45, 100x30, 80x24, self-review and commit for review.
Interface note: derive transient repr-hidden mixin_usage_count on message projections only from verified operational receipt publication/off-loop hydration. Marked None is unavailable; zero is recorded-empty. Retry still validates operational records, never trusts display count. No schema or generic metadata changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Mix-ins as an additional section of the existing Prompt Workbench/ACTION_PROMPTS route; Browse remains the landing view per the controller ruling. The section uses typed exact-origin session/incarnation/source/preference-revision intents, runtime CAS/enable_reference, preserved disabled ordering, group-end placement reconciliation, the shared Library editor and existing settings-save recovery. Preview reuses the exact read-only Next send builder once. Historical inspection reads operational attempt snapshots; verified transient usage counts drive original/empty/unavailable Retry copy without granting replay authority. Missing originals expose current-choice Regenerate, while valid-original transport/new-write failures retain Retry.
ADR required: no new ADR; direct implementation of backlog/decisions/286-console-prompt-mix-ins.md. The approved transient display projection introduces no schema or generic metadata.
Validation: 94 passed in final risk-focused run; 9 additional unavailable/current-B/valid-original failure cases passed; 25 native/factory/inspection cases passed; final source-qualified native captures passed at 160x45, 100x30 and 80x24. Token/bundle guards pass, full-file Ruff comparison against a1fed4545b introduces zero diagnostics, new files format cleanly, diff/syntax checks pass. Console guide updated.
The required three-file UI run has six pre-existing failures: four exact nodes were qualified in Task1, two reproduce on exact Task8 BASE. Ten additional Canvas/tool-label failures also reproduce on Task8 BASE; their origin before that base is not claimed. One newly exposed deferred-focus race was fixed and reverified. Full suite was not run.
Self-review and all exact commands/logs, source hashes, captures and verification limits are in .superpowers/sdd/2026-11-01-prompt-mix-ins-implementation/task-8-report.md and task-8-evidence/. Keep In Progress for independent review.
Fix round 1 addresses review I1/I2: Mix-ins projection/action/inspect/save/preview completions now require their captured mode generation, rendered view and editor ownership. Navigation, Back and unmount invalidate earlier completions. Library Save returns the exact saved record/version; later raw typing remains mounted and dirty against that saved baseline, with a revised editor intent for a second conditional Save. Rendered selection snapshots remain immutable. Deterministic held-project/navigation and held-save/new-draft regressions reproduce the prior defects, then pass; all already-added delayed success/error controls pass. Covering Workbench file: 19 passed; final held-completion subset: 9 passed. Whole-file per-path Ruff comparison against FIX_BASE 3c77bbbdd7 introduces zero diagnostics (42 baseline/current); formatter, syntax and diff checks pass. No styles/layout changed or screenshots refreshed. Keep In Progress for scoped re-review; generation-variant count audit stays with Task9. Exact evidence appended to task-8-report.md.
Independent task review and scoped re-review approved spec and quality after 0dcf71c2e5; I1/I2 closed with no new breakage. Native UI finish reviewed the source-qualified three-size captures. Targeted baseline attribution now shows no current-only failures among 26 selected predecessor nodes (original d83ca versus 3c77bbb with identical native-profile qualification); six Canvas cases pass both, one workspace assertion is intermittent. Generation-variant display projection remains an explicit integrated-qualification audit. ADR-286 applies. No full suite, cloud-provider hit claim, push or merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
