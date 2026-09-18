---
id: TASK-32820
title: Frame Roleplay draft recovery actions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 20:34'
updated_date: '2026-09-18 21:00'
labels:
  - design-system
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CSS consolidation preserved the existing unstyled full-screen Roleplay partial-save recovery dialog. Make the failure and its recovery choices readable and contained within the established modal pattern while preserving draft-safe navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The recovery dialog frames its title, complete failed-domain list and Retry/Stay actions at 52x20, 80x24 and 170x48 in both supported themes.
- [x] #2 Keyboard focus and pointer activation expose Retry, Stay and Escape with the existing results and no clipping.
- [x] #3 The mounted partial-save navigation flow still preserves unsaved drafts on Stay and existing aggregate navigation checks pass.
- [x] #4 Styles use the shared dialog component classes and central tokens; generated CSS reproduces within unchanged source, selector and byte ceilings.
- [x] #5 Targeted checks and native compact/wide visual evidence are recorded with source and lifecycle boundaries.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md; design language ADR-150 and component patterns ADR-161. Reason: bounded visual repair implementing existing draft recovery and modal contracts. 1. Preserve the native unstyled baseline and add a mounted failure/domain/action containment check with current app styles. 2. Compose shared dialog-title and action-row classes and add token-backed frame rules in the dialog source module; rebuild generated CSS. 3. Verify all supported compact/wide cells, original Retry/Stay/Escape results and incumbent partial-save/aggregate navigation tests serially in private profiles. 4. Run token, bundle, CSS budget and source checks; inspect native dark/light captures and clean lifecycle. 5. Record evidence, independent review, completion ledger and draft PR update. Allocation: all-ref/33-worktree sweep found max32819; local CLI file renumbered immediately to32820 before references or implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced unstyled full-screen Roleplay recovery with the shared dialog-title/action-row grammar and token-backed centered frame. Full four-domain failure text and right-aligned Retry/Stay remain visible at 52x20, 80x24 and 170x48 in dark/light. Kept original handlers and results. Reviewer caught shared row centering; a scoped rule fixes this modal only. Selector guard caught 275/274; narrowing the same text-color targets restores 274 without changing limits. Thirty-one distinct targeted cases qualify layout, incumbent partial-save draft retention and aggregate navigation, hover/disabled/Cancel, tokens/bundle and original CSS source/byte/selector budgets. Six native captures inspected; exact source hashes, clean shutdown, released lock, ten healthy private DBs and unchanged defaults verified. Ruff and changed-range format pass; independent review has no remaining finding. Evidence: Docs/superpowers/qa/2026-09-18-roleplay-recovery/README.md. Production changes: navigation dialog composition, components/_dialogs.tcss and rebuilt app bundle. Native opens the modal directly; actual partial-save entry remains bounded to mounted fixtures. No full suite. ADR required: no; implements ADR-120, ADR-150 and ADR-161. TASK-31243, shared action-row alignment across other consumers and wider component review remain open. Draft PR2707 stays unmerged.
<!-- SECTION:NOTES:END -->
