---
id: TASK-31244
title: Add Character conversations to Console Context
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 02:08'
updated_date: '2026-09-06 06:47'
labels:
  - console
  - context
  - characters
  - ux
dependencies:
  - TASK-31243
references:
  - >-
    Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
  - >-
    Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make recent character conversations discoverable in the Console Context rail through a bounded, date-sorted, local-only Character section that preserves first-use comprehension and expert state.
<!-- SECTION:DESCRIPTION:END -->

## Renumbering provenance

Renumbered from TASK-31236 on 2026-09-04. The final pre-commit worktree sweep
found the older `Review set Dismiss gets an Undo receipt` task created at 01:50;
it keeps TASK-31236 under the older-arrival rule. This unshipped task moves with
all plan and dependency references.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Character is always composed directly after Conversations and before Model, independent of avatar-image visibility.
- [ ] #2 At most four character headers render; the current character is included even with zero chats and the Unavailable group consumes one header slot when present.
- [ ] #3 Only one group is expanded; first use opens current or most recent, while explicit saved disclosure preference wins thereafter without responsive-state persistence.
- [ ] #4 Each ordinary nonempty group shows at most five recent chats and ends with the exact View all N in Roleplay action.
- [ ] #5 Global Keyword search returns at most eight local character-chat results and clearing or escaping restores browse disclosure, focus, and scroll.
- [ ] #6 Unavailable rows offer only valid Library recovery; empty state offers Open Roleplay; exact chat Enter uses the shared typed activation contract.
- [ ] #7 This PR renders no Continue search in Character chats control and makes no narrow-terminal claim before the Ctrl+K fallback exists.
- [ ] #8 Production CSS and tests cover 52x20, standard widths, keyboard, pointer, truncation, empty, failure, preference migration, and exact activation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md and ADR-083. Reason: direct implementation of accepted Context ownership and bounds. Prepare a separate PR on latest dev after PR2446. Adapt preserved Task4 implementation and owning fixes to merged projection, paginated repair, and strict typed navigation contracts. This PR must not render Continue search in Character chats. Verify focused baseline, new adaptation RED/GREEN, full targeted Context and reachable Library/avatar tests, production geometry, startup/static/CSS checks, then independent review. Record native/platform gaps honestly; no full repository sweep or semantic feature work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the bounded, avatar-independent Character Context browser, exact typed activation and complete Roleplay/Library handoffs. Reused reviewed Task4 and owning focus/avatar lifetime fixes, adapted paginated repair totals and strict lazy Pydantic payloads, and kept unavailable navigation in the existing Library module boundary (ADR120/ADR083; no new ADR). Search rows retain visible title, character and Local/age metadata; existing outer scrolling follows the exact focused Character control. Targeted aggregate:355 passed/3 inherited-evidence failures; final scoped paint/avatar/rail/preference confirmation:137 passed; final scoped static/CSS/diagnostics:12 passed. No cap raises. Native/platform and aggregate FD attribution remain unverified; retain In Progress and unchecked AC pending controller qualification. Full evidence: .superpowers/sdd/2026-09-05-character-keyword-release-isolation/task-4-report.md.

Fix round1 closes post-await generation admission, retained-profile Start, off-loop Library authority reads, recompose focus ownership and route-owned Back to Console. ADR120 now clarifies the two new routes originate only in Console Context; incumbent repair origins remain unchanged. Return reveal is transient, preserves manual disclosure policy, and keeps the semantic anchor with visible composer fallback at52x20 without resize focus theft. Search copy is Search chats. Covering gate231 passed/3 fixture-wait failures; corrected final ownership/return confirmation19 passed; final static/startup/CSS/canary17 passed. The new unmerged helper initial pin is803 (+90), explicitly approved; all incumbent caps fixed. Native/platform/resource attribution remains unqualified; no Done or AC change. Full fix evidence and capture/source manifests: .superpowers/sdd/2026-09-05-character-keyword-release-isolation/task-4-fix-1-report.md.

Fix round 2 preserves the committed unavailable Library route, browse projection, and Back action through invalid or save-vetoed incoming navigation. Validated candidates are staged separately, with monotonic attempt generations and promotion only after guarded admission; a late superseded save cannot resurrect ownership. Direct ADR120 implementation, no new ADR. Genuine RED: 4 failures; final affected Library/payload/diagnostic/complete-return gate: 85 passed, 1 known incumbent empty-preview deselected; scoped size/slack gate: 4 passed. The still-unmerged helper initial pin is 811 (+8), explicitly approved; incumbent caps and both screens are unchanged. No new Ruff diagnostics; four incumbent diagnostics remain. Native/platform/resource qualifications remain open; status and AC unchanged. Full evidence: .superpowers/sdd/2026-09-05-character-keyword-release-isolation/task-4-fix-2-report.md.

PR2452 fix round 3 reuses strict shared 512-character Console query validation in the widget and controller before trimming or DB work. Invalid edits restore the last valid visible query with bounded native feedback; clear, Enter safety, and later valid search are covered. Added Google-style documentation to all four new unavailable route APIs. Removed the redundant cold-resume Character worker through the existing mount-visit guard, preserving mounted initial groups and warm refresh; no allowlist or cap change. Direct ADR120/shared-validation implementation, no new ADR. Focused RED/GREEN 10 failures to 12 passes; final covering 136 passed/1 unchanged pre-Console splash/stagger observation failure, also isolated. Worker census, exact warm handoff, and UI-ready 971/972 pass; no new lint/format diagnostics. Status/AC and qualification gaps unchanged. Full evidence: .superpowers/sdd/2026-09-05-character-keyword-release-isolation/task-4-fix-3-report.md.
<!-- SECTION:NOTES:END -->
