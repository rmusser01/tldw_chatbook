---
id: TASK-31747
title: Restore readable File Notes error text across shipped themes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 20:02'
updated_date: '2026-09-05 20:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Save failed readable under the application stylesheet in every shipped theme without changing error copy, draft behavior, geometry or Git commit-error styling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Save failed painted foreground and background meet at least 4.5:1 in all shipped themes at both existing terminal sizes
- [x] #2 File Notes error and recovery text, draft preservation and click targets remain unchanged; Git commit-error styling is untouched
- [x] #3 Source CSS and generated bundles are synchronized, relevant tests and checks pass, and no design-context metadata is changed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve existing two-size RED evidence and expand painted-text coverage over all 72 shipped/built-in themes; inspect the app-tier override and computed foreground/background. Stop only the characterization fixture poll after initial scan so seeded states remain stable.
2. Separate only File Notes save error from the Git error selector. Use existing $ds-text-primary / $surface semantic pair with full opacity after the all-theme probe showed the original error tint fails on litestep; preserve bold labels and geometry. Regenerate CSS with the repository builder.
3. Verify all 72 themes at both sizes with the unchanged enforced 4.5 threshold, check Git styling unchanged and focused draft/recovery/live-poll coverage, run scoped checks, request parent review and commit independently.
ADR required: no
ADR path: N/A
Reason: Bounded error-state contrast repair preserves the existing design system, ownership and UI contract; no architecture or redesign decision. Impeccable harden and Textual testing guidance apply.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Characterization: the original app-tier pair paints Save failed at 1.498:1 in textual-light at both sizes. The first incumbent tint correction fixes that but all-theme coverage exposed litestep at 3.516:1 (#ededed on #7d7d7d); Textual automatic text uses a brightness heuristic. A test-local $surface candidate probe measured all 72 shipped/built-in themes at both sizes with no failing Save failed paints. These probes collect diagnostics and are not final test-pass evidence. The expanded loop also crossed the fixture polling interval, so its explicitly seeded state characterization now stops the real poll timer after initial scan; live polling tests are unchanged.

Implemented the smallest separate Save failed app-tier rule using existing $ds-text-primary and $surface; Git commit-error declarations are byte-identical to baseline. The original tint candidate was revised after measured litestep failure, with parent approval before the second edit. Bold explicit copy, draft behavior, geometry and live polling are unchanged.
Verification: actual CSS all72 themes at120x40 and40x20 plus explicit state labels, draft recovery, real polling/editor retention, bundle reproduction and boot CSS budget:12 passed104.50s (/private/tmp/library-contrast-final-31747.xml). Independent enforcing wide minimum recorder: Save failed5.070:1 earthy_nature; Offline/Conflict5.117:1 litestep. Source builder regenerated only the expected bundle hunk; budget792135/804000. Ruff and changed-range formatting pass, git diff --check passes. Impeccable harden informed all-theme painted-cell verification; one manual detector run flagged only preexisting literal colors at unrelated lines, no new literal colors or design metadata changes. ADR check remains no/N/A: routine correction under existing semantic design.

Parent reviewed the exact CSS/test diff and approved the scoped commit; all acceptance criteria satisfied. A generalizable lesson is that automatic text selection is not itself WCAG proof: the measured litestep midpoint selected white on gray, while the existing semantic surface preserved legibility in every shipped theme. This incident and the seeded-state polling isolation are documented here without altering shared design metadata.
<!-- SECTION:NOTES:END -->
