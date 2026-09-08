---
id: TASK-32012
title: Avoid repainting unchanged Console send status text
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-08 00:38'
updated_date: '2026-09-08 01:00'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During a pending Console send, the transcript heading, composer reason and footer hints repaint unchanged text on every 200ms sync. Measured in-flight partial paints disappear when those duplicate writes are suppressed. Remove that unnecessary terminal output while preserving changing content and responsive layout; the external flicker report remains unconfirmed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unchanged in-flight Console syncs do not repaint the transcript heading, composer reason or footer hint text.
- [x] #2 Changed guidance, setup actions, footer hints and terminal widths still update correctly.
- [x] #3 Mounted Enter regression and focused component checks pass; existing send completion and cancellation behavior is preserved.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: routine equality guards in existing rendered widgets; unchanged ownership, content, privacy and responsive layout.
1. Preserve the in-flight Enter experiment and add a mounted regression that measures real compositor spans for the three unchanged visible surfaces while validation is active. The existing per-keystroke TASK-21120 dismissal concerns a different workload; this finding measures repeated idle-send paints.
2. Compare each newly rendered text value with its mounted Static content before update, including styled setup-action content. Keep width, visibility, state and empty-card sync running.
3. Verify changed text and widths through focused component tests; re-run the mounted real-paint test and failed-send regression; measure post-fix in-flight painting, lint changed ranges and self-review. State that the external terminal flicker remains unconfirmed.
4. Address PR #2496 review: manage the mounted regression database with contextlib.closing from acquisition, retaining asynchronous teardown for setup and runtime failures; document the guidance arguments; rerun targeted regressions and preflight before merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added equality checks against each mounted Static widget's current content before writing the transcript heading (both its title and guidance writers), composer disabled-reason content and responsive footer hints. Styled setup actions still participate in equality, and responsive width/visibility calculations and empty-card synchronization still run. No screen-wide throttling, new cache, capture tool, logging surface, ownership change or dependency.

Investigation evidence: holding provider validation open after Enter at four sizes with short and multiline drafts found zero full-screen updates. Apparent rail movement was a single one-row adjustment, not a repeating geometry cycle. Actual partial spans repainted the unchanged heading/reason/footer approximately five times per second; the 128x24 experiment measured 6942 repainted cells over its five-second baseline. Suppressing duplicate writes at just those three widgets left only five caret paints (150 cells total) in a three-second comparison window. This is a demonstrated redraw source; the external reporter's exact visible flicker remains unconfirmed.

The new mounted Enter regression measures actual full/partial compositor output during forced unchanged in-flight syncs and then releases validation and verifies send completion and polling shutdown. Before the fix, 128- and 160-column cases repainted all three surfaces four to five times; the 80-column reason strip is intentionally hidden and its geometry precondition was corrected. The initial implementation left a second heading writer unguarded; all three cases stayed red on heading paint until both writers were covered.

Verification: 15 mounted send, real-paint, diagnostic, title and inline-guidance tests passed after the final change. Another 34 composer-gating, markup, reason-width and footer-content/resizing checks passed (49 targeted checks total). Two broader pre-existing layout tests fail with the original three widget files restored from a846f01b6: test_console_transcript_header_sits_at_top_of_center_panel expects an obsolete one-row spacing, and test_console_transcript_header_and_tabs_have_distinct_visual_roles expects an obsolete tab-strip class. No full test sweep was run.

Static verification: the new test passes Ruff and formatting; changed production ranges format cleanly, git diff --check and the task-ID guard pass. Whole-file Ruff has unchanged existing findings: session surface 20, composer 5, footer 5; no introduced findings. Task remains In Progress under the repository's strict all-green Definition of Done. Added an incident-backed testing lesson and completed self-review. ADR required: no; routine render equality guards preserve existing contracts. The discarded experiment remains outside repository test discovery.

PR #2496 Qodo follow-up: wrapped the mounted regression database in contextlib.closing immediately at acquisition and moved all conversation/provider/harness setup under the existing asynchronous cleanup. Database closure now survives setup and teardown exceptions. Added the Google-style Args contract for sync_inline_guidance. The 17 affected lifecycle, mounted diagnostic and compositor tests pass after these changes; the test file passes Ruff and formatting, and the production surface retains the same 20 baseline lint findings with none introduced. Self-review complete; these review fixes preserve the existing ADR decision and acceptance criteria.
<!-- SECTION:NOTES:END -->
