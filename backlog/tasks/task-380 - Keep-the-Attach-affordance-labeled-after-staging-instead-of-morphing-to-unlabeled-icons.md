---
id: TASK-380
title: Keep the Attach affordance labeled after staging instead of morphing to unlabeled icons
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Idle composer shows 'Send  Attach  Save'. Once anything is staged the middle button re-labels to '(paperclip)(check)' and an extra '✕' appears. The check-glyph reads as a status ('attached OK') rather than the action 'attach another', and nothing labels ✕ as 'clear all attachments'.

**Repro:** Stage one file and compare the composer button row to its idle state.

**Verifier note:** Code-confirmed: attach button relabels to '📎✓' with tooltip 'Attached: … Press to replace.' (console_composer_bar.py:1752-1753) and the ✕ tooltip is 'Remove the pending attachment.' (line 1883) — both tooltips are stale single-attachment-era copy now that staging appends up to 5 (task-217), which reinforces this as an unrecorded polish defect rather than settled design. P3 correct.

**Source:** Console UX expert review 2026-07-20 (finding j3-attach-button-morphs-to-icon; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J3 attachments journey. Evidence: `j3-16-click-chip.png`, `j3-40-two-images-chip.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keep the verb label ('Attach' or 'Attach more (1/5)') and give the clear control an explicit label or count-accurate tooltip
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Staging no longer morphs the Attach button into the status glyph "📎✓" (which read
as "attached OK", not a control). It keeps the action verb ("Attach +") and the
tooltips are now count-accurate now that staging appends up to 5 (task-217):
attach tooltip "Attach another file (N of 5 staged)." and the clear (✕) control
"Clear all N attachments." / "Clear the attachment." `set_pending_attachment_label`
gained `count`/`total` params (default 0 = generic copy, backward compatible),
supplied by `_sync_console_composer_action_state` from `len(pendings)` +
MAX_PENDING_ATTACHMENTS. Harness test asserts the verb survives + count tooltips.
<!-- SECTION:NOTES:END -->
