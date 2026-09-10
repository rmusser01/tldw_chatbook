---
id: TASK-32223
title: >-
  Library Skills list cannot distinguish an approved skill from an unapproved
  one
status: Done
assignee: []
created_date: '2026-09-10 14:55'
updated_date: '2026-09-10 20:20'
labels:
  - library
  - skills
  - ux
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rows for a trust-approved and an unapproved skill paint identically in the list; only the editor's trust panel tells. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 20.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The list row carries the trust state as a text label or glyph from the legend
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test: an approved and an unapproved skill row paint the same text.
2. Add SkillListRow.trust_label (trusted / needs review / locked) in Library/library_skills_state.py, derived from the record's trust_status with the trust_blocked flag as the fallback.
3. Render it on the row label in library_skills_canvas.py.
4. Docs + live-verify.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`SkillListRow` gains a `trust_label` beside the `trust_glyph` it explains, and
the canvas appends it to the row label: "✓ code-review · trusted",
"⚠ summarize-meeting · needs review", "· locked" for a locked skill. Three
states, not the eight the editor's trust panel carries -- a list row only has
to answer "can I use this".

Derived from the record's `trust_status`, with `trust_blocked` as the fallback
for an absent or unrecognised status. Fix round 1 (review finding 10): that
fallback no longer says "trusted" in the not-blocked direction. A blocked row
still reads "needs review"; anything unrecognised keeps the glyph and says no
word at all, because the glyph is decoration and the word is a claim, and
over-claiming trust is the wrong direction to fail in.

Live on the seeded profile at 235x52 both rows read "· needs review" (that
profile has trust unset, so it has no trusted skill to contrast with); the
two-state contrast and the fallback table are pinned in
Tests/UI/test_library_crit9_shell.py.

Files: tldw_chatbook/Library/library_skills_state.py,
tldw_chatbook/Widgets/Library/library_skills_canvas.py,
Tests/UI/test_library_crit9_shell.py, Docs/User_Guide/library/skills.md.
<!-- SECTION:NOTES:END -->
