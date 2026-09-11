---
id: TASK-32363
title: >-
  Library Skills: a trust-approved skill reads 'needs review' when the trust
  store is not set up, with no explanation
status: Done
assignee: []
created_date: '2026-09-11 06:19'
updated_date: '2026-09-11 07:43'
labels:
  - library
  - skills
  - copy
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The seeded trust-approved skill renders '⚠ needs review' beside the unapproved one under 'Skill trust isn't set up — set it up to review and use skills.' (B D5 cap 51; A cap 59). The precedence (no trust store ⇒ prior approvals ignored) is never stated. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The list distinguishes an approved skill from an unapproved one, or the banner states that approvals apply once trust is set up
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reword skill_trust_header_line's needs_setup branch to state the precedence (no trust store => every skill reads needs review).
2. Update the pin in Tests/Library/test_library_skills_state.py to the exact string.
3. Update Docs/User_Guide/library/skills.md + stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Took AC#1's second branch: the banner states the precedence.

The list genuinely cannot distinguish an approved skill from an unapproved one under the `needs_setup` posture — with no trust store there is nothing to verify an approval against, so both render `⚠ needs review` and the list looked wrong rather than unverifiable. `skill_trust_header_line`'s `needs_setup` branch now returns:

> Skill trust isn't set up, so every skill reads "needs review" — set it up to review and use skills.

The action id is unchanged (`setup`), so the **Set up skill trust** button and every posture around it are untouched. `grep -rn "Skill trust isn't set up"` found one product occurrence, one pin (`test_skill_trust_header_line_maps_postures`, tightened from a substring to the exact string) and one guide row (`Docs/User_Guide/library/skills.md`); the two `Docs/superpowers/` hits are the historical plan and design record and were left as written.

One cost worth naming: at 60 columns the sentence wraps to 4 lines instead of 2. It fits because task-32354 gave the same canvas 3 rows back from the pager on the same branch (`crit10/wave/pagers/caps/09-skills-60x24.txt` — both skills still visible).

## Files

- `tldw_chatbook/Library/library_skills_state.py`, `Tests/Library/test_library_skills_state.py`, `Docs/User_Guide/library/skills.md`.
<!-- SECTION:NOTES:END -->
