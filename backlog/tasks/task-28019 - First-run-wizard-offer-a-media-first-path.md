---
id: TASK-28019
title: First-run wizard - offer a media-first path
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
updated_date: '2026-09-02 21:07'
labels:
  - onboarding
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The five-step wizard (Welcome, Provider, Model, Voice, Summary) is entirely LLM and voice setup; a user whose goal is ingesting media must skip everything (Esc plus confirm) and find Import on their own. Home's card helps, but the wizard never mentions the Library ingest loop. Related live observation: the Check-model-lists-online consent modal fired on top of the user's first navigation - consider sequencing startup modals so they do not interrupt.

Re-verified 2026-09-02: wizard is now six steps (Welcome/Provider/Model/Voice/Protect/Summary) - still no media/ingest path; the "Check model lists online?" modal fired on first Library entry in the live run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The wizard surfaces the import or Library path as a first-class option or step
- [ ] #2 Skipping remains one gesture
- [ ] #3 Startup modals do not interrupt the user's first navigation
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RECON (attempted, then DEFERRED — the "contained" placement is not actually contained): the routing works cleanly — a Summary-step "Import" exit button -> _finish(TAB_INGEST) + the app-side completed-exit gate (_handle_first_run_wizard_result) accepting TAB_INGEST alongside TAB_CHAT/TAB_HOME navigates correctly (verified by a unit test). BUT a FOURTH action button does not fit the Summary step's docked `.setup-summary-actions` row at the wizard's tested 80x24 budget: "Review provider setup" (21 chars) + 3 more actions + the wizard's border/padding exceed 80 cols, so Textual truncates the last label ("Review setti"). Tried min-width:0 (buttons size to label — helped but not enough), compact buttons, and tightened row padding/margin; none fit four full labels because the existing "Review provider setup" is the long pole and is not this task's label to change. A clean fix needs a DIFFERENT placement (a Welcome-step media/import track, or a summary-body affordance) — a design decision, not a drop-in. `test_summary_three_actions_visible_and_focused_on_full_track[size0=(80,24)]` enforces all actions painted at 80x24 and is the guardrail to respect. AC#3 (startup consent modal) remains a separate startup-sequencing concern (see the recon in the original filing).
<!-- SECTION:NOTES:END -->
