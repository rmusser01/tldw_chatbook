---
id: TASK-33627
title: >-
  Console unification: give each screen region one job per the region-ownership
  ADR
status: To Do
assignee: []
created_date: '2026-09-30 03:04'
labels:
  - console
  - ux-review-2026-09-29
  - unification
dependencies: []
references:
  - qa/console-ux-review-2026-09-29/report.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: no Console region has a single job, so information repeats and the screen overflows. The left rail mixes a Terminal launcher, workspace navigation, a chat list with five or six search fields, character, model, agent log and storage details. The Inspect rail is one scroll of about 15 sections over about 90 rows, uses four header grammars and includes debug payloads. The active chat's title appears up to six times and provider/model three or four times. There are three create buttons, one of them a worker-mounted alias that moves around. The header action row duplicates controls found elsewhere. At 80x24, chrome takes half the rows. The complete next-send preview is hidden behind the cost chip or an undocumented key.

What unifying means here: one region-ownership decision that assigns each region one job. Header: an identity and authority breadcrumb (workspace, chat, local/server, state) plus at most three actions. Tab strip: owns chat identity, with New and Temporary pinned outside the scroll area. Left rail: one Chats browser with one search. Inspect: three fixed groups (Next send, derived from the prepared request; Run; Selected turn). Status strip: at most five priority chips. Composer: the draft plus run controls. Terminal, Details and live-work diagnostics move out of the default view. The decision is being proposed as ADR-210, a proposal document in its own PR that is pending owner approval. It would revise recorded decisions, including ADR-017 (left-rail usability), ADR-083 (edge rails and workspace-tree ownership), TASK-23196 and TASK-24611. Implementation under this umbrella is gated on that approval, and no child work should start until ADR-210 is accepted. It also depends on the run-state, action-register, attention-registry and width-priority unifications in this review. The theme has no P0/P1 findings.

Evidence: qa/console-ux-review-2026-09-29/report.md (theme 'Regions have no single job') and qa/console-ux-review-2026-09-29/findings.md / findings.json.

Related P2/P3 in the ledger: G2-36, G2-17, G2-15, G2-13, G2-14, G2-19, G2-18, G4-55, G4-36, G2-34, G2-35, G2-26, G2-28, G2-29, GAP5-08, G2-12, G2-11, G2-30, G2-41, G4-42, GAP5-22, GAP4-13, GAP4-03, GAP4-18.

Source: Console UX review 2026-09-29 — qa/console-ux-review-2026-09-29/report.md (themes) and findings.md (ledger).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ADR-210 (Console region ownership) is accepted by the owner before any implementation under this umbrella begins, and the umbrella's notes link the accepted decision
- [ ] #2 In the default Console view, the active chat's title and its provider/model each appear in only the region the accepted ADR assigns them
- [ ] #3 The left rail offers one chat browser with one search field, and there is exactly one visible create-chat entry point in the default view
- [ ] #4 The Inspect rail presents only the groups the accepted ADR defines, in a fixed order, with no raw internal fields or debug payloads in the default view
- [ ] #5 At 80x24 the transcript receives at least the row budget the accepted ADR specifies, verified by a geometry test on the rendered screen
- [ ] #6 All child tasks filed under this umbrella are Done (the theme has no P0/P1 findings, so its children are the ADR-210 implementation work)
<!-- AC:END -->
