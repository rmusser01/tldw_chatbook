---
id: TASK-25714
title: Console Context rail still overflows below 140x40
status: To Do
assignee: []
created_date: '2026-08-31 14:27'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-23193 made all Context rail section headers reachable without scrolling at 160x48, and its implementation notes record that 140x40 and below still overflow as follow-up work rather than something silently dropped. This is that follow-up. The rail's fit was bought at 160x48 by merging Sessions into Conversations and trimming section content; the smaller sizes were never re-measured after those changes landed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Context rail's section headers are reachable without scrolling at 140x40
- [ ] #2 Behaviour below 140x40 is either made to fit or given a deliberate, documented degradation rather than an unbounded overflow
- [ ] #3 A test pins the smallest size the rail is claimed to fit, so the claim cannot rot silently
<!-- AC:END -->

## Context

Two constraints from TASK-23193 that this task inherits, so they are not
rediscovered from scratch:

- The obvious lever -- dropping the section headers' border-top rule and their
  2-row min-height -- **was built and did make 140x40 fit (25 rows of content
  in 25 available)**. It was reverted because
  `test_context_section_headers_match_inspector_title_band` deliberately ties
  Context section headers to the Inspector's title band, and on review of both
  screenshots the reviewer chose to keep the rule. So the fit is achievable;
  what blocked it was a visual contract, not layout.
- That contract test is itself now red on dev for an unrelated reason
  (TASK-25715: #2220 changed the Inspector side's padding to 0 while the
  Context side kept 1). Whoever takes this on should settle TASK-25715 first --
  if the shared band is no longer real, the trade TASK-23193 made to protect it
  should be re-decided rather than inherited.

Sizes worth measuring: 140x40, 120x36, and the 80x24 floor.
