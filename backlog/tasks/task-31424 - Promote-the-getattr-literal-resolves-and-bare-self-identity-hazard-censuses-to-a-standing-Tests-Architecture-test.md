---
id: TASK-31424
title: >-
  Promote the getattr-literal-resolves and bare-self-identity hazard censuses to
  a standing Tests/Architecture test
status: To Do
assignee: []
created_date: '2026-09-05 01:07'
labels:
  - library
  - architecture
  - tech-debt
  - testing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recipe section 3's sixth bypass shape (bare self used as an identity-compared/screen-identity argument, and its close cousin the unbound-attribute escape via getattr(self, "<literal>", default)) was found twice by one-off, hand-run AST censuses during the wave-4 skills series -- once pre-landing (bare-self-identity) and once only by post-landing review (the getattr/focused escape), the latter surviving a full green verification battery. Every future Library_Modules controller-move series (and any future subsystem's controller) is exposed to the same two silent-hazard shapes, with no automated guard re-running either census against new movers. Promoting both censuses into a standing test closes that gap for every controller under Library_Modules going forward, not just the skills series.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Tests/Architecture test runs the getattr(self, "<literal-string>", default)-with-no-corresponding-property census and the bare-self-passed-as-an-identity-argument census over every controller module under tldw_chatbook/UI/Library_Modules/, not just library_skills_controller.py
- [ ] #2 The new test fails against a reintroduced instance of either hazard shape (a negative-control fixture proves this) and passes on the current tree
<!-- AC:END -->
