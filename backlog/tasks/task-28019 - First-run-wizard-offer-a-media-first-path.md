---
id: TASK-28019
title: First-run wizard - offer a media-first path
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The wizard surfaces the import or Library path as a first-class option or step
- [ ] #2 Skipping remains one gesture
- [ ] #3 Startup modals do not interrupt the user's first navigation
<!-- AC:END -->
