---
id: TASK-32871
title: Add registered Chatbooks and sharing to Library
status: To Do
assignee: []
created_date: '2026-09-19 22:00'
labels:
  - library
  - artifacts
dependencies:
  - TASK-32870
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Browse registered Chatbooks and Console saved responses in the Library artifact reader, with capability-based actions and a visible link to the existing ZIP-pack manager. Keep multi-item sharing and its application-owned lifetime. Governed by ADR-172 and stage 2 of Docs/superpowers/plans/2026-09-19-library-artifacts.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All matching registered Chatbooks are reachable, Console response excerpts are labeled, and sharing is offered only for usable exported bundles.
- [ ] #2 Manage Chatbook packs opens the existing manager with its creation, import, template, export, and delete workflows intact.
- [ ] #3 Multi-item sharing remains available and Library exposes Manage and Stop across types, filters, pane collapse, and navigation.
- [ ] #4 Targeted registry, manager-link, sharing, production-CSS, and worker-lifecycle checks pass without altering storage or publication authority.
<!-- AC:END -->
