---
id: TASK-2803
title: Rewrite README for newcomers
status: In Progress
assignee: []
created_date: '2026-07-24 00:43'
updated_date: '2026-08-24 05:04'
labels:
  - docs
  - onboarding
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-07-23-newcomer-first-readme-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the project landing page easy for new users to understand, set honest expectations about the Alpha project state and goals, and guide a source-checkout user from installation to a first hosted or local-model conversation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A newcomer can quickly understand what tldw_chatbook is, who it serves, and its local-first goal.
- [ ] #2 A clearly labeled Alpha status section distinguishes dependable current capabilities from evolving or optional workflows.
- [ ] #3 The primary quick start installs and launches the latest source checkout in a virtual environment.
- [ ] #4 Hosted-provider and local-model setup paths both lead to a clearly described first Console message.
- [ ] #5 Advanced extras and configuration are concise and linked without obscuring the newcomer path.
- [ ] #6 Stale or duplicated README content is removed and remaining commands and local links are verified.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify README-facing package metadata, entry points, current navigation, setup-wizard recovery paths, and maintained documentation targets.
2. Rewrite README.md around the approved newcomer-first structure: introduction, Alpha status, source quick start, hosted/local first conversation, concise capabilities, direction, optional extras, configuration/data, troubleshooting/docs, and contribution/license.
3. Validate commands, metadata claims, Markdown structure, relative links, and the focused runtime/recovery baseline; review the diff for stale or duplicated material.
4. Record implementation notes, ADR outcome, verification evidence, check all acceptance criteria, and close TASK-2803.

ADR required: no
ADR path: N/A
Reason: documentation-only alignment with existing behavior and accepted product/navigation decisions; no architecture or runtime policy changes.
<!-- SECTION:PLAN:END -->
