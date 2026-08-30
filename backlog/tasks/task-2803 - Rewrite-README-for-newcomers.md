---
id: TASK-2803
title: Rewrite README for newcomers
status: In Progress
assignee: []
created_date: '2026-07-24 00:43'
updated_date: '2026-08-30 17:16'
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
- [x] #1 A newcomer can quickly understand what tldw_chatbook is, who it serves, and its local-first goal.
- [x] #2 A clearly labeled Alpha status section distinguishes dependable current capabilities from evolving or optional workflows.
- [x] #3 The primary quick start installs and launches the latest source checkout in a virtual environment.
- [x] #4 Hosted-provider and local-model setup paths both lead to a clearly described first Console message.
- [x] #5 Advanced extras and configuration are concise and linked without obscuring the newcomer path.
- [x] #6 Stale or duplicated README content is removed and remaining commands and local links are verified.
- [ ] #7 The original layered README is restored as the source document rather than the short replacement being expanded.
- [ ] #8 The first two screenfuls show the product, explain its value and Alpha state, and provide a five-minute path to launch and a first conversation.
- [ ] #9 Useful detailed feature, optional-install, configuration, troubleshooting, and development reference material remains available below the newcomer path.
- [ ] #10 The corrective pull request contains no unrelated repository cleanup, generated artifacts, or application-code changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Correct the approved design to restore the original README's layered structure and submit it to focused review before implementation.
2. Restore the README immediately before PR #2045 as the source document, then preserve its useful depth while repairing stale, duplicated, inaccurate, and poorly ordered material.
3. Verify package metadata, entry points, current navigation, setup-wizard recovery paths, optional extras, maintained documentation targets, and the selected landing-page screenshot.
4. Validate commands, Markdown structure, relative links, focused documentation/runtime checks, and exact PR scope.
5. Record the corrective implementation notes and verification evidence, check the corrective acceptance criteria, and close TASK-2803 again.

ADR required: no
ADR path: N/A
Reason: documentation-only alignment with existing behavior and accepted product/navigation decisions; no architecture or runtime policy changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rewrote the project landing page around a newcomer-first path: plain-language purpose, honest Alpha status, source-checkout quick start, hosted and local provider setup, first Console message, concise capability and goals sections, configuration and profile-owned data locations, troubleshooting, documentation, contribution, and license guidance. Removed stale duplicate material and clarified that `~/.local/share/tldw_cli/` is the base storage directory while fresh installs use its `default_user/` profile child.

Cleaned obsolete root artifacts, plans/PRDs, and QA screenshots requested for the same PR while preserving maintained project documentation and referenced evidence. Rebased onto the latest `dev`; the previously local safe run-log CI repair was omitted because equivalent newer fixes and regression coverage are already present upstream. ADR: no ADR required because this is documentation and repository-hygiene work aligned with existing behavior.

Corrective follow-up opened 2026-08-30 after user review rejected the short replacement. The corrective design restores the pre-PR README as the source document, keeps its layered technical depth, and limits the follow-up PR to README-specific documentation and any deliberately selected landing-page asset.
<!-- SECTION:NOTES:END -->
