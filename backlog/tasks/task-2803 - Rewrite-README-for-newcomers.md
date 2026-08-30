---
id: TASK-2803
title: Rewrite README for newcomers
status: Done
assignee: []
created_date: '2026-07-24 00:43'
updated_date: '2026-08-30 18:05'
labels:
  - docs
  - onboarding
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-07-23-newcomer-first-readme-design.md
  - Docs/superpowers/plans/2026-08-30-task-2803-layered-readme-restoration.md
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
- [x] #7 The original layered README is restored as the source document rather than the short replacement being expanded.
- [x] #8 The first two screenfuls show the product, explain its value and Alpha state, and provide a five-minute path to launch and a first conversation.
- [x] #9 Useful detailed feature, optional-install, configuration, troubleshooting, and development reference material remains available below the newcomer path.
- [x] #10 The corrective pull request contains no unrelated repository cleanup, generated artifacts, or application-code changes.
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
Corrective follow-up started 2026-08-30 after user review rejected the short replacement. Restored the exact 886-line README at `d2ff9c05ca91d7f7b7be80a2401f78f7142e1aff` as the editing source, then rebuilt it as a 700-plus-line layered landing page instead of expanding the rejected memo.

The repaired opening now explains the product, Alpha state, current/evolving/goal boundaries, source-checkout setup, first-run wizard, and hosted/local first-conversation paths. The lower reference retains detailed workflow, optional dependency, speech, model, configuration/data, browser serving, troubleshooting, project structure, development, documentation, contribution, license, and contact material while removing duplicate recommendations, obsolete navigation, brittle config blocks, and the all-extras install command.

Added `Docs/static/tldw-chatbook-console.png`, rendered from the maintained neutral Console SVG used by the current User Guide and inspected at its original 1848×1124 resolution. This avoided reading or displaying any user profile, key, conversation, path binding, or live local state. The opening and reference were committed together after the restored source was repaired; this is the only meaningful sequencing deviation from the plan.

Verification on the rebased content head: isolated no-download editable install reports version `0.1.8.0` and both `tldw-cli`/`tldw-serve`; metadata/extras, Markdown fences/headings, local links/images, and whitespace audits pass; the focused runtime/documentation selection passes 13 tests. A full fail-fast run collected 68,811 tests and stopped on the upstream Actor Pack failure `test_create_new_persona_preserves_incoming_uuid`; the failing test and implementation blobs exactly match `origin/dev`, so no unrelated fix was added.

Current scope is exactly the five corrective files named in the implementation plan: README, one Console screenshot, corrected design, restoration plan, and this task record. TASK-2803 remained In Progress until PR review and CI completed. ADR required: no; this documents existing behavior and accepted product/navigation decisions without changing runtime or architecture.

PR #2235 content-head closeout: Qodo's final assessment endorsed the progressive-disclosure restoration and posted no inline review threads. The Backlog Guard passed; CodeRabbit skipped review on the non-default `dev` target; Cubic completed neutral; platform evidence was intentionally skipped; GitHub reports the PR mergeable. All acceptance criteria were rechecked against the final five-file scope before setting TASK-2803 to Done.
<!-- SECTION:NOTES:END -->
