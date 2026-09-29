---
id: TASK-33163
title: Add Console hook settings and consent review
status: In Progress
assignee:
  - codex
created_date: '2026-09-28 03:03'
updated_date: '2026-09-28 03:06'
labels:
  - hooks
  - console
  - settings
dependencies:
  - TASK-33151
references:
  - backlog/decisions/197-console-hook-configuration-review.md
documentation:
  - Docs/superpowers/specs/2026-09-27-console-hook-settings-and-review-design.md
  - Docs/superpowers/plans/2026-09-27-console-hook-settings-and-review.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let Console users inspect and approve hook commands before execution, and manage hook configuration in canonical Settings, following the approved native design and ADR-197.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A persistent Console Hooks action displays enabled pending and error states and opens exact hook details and current permissions with mouse and keyboard access at supported terminal widths.
- [ ] #2 Existing, new, and changed enabled hooks require persisted exact-definition review on the next Send; cancelling or navigating preserves drafts and stale callbacks cannot send twice or into another chat.
- [ ] #3 Shared foreground, queued, durable, recovered, and viewless admission and subprocess launches enforce consent, including revocation races, failed persistence, legacy duplicates, and notification target ownership.
- [ ] #4 Canonical Settings provides staged hook add/edit/enable/disable/remove and Save/Revert, validation and impact copy, permission review and deep links, preserving concurrent edits and unknown configuration fields.
- [ ] #5 Targeted runtime and UI checks, private storage and sensitive-path checks, design-token and generated-style checks, and isolated live verification pass with documented limits and updated user guidance.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/197-console-hook-configuration-review.md
Reason: implement accepted ADR-197 and the reviewed spec; preserve ADR-148, ADR-033, and ADR-150.
1. Finish and self-review Docs/superpowers/plans/2026-09-27-console-hook-settings-and-review.md, grounding each interface and check in the actual repository.
2. At execution, reuse a suitable managed worktree or create one from the committed baseline; preserve the shared checkout and recheck caller seams before edits.
3. Implement lossless hook identity and config transactions, persistent consent and serialized launch authority, and shared Send/queue admission with targeted regressions.
4. Implement the native review modal, persistent Console action, and staged canonical Hooks Settings category using existing UI and token patterns.
5. Run the plan-scoped checks and isolated live verification; update user guidance, review the implementation, and record evidence before marking acceptance criteria complete.
Renumbering provenance: CLI assigned TASK-33152; the live all-ref/history and 87-worktree sweep found TASK-33162 as maximum, so the new task became TASK-33163 before cross-references were added.
<!-- SECTION:PLAN:END -->
