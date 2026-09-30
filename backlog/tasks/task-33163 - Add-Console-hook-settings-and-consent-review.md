---
id: TASK-33163
title: Add Console hook settings and consent review
status: Done
assignee:
  - codex
created_date: '2026-09-28 03:03'
updated_date: '2026-09-30 01:27'
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
- [x] #1 A persistent Console Hooks action displays enabled pending and error states and opens exact hook details and current permissions with mouse and keyboard access at supported terminal widths.
- [x] #2 Existing, new, and changed enabled hooks require persisted exact-definition review on the next Send; cancelling or navigating preserves drafts and stale callbacks cannot send twice or into another chat.
- [x] #3 Shared foreground, queued, durable, recovered, and viewless admission and subprocess launches enforce consent, including revocation races, failed persistence, legacy duplicates, and notification target ownership.
- [x] #4 Canonical Settings provides staged hook add/edit/enable/disable/remove and Save/Revert, validation and impact copy, permission review and deep links, preserving concurrent edits and unknown configuration fields.
- [x] #5 Targeted runtime and UI checks, private storage and sensitive-path checks, portable-backup exclusion of hook consent, design-token and generated-style checks, and isolated live verification pass with documented limits and updated user guidance.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the persistent Console Hooks action, expandable exact-command review, next-Send consent gate and canonical Settings → Expert → Hooks staged editor. Cancellation preserves the draft. Shared foreground/queued/durable/viewless admission and subprocess creation enforce persisted exact-definition authority. Guarded saves preserve unknown and invalid configuration, concurrent edits and complete unchanged legacy groups; revocation and failed refresh fence future launches.

Integrated only the hook design/plan and four implementation commits onto dev e5ac111967 on codex/hook-review-dev. Preserved dev's token compiler, responsive modal rules, runtime callbacks and Settings contracts. The concrete JSON/lock owner participates in ADR-126 nested raw-file admission while retaining the outer config lease and its narrow helper scope. Backup recognizes but excludes local consent and refuses restore input. Added resettable handling for decoder depth exceptions.

ADR check: direct implementation of backlog/decisions/197-console-hook-configuration-review.md and ADR-126; no new ADR, dependency or database migration. Original codex/hook-review and its earlier 417-case evidence remain historical. The shared modal/editor and dev adaptations form one final integration unit.

Previous e5ac111967-based validation: 201 distinct feature and Settings metadata cases passed across targeted runs. Core runtime/config/admission/viewless run: 153 passed. Latest consent owner suite: 42 passed. Inventory: 27 passed. Mounted review/editor: 13 passed. Real Settings metadata: seven passed. All eight token and 29 CSS-integrity cases passed; styles rebuilt from source. Authored Ruff/format checks are clean across 48 changed Python files; all three QA scripts pass full Ruff/format checks. Real TldwCli passed six private-profile checks at actual 80×24 and 120×40 using a harmless real hook and recording gateway.

Eleven broader failures reproduce on an untouched archive of the exact dev base. Interrupted exploratory bundles qualify only recorded completed cases. No full suite, real provider generation, push or merge was performed. The current branch's named results, captures, baseline comparisons and limits are in Docs/superpowers/reviews/2026-09-27-console-hook-settings-and-review.md and its QA artifacts.

Self-review completed for launch/revocation ownership, narrow nested storage scope, excluded restore authority, callbacks, lossless Settings writes and generated styles. Changed core hook inventory/config transactions and consent authority, Console runtime/admission/UI routing, shared review modal, canonical Settings editor/registry/search, raw participant and private-path integration, backup declarations and targeted tests. Updated the guide, accepted plan/ADR and evidence-based lessons.

Current-dev continuation (2026-09-29): carried only four implementation commits onto 857b3dd7d0 as codex/hook-review-current. Rebuilt CSS from current sources. Targeted evidence: 367 distinct passing cases including 201 feature/category cases; two Settings search assertions reproduce on current dev application sources. Real TldwCli passed all six private-profile checks at actual 80×24 and 120×40. QA now isolates the HOME-based recovery bootstrap; the queue-controller test module uses the existing bound-profile fixture. Authored-line Ruff/format checks and QA-script lint/format passed. See the updated implementation review and machine-readable QA results. No full suite, real provider generation, push or merge.
<!-- SECTION:NOTES:END -->
