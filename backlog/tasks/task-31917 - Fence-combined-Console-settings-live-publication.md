---
id: TASK-31917
title: Fence combined Console settings live publication
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06'
updated_date: '2026-09-06 01:39'
labels:
  - tests
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The remaining fork-transition census identifies combined Console settings commits
as an unclassified live-mutation route. Its generation-settings setter is fenced,
but releases that fence before the context policy is updated, allowing a fork to
observe only half of the submitted configuration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fork eligibility and fence issuance reject the source between generation-settings and context-policy publication.
- [x] #2 Other sessions remain forkable; the source becomes forkable after success or an exception without leaking transition ownership.
- [x] #3 Exact-origin, duplicate-submission and existing settings persistence behavior are preserved.
- [x] #4 Targeted complete-file tests, scoped static checks and review qualify the repair; unrelated census failures remain explicitly open.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In `Tests/Chat/test_console_settings_fork_publication.py`, build real process-local Console sessions and submit a typed settings draft. Wrap the real context-policy publisher to attempt fork eligibility/fence issuance after generation settings changed but before policy publication. Check nonempty observations, other-session isolation and success/exception release. Run the file and require the missing fence to fail.
2. In `ConsoleChatStore.commit_console_settings_live`, enter `self._fork_source_transition(session.id)` around the existing preparation-lock publication block. Reuse its nested transition counting; do not change validation, durable writers, rollback semantics or APIs.
3. Run the regression and the complete settings-apply, settings-persistence and fork files. Register only the newly repaired `commit_console_settings_live` route in `Tests/Chat/test_console_fork_transition_census.py` after its existing owner-aware scanner confirms the boundary. Rerun the census diagnostically, retaining its other unclassified-route failures rather than adding blanket exceptions.
4. Run scoped Ruff and changed-range formatting, review the diff, record evidence and update the draft PR. Leave detached display-name plan lifetime and the other census routes for separate contract review.

ADR required: no
ADR path: backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md
Reason: Restore the existing immutable fork-configuration boundary using its existing guard; no new runtime or ownership contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the reproduced partial settings publication with one outer existing fork transition around the combined live commit. Both success/failure regression variants failed before the fix because real fork eligibility and fence issuance admitted partial configuration. The regression now checks isolation, release and stale-fence invalidation. Six complete settings/fork/first-send files: 330 passed in 35.61s, two existing warnings; /private/tmp/tldw-settings-fork-final.xml. Scoped Ruff, test formatting, exact changed-range store formatting and diff checks pass; independent review found no issues. All 81 store diagnostic statements are unchanged. Registered only the repaired route in the census; other routes remain under review. No persistence or rollback semantics changed. Added the testing-evidence lesson about nested guards not covering caller publication. ADR check: existing ADR-092, no new ADR required.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31798 was renumbered to TASK-31917 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
