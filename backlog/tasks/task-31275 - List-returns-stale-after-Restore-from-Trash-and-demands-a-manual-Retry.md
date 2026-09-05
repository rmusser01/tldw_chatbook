---
id: TASK-31275
title: List returns stale after Restore from Trash and demands a manual Retry
status: Done
assignee: []
created_date: '2026-09-04 13:54'
updated_date: '2026-09-04 20:48'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P2: after restoring an item in Trash and pressing `‹ Media`, the list comes back with `Media changed; retry to load a current page.`, every row and action rendered `○`, and `Page boundary is unknown.` until the user presses Retry (B cap_104-110). The app itself made the change, so it knows the page is stale; the honest-stale gate exists for external changes, not for the app's own mutations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Restore or permanent delete in Trash, returning to Media shows a fresh list without pressing Retry
- [x] #2 The stale-list gate still fires for changes the app did not make itself
- [x] #3 Test plus live verification
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: real-DB test — trash, open Trash, Restore, ‹ Media → list fresh (no Retry, no ○ rows); reuse the browse-controller's external-mutation test as the negative control
2. GREEN: route Restore and permanent delete through the shared committed-mutation completion (reconcile_committed_mutation + page/facet re-request), the same path every other Media write takes
3. Live tmux 235x52
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both Trash mutators now take the one completion path every committed Media write takes (_complete_library_media_mutation → reconcile_committed_mutation → re-request page/facets); Restore passes the same bounded summary the bulk-delete Undo uses; the refresh_normal_media/stale_normal_media flags had no other callers and are gone, so there is one committed-mutation path instead of three. The controller is untouched, so the honest-stale gate still fires for changes the app did not make (pinned by the existing browse-controller shrink test). Live: Restore → ‹ Media → 'Media (3) / 1-3 of 3', permanent delete → 'Media (2)', zero hits for 'Media changed', 'Retry', 'Page boundary is unknown', '○'. Deferred: mark_stale_after_trash_restore is now production-dead (tests only); the permanent-delete reconcile is a content no-op that still flips the page stale when has_authority is false (controller change barred here); the guide's 'no permanent-delete action' sentence needs correcting.
<!-- SECTION:NOTES:END -->
