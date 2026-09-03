---
id: TASK-28014
title: Library rail - media counts stale after Trash restore
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
updated_date: '2026-09-02 21:08'
labels:
  - library
  - bug
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-verified 2026-09-02 live on dev tip (worktree media-ux-fixes @ b7e89b6de, tmux scratch-profile run). Confirmed and worse than filed: deleting updates rail and canvas counts together, but after RESTORE from Trash the media canvas enters a degraded state - header shows bare "Media" with "Media changed; retry to load a current page", the restored item is missing from the list, and the pager shows "List may be out of date / Page boundary is unknown" with a manual Retry button. After clicking Retry the canvas shows Media (3) but the rail STILL says Media (2) - rail and canvas disagree even post-retry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rail media count and Details tally match the canvas immediately after restore and other Trash mutations
- [ ] #2 A pinning test covers the restore-count path
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RECON (not started — needs reproduction-based debugging): rail/Details count = cached _local_source_counts['media'] (delta-mutated, not a live query). Delete = unconditional -1 (library_screen.py:42135/27367) + refresh_normal_media=True -> authoritative controller.request. Restore (_restore_library_media_from_trash ~28118) = GUARDED +1 behind 'if _source_record_id(restored_record) not in existing_ids' + refresh_normal_media=False (defers canvas to manual Retry, which never writes _local_source_counts). CAVEAT: static analysis does NOT cleanly explain the 'rail stuck low' symptom -- the restore summary emits id as a BARE INT (_validated_library_media_trash_restore_summary:17963 {'id': restored_id:int}) while _local_source_records use canonical 'local:media:N' strings, so _source_record_id(restored)='5' should NOT match existing {'local:media:5'} -> guard TRUE -> +1 SHOULD fire (opposite of symptom). Needs a reproduction test driving delete->restore that asserts _local_source_counts['media'] to see the real mechanism before fixing. Sibling _undo_library_media_bulk_delete (~27547) has the same guarded-increment pattern. Only existing test (test_library_media_trash.py:1771) asserts nothing about the count.
<!-- SECTION:NOTES:END -->
