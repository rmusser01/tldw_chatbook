---
id: TASK-34396
title: Trim notes sync watcher and reconciliation hot paths
status: Done
created_date: 2026-10-12 03:20
labels:
- performance
priority: medium
assignee:
- '@zcode'
updated_date: 2026-10-12 04:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
notes_sync_runtime observe_root (lines ~679-689) rescans every discovered file with a fresh SHA-256 (stable_identity_digest) per missing binding on mass rename/move - O(B_missing x F) hashing per pass; the poll watcher (changed_root_ids ~838-854 -> note_import_discovery discover_import_sources ~296-359) re-walks every root with per-entry stat + full sorted() + a Path constructed per candidate for a suffix check every tick (1-10s backoff) even when nothing changed. Chat/chat_persistence_service.py save_history (~3736-3743) takes an O(passes x n) transitive variant closure and does ~2n queries per resave via update_message_content re-reading each row.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Binding reconciliation uses a prebuilt identity_digest index instead of rescanning all discovered files per missing binding,Discovery signature pass no longer sorts the full candidate list or constructs Path objects for suffix checks (hash the entry set / compare suffixes directly),save_history computes the variant closure from an adjacency index in O(n) and skips per-message re-reads when the row is already loaded,Existing notes-sync and chat-persistence tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Identity-digest index for binding reconciliation 2. cheaper discovery signature 3. save_history adjacency closure + transactional resave 4. targeted tests + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented and verified (897 consolidated tests green); see task file Implementation Notes; PR #3083
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
