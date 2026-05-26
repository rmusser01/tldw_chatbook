---
id: TASK-70.5
title: Add Sync v2 Notes and Chat restore QA plan
status: To Do
labels:
- sync
- sync-v2
- restore
- qa
priority: high
parent_task_id: TASK-70
documentation:
- Docs/superpowers/specs/2026-05-26-chatbook-sync-v2-completion-roadmap-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate that manually synced Notes and Chat messages can be restored onto a clean or repaired Chatbook profile with correct content, ordering, identity, and recovery-key behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Notes restore preserves title, body, status, deletion/tombstone behavior, and profile ownership.
- [ ] #2 Chat restore preserves conversation identity, message order, roles, parentage, and variants required for continuity.
- [ ] #3 Recovery-key restore decrypts selected envelopes without persisting recovered secrets incorrectly.
- [ ] #4 Restore failure stops before applying corrupt or partial user-visible state.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
