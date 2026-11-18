---
id: TASK-41668
title: Bound Library conversation message counts by the requested page
status: In Progress
created_date: 2026-11-18 05:47
assignee:
- '@codex'
labels:
- performance
- library
- sqlite
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3610
documentation:
- Docs/QA/console-library-responsiveness/README.md
updated_date: 2026-11-18 06:08
modified_files:
- tldw_chatbook/DB/ChaChaNotes_DB.py
- Tests/ChaChaNotesDB/test_chat_conversation_parity.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cold Library and saved conversation browsing spend avoidable time counting messages for a bounded page. Reduce that work while preserving the exact per-conversation counts and storage lifetime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A native SQLite page count keeps the existing soft-delete options, zero counts and result identity while bounding work attributable to unrelated conversations.
- [x] #2 Matched sealed Terminal runs record cold and warm Library paint, startup and first input, full event-loop windows, and ordinary SQLite retirement.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the excess SQLite VM work on a current database with the real count method and a fixed set of visible conversation IDs.
2. Filter the existing join by the selected conversation primary keys. Preserve count, deletion, projection, policy and native connection ownership.
3. Run the affected native storage, local conversation service and Library entry tests. Compare three alternating sealed Terminal pairs with the same full windows and capture startup stages and resource retirement.
ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: This is a SQL filter placement correction inside the existing admitted read and retirement boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Filter the requested conversation IDs on the parent side of the existing message-count join. The stock SQL regression exceeded its SQLite work bound at 620 groups of 100 instructions and passed with the filter. The test also covers duplicate, deleted and missing IDs and soft-deleted messages. ADR-126 and the existing native lifetime apply.

The default bundled-tokenizer PTY pairs measured count CPU 41.05 to 2.03 ms, median cold paint 806 to 736 ms, and warm paint 41 to 50 ms. The earlier empty-cache pairs and all blocked download attempts are retained. Every process kept all three 13-second windows and quit with zero scoped SQLite descriptors. Launch to first typed echo ranged 5.52 to 8.85 s with the shipped assets. The largest measured stage was the initial Console push through first usable composer.

The database and service selection passed 119 cases. Two legacy migration failures reproduced on the frozen baseline. The Library checks retain the original pre-mount config refusals and matched bootstrap-profile focus failure. Both changed ranges passed formatting. Ruff reported zero additional findings. Source and evidence remain local pending publication and CI. See the QA receipts in Docs/QA/console-library-responsiveness/README.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
