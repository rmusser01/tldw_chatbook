---
id: TASK-34399
title: DB layer small efficiency batch
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
Six verified small items from the DB review: (1) list_notes/_list_generic_items (ChaChaNotes_DB ~16740,17669) SELECT * full note bodies for list surfaces while the newer library path projects substr(content,1,?) - callers pass limit=10000; (2) search_messages_by_content (~16411-16475) always transfers m.image_data BLOBs though both callers consume text only (sibling loaders have include_image_data=False seam); (3) subscriptions.source has no index - full scans inside BEGIN IMMEDIATE during OPML import (Subscriptions_DB _find_exact_source_id ~2063, get_subscription_id_by_source ~4817, resolve_collection_candidates ~3701); (4) eval_results lacks (run_id, created_at) index so get_run_results (~1964) re-sorts the run per page and json-decodes logprobs for every row; (5) _persist_chunks (Client_Media_DB_v2 ~4641-4720) loops 2 statements per chunk instead of executemany and embeds chunk text in sync payloads; (6) ChaChaNotes execute_query (~4018-4107) pays unconditional log_histogram/log_counter + eager debug f-string formatting per query - gate on metrics-enabled; (7) get_sync_log_entries default limit=None is unbounded (~20742).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 list_notes accepts a projection/preview mode and the limit=10000 caller uses it,search_messages_by_content gains include_image_data: bool = True and text-only callers pass False,Index on subscriptions(source) added via migration following house migration patterns,eval_results gains (run_id, created_at) index,_persist_chunks uses executemany for chunk inserts (and sync-event inserts),execute_query metric emission gated behind a cheap enabled check with lazy formatting,get_sync_log_entries gets a sane default limit (documented),All DB migration/schema tests and the affected store tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. list_notes projection 2. include_image_data seam 3. two indexes via house migration pattern 4. executemany chunk persist 5. metrics gate 6. sync-log default limit 7. tests + notes
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
