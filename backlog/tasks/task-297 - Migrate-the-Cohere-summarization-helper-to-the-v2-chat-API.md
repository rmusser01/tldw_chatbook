---
id: TASK-297
title: Migrate the Cohere summarization helper to the v2 /chat API
status: Done
assignee:
  - '@claude'
created_date: '2026-07-17 21:47'
updated_date: '2026-07-17 23:27'
labels:
  - providers
  - maintenance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-267 migrated chat_with_cohere to Cohere v2 /chat, but Summarization_General_Lib.py (lines ~976/1029) still calls v1 /chat — the two Cohere code paths now diverge on API version (final-review finding, PR for task-267). Migrate the summarization path to v2 for consistency before Cohere retires v1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No tldw_chatbook code path calls Cohere v1 /chat
- [x] #2 Summarization behavior pinned by tests before and after the migration
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
summarize_with_cohere migrated v1->v2 /chat (repo's last v1 caller after task-267): messages array w/ system inline (blank omitted), stream flag, parts-array content parse, streaming content-delta loop. Error string formats preserved for the dispatcher; inline default model aligned to config's command-a-03-2025; dead second system_message default removed; streaming 4xx now reports the same pinned error format as non-streaming. KEY FINDING (live smoke caught a review-introduced regression): with 'accept: application/json' the REAL v2 stream is raw JSON lines with NO SSE framing (unlike chat_with_cohere which negotiates SSE) — a strict data:-only filter dropped every line to zero chunks; parser handles both framings, pinned by framing-specific tests. 10 tests at the requests boundary; grep-verified zero cohere.ai/v1 references remain. Live smoke: both paths return real summaries (streaming 18 chunks). Review (sonnet) 'with fixes' — Important (event-line skip) + 2 adjacent Minors applied; latent test-env coupling noted not blocking. Branch claude/cohere-summarize-v2-297.
<!-- SECTION:NOTES:END -->
