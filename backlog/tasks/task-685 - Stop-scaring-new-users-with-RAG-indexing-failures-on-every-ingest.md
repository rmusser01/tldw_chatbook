---
id: TASK-685
title: Stop scaring new users with RAG indexing failures on every ingest
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-26 04:05'
updated_date: '2026-07-26 23:57'
labels:
  - ingest
  - ux
  - rag
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a fresh install with no embedding model downloaded, every successful ingest raises a red 'RAG indexing failed ... All chunks failed embedding generation' notification. The import itself worked, so the first thing a new user sees after their first successful action is a failure they did not cause and cannot act on. Observed during the ingest UAT once folder ingest started completing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh install with no embedding model does not report an ingest as failing when only indexing was skipped
- [x] #2 The user is told what indexing gives them and how to enable it, rather than shown a raw failure
- [x] #3 Genuine indexing failures on a configured install are still surfaced
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A fresh install no longer presents its first successful ingest as a failure.

With the embeddings_rag deps installed but no model downloaded, every chunk fails to embed, and the indexer surfaced 'RAG indexing failed for N item(s): All chunks failed embedding generation' as a warning toast after every otherwise-successful import. The import worked; the first thing a new user saw after their first working action was a failure they did not cause and could not act on.

The discriminator is deliberately NOT the error text alone. A configured install can legitimately fail to embed one bad document, and that is a real failure worth surfacing (AC#3). Guidance is offered only when embeddings have never worked -- nothing indexed in this process OR in the current batch -- and every error in the batch is the embeddings-unavailable one. Anything else reports exactly as before.

Counting the current batch as well as process history matters: if this batch embedded anything, embeddings demonstrably work whatever the history says. That case was what turned my first implementation red.

The message says what indexing gives and how to enable it rather than what broke: 'Saved, but not added to semantic search yet -- no embedding model is set up. Download one in Settings to search this content by meaning as well as by keyword.'

Guidance travels on its own notifier so the app can render it as information rather than a warning; when no guidance notifier is supplied it falls back to the failure channel, so the message is never simply lost.

Mutation-checked: removing the have-embeddings-ever-worked guard downgrades a genuine failure to guidance and fails the AC#3 test.

Files: RAG_Search/ingestion_indexing.py, app.py, Tests/RAG/test_ingestion_indexing.py.
<!-- SECTION:NOTES:END -->
