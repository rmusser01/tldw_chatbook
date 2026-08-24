---
id: TASK-21700
title: >-
  RAG cache-hit log writes the user search query into a persistent sink
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - privacy
  - diagnostics
  - rag
priority: medium
---

## Description

`RAG_Search/simplified/rag_service.py` logs a cache hit at INFO level with the first 50
characters of the user's search query interpolated into the message. Search queries are user
content. The diagnostic-privacy work in this repo (TASK-15103/15600, and the persistent
diagnostic inventory that guards it) exists to keep exactly this class of value out of
persistent sinks.

Pre-existing — it was noticed while re-pinning the inventory after an unrelated RAG refactor
re-wrapped the line, and the statement text is byte-identical to what was there before. Filing
rather than fixing in that commit, because the fix is a behaviour change and deserves its own
review.

## Acceptance Criteria

- [ ] It is determined and recorded whether this statement actually reaches a persistent sink at the shipped default log level, or is filtered before it lands
- [ ] If it does reach one, the query text is removed, hashed, or length-only — the diagnostic keeps its debugging value without carrying user content
- [ ] The whole `RAG_Search` tree is swept for the same shape: any diagnostic interpolating a query, a document body, a chunk, or a filename
- [ ] Anything found is fixed in the same pass, or filed with a reason it is safe
- [ ] A test or census guard pins the outcome so a future refactor cannot reintroduce it
- [ ] If the conclusion is that this is acceptable, that reasoning is written down — an unreviewed "it's fine" is what let it sit this long

## Evidence (verified on dev 1daa47f0a, 2026-08-24)

`tldw_chatbook/RAG_Search/simplified/rag_service.py:1340`:

```python
logger.info(
    f"[{correlation_id}] Cache hit for query: '{query[:50]}...'"
)
```

The inventory checker flags added statements with exactly this prompt — *"does it interpolate
user content, a secret, a path, or a URL?"* — and this one does. It was previously at line 1315
with identical text, so the refactor moved it rather than introducing it.

Note the truncation to 50 characters limits the volume but not the kind: a 50-character prefix of
a search query is still the user's words.
