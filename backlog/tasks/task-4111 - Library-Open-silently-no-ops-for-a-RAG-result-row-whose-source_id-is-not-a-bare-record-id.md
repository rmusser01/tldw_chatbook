---
id: TASK-4111
title: >-
  Library Open silently no-ops for a RAG result row whose source_id is not a
  bare record id
status: To Do
assignee: []
created_date: '2026-08-09 20:23'
labels:
  - library
  - rag
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found reviewing plan Tasks 4 and 5 of the hybrid-fusion cluster; PRE-EXISTING, not introduced by that work. The Library RAG panel's Open action routes through _open_library_item_by_id (UI/Screens/library_screen.py), which hands the row's source_id straight to the per-type detail route. Every one of those routes assumes a bare record id: the prompt branch does int(record_id) inside a try/except (TypeError, ValueError) that simply returns, and the media branch passes the value to the media detail fetch whose own broad except Exception logs and sets detail to None. A row whose source_id carries the retrieval layer's prefixed document id (media_15 rather than 15) therefore produces no navigation, no error, and no message - the user presses Open and the screen does not change. The engine's keyword leg used to emit exactly that shape, because SearchResult.id is a document id and the source_id key the row mappers read was never populated; TASK-3996 fixed that one instance by stamping a bare source_id on keyword rows, and the eval harness's canonicalizer compensates for the same mismatch on its own side. The general fragility is untouched: any future row builder, any fallback path that stamps a composite id (_fusion_doc_key still falls back to the row id when metadata is missing), and any id space that is not an integer reaches the same silent dead end. The remedy is for the open route to resolve or reject explicitly rather than swallow - normalise a prefixed id, or tell the user the row cannot be opened - so the failure is never invisible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening a RAG result row whose source_id carries a source-type prefix lands on the item's detail surface
- [ ] #2 A row that genuinely cannot be resolved reports why instead of doing nothing
- [ ] #3 No open path swallows a parse failure into a silent return or a None detail
- [ ] #4 A regression test covers a prefixed source_id and an unresolvable id, for media and for prompts
<!-- AC:END -->
