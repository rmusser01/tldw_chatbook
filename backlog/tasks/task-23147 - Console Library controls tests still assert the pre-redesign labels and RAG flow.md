---
id: TASK-23147
title: >-
  Console Library controls tests still assert the pre-redesign labels and RAG
  flow
status: To Do
assignee: []
created_date: '2026-08-28'
labels:
  - tests
  - console
priority: medium
dependencies: []
---

## Description

13 tests assert the Console Library controls as they behaved before the approved redesign — 11 in
`Tests/UI/test_console_internals_decomposition.py` and both snapshots in
`Tests/UI/test_workbench_visual_snapshots.py`. Every symptom is specified behaviour in
`Docs/superpowers/specs/2026-08-22-console-library-controls-design.md`; these are stale tests, and
the redesign PR updated ~18 other test files but not these two.

Three sub-symptoms:
1. **Chip label** — tests expect `Library search: off`; the spec's new strings (lines 1038-1041)
   are `Library · Auto off · Agent blocked` and siblings.
2. **RAG staging** — both Run controls now open a modal instead of running retrieval, so nothing is
   ever staged and 6 tests time out waiting for staged evidence (spec §5, line 656ff).
3. **Path-shaped-draft prefill** — the query is now always prefilled with the exact composer draft,
   and `_console_draft_looks_like_rag_query` was **deliberately deleted** by the spec (lines
   669-670) and the plan's Step 4. Its guard test is now testing a removed behaviour and should be
   deleted, citing those spec lines. This was checked precisely because it looks like a regression
   and is not one.

## Acceptance Criteria

- [ ] Label assertions match the four chip strings the spec specifies
- [ ] The 6 RAG tests drive the modal (accepting a search result) rather than expecting the button
  to retrieve directly
- [ ] The path-shaped-draft guard test is deleted, with the spec lines cited in the deletion
- [ ] Both workbench snapshots regenerate against the redesigned controls

## Evidence

`tldw_chatbook/Chat/console_display_state.py:752` `rag_label=library_display.chip_label`, built at
`:624-627`. Modal routing: `chat_screen.py:2771` and `:9999-10003`. Unconditional prefill:
`tldw_chatbook/UI/Console_Modules/retrieval.py:407-411`.

Introduced by `d7bb844d9b` (2026-08-27) "fix(console): restore explicit Library policy and search
controls (#2143)".
