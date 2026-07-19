---
id: TASK-279
title: Wrap chatbook import in a single transaction
status: Done
assignee: []
created_date: '2026-07-16 14:30'
updated_date: '2026-07-17 00:20'
labels: [performance, chatbooks]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
chatbook_importer._import_conversations commits per add_conversation/add_message/set_message_attachments — ~1,500 commits for a 50-conversation × 30-message import. TransactionContextManager is reentrant (ChaChaNotes_DB.py:9703-9722), so one outer transaction per chatbook (or per conversation) is a pure win. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P0 A5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import runs under an outer transaction; commit count measured before/after on a synthetic chatbook
- [x] #2 Round-trip + conflict-resolution tests green; partial-failure semantics documented
<!-- AC:END -->

## Implementation Notes

Wrapped each conversation's writes (`add_conversation` + its whole
`add_message` loop, including `set_message_attachments` and the RAG
citation-context persist call) in `Chatbooks/chatbook_importer.py`
`_import_conversations` in a single `with db.transaction():`, scoped per
conversation (not per chatbook) to preserve the existing error-isolation
semantics: the per-conversation `try/except` in the caller loop is
unaffected, so one bad conversation still fails alone while the rest of
the chatbook imports. `TransactionContextManager` is depth-tracked/
reentrant, so the nested `with self.transaction():` calls already inside
`add_conversation`/`add_message`/`set_message_attachments` no longer each
open+commit their own top-level transaction -- only the outer block
commits, once, per conversation.

Partial-failure semantics (documented in a code comment at the wrap site):
before this change, a failure partway through a conversation's message
loop left a *partially* imported conversation (the conversation row plus
whatever messages had already individually committed) marked as a failed
item. After this change, that same failure rolls back the *entire*
conversation's writes -- an isolation improvement (no partial state is
ever left behind for a conversation counted as failed), not a behavior
regression, since the caller already treated that case as
`failed_items += 1` either way.

New test `Tests/Chatbooks/test_import_transactions.py`: imports a
synthetic 3-conversation x 5-message chatbook (no attachments) through the
real `ChatbookImporter` + a real tmp_path-backed `CharactersRAGDB`, and
counts actual top-level commits by wrapping
`TransactionContextManager.__exit__` (counts only when
`is_outermost_transaction` and no exception -- distinguishes "committed" from
the many nested `.transaction()` calls that don't). RED before the fix:
measured 19 top-level commits (3 `add_conversation` + 15 `add_message` +
1 from `CharactersRAGDB`'s own schema-init transaction) against an
asserted ceiling of `NUM_CONVERSATIONS + 3 = 6`. GREEN after: 2 passed
(commit count now well under the ceiling; a second test independently
confirms every conversation and all 5 messages each still round-trip
correctly). Full regression: `Tests/Chatbooks/` -- 137 passed, 1 skipped
(pre-existing, `--run-slow`-gated, unrelated).
