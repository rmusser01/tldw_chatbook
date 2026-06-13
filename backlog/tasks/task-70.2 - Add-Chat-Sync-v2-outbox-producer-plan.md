---
id: TASK-70.2
title: Add Chat Sync v2 outbox producer plan
status: Done
labels:
- sync
- sync-v2
- chat
- local-first
priority: high
parent_task_id: TASK-70
documentation:
- Docs/superpowers/specs/2026-05-26-chatbook-sync-v2-completion-roadmap-design.md
modified_files:
- tldw_chatbook/Sync_Interop/chat_outbox_producer.py
- tldw_chatbook/Sync_Interop/envelope_builder.py
- tldw_chatbook/Sync_Interop/domain_adapters/chat.py
- tldw_chatbook/Sync_Interop/__init__.py
- tldw_chatbook/Chat/console_chat_store.py
- Tests/Sync_Interop/test_chat_outbox_producer.py
- Tests/Sync_Interop/test_envelope_builder.py
- Tests/Sync_Interop/test_envelope_applier.py
- Tests/Chat/test_console_chat_store.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Chat Sync v2 content-producing path so conversation and message changes can be represented as ordered encrypted outbox envelopes that preserve resumed-chat continuity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User and assistant Chat messages enqueue ordered Sync v2 envelopes only after they represent durable local message state.
- [x] #2 Conversation identity, message roles, parentage, ordering, and message variants have restore-compatible metadata.
- [x] #3 Streaming, failed-send, and regenerated-message cases do not produce misleading final-message envelopes.
- [x] #4 Tests cover local-first success, envelope shape, ordering, variants, and pending profile summary counts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for a pure Chat Sync v2 outbox producer and ConsoleChatStore post-durable-message enqueue hooks.
2. Extend the Chat envelope builder with restore metadata for parentage, transcript ordering, and selected regenerated variants while keeping message content encrypted.
3. Implement `ChatSyncV2OutboxProducer` using the same local-first profile and dataset-key readiness contract as the Notes producer.
4. Wire `ConsoleChatStore` to an injected producer only after messages have durable persisted IDs and complete status, leaving streaming, stopped, and failed assistant content unsynced.
5. Run focused Chat/Sync verification and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added `ChatSyncV2OutboxProducer` as the Chat content-producing Sync v2 boundary. The producer reads the active local-first profile from `SyncStateRepository`, requires device/dataset identity plus an in-memory dataset key, encrypts chat message role/content, and persists deterministic pending outbox envelopes through the existing durable outbox API. Non-local-first, unconfigured, missing-identity, or missing-key profiles return explicit skipped states and do not dispatch remote sync.

Extended `SyncEnvelopeBuilder.build_chat_message()` with restore-compatible clear routing metadata for conversation identity, parent message, transcript sequence, selected variant ID, variant turn ID, selected variant index, and variant count. Message content remains encrypted and does not appear in serialized envelope JSON.

Wired `ConsoleChatStore` to an optional injected Chat producer after durable local persistence succeeds. User messages enqueue immediately after persistence; assistant messages enqueue only when complete. Streaming chunks, failed assistant sends, stopped assistant sends, empty/pending messages, and producer exceptions do not block local chat persistence or create misleading final-message envelopes. Regenerated variants enqueue the selected variant content with variant metadata.

PR review follow-up added Google-style public API docstrings, preserved Loguru structured exception context with `logger.bind()`, made sequence metadata count only sync-eligible persisted complete messages, and added versioned Chat message update semantics. The store now tracks the last enqueued payload hash per stable message key and passes it as `base_version` for changed content, while `ChatSyncAdapter` applies updates when `base_version` matches the current local hash and still conflicts on divergent changes.

Verification:
- Red tests failed on the missing Chat producer module and missing ConsoleChatStore Sync v2 enqueue contract.
- `../../.venv/bin/python -m pytest -q Tests/Sync_Interop/test_chat_outbox_producer.py Tests/Sync_Interop/test_envelope_builder.py Tests/Chat/test_console_chat_store.py --tb=short` passed with 27 tests.
- `../../.venv/bin/python -m pytest -q Tests/Sync_Interop/test_local_first_sync_service.py Tests/Sync_Interop/test_restore_service.py Tests/Sync_Interop/test_sync_state_repository.py --tb=short` passed with 61 tests.
- `../../.venv/bin/python -m pytest -q Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py --tb=short` passed with 43 tests.
- `../../.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase1_harness.py::test_backlog_task_frontmatter_ids_are_unique --tb=short` passed with 1 test.
- `../../.venv/bin/python -m compileall tldw_chatbook/Sync_Interop/chat_outbox_producer.py tldw_chatbook/Sync_Interop/envelope_builder.py tldw_chatbook/Chat/console_chat_store.py` passed.
- `git diff --check` passed.
- `git rebase origin/dev` reported the branch was up to date because `origin/dev` is already an ancestor of the stacked roadmap branch.
- `../../.venv/bin/python -m pytest -q Tests/Sync_Interop/test_chat_outbox_producer.py Tests/Sync_Interop/test_envelope_builder.py Tests/Sync_Interop/test_envelope_applier.py Tests/Chat/test_console_chat_store.py --tb=short` passed with 33 tests after PR review fixes.
- `../../.venv/bin/python -m pytest -q Tests/Sync_Interop/test_local_first_sync_service.py Tests/Sync_Interop/test_restore_service.py Tests/Sync_Interop/test_sync_state_repository.py --tb=short` passed with 61 tests after PR review fixes.
- `../../.venv/bin/python -m pytest -q Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py --tb=short` passed with 44 tests after PR review fixes.
- `../../.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase1_harness.py::test_backlog_task_frontmatter_ids_are_unique --tb=short` passed with 1 test after PR review fixes.
- `../../.venv/bin/python -m compileall tldw_chatbook/Sync_Interop/chat_outbox_producer.py tldw_chatbook/Sync_Interop/envelope_builder.py tldw_chatbook/Sync_Interop/domain_adapters/chat.py tldw_chatbook/Chat/console_chat_store.py` passed after PR review fixes.
- `git diff --check` passed after PR review fixes.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
`TASK-70.2` adds the Chat Sync v2 producer and durable Console store hook needed for ordered, encrypted local-first Chat message outbox entries. The slice remains opt-in through injected producer/profile context and does not add background sync, automatic dispatch, or UI/config wiring.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests and verification recorded
- [x] #3 Documentation updated in task record
- [x] #4 No background sync or automatic dispatch added
<!-- DOD:END -->
