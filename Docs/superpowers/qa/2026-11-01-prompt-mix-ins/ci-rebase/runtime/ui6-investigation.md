# PR3346 UI6 causal investigation

Reviewed product HEAD `79e659dcde6f5360edb0070f9d184713bd8584d1`, rebased on dev `62a169f906508ed02bd7546ab278d916638d1607`. Read-only repository investigation; root-owned working changes were preserved.

## Finding and recommended minimal fix

`tldw_chatbook/Chat/console_prompt_mixin_service.py:592` unconditionally schedules `run_owned_db_call(persistence.db, callback)` for the operational receipt save. That helper always invokes `asyncio.to_thread`; its memory branch changes connection cleanup, not scheduling. The real `CharactersRAGDB(":memory:")` used by `Tests/UI/app_factory.py::attach_chachanotes_db` has thread-local connections. The callback therefore gets a different empty database on the executor thread.

The actual exception, observed without changing callback behavior, was `OperationalError: no such table: messages` from `ChatPersistenceService.save_conversation_mixin_attempt`. The service converts that failure to `MixinSnapshotUnavailableError`; the pre-provider context block then encounters an already accepted durable checkpoint and raises `ConsoleDispatchSettlementError("Dispatch terminal settlement previously failed.")`. CI diagnostics show durable commit success followed by dispatch-settlement refusal at approximately307ms; local reproduction showed335ms. The10-second timeout is a symptom, not slow preparation.

Use the existing memory-owner scheduling pattern in `console_native_commit.py:82`: execute the real callback on the caller loop when its exact persistence DB is memory-backed; retain the existing worker/shield/drain path for file-backed DBs. Preserve error conversion, exact anchors, post-write session/store/persistence ownership checks, accepted publication and cancellation behavior. Do not replace this harness with a file-backed DB: that changes its deliberate direct-provider versus agent-bridge behavior. Do not skip empty receipts, relax the timeout, or weaken persistence authority.

## Exact-node evidence

Selected original node each time, preserving repository conftest/bootstrap_profile admission and real-profile guard:

`Tests/UI/test_console_composer_token_enter.py::test_one_enter_sends_a_token_and_typed_question_with_the_full_text[paste]`

Interpreter `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`; `PYTHONDONTWRITEBYTECODE=1`; pytest options `-q --tb=short -p no:cacheprovider`, separate private basetemp per run.

- Original unchanged test:1 failed in16.58s (`mixins-ui6-original.log`).
- Runtime-only exception-constructor observer:1 failed in16.68s, confirms dispatch-settlement exception (`mixins-ui6-observe.log`).
- Runtime-only real persistence callback observer:1 failed in15.57s, confirms `no such table: messages`, executor thread, memory DB (`mixins-ui6-db-observe.log`).
- Single-variable control:1 passed in6.79s, call4.51s (`mixins-ui6-memory-control.log`). A pytest collection plugin wrapped `DB.base_db.run_owned_db_call` only for a partial whose bound function is exactly `ChatPersistenceService.save_conversation_mixin_attempt` and whose DB is memory-backed, returning its real callback directly. Every other operation used the original helper. No test assertions, timeout, provider, receipt write or authority checks changed.

No product fix was applied. No full suite, other failing-node reruns, or dev comparison was needed to isolate this newly introduced callback scheduling regression. Related UI9/UI14 failures may share it but were not independently proven here.

## Owned artifacts

Report: this file.

Logs (all `/private/tmp/`):
- `mixins-ui6-original.log`
- `mixins-ui6-observe.log`
- `mixins-ui6-db-observe.log`
- `mixins-ui6-memory-control.log`

Corresponding pytest basetemp directories:
- `/private/tmp/mixins-ui6-original`
- `/private/tmp/mixins-ui6-observe`
- `/private/tmp/mixins-ui6-db-observe`
- `/private/tmp/mixins-ui6-memory-control`

No repository file/index/HEAD, manifest, commit, push or PR comment was created or modified by this investigation. The input CI log is parent-owned. Fixture-created bootstrap roots were managed by existing teardown.
