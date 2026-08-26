# Task 3 implementation report

## Summary

Task 3 now purges only the target session's Full capture owners, commits the durable delete and authoritative live swaps on the owner loop under one quiescence lease, and fails closed for missing targets, active leases, stale revisions, and Inspector remount races. Staging is recursively immutable; removed counts use the durable/live identity union; Safe captures and unrelated sessions remain unchanged. Store and persistence-wrapper exchange-flush failure logs are content-free.

## Review finding mapping

- Global snapshot corruption: two-session preservation and overlapping-stage regressions prove target-only cache/tag/revision publication, exact surviving cache keys, monotonic per-session revisions, and no stale Full access.
- Worker-thread live swaps: a thread-identity/delete-hook regression proves the delete and live publication execute on the caller owner loop and the lease prevents capture attachment during the indivisible section.
- Async Inspector remount: a two-call race advances the revision during the first awaited mount and proves all decoded maps and mounted call nodes are removed before a second mount.
- Missing immutable target: direct missing and closed-session cases prove availability and purge return content-free `target_missing`, zero removed, and revision sentinel `-1`.
- Removed-count under-reporting: a durable-only plus live-only regression proves the authoritative identity union count while Safe remains; the DB mismatch test proves staged-count validation rolls the delete back before commit.
- Stage hardening: mutation-resistance assertions cover the tuple assignments, mapping proxies, and frozensets.
- Log gaps: semantic canaries prove `exchange_flush_failed` and `exchange_append_failed` record only stable categories and exception types, never exception text.

## TDD and verification evidence

- Review RED: the focused 31-item reproduction matrix reported **10 failed / 21 passed**. Failures covered each review finding above; existing controls stayed green.
- Review GREEN: the same focused matrix reported **31 passed**, 1 dependency warning.
- Exact Gate 3: `python -m pytest Tests/DB/test_chachanotes_message_exchanges.py Tests/Chat/test_console_chat_store_exchanges.py Tests/Chat/test_console_capture_purge.py Tests/Chat/test_console_chat_controller_exchanges.py Tests/UI/test_console_conversation_inspector.py Tests/UI/test_chat_screen_console_inspector_loader.py -q` reported **125 passed**, 1 dependency warning in 20.16s.
- Changed-file Ruff: passed.
- Production `py_compile`: passed.
- `git diff --check`: passed.

### Fix round 2/5 — persistence-wrapper log redaction

- RED: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_chat_store_exchanges.py::test_append_message_exchanges_service_wrapper_logs_and_returns_false -q` reported **1 failed**, 1 dependency warning in 0.33s. The captured `exchange_append_failed` event contained the unique semantic request/response canary inside `error=repr(exc)` and had no `error_type`.
- GREEN: the same focused command reported **1 passed**, 1 dependency warning in 0.26s. The wrapper still returned `False`; the captured event retained stable category `exchange_append_failed`, `message_id`, and `error_type=RuntimeError`, while the semantic canary, complete exception repr, and capture bytes were absent.
- Exact Gate 3: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/DB/test_chachanotes_message_exchanges.py Tests/Chat/test_console_chat_store_exchanges.py Tests/Chat/test_console_capture_purge.py Tests/Chat/test_console_chat_controller_exchanges.py Tests/UI/test_console_conversation_inspector.py Tests/UI/test_chat_screen_console_inspector_loader.py -q` reported **125 passed**, 1 dependency warning in 22.61s.
- Changed-file Ruff: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Chat/chat_persistence_service.py Tests/Chat/test_console_chat_store_exchanges.py` reported `All checks passed!`.
- Production compile: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile tldw_chatbook/Chat/chat_persistence_service.py` exited 0 with no output.
- `git diff --check` exited 0 with no output.

The original Task 3 RED/GREEN evidence remains in commit `16cfc0991a`. This review correction is committed as `fix(console): harden full capture purge isolation` and includes the production, test, backlog-note, and report changes listed by `git show --stat`.

## Concerns and deviations

- The repository environment emitted its existing `requests` dependency warning and pytest temporary-directory cleanup warnings; neither affected outcomes.
- The review-required owner-loop correction intentionally replaces the original worker-offloaded DB commit. It adds no callback, await, dependency, second pipeline, or speculative abstraction after the durable commit.
- ADR-089 remains governing; no new ADR or lesson entry was needed.

Task `TASK-22507.3` remains **In Progress** for independent review. Acceptance criteria remain intentionally unchecked.

## Controller closure

- Final scoped re-review: all findings addressed, with no new Critical or
  Important breakage; the semantic-canary wrapper regression independently
  passed.
- The Backlog acceptance criteria are checked and `TASK-22507.3` is **Done**.
- Final implementation range: `16cfc0991a..2f2be5d7ad` plus the
  Backlog/report closeout commit.
