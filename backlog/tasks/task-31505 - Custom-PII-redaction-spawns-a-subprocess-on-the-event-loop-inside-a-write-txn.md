---
id: TASK-31505
title: Custom-PII redaction spawns a subprocess on the event loop inside a write txn
status: To Do
assignee: []
created_date: '2026-09-04 19:30'
labels:
  - performance
  - console
  - chat
dependencies: []
priority: medium
---

## Description (the why)

With `pii_redaction_enabled` and a custom ruleset (opt-in),
`run_custom_pii_batch` (`Chat/console_trace_regex_worker.py:126-225`) spawns
`Popen([sys.executable, "-I", ...])` and blocks on `communicate()` (up to
500 ms) -- and it is called (a) from the streaming `finally:` via synchronous
`_complete_scoped_exchange` (`console_provider_gateway.py:1133`) ON THE EVENT
LOOP once per completed exchange, and (b) from
`_build_durable_trace_request`'s per-saved-row loop
(`console_chat_controller.py:8587`) while `transaction(immediate=True)` is
held (`:8559`) -- a fresh interpreter spawn per message row with the write
lock held, on the event loop. When the feature is on, every send freezes the
UI. The `-I` subprocess isolation is presumably deliberate (regex sandboxing)
and should be kept; the wait belongs off-loop and outside the transaction.
Evidence: `Docs/Design/2026-09-04-holistic-perf-review.md` section 6.

## Acceptance Criteria (the what)

- [ ] No subprocess spawn/wait occurs on the Textual event loop on the send or exchange-completion paths
- [ ] No subprocess spawn/wait occurs while a database write transaction is held (redactions computed before the transaction opens)
- [ ] Redaction semantics unchanged: redact-before-persist ordering, deadline handling, and continuation masks behave as before (existing custom-PII tests stay green)
