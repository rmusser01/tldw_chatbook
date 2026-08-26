# Task 2 implementation report

## Summary

Implemented admission-time capture-policy resolution and immutable provider
threading. Session next-send/conversation policy is revision fenced, persisted
overrides hydrate and flush on promotion, accepted ordinary and durable turns
carry one frozen signal object, and generic/llama.cpp capture uses Task 1's
sanitizer and one shared budget. Global Safe/Off and Full/On mutations follow
the required privacy-preserving publication order.

## RED evidence

- Initial focused behavior run:
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_chat_store_exchanges.py Tests/Chat/test_console_chat_controller_exchanges.py Tests/Chat/test_console_provider_gateway.py Tests/test_config_save_settings_semantics.py -q`
  -> **9 failed, 299 passed, 2 skipped**. Failures were the intentionally
  absent store policy APIs (3), controller policy/admission APIs (2), signal
  detail/shared budget behavior (2), and config mutation APIs (2).
- Persistence lifecycle tests before implementation:
  `...pytest Tests/Chat/test_console_chat_store_exchanges.py -q -k 'capture_policy_hydrates or failed_staged_safe'`
  -> **2 failed, 21 deselected** (missing hydration API and pending-save state).
- Semantic provider-boundary regression before its fix:
  `...pytest Tests/Chat/test_console_provider_gateway.py -q -k 'generic_capture_threads_detail'`
  -> **1 failed, 1 passed, 271 deselected**. Safe capture exposed a tagged
  project-instruction body after provider preparation split it into
  `system_message`.

## GREEN / verification evidence

- Exact Gate 2 matrix from the brief:
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_exchange_capture.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_chat_controller_exchanges.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store_exchanges.py Tests/test_config_save_settings_semantics.py -q --tb=short`
  -> **541 passed, 2 skipped, 0 failed** in 24.98s. Both skips are existing
  loopback-listener tests unavailable under sandbox permissions.
- Changed-file Ruff, including every owned Python/test path in the brief:
  -> **All checks passed**.
- `python -m py_compile` for the four production modules -> exit 0.
- `git diff --check` -> exit 0.

## Files changed

- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_provider_gateway.py`
- `tldw_chatbook/config.py`
- `Tests/Chat/test_console_chat_controller_exchanges.py`
- `Tests/Chat/test_console_chat_store_exchanges.py`
- `Tests/Chat/test_console_provider_gateway.py`
- `Tests/test_config_save_settings_semantics.py`
- `backlog/tasks/task-22507.2 - Freeze-scoped-capture-policy-across-Console-provider-runs.md`

## Commits

- `cf586ca823fdf231ef17e2ce8401124ed70f7cd1` —
  `feat(console): freeze capture policy at admission`

## Concerns / deviations

- No scope deviation and no new dependency or trace pipeline.
- The repository-wide suite was intentionally not run; the brief and
  repository instructions require the focused Gate 2 matrix only.
- Test output includes a pre-existing `requests` dependency warning and pytest
  temporary-directory cleanup warnings caused by sandbox filesystem
  permissions; neither affected the test result.
- This report was written after the implementation commit so it can record the
  exact commit hash; it is intentionally the only remaining uncommitted file.

## Status

Task `TASK-22507.2` remains **In Progress** for independent controller review
and closure. It was not marked Done and its acceptance checkboxes remain open.

## Fix round 1

### Summary

- Added one public arbitrary-value sanitizer that recursively removes structured
  credentials and stubs explicit base64/data URIs at every size; llama.cpp uses
  it for both initial and retry wire captures.
- Reserved the shared policy revision before conversation/config persistence,
  blocked sibling mutations while durable work is in flight, and shielded then
  reconciled conversation persistence before cancellation propagates.
- Bounded response accumulation when chunks arrive and made in-flight capture
  projection pure/idempotent. Full generic capture now preserves the exact
  adapter message/system kwargs; only Safe substitutes tagged semantic rows.
- Allowed confirmed ephemeral Full overrides to stage for promotion and made
  exchange-attach failure logging content-free.

### RED evidence

- Public sanitizer canaries initially failed during collection with
  `ImportError: cannot import name 'sanitize_capture_value'` (exit 2).
- Focused provider regressions then produced **4 failed, 1 passed**: retained
  content was **800 bytes > 180-byte budget**, Full messages had the substituted
  tagged semantic shape, and nested credential canaries remained in both Safe
  and Full llama.cpp wire payloads.
- The retry canary additionally failed before its fixture carried structured
  input, demonstrating the retry request needed the same sanitizer boundary.

### GREEN / verification evidence

- Shared sanitizer focus: **3 passed, 30 deselected**.
- Conversation reservation/stale/cancellation/ephemeral and content-free log
  focus: **8 passed, 284 deselected**, followed by **4 passed, 11 deselected**
  after the explicit stale-owner regression was added.
- llama.cpp initial and real stream-to-complete retry focus: **5 passed, 273
  deselected**.
- Exact Gate 2 command from the brief: **549 passed, 2 skipped, 0 failed** in
  23.48s. The two skips are the existing sandbox-denied loopback listeners.
- Changed-file Ruff: **All checks passed**. Production `py_compile`: exit 0.
  `git diff --check`: exit 0.

### Files changed in fix round 1

- `tldw_chatbook/Chat/console_exchange_capture.py`
- `tldw_chatbook/Chat/console_provider_gateway.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Chat/console_chat_controller.py`
- `Tests/Chat/test_console_exchange_capture.py`
- `Tests/Chat/test_console_provider_gateway.py`
- `Tests/Chat/test_console_chat_controller_exchanges.py`
- `backlog/tasks/task-22507.2 - Freeze-scoped-capture-policy-across-Console-provider-runs.md`

### Commit / concerns

- Fix-round commit: `aa285db7d7` —
  `fix(console): harden semantic capture policy`.
- All seven independent findings are addressed. No dependency, schema, or
  scope deviation was introduced. Task remains **In Progress**; acceptance
  checkboxes remain open for independent review and closure.
