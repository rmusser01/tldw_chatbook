# Phase 2.1 Grounded Console Response Contract

Date: 2026-05-05
Task: TASK-9.1
Branch: codex/product-maturity-phase2-1-grounded-console
Workflow: Search/RAG staged context -> Console send -> model-bound request

## What Was Verified

Phase 2.1 verifies the first Phase 2 gate: staged Search/RAG context is not only visible in Console, but also reaches the provider-bound request when the user sends a prompt.

Verified contract:

- Staged Search/RAG context is applied to the `chat_wrapper(message=...)` payload.
- Source authority survives into the model-bound prompt through source, source ID, content ref, body, and metadata lines.
- The user prompt is preserved under the staged-context block.
- If the send handler exits before generation work starts, the handoff remains `staged` so recovery does not silently drop source context.

## Automated Evidence

Initial red run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py -q
```

Result:

- Failed as expected in `test_tab_send_preserves_handoff_payload_when_original_handler_does_not_dispatch`.
- Failure: payload status became `sent` after a blocked handler path instead of remaining `staged`.

Focused verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py Tests/Event_Handlers/Chat_Events/test_chat_events.py -q
```

Result:

- `26 passed, 16 skipped, 1 warning in 4.66s`.

Adjacent handoff and product-maturity verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_chat_first_handoffs.py Tests/UI/test_product_maturity_phase1_core_loop.py -q
```

Result:

- `26 passed, 3 warnings in 9.81s`.

Full Phase 1 product-maturity regression sweep:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_harness.py Tests/UI/test_product_maturity_phase1_first_run.py Tests/UI/test_product_maturity_phase1_navigation_smoke.py Tests/UI/test_product_maturity_phase1_keyboard_focus.py Tests/UI/test_product_maturity_phase1_visual_audit.py Tests/UI/test_product_maturity_phase1_empty_setup_states.py Tests/UI/test_product_maturity_phase1_core_loop.py -q
```

Result:

- `42 passed, 1 warning in 24.05s`.

Collection note:

- The same adjacent UI command run through the `pytest` console script failed before collection with `ModuleNotFoundError: No module named 'Tests'`.
- The module form is the verified command because it preserves the worktree import root for `Tests` package imports.

## Defects Fixed

- `workflow-degradation`: tab-aware send consumed staged context even when validation or runtime setup stopped the send before generation work started. Fixed by marking handoff context `sent` only after generation work is dispatched.
- `recoverability`: legacy selector mapping accepted a mock-like `_current_chat_tab_id`, producing invalid tab-specific selectors in event-handler tests. Fixed by using tab-specific selectors only when the tab ID is a non-empty string.

## UX Notes

- This slice preserves user control: staged source context is consumed only when a request is actually dispatched.
- The blocked path keeps the source context available for retry after provider, model, or runtime recovery.
- This is a contract-level proof, not a full visual walkthrough of generated answers.

## Residual Risk

- Full grounded generation against a live provider or local model remains unverified in this gate.
- Artifact or Chatbook save, reopen, and Home resume controls are intentionally deferred to later Phase 2 gates.
- Missing provider/model/runtime copy is preserved by not dropping context, but this gate does not redesign those recovery messages.

## Exit Decision

Pass for Phase 2.1. The model-bound Console request contract is covered, and blocked send paths preserve staged context for recovery. Phase 2 should continue with Artifact/Chatbook creation and reopen/resume gates.
