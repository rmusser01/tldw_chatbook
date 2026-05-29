# Console Provider Parity Screenshot QA

Date: 2026-05-28
Branch: `codex/settings-config-next-slice-4`
Screen: Console
Scope: Console provider execution parity
Capture method: Actual textual-web rendering captured through headless Chromium / Playwright.
User approval: approved in Codex thread after reviewing the actual rendered screenshots.

## Approved Captures

| Scenario | Screenshot | Result |
| --- | --- | --- |
| Generic provider success | `Docs/superpowers/qa/product-maturity/screen-qa/console/provider-parity/generic-success-2026-05-28.png` | Console sends through a generic provider path and renders a completed assistant message. |
| Missing key blocked | `Docs/superpowers/qa/product-maturity/screen-qa/console/provider-parity/missing-key-blocked-2026-05-28.png` | Console blocks send, shows OpenAI missing-key recovery, and preserves the composer draft. |
| Unsaved base URL override blocked | `Docs/superpowers/qa/product-maturity/screen-qa/console/provider-parity/base-url-override-blocked-2026-05-28.png` | Console blocks the send, tells the user to save the endpoint in Settings, and preserves the composer draft. |

## QA Notes

- A temporary textual-web harness initially reused the already-active Console session through `ensure_session(...)`, so the base URL override scenario did not apply the unsaved endpoint. The harness was corrected to create a fresh session for that scenario before recapturing the approved screenshot.
- Production gateway behavior already blocked differing generic base URL overrides; no production code change was needed for the screenshot correction.
- The stale early generic screenshot in this folder is not approval evidence and should not be staged for the provider-parity closeout.
- Follow-up provider-sweep regressions now assert every `chat_api_call()` handler key resolves through Console provider support, Console settings' static execution-key list stays aligned with `API_CALL_HANDLERS`, and the send gateway does not route any supported handler key to the old WIP/unsupported-provider recovery copy.

## Verification

- `python -m pytest -q Tests/Chat/test_console_provider_gateway.py::test_resolve_for_send_blocks_generic_base_url_override_that_differs_from_config --tb=short`
- Result: `1 passed, 1 warning`
- `python -m pytest -q Tests/Chat/test_console_provider_support.py Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_provider_gateway.py Tests/UI/test_console_native_chat_flow.py --tb=short`
- Result: `121 passed, 8 warnings`
- `python -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_session_settings.py --tb=short`
- Result: `132 passed, 1 warning`
- `python -m pytest -q Tests/Chat/test_console_provider_support.py::test_all_chat_api_call_handlers_resolve_to_supported_console_identity Tests/Chat/test_console_session_settings.py::test_settings_execution_provider_keys_match_chat_api_handlers Tests/Chat/test_console_provider_gateway.py::test_resolve_for_send_all_chat_api_handlers_are_console_supported --tb=short`
- Result: `3 passed, 1 warning`
- `git diff --check`
- Result: clean

## Residual Risks

- These captures verify deterministic fake/local provider paths and blocked recovery states. Live third-party provider smoke tests still require real credentials or local provider runtimes.
