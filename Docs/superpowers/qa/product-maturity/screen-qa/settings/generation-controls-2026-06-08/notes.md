# Settings Generation Controls QA

Date: 2026-06-08
Branch: `codex/settings-generation-controls`

## Scope

- Providers & Models per-model sampler defaults.
- Console Behavior global generation fallback defaults.
- Provider-gated OpenAI reasoning and Anthropic thinking controls.

## Screenshot Evidence

- `01-settings-overview-generation-controls.png`: Settings overview at the start of the slice.
- `02-providers-models-generation-controls.png`: Providers & Models expanded generation controls before final provider-gating polish.
- `03-console-behavior-generation-controls.png`: Console Behavior global fallback controls.
- `04-providers-models-provider-gated-controls.png`: Provider-specific unavailable controls; approved by the user.

## Verification

- `python -m pytest -q Tests/UI/test_console_session_settings.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_chat_functions.py::TestChatApiCall Tests/Chat/test_chat_functions.py::TestProviderRequestPayloads Tests/UI/test_settings_configuration_hub.py --tb=short`
- Result: `317 passed, 8 warnings`.
- Added provider API contract regressions for OpenAI `none` reasoning and Anthropic thinking payload compatibility.
- Added opt-in live provider validation: `python -m pytest -q Tests/Chat/test_live_thinking_provider_apis.py --tb=short`
- Local result: `2 skipped` because `OPENAI_API_KEY`, `TLDW_LIVE_OPENAI_REASONING_MODEL`, `ANTHROPIC_API_KEY`, and `TLDW_LIVE_ANTHROPIC_THINKING_MODEL` are not exposed in this shell.
- To run against live APIs, set the required provider API key and model env vars above. Optional overrides: `TLDW_LIVE_OPENAI_API_BASE_URL`, `TLDW_LIVE_OPENAI_REASONING_EFFORT`, `TLDW_LIVE_OPENAI_REASONING_SUMMARY`, `TLDW_LIVE_OPENAI_VERBOSITY`, `TLDW_LIVE_ANTHROPIC_API_BASE_URL`, `TLDW_LIVE_ANTHROPIC_THINKING_EFFORT`, and `TLDW_LIVE_ANTHROPIC_THINKING_BUDGET_TOKENS`.
- `python -m py_compile tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Chat/console_session_settings.py`
- `git diff --check`
