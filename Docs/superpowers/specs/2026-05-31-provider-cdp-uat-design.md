# Provider CDP UAT Design

Date: 2026-05-31
Status: Approved by user for spec review
Primary repo: `tldw_chatbook`
Scope: Manual CDP/Textual-web user acceptance testing for Chatbook Console API provider support

## Purpose

Verify, through the rendered Chatbook app rather than adapter-only calls, that Console can use every API provider currently supported by the codebase. A provider is accepted only when a two-turn conversation succeeds in the same Console session and the second assistant reply is visible in the app.

This pass is intentionally user-acceptance oriented. CDP/browser automation may click, type, wait, and capture evidence, but the acceptance surface is the running Textual-web app.

## Current Context

Relevant provider and Console files:

- `tldw_chatbook/Chat/Chat_Functions.py`
- `tldw_chatbook/Chat/console_provider_gateway.py`
- `tldw_chatbook/Chat/console_provider_support.py`
- `tldw_chatbook/Chat/provider_readiness.py`
- `tldw_chatbook/Chat/console_session_settings.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/config.py`

Observed state:

- `Chat_Functions.API_CALL_HANDLERS` is the current execution source for supported chat providers.
- Console now has a generic provider gateway path plus direct llama.cpp handling.
- Settings Providers and Models owns provider/model config, credential source display, model defaults, and Console-readiness agreement.
- The adjacent server `.env` at `../tldw_server2/tldw_Server_API/Config_Files/.env` contains keys for many hosted LLM providers.
- Some local and custom providers have no key or need a live localhost endpoint. Those are not accepted or failed unless a reachable endpoint exists.

## Goals

- Launch the real Chatbook app through Textual-web/CDP.
- Use an isolated temporary `HOME`/`XDG_CONFIG_HOME`/data profile so real user config and databases are not mutated.
- Load provider API keys from the adjacent server `.env` into the app process environment.
- Exercise provider setup, provider/model selection, and two-message Console send behavior through the rendered app.
- Cover every provider currently in Chatbook's execution surface when credentials or reachable endpoints make it testable.
- Fix provider support when failure evidence points to Chatbook code rather than external auth, quota, model availability, or endpoint reachability.
- Capture durable, redacted QA evidence for passes, skips, failures, and reruns.

## Non-Goals

- Do not mutate the user's normal Chatbook config, keyring, or databases.
- Do not print, screenshot, or commit raw API key values.
- Do not count adapter-only direct calls as UAT success.
- Do not mark local/custom providers failed when no live endpoint is reachable.
- Do not replace the Settings provider system or Console provider gateway architecture as part of UAT.
- Do not force custom OpenAI-compatible entries to hosted provider endpoints unless the user separately requests it.

## Provider Scope

The provider inventory starts from `Chat_Functions.API_CALL_HANDLERS` and the Console provider identity helpers. At the time of design, this includes:

- Hosted/API providers: `openai`, `anthropic`, `cohere`, `deepseek`, `google`, `groq`, `huggingface`, `mistral`, `mistralai`, `moonshot`, `openrouter`, `zai`.
- Local/custom providers: `llama_cpp`, `koboldcpp`, `oobabooga`, `tabbyapi`, `vllm`, `local-llm`, `ollama`, `aphrodite`, `custom-openai-api`, `custom-openai-api-2`, `mlx_lm`, `local_llamacpp`, `local_llamafile`, `local_ollama`, `local_vllm`, `local_mlx_lm`.

Hosted providers are in scope when the adjacent `.env` has a non-empty matching key. Provider aliases must use the same readiness and execution identity rules as Console.

Local and custom providers are in scope only when their configured endpoint is reachable from the isolated app process. Unreachable endpoints are recorded as skipped with the endpoint category, not as provider failures.

## Credential Handling

The UAT process reads the adjacent `.env` and maps non-empty values into the app environment. Examples of relevant names include `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `HUGGINGFACE_API_KEY`, `OPENROUTER_API_KEY`, `MOONSHOT_API_KEY`, and provider-specific local/custom key names.

Rules:

- Raw keys stay only in process environment or isolated config if the UI requires a save path.
- Evidence records provider name, key source label, and masked readiness state only.
- Screenshots must be inspected before being archived to ensure no raw key is visible.
- Logs or failure summaries must categorize credential issues without echoing token values.

## CDP UAT Flow

For each in-scope provider:

1. Launch or reuse the isolated Textual-web app with splash disabled and default tab set to Console/Chat.
2. Open Settings through the rendered app when provider setup or credential-source confirmation is needed.
3. Select the provider in Providers and Models, verify or save the credential env var name, select a low-cost model, and return to Console.
4. In Console settings, select the provider and model for the active session.
5. Send the first short message.
6. Wait until the first assistant response is visible and the run is no longer active.
7. Send the second short message in the same session.
8. Wait until the second assistant response is visible and the run completes.
9. Capture pass/fail/skip evidence, including screenshot paths when useful and redacted log excerpts.

The recommended prompts should be short and cheap, for example:

- First message: `Reply with one short sentence: provider UAT turn one.`
- Second message: `Reply with one short sentence: provider UAT turn two.`

The exact response text is not asserted. The visible second assistant reply is the acceptance condition.

## Success Criteria

A provider passes only when all of the following are true:

- The rendered app shows the selected provider and model in the Console/Settings flow.
- The first user message receives an assistant response from the provider.
- A second user message in the same session receives a second assistant response.
- The app remains usable after the run.
- No raw API key appears in screenshots, logs, or QA notes.

## Failure Classification

Failures are categorized before any fix:

- `missing_key`: no usable key exists in `.env` or isolated settings.
- `auth`: provider rejects the key.
- `quota_or_rate_limit`: provider reports quota, billing, or rate-limit exhaustion.
- `model_unavailable`: configured model is invalid, retired, unavailable to the key, or incompatible.
- `request_shape`: Chatbook sends invalid provider-specific payload or parameters.
- `response_shape`: Chatbook receives a valid provider response but fails to normalize/display it.
- `streaming`: streaming path fails where non-streaming should work or fallback should occur.
- `endpoint_unreachable`: local/custom endpoint is unavailable.
- `console_ui`: CDP cannot complete the selection/send workflow because of UI behavior.
- `unknown`: insufficient evidence; requires logs or a focused repro.

Only `request_shape`, `response_shape`, `streaming`, and `console_ui` are default candidates for Chatbook fixes in this UAT pass. External provider issues are documented and skipped or failed according to evidence.

## Fix And Rerun Policy

When a failure points to Chatbook support:

1. Preserve the failure evidence.
2. Add or update focused automated coverage around the failing provider path when practical.
3. Make the smallest provider-specific or gateway fix that preserves existing behavior.
4. Run focused tests for the changed area.
5. Rerun the failed provider through CDP until it passes or is reclassified.

Do not broaden the provider architecture or perform unrelated refactors during UAT fixes.

## Evidence

Create a QA evidence folder under:

`Docs/superpowers/qa/provider-cdp-uat/`

The QA report should include:

- Date, branch, app launch command shape, and Textual-web/CDP URL.
- Isolated config/data directory locations.
- Provider inventory table with status: pass, skip, fail, fixed-then-pass.
- Provider/model used for each attempted provider.
- Redacted key source for each hosted provider.
- Failure category and short diagnosis for failed providers.
- Screenshot paths or CDP notes for representative passes and every UI/config failure.
- Focused test commands and results for any code fix.
- Residual risks, including providers skipped because no key or no reachable endpoint exists.

Screenshots are evidence of the rendered app state, not a substitute for the two-turn acceptance condition.

## Risks And Mitigations

- Provider keys may be invalid or quota-limited. Mitigation: classify as external auth/quota and avoid code churn.
- Current model lists may include retired models. Mitigation: choose known low-cost default models from config first, then update model selection only when evidence shows model unavailability.
- Textual-web/CDP may be flaky. Mitigation: use screenshots, app logs, and repeatable CDP steps; retry only after identifying whether the flake is app behavior or browser control.
- Hosted providers may stream in incompatible formats. Mitigation: prefer non-streaming for UAT unless the provider path is explicitly testing streaming behavior.
- Large provider sweeps can run long. Mitigation: keep prompts short, cap per-provider wait time, and record partial progress in the QA report.

## Open Follow-Up

After this spec is reviewed and approved, write an implementation/UAT plan that includes:

- exact isolated launch commands;
- the provider/key/model inventory extraction step;
- the CDP operation sequence;
- evidence report template;
- focused test list for likely provider-gateway fixes;
- rerun and closeout steps.
