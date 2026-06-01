# Provider CDP UAT

Date: 2026-06-01
Branch: codex/provider-cdp-uat-execution
Spec: Docs/superpowers/specs/2026-05-31-provider-cdp-uat-design.md
Backlog task: TASK-75 - Provider CDP UAT sweep
Textual-web URL: http://127.0.0.1:8897
Isolated QA root: `${TMPDIR:-system temp}/tldw-chatbook-provider-cdp-uat` via `run_textual_web_with_env.py`
Isolated HOME: `<qa-root>/home`
Isolated XDG config: `<qa-root>/config`
Isolated data: `<qa-root>/data`
App log: `<qa-root>/home/.local/share/tldw_cli/default_user/tldw_cli_app.log`

## Provider Inventory

| Display | Readiness key | Execution key | Model | Model source | Key source | Endpoint source/status | Status | Classification | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic | anthropic | anthropic | claude-3-5-haiku-20241022 | override:anthropic | env_file:ANTHROPIC_API_KEY ***REDACTED*** | not_applicable | pending_cdp | ready_for_cdp | provider-inventory.json; provider-inventory.md |
| aphrodite | aphrodite | aphrodite | aphrodite-engine | configured_models:aphrodite | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:2242/v1/chat/completions | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| cohere | cohere | cohere | command-r-08-2024 | override:cohere | env_file:COHERE_API_KEY ***REDACTED*** | not_applicable | pending_cdp | ready_for_cdp | provider-inventory.json; provider-inventory.md |
| custom_openai_api | custom | custom-openai-api | custom-model-alpha | configured_models:custom | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:1234/v1/chat/completions | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| custom_openai_api_2 | custom_2 | custom-openai-api-2 | custom-model-gamma | configured_models:custom_2 | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:5678/v1/chat/completions | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| deepseek | deepseek | deepseek | deepseek-chat | override:deepseek | env_file:DEEPSEEK_API_KEY ***REDACTED*** | not_applicable | pending_cdp | ready_for_cdp | provider-inventory.json; provider-inventory.md |
| google | google | google | gemini-2.0-flash-lite | override:google | env_file:GOOGLE_API_KEY ***REDACTED*** | not_applicable | pending_cdp | ready_for_cdp | provider-inventory.json; provider-inventory.md |
| groq | groq | groq | llama-3.1-8b-instant | override:groq | env_file:GROQ_API_KEY ***REDACTED*** | not_applicable | pending_cdp | ready_for_cdp | provider-inventory.json; provider-inventory.md |
| huggingface | huggingface | huggingface | meta-llama/Meta-Llama-3.1-8B-Instruct | configured_models:huggingface | env_file:HUGGINGFACE_API_KEY ***REDACTED*** | not_applicable | pending_cdp | ready_for_cdp | provider-inventory.json; provider-inventory.md |
| koboldcpp | koboldcpp | koboldcpp |  | server_default | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:5001/api/v1/generate | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| llama_cpp | llama_cpp | llama_cpp |  | server_default | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:8080/completion | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| local_llm | local_llm | local-llm |  | server_default | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:8000/v1/chat/completions | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| local_llamacpp | local_llamacpp | local_llamacpp | custom-model-gamma | config:model | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:8001/v1/chat/completions | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| local_llamafile | local_llamafile | local_llamafile |  | server_default | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:8001/v1/chat/completions | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| local_mlx_lm | local_mlx_lm | local_mlx_lm | custom-model-gamma | config:model | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:5678/v1/chat/completions | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| local_ollama | local_ollama | local_ollama | custom-model-gamma | config:model | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:5678/v1/chat/completions | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| local_vllm | local_vllm | local_vllm | custom-model-gamma | config:model | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:8008/v1/chat/completions | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| mistral | mistral | mistral | open-mistral-nemo | override:mistral | env_file:MISTRAL_API_KEY ***REDACTED*** | not_applicable | pending_cdp | ready_for_cdp | provider-inventory.json; provider-inventory.md |
| mistralai | mistralai | mistralai | open-mistral-nemo | override:mistralai | env_file:MISTRAL_API_KEY ***REDACTED*** | not_applicable | pending_cdp | ready_for_cdp | provider-inventory.json; provider-inventory.md |
| mlx_lm | local_mlx_lm | local_mlx_lm | custom-model-gamma | config:model | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:5678/v1/chat/completions | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| moonshot | moonshot | moonshot | kimi-latest | override:moonshot | env_file:MOONSHOT_API_KEY ***REDACTED*** | not_applicable | pending_cdp | ready_for_cdp | provider-inventory.json; provider-inventory.md |
| ollama | ollama | ollama | gemma3:12b | configured_models:ollama | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:11434/v1/chat/completions | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| oobabooga | oobabooga | oobabooga |  | explicit_model_missing | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:5000/v1/chat/completions | skip | explicit_model_missing | provider-inventory.json; provider-inventory.md |
| openai | openai | openai | gpt-4o-mini-2024-07-18 | override:openai | env_file:OPENAI_API_KEY ***REDACTED*** | not_applicable | pending_cdp | ready_for_cdp | provider-inventory.json; provider-inventory.md |
| openrouter | openrouter | openrouter | openai/gpt-4o-mini | override:openrouter | env_file:OPENROUTER_API_KEY ***REDACTED*** | not_applicable | pending_cdp | ready_for_cdp | provider-inventory.json; provider-inventory.md |
| tabbyapi | tabbyapi | tabbyapi | tabby-model | configured_models:tabbyapi | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:8080/v1/chat/completions | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| vllm | vllm | vllm | vllm-model-z | configured_models:vllm | not_required | api_url; reachable=false; probe=unreachable:URLError; endpoint=http://localhost:8000/v1/chat/completions | skip | endpoint_unreachable | provider-inventory.json; provider-inventory.md |
| zai | zai | zai | glm-4.5-flash | override:zai | missing | not_applicable | skip | missing_key | provider-inventory.json; provider-inventory.md |

## Run Notes

Final full sweep: `Docs/superpowers/qa/provider-cdp-uat/provider-sweep-results.json`

Generated at: 2026-06-01T07:31:58.333Z

Attempted hosted providers with usable keys: 11

Status counts:

- success: 7
- fail_external: 4
- fail_chatbook: 0

Acceptance rule: a provider is accepted only when the second assistant reply is visible in the same rendered Console session after selecting provider and model through dropdown controls.

| Provider | Execution key | Model | Result | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| anthropic | anthropic | claude-3-5-haiku-20241022 | fail_external | screenshots/anthropic-turn-1-fail_external.png | Entered provider path but remained in streaming state until timeout. |
| cohere | cohere | command-r-08-2024 | success | screenshots/cohere-success.png | Second assistant reply completed. |
| deepseek | deepseek | deepseek-chat | success | screenshots/deepseek-success.png | Second assistant reply completed. |
| google | google | gemini-2.0-flash-lite | fail_external | screenshots/google-turn-1-fail_external.png | Provider returned HTTP 400 through the rendered Console flow. |
| groq | groq | llama-3.1-8b-instant | fail_external | screenshots/groq-turn-1-fail_external.png | Provider returned HTTP 400 through the rendered Console flow. |
| huggingface | huggingface | meta-llama/Meta-Llama-3.1-8B-Instruct | fail_external | screenshots/huggingface-turn-1-fail_external.png | Provider returned HTTP 400 through the rendered Console flow. |
| mistral | mistral | open-mistral-nemo | success | screenshots/mistral-success.png | Second assistant reply completed after readiness fix. |
| mistralai | mistralai | open-mistral-nemo | success | screenshots/mistralai-success.png | Second assistant reply completed. |
| moonshot | moonshot | kimi-latest | success | screenshots/moonshot-success.png | Second assistant reply completed. |
| openai | openai | gpt-4o-mini-2024-07-18 | success | screenshots/openai-success.png | Second assistant reply completed. |
| openrouter | openrouter | openai/gpt-4o-mini | success | screenshots/openrouter-success.png | Second assistant reply completed. |

Skipped rows were not attempted through CDP because inventory classified them before launch:

- local/custom endpoints with unreachable probes: aphrodite, custom_openai_api, custom_openai_api_2, koboldcpp, llama_cpp, local_llm, local_llamacpp, local_llamafile, local_mlx_lm, local_ollama, local_vllm, mlx_lm, ollama, tabbyapi, vllm
- explicit model missing: oobabooga
- missing key: zai

## Fixes And Reruns

- Added Console Settings provider options for all Console-sendable handler providers, including providers that are not represented by the model registry.
- Changed Console Settings model selection to remain dropdown-backed at all times. When no configured model exists, the visible control is a disabled dropdown with `No configured models`; the freeform input is hidden/internal only.
- Added UAT-first model configuration in the isolated Textual-web launch helper so provider/model dropdowns have deterministic UAT choices.
- Fixed keyed-provider readiness fallback to use conventional env var names when config supplies only a model. This fixed `mistral` with `MISTRAL_API_KEY`; `mistralai` also aliases to `MISTRAL_API_KEY`.
- Hardened the CDP runner to use provider/model dropdown keyboard navigation, wait for `Run: Response complete.` before sending turn 2, and focus the composer before each send.
- Reruns:
  - OpenAI canary passed after composer focus fix.
  - Cohere passed after response-complete gating.
  - Mistral passed after readiness env-var fallback.
  - Final full sweep completed with no `fail_chatbook` rows.

## Residual Risks

- Anthropic remained in streaming state until the 120s harness timeout. It was attempted with a ready credential and selected model, but did not meet acceptance.
- Google, Groq, and HuggingFace returned HTTP 400 from their provider calls. The rendered flow reached provider execution, but the UI/log evidence does not expose enough provider response detail to distinguish model availability from provider-specific request validation.
- Local/custom providers were not exercised because endpoint probes were unreachable in the isolated profile.
