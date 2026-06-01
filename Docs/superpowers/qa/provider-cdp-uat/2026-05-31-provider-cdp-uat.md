# Provider CDP UAT

Date: 2026-05-31
Branch:
Spec:
Backlog task:
Textual-web URL:
Isolated HOME:
Isolated XDG config:
Isolated data:
App log:

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

## Fixes And Reruns

## Residual Risks
