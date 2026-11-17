---
id: TASK-41131
title: "chat_api_call callers outside Console skip the provider endpoint contract"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-16 21:50'
created_date: 2026-11-15 02:20
labels:
- follow-up
- security
- providers
priority: medium
references:
- tldw_chatbook/Chat/Chat_Functions.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 endpoint stream fu/r14-endpoint 6cb8e34461 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-40300 made Console's send, readiness, registry and project identity refuse every endpoint the provider endpoint contract refuses (zone ids, leading-hyphen hosts, 127.1, IDN, query or fragment). The same configured endpoint still reaches the network from every other caller of the unified dispatcher.

**Evidence.** `chat_api_call` (`Chat/Chat_Functions.py:1029`) applies no contract check (`endpoint_refusal` is not referenced in the file). On the endpoint branch only `chat_with_llama` checks it (`LLM_API_Calls_Local.py:899-910`). Callers outside Console reported by the endpoint stream: `Chat_Functions.chat` (legacy), `Chat/document_generator.py`, `AgentService`'s default chat call, Dreams, Summarization, Local_Ingestion, Research, Evals, Library, WebSearch, MCP sampling and the Personal Context interview.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every provider call that resolves a local or custom endpoint is refused before any network I/O when the contract refuses the endpoint, with the contract's reason
- [x] #2 The check sits where all callers route (for example chat_api_call or the provider handlers), not in each caller
- [x] #3 Tests cover at least the dispatcher plus two non-Console callers with the refused shapes, red before the change
<!-- AC:END -->

## Implementation Plan

1. Put the contract check where every local and custom adapter's request is built, on the URL it actually posts to: the shared OpenAI-compatible local transport (`_chat_with_openai_compatible_local_server`: Local LLM, the llama.cpp family, vLLM, Ollama, Oobabooga, TabbyAPI, Aphrodite, MLX-LM, Custom OpenAI API 1 and 2), KoboldCpp's native send, and the hosted engine's base-URL resolution (custom-hosted and every engine preset). The check runs on the value the adapter sends to, so the checked URL is the one used whichever table it came from, and no destination changes.
2. Refuse with a status-less ChatConfigurationError carrying the contract's reason, before any session or request.
3. Tests in Tests/Chat/test_local_server_discovery.py (lane-run): chat_api_call for every local and custom dispatch name with each refused shape, from config and pinned; Dreams and Summarization with a refused endpoint; keeps-working checks on `_merge_user_config_over_shipped_defaults(...)` that every shipped local endpoint and engine preset base still resolves to the same URL. Red first.

## Implementation Notes

The check sits on the URL each local and custom adapter is about to use, inside the adapters every `chat_api_call` caller routes through, not in `chat_api_call` and not in callers:
- the shared OpenAI-compatible local transport `_chat_with_openai_compatible_local_server` (Local LLM, llama.cpp/llamafile/Strata/PolyStrata, vLLM, Ollama, Oobabooga, TabbyAPI, Aphrodite, MLX-LM, Custom OpenAI API 1 and 2), with the contract's generic reading, which every one of those adapters shares (none is an engine preset or route-owning);
- `chat_with_kobold`'s native send, with the `koboldcpp` reading;
- the hosted engine's `_resolve_base_url` (custom-hosted, read relaxed as its engine reads it, and every engine preset, with its preset reading), before the engine's own malformed-URL copy, so the reason is the contract's.

One helper, `_refuse_a_refused_endpoint`, raises a status-less `ChatConfigurationError` "<adapter> API URL: <contract reason>" before any session or request; `chat_with_llama`'s round-14 check now uses it too. `chat_api_call` re-raises it unchanged (only Console runs sensitive requests, and Console refuses first), and the Summarization path returns it in-band.

**Refusing and pinning.** Console resolves an endpoint, checks it and pins it so the adapter cannot use another. Here the check runs on the value the adapter actually sends to, after the adapter's own resolution (pinned argument, else its own table, MLX's host/port), so the checked URL is the used URL by construction and there is nothing left to pin; no lookup table had to be duplicated and no caller's destination changes. That is why the keeps-working checks pass unchanged.

**Keeps working.** `test_an_ordinary_local_config_still_posts_where_it_did` posts through `chat_api_call` (fake session) on `_merge_user_config_over_shipped_defaults(user)` for 18 configs: every shipped local table, the adapters' own tables the template does not ship, and accepted user values (`gpu-box.lan`, `kobold_cpp`, `[fe80::1]`, MLX host/port); each posted URL is pinned as a literal. `test_every_engine_preset_still_resolves_its_shipped_base` resolves every engine preset's shipped base and a relaxed custom-hosted base (`/v1/openai`). Both were green with the base adapters and stay green.

**Red/green** (all in `Tests/Chat/test_local_server_discovery.py`, lane-run; refused shapes are round 14's plus `0x7f.1`; each test loops over the shapes, from the adapter's own table and pinned):
- first red run (one node per dispatch x shape): `test_chat_api_call_never_reaches_an_endpoint_the_contract_refuses` 117 of 126 red at base (a POST recorded, or the engine's own copy instead of the contract's reason; the 9 llama.cpp nodes were already refused by round 14), the MLX host test 10 of 10, Dreams 9 of 9 (`DID NOT RAISE`), Summarization 9 of 9 (a POST recorded). 173 of 173 green after.
- that took 232 s serially in the lane step at load 50, so the shapes were folded into each body (36 nodes for all five tasks' new tests); re-shown red with every source file temporarily restored to base (29 failed; the 7 that passed are the controls and keeps-working checks), green at head (36 passed). Serial lane cost of the 36: 93 s at load 100-111.
- Existing tests: `Tests/Chat/test_sensitive_llm_logging.py`'s shared-transport and KoboldCpp canary tests put a canary in a query, which the transport now refuses before the request, so they stopped reaching the error-body path they guard (one failed, one passed trivially). The canary now rides the path, as round 14 did for llama.cpp; a probe confirmed the shared-transport case reaches the 500 path again.

**Covering run** (70 files that call the local/engine adapters, `chat_api_call`, the summary, ingest analysis, Dreams or Detach; `-n 4`, load 65-113): 139 failed / 4053 passed at head against 134 / 3869 in an earlier run of the same files. Nearly all failures in both are the known local `RecoveryRequired: raw_source_selection_changed` trap (plain `Tests/UI/test_console_session_settings.py`, `test_settings_custom_endpoints.py`, `test_chat_functions.py`) or are red at base for other reasons (`test_custom_openai_execution_keys_constant` has the identical message with base sources). The head-only reds: the two sensitive-logging canary tests (fixed above); `test_console_review_wake_transport::test_actual_handler_review_wake_waits_for_pinned_capacity_and_finishes` (`review_context_capacity_unknown`) failed twice at load 65-110 and passed 4 of 4 at head at load 8-40 (3 serial, once in its file under `-n 4`) and once with base sources at load 8 — a timed capacity wait, not this change (its endpoints are 127.0.0.1 and cloud hosts the contract accepts); two Permission summaries journey nodes and one Console session-settings node passed on re-run. The two Detach UI nodes in `test_settings_custom_endpoints.py` still hit `RecoveryRequired` under a temporary `@private_profile_test` marker (marker removed again), so they are covered only by the lane tests.

**Not covered (outside AC#1's local/custom scope):** first-party cloud adapters (OpenAI, Anthropic, Cohere, Google, HuggingFace, Groq, OpenRouter, DeepSeek, Mistral, Moonshot, Z.ai, QwenCloud) still send to a base-URL override in their own config table unchecked when a non-Console caller pins nothing. Proposed as a follow-up.

**Side effect, not a regression.** With `[console] custom_endpoints_use_engine = false`, an OpenAI-compatible custom entry whose base only the relaxed reading accepts (`/v1/openai`) goes through the legacy custom slot, whose transport now refuses it ("Endpoint path has ambiguous API suffixes.") instead of posting to `.../openai/chat/completions/v1/chat/completions`, which never worked.

Files: `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py`, `tldw_chatbook/LLM_Calls/hosted_provider_engine.py`, `tldw_chatbook/Provider_Contract/provider_endpoint_contract.py` (docstring), `Tests/Chat/test_local_server_discovery.py`, `Tests/Chat/test_sensitive_llm_logging.py`.

**Review r15ep-rev1 round (2026-11-16).**
- M-1: inserting the helper had moved `@_provider_recovery.unqualified` off `_chat_with_openai_compatible_local_server` onto `_refuse_a_refused_endpoint`, so local sends and streams lost recovery admission. The helper now sits above the comment, undecorated. `test_the_shared_local_transport_keeps_its_recovery_wrapper` was red at the round's head (`AttributeError: __wrapped__`), green after.
- M-2: the shared transport takes the adapter's `provider_key` for the refusal's `provider`, and Custom OpenAI API 2 and MLX-LM pass display names ("Custom OpenAI API 2", "MLX-LM"). `test_a_refusal_names_the_adapter_and_carries_its_key` (14) and `test_an_mlx_refusal_names_mlx_lm_and_carries_its_key` (2): 11 were red (a display name as provider; "Custom_openai_api_2"/"mlx_lm" copy), green after; `Tests/LLM_Management/test_mlx_lm.py`'s four call pins follow.
- I-2: `Evals/eval_runner.ErrorHandler.with_retry` caught a `ChatConfigurationError` in its generic API-error branch, retried it and reported "Failed to connect". It now raises a non-retryable `ModelConfigurationError.provider_refused` at once, carrying the refusal's message. `test_an_eval_never_retries_an_endpoint_refusal` was red (4 attempts), green after; `Tests/Evals` (5 files, 72 passed).
  The new `logger.error(f"Configuration error for {provider}: {type(e).__name__}")` logs the provider key and the exception's class only (no URL, value or message); `Docs/security/production-diagnostic-inventory.json` re-pinned after reading it (`--statements` showed that one added call).
- I-3 (coordinator ruling: in scope): the Evals word bench posts to `[api_settings.llama_cpp] api_url` through its own httpx client. `UI/Evals/sample_bench.configured_llama_cpp_url` returns None for a URL the contract refuses, which shows the bench's existing not-configured state. Deviation from the review's wording: it checks the URL the bench sends to (the normalised root, with the typed host, user info, query and fragment kept), not the raw value, because the raw check refused `http://localhost:8080/v1/completions`, the bench's own route, which `Tests/UI/test_evals_empty_states.py` pins as working. `test_the_word_bench_never_targets_an_endpoint_the_contract_refuses` (127.1, hex, user info, `-bad.corp`, a query) was red at base for all 5, green after; two controls stay green. Known limit: with an existing unsteered llama.cpp bench target and a refused URL, `_default_client_factory` still falls back to `http://localhost:8080` (pre-existing behaviour for "not configured").
- Recommendation: the `derived-artifacts.yml` comment for this file now gives the measured serial time.
