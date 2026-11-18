---
id: TASK-41700
title: "Cloud provider base-URL overrides accept gateway paths"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 03:38
updated_date: 2026-11-18 05:35
labels:
- follow-up
- providers
priority: high
references:
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/LLM_Calls/LLM_API_Calls.py
dependencies:
- TASK-41522
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner decision 2026-11-18 ("Allow gateway paths"), taken while reviewing TASK-41522. The provider endpoint contract refuses a cloud provider's base-URL override that carries a gateway path, for example a Cloudflare AI Gateway base `https://gateway.ai.cloudflare.com/v1/<account>/<gateway>/openai` ("Endpoint path has ambiguous API suffixes"). Console has refused such bases since TASK-40300, and TASK-41522 makes every other caller refuse them too. The owner wants gateway-style bases to work for the first-party cloud providers, in Console and everywhere else, the way the contract's custom-hosted reading already accepts a route-bearing base. The host checks (hex, octal and short IPv4, IDN, user info, zone id, query, fragment) stay as they are.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A first-party cloud provider's base-URL override with a gateway path (a Cloudflare AI Gateway base for OpenAI is the worked example) is accepted by the endpoint contract and sends to that exact base, in Console and in every non-Console caller (Dreams, ingest, Evals)
- [x] #2 Every refusal TASK-41131, TASK-41132 and TASK-41522 pin for hosts, user info, zone ids, queries and fragments still refuses
- [x] #3 Tests pin the accepted gateway base per adapter family and show it red before the change
- [x] #4 The endpoint docs (`Docs/Architecture/llm-providers.md`) and the provider settings User Guide page say gateway bases are accepted
<!-- AC:END -->

## Implementation Plan

1. One checker: in the endpoint contract, a first-party cloud provider's key (the registry's non-engine cloud records, `mistralai` included) reads a path the generic suffix rule calls ambiguous as its API base, as the custom-hosted reading does, unless the path names a route. Every value the contract accepts today resolves as before; every host and shape refusal is untouched (they are decided before the path).
2. Console's readiness, send, Make default, Settings save and project identity, and every adapter's check (TASK-41522), already decide with this contract, so they accept the same bases. Console pins a cloud provider's configured base unchanged, and Settings saves `<base>` from `<base>/chat/completions`.
3. Display: `safe_endpoint_display` also tries the relaxed reading, so a gateway base shows as itself rather than "invalid endpoint".
4. Tests (red at TASK-41522's head, green after): per adapter family through `chat_api_call` (the seam Dreams, ingest and Evals call), the exact URL posted for a gateway base; Console readiness and send pin the exact base; Settings saves it unchanged; the refused shapes from TASK-41131/41132/41522 still refuse with a gateway path.
5. Docs: `Docs/Architecture/llm-providers.md` and the provider settings User Guide page.

## Implementation Notes

One checker, one new reading: in `Provider_Contract/provider_endpoint_contract.py`, a first-party cloud provider's key (`_CLOUD_BASE_KEYS`: the registry's non-engine cloud records, so OpenAI, Anthropic, Cohere, Google, Hugging Face, Groq, OpenRouter, DeepSeek, Mistral and `mistralai`, Moonshot, Z.ai, QwenCloud) reads a path the generic suffix rule calls ambiguous (a `v1` that is not last) as its API base, through the custom-hosted branch that already does this, unless the path names a route. Every value the contract accepted before resolves exactly as before; the host and shape rules run first and are untouched. Console's readiness and send, Make default, the Settings save, project identity and every adapter's TASK-41522 check already decide with this contract, so all of them now accept the same bases. Console pins a cloud provider's base unchanged, the Settings save writes `<base>` (from `<base>/chat/completions`), and each adapter appends its own route.

`_display_resolution` also tries the relaxed reading, so a gateway base displays as itself in Console's endpoint labels instead of "invalid endpoint" (a refused one still reads "invalid endpoint"; one with user info, a query or a fragment displays without them, as other accepted shapes already did).

**Tests** (`Tests/Chat/test_local_server_discovery.py`, lane-run; red with TASK-41522's head sources, ae2a2368e1, green after; 25 of 28 nodes red, the 3 green ones are Z.ai's adapter and Console nodes, whose preset reading already took a path as its base, and the refusal check, green before and after):
- `test_a_cloud_adapter_sends_to_a_gateway_base_exactly` (14 nodes, one per adapter case): through `chat_api_call`, the seam Dreams, ingest analysis and Evals call, from the adapter's own table and pinned, the exact URL posted: OpenAI `<gw>/openai/chat/completions` and DeepInfra `/v1/openai/chat/completions`; Anthropic `<gw>/anthropic/messages` and `<gw>/anthropic/v1/messages`; Cohere `<gw>/cohere/v2/chat`; Google `<gw>/google-ai-studio/v1beta/models/m:generateContent`; Hugging Face `<gw>/huggingface/v1/chat/completions` and, in router mode, `<gw>/huggingface/models/m/v1/chat/completions`; the strict adapters `<base>/chat/completions`; QwenCloud `<base>/responses`. Red: the contract's refusal, nothing posted.
- `test_console_sends_to_a_gateway_base_exactly` (12 nodes): from the config and from the session, readiness is not "Invalid URL", the resolution is Ready with that exact base, and the adapter is called with it. Red: "Invalid URL", not ready.
- `test_a_gateway_base_saves_and_displays_as_itself`: Settings' `persisted_endpoint_for_key` writes the base (also from a pasted `/chat/completions`), and the display shows it. Red: nothing written.
- `test_a_gateway_path_keeps_every_host_and_shape_refusal` (AC#2): every cloud key still refuses each TASK-41131/41132/41522 shape with a gateway path (zone id, leading hyphen, `127.1`, IDN, `localhost..`, user info, hex, octal and long-hex IPv4, query, fragment, `//`, `..`), and all but Z.ai refuse a route (`/responses`, `/chat/completions/v1`).
- Rewritten on purpose: `test_make_default_never_writes_an_endpoint_the_contract_refuses` pinned `/v1/models/v1` and `/v1/api/v1/generate` as refused for OpenAI; they are now gateway-shaped bases for a cloud provider and are written (still refused for the local providers in that test).
- Serial cost of all 43 TASK-41522/41700 nodes: 36 s at load 45-49.

**Covering run** (114 files: the TASK-41522 set plus every file that calls the contract, the display or the Settings endpoint save; `-n 4`, load 26-46): 575 failed / 7623 passed. Re-run of the 524 failing functions with the contract change reverted: 522 fail the same way (mostly the known `RecoveryRequired` trap, 295 in `test_console_native_chat_flow.py` and 97 in `test_console_session_settings.py`). The other two passed at head on a re-run (load flakes), and `test_open_console_settings_real_callback_stages_typed_credential_route`, which passed in the reverted batch, fails alone both with and without the change (order-dependent). The three TASK-40300 files outside that set (`test_custom_endpoint_registry.py`, `test_settings_custom_endpoint_default.py`, `test_console_endpoint_template_modal.py`): 93 passed. `test_first_run_provider_check.py` (reads `llm-providers.md`): 66 passed, 1 failed the same way with the change reverted (an API-key help line).

**Other checks.** Size ratchets 232 passed (serial); `ruff_gate.py` OK; pre-import LOC budget passed; preflight passed.

**Docs.** `Docs/Architecture/llm-providers.md`: a "Base-URL overrides and the endpoint contract" section and gotcha 6. `Docs/User_Guide/settings.md`: the Endpoint row accepts a cloud gateway base, with the route each adapter adds.

**Not changed (pre-existing, filed as TASK-41704).** The generic reading turns a cloud base whose path has no `v1` into `<path>/v1/chat/completions`, so Settings saves Google's own `https://generativelanguage.googleapis.com/v1beta` as `.../v1beta/v1` and Cohere's `https://api.cohere.com` as `.../v1`, which those native adapters then post under. Reading every cloud path as its base would fix it but changes the canonical connection identity of values accepted today (project-instruction keys), so it is left for its own task.

Files: `tldw_chatbook/Provider_Contract/provider_endpoint_contract.py`, `Tests/Chat/test_local_server_discovery.py`, `Docs/Architecture/llm-providers.md`, `Docs/User_Guide/settings.md`.

**Review r16cu-rev2 round (2026-11-18).**
- I-1: a native adapter's own route passed as a base (`https://api.anthropic.com/v1/messages`, a gateway's `.../anthropic/v1/messages`, `/v1/complete`, Google's `.../models/x:generateContent`): Settings saved it, Console read Ready, and the adapter posted `.../v1/messages/messages` (404). `_names_a_native_route` now refuses, for its key and whatever the path, Anthropic's trailing `messages` or `complete`, Cohere's trailing `chat` (`/v2/chat`), and any Google segment with `:generateContent` or `:streamGenerateContent`, with the copy route refusals already use ("Endpoint path has ambiguous API suffixes."). This also refuses the plain forms the generic rule accepted before TASK-41700 (`https://api.anthropic.com/messages`, `https://api.cohere.com/v2/chat`, Google's own `.../v1beta/models/<model>:generateContent`); each of those was sent with the route twice. `test_a_gateway_path_keeps_every_host_and_shape_refusal` gained a row set per family: at 4dab01626b eight of the nine rows were accepted (all but the Cohere gateway form, already refused); green after. The User Guide line now names both kinds of refused route, which is true.
- M-1: `Docs/Architecture/llm-providers.md` scopes "saved unchanged" to a gateway base whose `v1` is not last, names the dropped `/chat/completions` or `/models`, and names the `/v1` the generic rule appends to a path with no `v1` (TASK-41704). The User Guide says the same, with the config-file workaround.
- M-2: `console_session_settings._invalid_endpoint_kind` branched on the provider-less display, which the relaxed display now accepts more paths for, so a refused path typed without a scheme read "add http:// or https://". It now first asks the provider's own reading of `http://<value>`, and a value that reading still refuses is "invalid". New case in `test_the_endpoint_line_tells_only_an_address_to_add_a_scheme`: `gpu-box.lan:8080/v1/models/v1` (llama.cpp's reading), red at 4dab01626b ("Endpoint: Invalid (add http:// or https://)"), green after.
- M-3: the user-info traceback test moved from `Tests/LLM_Calls/test_hosted_chat.py` (nightly only) to `Tests/Chat/test_local_server_discovery.py::test_the_strict_transport_refuses_a_user_info_base_without_rendering_it`, with its `HostedHTTPTransportConfig` built inline. Negative control re-run there: with the refusal temporarily raised `from ValueError(<url>)` it failed on the assertion (the canary rendered in the chained `ValueError`); green after reverting.
- Filed TASK-41704 (To Do) for the `/v1` append, at the coordinator's request.
- Reruns: the gateway, Make-default, cloud-adapter, scheme-line and moved nodes, 207 passed (`-n 4`). Covering set (the same 114 files, `-n 4`, load 8-42): 575 failed / 7624 passed (one more pass: the moved test). Against the previous round's head run, two failing functions are new (`test_console_native_chat_flow.py`'s two delete-preview tests, the known `RecoveryRequired` family); both also fail with this round's source changes reverted, and two earlier failures passed this time. Serial size ratchets 232 passed; `ruff_gate.py` OK; preflight passed. `console_session_settings.py` has no size row.

