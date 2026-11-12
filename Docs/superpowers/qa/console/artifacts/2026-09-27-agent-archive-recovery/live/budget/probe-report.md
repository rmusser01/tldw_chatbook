# Bounded archive output-budget live qualification

Status: **incomplete / constraint-violating probe**. TASK-33020 AC2 remains unqualified. No model-issued restoration, confirmation, receipt Open, or natural restore follow-up was completed. This is a fresh bounded probe, not evidence that changing output tokens fixes the historical failures.

Scratch: `/private/tmp/archive-budget-live-764p7lar`. Report generated after native Quit. Production HEAD `f83f8975c88b762eb61e927247d76a20c466ea2f`; feature source `/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook`; interpreter `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` (3.12.11). No production source/test/config/task/plan/QA edits, commits, tests, reviews, subagents, normal-profile changes, server changes, or cleanup were performed by this probe. Root-owned dirty documentation paths remained the same.

## Isolation and controls

Only the earlier scratch TOML was cloned into the new root; no old data/profile state was copied or mutated. HOME, XDG_CONFIG_HOME, XDG_DATA_HOME, XDG_CACHE_HOME, TLDW_CONFIG_PATH, and paths.data_dir all resolve into the new scratch. The source/config/data paths and `keyring.backends.null.Keyring` were verified before importing/running the native app. Scratch config parses as TOML. Capture-Off, native tools/runtime, RAG-only (`direct_library_tools=false`) and default Assistant Library allowed were observed. A private tmux server ran the production TldwCli terminal at 160×48. No original task7 socket/session was used.

Scratch-only controls: llama_cpp `max_tokens=4096`; `api_retries=0` and compatibility `retries=0`; Console `agent_max_model_turns=10`, `agent_max_steps=80`, `agent_max_wall_seconds=180`, `agent_max_model_retries=0`. `console_run_budget()` verified 10/80/180/retries0; persisted AgentService run budget agreed. No AgentService initializer patch was needed. Existing successful tool/request shaping was unchanged. The ephemeral launcher wrapped existing requests Session.post and the returned response.iter_lines, observed metadata only, and blocked an eleventh local chat POST before transport as a whole-probe backstop. It never rewrote content, messages, schemas, response lines, or arguments.

Before the first send, localhost9099 reported exactly one idle slot, n_ctx64000, and the sole listed existing Gemma model `../../../Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`. All observed inference was serial on that endpoint. Metadata observer recorded max_tokens, scalar stream/reasoning controls, role counts, tool names/schema digest, HTTP status/content type, aggregate content/reasoning byte totals, named-call counts, finish_reason, usage presence, and SSE error presence. No headers, keys, tool args, message/answer/reasoning bodies were recorded by the observer. Native UI capture artifacts separately contain the displayed synthetic UI, including thinking panels; these are not wire/body exports.

## Exact native interaction and stopping

Chat2 was created through the native New tab button. Native Paste placed the 29-character unsent draft `Unrelated draft stays unsent.`; `unrelated-draft-settled.txt/.ansi` shows it. Source Chat1 was activated with its capture-derived tab coordinates and native Enter.

Initial unsent search paste: `Find my archived chat named Cedar budget archival notes. Show the archive search result so I can choose it.` This exceeded the 50-character collapse threshold. Enter opened the native `Expand?` prompt; no model event or HTTP request occurred. It was replaced through native Ctrl+A/Backspace and Paste. The only actual submitted user prompt was exactly `Find my archived Cedar budget archival notes.` (45 characters), captured in `search-draft-short.txt/.ansi`, then sent with native Enter. No opaque reference or tool argument payload was supplied as user input.

The model first discovered/loaded `search_library_rag`, executed it, then made three successive find_tools calls before loading archive search. Archive search subsequently executed twice. The final load_tools working set switched back to search_library_rag. No request_conversation_restore schema was observed in any sent wire tool set, and no such tool was proposed or executed. No completed final answer or selectable archive result card was captured before stopping. No natural named/ordinal restore follow-up was sent.

The requested stop rule was the first model-error/discovery loop, with at most10 logical attempts over the whole probe. Repeated find_tools were detected while inspecting native/runtime state; native Ctrl+Q was dispatched. Fast continuation had reached the 10-turn bound before Quit completed. Thus this run stopped at the hard10 bound rather than demonstrating an immediate first-loop cutoff. Exact final database count: 10 model_request_started, 10 model_response_completed, 10 model rows, 10 tool proposals/executions/results; one run, ID `ab2dd39c-b7bc-4485-8559-2ca589aa1b93`, final status `stuck`. Runtime recorded `model-turn budget exhausted`, then `Budget wrap-up call failed; terminating without one`. Those are local runtime outcomes, not a raw HTTP/provider failure. No provider error or equivalent retry was observed. No eleventh HTTP POST was sent.

Native Ctrl+Q exited directly; no Quit modal remained to confirm. A subsequent capture failed because the private server had exited. `shutdown.json` records `no server running` and the sole server slot idle. No further request, relaunch, manual restore, or remediation was attempted.

## Constraint violation: unintended model download and index initialization

**The intended no-download/no-index constraint was violated by the model-selected production search_library_rag tool.** The native terminal emitted an unauthenticated Hugging Face Hub warning. Post-Quit scratch inspection found six MiniLM embedding-cache blobs totaling **91,567,205 bytes** under `data/task7_live/models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/`, plus a scratch `data/task7_live/chromadb/chroma.sqlite3` containing one collection and zero embedding rows. The Xet log is under scratch XDG_CACHE_HOME. The probe did not intentionally call a remote inference provider or download API, but that does not satisfy the requested no-download condition. No downloaded files/index were removed, and no normal settings/backend/server were changed to remediate it. This artifact set must not be labeled a successful no-index qualification.

## Actual wire results

Every observed request used max_tokens4096, stream=true, empty explicit reasoning controls, and transport Retry.total0. All ten raw HTTP statuses were200/text-event-stream. Every completion ended with finish_reason tool_calls, DONE=true, one named call, zero visible content bytes, and no SSE error or invalid JSON line. Usage was absent for every response, so prompt/completion token counts are unavailable. Zero visible bytes here are paired with usable native tool calls; they do not establish reasoning exhaustion. No finish_reason length was observed. This probe cannot diagnose the old512-token carried502 failures or evaluate restore-schema compatibility, because restore was never sent.

| Attempt | UTC request time | Roles | Response tool | Reasoning bytes | HTTP / finish |
| --- | --- | --- | --- | ---: | --- |
| 1 | 2026-09-28T00:09:18.300+00:00 | system:1, user:1 | find_tools | 346 | 200 / tool_calls |
| 2 | 2026-09-28T00:09:21.185+00:00 | assistant:1, system:1, tool:1, user:1 | load_tools | 440 | 200 / tool_calls |
| 3 | 2026-09-28T00:09:24.195+00:00 | assistant:2, system:1, tool:2, user:1 | search_library_rag | 87 | 200 / tool_calls |
| 4 | 2026-09-28T00:09:46.686+00:00 | assistant:3, system:1, tool:3, user:1 | find_tools | 825 | 200 / tool_calls |
| 5 | 2026-09-28T00:09:50.408+00:00 | assistant:4, system:1, tool:4, user:1 | find_tools | 118 | 200 / tool_calls |
| 6 | 2026-09-28T00:09:51.451+00:00 | assistant:5, system:1, tool:5, user:1 | find_tools | 374 | 200 / tool_calls |
| 7 | 2026-09-28T00:09:53.801+00:00 | assistant:6, system:1, tool:6, user:1 | load_tools | 99 | 200 / tool_calls |
| 8 | 2026-09-28T00:09:55.535+00:00 | assistant:7, system:1, tool:7, user:1 | search_archived_conversations | 62 | 200 / tool_calls |
| 9 | 2026-09-28T00:09:58.987+00:00 | assistant:8, system:1, tool:8, user:1 | search_archived_conversations | 1207 | 200 / tool_calls |
| 10 | 2026-09-28T00:10:06.332+00:00 | assistant:9, system:1, tool:9, user:1 | load_tools | 669 | 200 / tool_calls |

Full request tool-name sets and schema SHA256 values are preserved in `wire-metadata.jsonl`; runtime event/tool timestamps and persisted budget are in `runtime-metadata.json`.

## Fixture, original identity, drafts and sends

One Default/global saved chat was seeded with production CharactersRAGDB APIs, title `Cedar budget archival notes`, ID `2e6b0b17-7e6f-4569-a9e6-c26675f1eb80`. Original message IDs before the run: `237a9a59-c7f1-4078-bae7-f003c1ae62b4` and `50213930-de23-4309-85ea-8a239157054e`; original active leaf `50213930-de23-4309-85ea-8a239157054e`. The archive transition used expected version1 and produced the archived fixture; post-run it remains archived1/version2, global workspace_idNULL. Both original message IDs, parent relation and active leaf remain exact. No duplicate restored fixture was created and no new message was added to that original chat. `fixture.json` and `post-state.json` contain the exact safe identities. No workspace-only second fixture was seeded.

The only persisted source conversation contains one explicitly submitted user message and one assistant record; one agent run exists. Chat2 stayed unsent at its pre-search native capture and has no stored conversation/messages. No restore/Open occurred and no implicit send was observed. This is evidence that the original fixture was untouched, not a restore-to-Open zero-send qualification. A source preservation draft was not prepared before the run reached its bound, and Chat2 draft retention after Quit was not re-opened/verified. Source-draft preservation, final unrelated-draft retention, exact UI confirmation/scope, restoration, original-ID Open, and zero implicit send across restoration/Open remain **unqualified**.

## Evidence files to copy

- `/private/tmp/archive-budget-live-report.md` (this report).
- `/private/tmp/archive-budget-live-764p7lar/wire-metadata.jsonl` (safe transport metadata, actual scalar output budget and all ten HTTP/finish outcomes).
- `/private/tmp/archive-budget-live-764p7lar/runtime-metadata.json` (safe runtime event/tool metadata, exact run ID/counts/budget).
- `/private/tmp/archive-budget-live-764p7lar/fixture.json` and `/private/tmp/archive-budget-live-764p7lar/post-state.json` (pre/post original IDs and unchanged archive state).
- `/private/tmp/archive-budget-live-764p7lar/shutdown.json`, `/private/tmp/archive-budget-live-764p7lar/preflight-server.json`, `/private/tmp/archive-budget-live-764p7lar/pre-send-slot.json` (slot and shutdown scalars).
- `/private/tmp/archive-budget-live-764p7lar/unrelated-draft-settled.txt/.ansi`, `/private/tmp/archive-budget-live-764p7lar/source-active.txt/.ansi`, `/private/tmp/archive-budget-live-764p7lar/search-draft-short.txt/.ansi`, `/private/tmp/archive-budget-live-764p7lar/search-after-rag.txt/.ansi` (native UI provenance; last includes HF warning and a thinking panel).
- `/private/tmp/archive-budget-live-764p7lar/launch.py`, `/private/tmp/archive-budget-live-764p7lar/seed.py`, `/private/tmp/archive-budget-live-764p7lar/start-terminal.sh`, `/private/tmp/archive-budget-live-764p7lar/drive.py` (ephemeral probe/observer controls).

No full suite, source patch, independent review, commit, cleanup, or additional inference is authorized/completed by this report.
