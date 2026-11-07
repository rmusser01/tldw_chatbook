# CHAT-bridge — tldw_chatbook/Chat/{console_trace_service,console_provider_gateway,console_runtime,console_agent_bridge}.py, 29,362 lines

Worktree `/Users/macbook-dev/Documents/GitHub/tldw-review` @ d8fb4053f9 (detached origin/dev). READ-ONLY; nothing modified. `ruff check --select E9,F63,F7,F82` over the four files → `All checks passed!`.

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| tldw_chatbook/Chat/console_trace_service.py | 6,580 | read in full (1-1700, 1700-3400, 3400-5000, 5000-6580) |
| tldw_chatbook/Chat/console_provider_gateway.py | 7,210 | read in full (1-1150, 1150-2300, 2300-3450, 3450-4600, 4600-5750, 5750-6900, 6900-7210) |
| tldw_chatbook/Chat/console_runtime.py | 4,930 | read in full (1-1250, 1250-2500, 2500-3750, 3750-4930) |
| tldw_chatbook/Chat/console_agent_bridge.py | 10,642 | read in full (1-1200, 1200-2400, 2400-3600, 3600-4800, 4800-6000, 6000-7200, 7200-8400, 8400-9600, 9600-10642) |

Supporting files read for evidence only (not reviewed): `Chat/console_prepared_request.py:100-130`, `Chat/console_trace_settlement.py:235-335, :355-475`, `Chat/console_trace_runtime.py:40-60`, `Chat/console_chat_store.py:1355-1368, :19338`, `Chat/console_chat_controller.py:5595-5603, :20286-20330, :20424, :20608-20618, :24735-24790, :26922-26940`, `DB/transaction_observer.py`, `Chat/console_voice_trace_gateway.py:343-356`, `Chat/console_trace_final_values.py:775-790`, `Chat/provider_readiness.py:37,501`, `Chat/custom_endpoint_registry.py:467`, `Internal_Prompts/resolver.py:48-80`, `Chat/console_voice_attempts.py:319-324`, `Chat/chat_conversation_scope_service.py:105-110`. Tests grepped as evidence only.

Tooling written to scratchpad: `fbimports.py` — AST inventory of every function-body import in a file, static module-scope reachability back to the importing module (cycle test), and `sys.modules` membership after importing the file alone (redundancy test). Run as `cd $WT && source $SCRATCH/env.sh && PYTHONPATH=$WT $PY $SCRATCH/fbimports.py tldw_chatbook/Chat/<file>.py`.

## Findings   (ordered P0→P3, then D1→D4; one ### each)

No P0. No P1.

### P2 [D3] — `ConsoleProviderGateway._chat_api_kwargs` has zero production callers; ten tests pin a kwargs builder the app never runs while its live twin has drifted
- Where: `tldw_chatbook/Chat/console_provider_gateway.py:6791-6865` (dead); live twin `:6711-6788 _chat_api_kwargs_from_prepared`; third copy `:5117-5165 _auxiliary_chat_api_kwargs`
- Evidence: `rg -n "_chat_api_kwargs\(" tldw_chatbook/ | rg -v "_from_prepared\(|_auxiliary_chat_api_kwargs\(|def _chat_api_kwargs"` → no output. `rg -n "\._chat_api_kwargs\(" Tests/` → `Tests/Chat/test_console_provider_gateway.py:1666, :3645, :3673, :7195, :7209, :7219, :7234, :7239, :7269, :7270`. The docstring at :6796-6813 cites "the cache-stability test in Tests/Chat/test_console_provider_gateway.py" as the reason its system-row join is deterministic — that test exercises the dead builder. Drift (read): `_from_prepared` adds `chat_template_kwargs` (:6749-6753), local `api_key_resolved` (:6754-6756), moonshot/zai transport + `provider_continuations` (:6760-6768), vllm/local_vllm base_url pinning (:6775-6776), openai `response_format` pinning (:6781-6787); `_chat_api_kwargs` has none. `_auxiliary_chat_api_kwargs` pins `api_base_url` for EVERY provider (:5131) where `_from_prepared` pins a list (:6754-6787), and omits `prompt_caching`/`tools`/`chat_template_kwargs`.
- Why it matters: the Anthropic prompt-cache stability property and the leading-system-row extraction are asserted against code the send path never executes; a regression in `_chat_api_kwargs_from_prepared` cannot be caught by those ten tests, and the three copies already disagree on which providers receive `api_base_url`.
- Recommended correction: delete `_chat_api_kwargs`; port the ten tests to `_chat_api_kwargs_from_prepared` (through `prepare_provider_request`); fold `_auxiliary_chat_api_kwargs` into `_from_prepared` behind the existing `prepared is not None` branch at :4976 (the auxiliary path already uses `_from_prepared` when a prepared request exists).
- Size: M · ADR: no · Confidence: verified
- Pinning test: the ten above state the *dead* builder's behaviour as a requirement (the deletion is a test-port, not a silent change)
- Already covered: none

### P2 [D3] — physical trace GC/compaction is silently disabled forever by a swallowed `ImportError` of three first-party modules
- Where: `tldw_chatbook/Chat/console_runtime.py:3277-3359` (`_schedule_legacy_trace_maintenance.run`): `from ...console_trace_maintenance import PhysicalTraceCompactor, TraceGarbageCollector`, `from ...console_trace_models import new_opaque_id`, `from tldw_chatbook.config import resolve_trace_compaction_policy` sit inside `try: … except ImportError: pass  # Narrow test doubles may provide only the legacy worker`
- Evidence: `fbimports.py console_runtime.py` → all three targets `exists=True` today; `tldw_chatbook.config` is ALREADY imported at module scope (:159), so its function-body import (:3285) can never raise ImportError in production. `rg -n "PhysicalTraceCompactor|only the legacy worker|ImportError" Tests/UI/test_console_runtime*.py Tests/Chat/test_console_runtime*.py` → no hits: nothing pins the swallow. The sibling `except Exception` two lines below DOES log; only the import failure is silent.
- Why it matters: a renamed/moved symbol in `console_trace_maintenance` or `console_trace_models` degrades to `pass` + `await asyncio.sleep(1.0)` forever — the ledger's GC and VACUUM never run again, with no log line, for every user.
- Recommended correction: hoist the three imports to the top of `run()` (`console_trace_models` is a leaf and already loaded — module scope), and give test doubles an explicit `physical_maintenance_enabled=False` seam instead of relying on ImportError.
- Size: S · ADR: no (097-boot-budget-ratchets.md governs *deferral*, not swallowing) · Confidence: verified (swallow), inferred (nobody has hit it yet)
- Pinning test: none
- Already covered: none

### P2 [D4a] — `thaw_json` (Chat/console_prepared_request.py:110) is re-rolled five times under private names; one copy drifts; two of the re-rolling modules already import from `console_prepared_request`
- Where: `console_trace_service.py:6565 _thaw`, `console_trace_final_values.py:779 _thaw`, `console_voice_trace_gateway.py:351 _thaw`, `console_provider_gateway.py:1745 _thaw_auxiliary_value` (all byte-identical modulo name), `console_runtime.py:586-591` nested `thaw` (DRIFTS: does not `str()` mapping keys); canonical `console_prepared_request.py:110 thaw_json`
- Evidence: `sed -n 100,130p console_prepared_request.py; sed -n 6565,6571p console_trace_service.py; sed -n 351,356p console_voice_trace_gateway.py; sed -n 779,784p console_trace_final_values.py; sed -n 1745,1752p console_provider_gateway.py; sed -n 586,591p console_runtime.py`. `console_trace_service.py:16` and `console_provider_gateway.py:65-76` already import from `console_prepared_request` (the gateway even imports `thaw_json` itself at :75 and still defines `_thaw_auxiliary_value`). `rg "_thaw\b|_thaw_auxiliary_value" Tests/` → 0 hits.
- Why it matters: `_artifact_bytes` (trace_service:6573) — the byte-identity oracle every trace verification compares against — depends on the private copy; an edit to one copy (e.g. handling `list`/`set`) silently changes which artifacts match in one verifier and not the others. The runtime copy already differs.
- Recommended correction: delete the five private copies; `from tldw_chatbook.Chat.console_prepared_request import thaw_json`. Canonical home: `Chat/console_prepared_request.py` (existing).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D1] — a failing `conversation_id_for_session` lookup makes the shutdown fleet fence key on the session id instead of the conversation id, silently
- Where: `tldw_chatbook/Chat/console_runtime.py:4416-4422` (`begin_dispose`: `except Exception: pass` → `conversation_id = session_id`), same shape at `:4606-4611` (`dispose`)
- Evidence: read only (mechanical candidate except_exception_pass :4421 confirmed). The fence written by `fence_fleet(conversation_id, generation=…)` (bridge :8637) is keyed by conversation id; a session id never matches, so the real conversation's fleet stays admittable during quit.
- Why it matters: only reachable if the controller's lookup raises (a controller bug), but the failure is invisible — no log — and the consequence is exactly what the fence exists to prevent.
- Recommended correction: `logger.warning` with `type(exc).__name__` at both sites, or let the fence be skipped explicitly rather than mis-keyed.
- Size: S · ADR: no · Confidence: inferred (no reproduction of a raising lookup)
- Pinning test: none
- Already covered: none

### P3 [D2] — provider readiness is offloaded on the generic path but runs inline on the event loop on the llama.cpp path
- Where: inline `console_provider_gateway.py:3959-3963` (`_resolve_for_send_unclassified`, `uses_direct_llama_path` branch) and `:3720-3722` (`_context_window_target`, sync, called from async `resolve_context_window` :3752-3756); offloaded `:4210-4215 await asyncio.to_thread(get_provider_readiness, …)`
- Evidence: `git log --oneline -S"get_provider_readiness," -- console_provider_gateway.py` → `0077b872f1 fix(console): preserve model settings and keep provider setup responsive` (the offload was a responsiveness fix; the llama branch kept the sync call). Measured under the isolated profile (`PYTHON_KEYRING_BACKEND=null`): `get_provider_readiness` = **0.01 ms/call** for llama_cpp/openai/anthropic — so the real-profile cost, if any, is keyring-bound and not reproduced here.
- Why it matters: only if a real keyring round-trip is on this path; the commit is the only evidence that it once was.
- Recommended correction: match :4210 (`await asyncio.to_thread`) at :3959 for consistency; measure under a real keyring before treating it as a stall.
- Size: S · ADR: no · Confidence: inferred
- Pinning test: none
- Already covered: none

### P3 [D2] — the no-sink trace settlement fallback runs SQLite on the calling event loop
- Where: `console_provider_gateway.py:2070-2083` (`_settle_trace_response`, non-voice branch `settler(envelope, outcome, usage)` = `ConsoleTraceCallBoundary.settle_response` → `ConsoleTraceService.submit_settlement` → coordinator `_settle_claimed` → `operation_owned_connection(database)` + writes)
- Evidence: read; the authors state it at `:769-771` ("the legacy no-sink path … repeating SQLite work on the event-loop thread"). Reachable when (a) no sink bound, (b) `prepare_response_settlement` returns `None` (its own `except Exception: return None`, trace_service:497-498), or (c) the sink returns False. Production binds a sink unconditionally at `console_chat_controller.py:24755-24760` (`store.register_provider_trace_settlement_async`, `async def` at `console_chat_store.py:19338`); the voice branch already offloads (`:2075 asyncio.to_thread`).
- Why it matters: on path (b) a trace-prepare failure is followed by a synchronous ledger write on the UI loop, exactly when the app is already degraded.
- Recommended correction: `await asyncio.to_thread(settler, envelope, outcome, usage)` at :2083 — one-line parity with :2075.
- Size: S · ADR: no · Confidence: verified (path), inferred (frequency)
- Pinning test: none
- Already covered: none

### P3 [D2] — streamed tool-call `arguments` are accumulated with string `+=` per SSE fragment
- Where: `console_provider_gateway.py:2371` (`_ToolCallAccumulator._merge`: `entry["function"]["arguments"] += arguments`)
- Evidence: read only. The string lives in a dict, so CPython's refcount-1 in-place append does not apply → O(total_len × fragments) on the provider worker thread.
- Why it matters: a large `write_file`-style call (tens of KB over hundreds of deltas) pays a quadratic copy; bounded, not a stall.
- Recommended correction: accumulate a `list[str]`, `"".join` in `calls()`.
- Size: S · ADR: no · Confidence: inferred
- Pinning test: none · Already covered: none

### P3 [D2] — first `ensure_chat_store` runs ledger recovery and media-reference retry synchronously on whichever loop asks (the UI loop on first Console mount)
- Where: `console_runtime.py:3059-3084` (`recover_console_trace_calls(db)` → `ConsoleTraceSettlementCoordinator.recover_open_calls`, `SELECT call_id FROM console_trace_calls WHERE state IN (…)` `.fetchall()` (settlement:364-374) + `advance_call_state` writes; `persistence.retry_recovered_media_references()`)
- Evidence: callers `rg -n "ensure_chat_store\(" tldw_chatbook/` → `UI/Screens/chat_screen.py:10041`, `Chat/console_launch_wake.py:196`, `UI/Console_Modules/character.py:267`, `UI/Console_Modules/fleet.py:120`. Cost not measured — see Left UNVERIFIED.
- Why it matters: once per process, on the mount leg task-2902/task-26834 already treat as latency-sensitive; the recovery query is bounded by the number of open calls (normally 0-1) so the write half is the only real cost.
- Size: S · ADR: no · Confidence: inferred · Already covered: adjacent to task-2902 (not the same work)

### P3 [D3] — `ensure_activity_receipt_service` documents "call via asyncio.to_thread"; its only caller is synchronous and runs from the UI
- Where: `console_runtime.py:3436-3476` (docstring :3439-3441) vs `:3514` (`ensure_agent_bridge`, sync; opens `AgentRunsDB(Path(db_path).parent / "agent_runs.db")` = sqlite connect + schema on the caller's thread)
- Evidence: `rg -n "ensure_activity_receipt_service\(" tldw_chatbook/` → only `:3514`; `rg -n "ensure_agent_bridge\(" tldw_chatbook/` → `UI/Console_Modules/agent.py:982`, `UI/Console_Modules/fleet.py:154`, `Chat/console_launch_wake.py:200` (UI callers).
- Recommended correction: offload in `ensure_agent_bridge` or fix the docstring.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D3] — two dead module-private helpers and one dead alias in console_trace_service.py
- Where: `console_trace_service.py:6359 _revision_only`, `:6509 _omission_source`, `:5075 reconstruct_logical_header = reconstruct_header`
- Evidence: `rg -n "_revision_only|_omission_source|reconstruct_logical_header" tldw_chatbook/ Tests/` → `_revision_only` hits only its definition and self-recursive call (:6366); `_omission_source` only its definition (`_surface_omission_source` in console_trace_final_values.py:670 is a different function); `reconstruct_logical_header` only :5075. `Tests/Chat/test_console_trace_service.py:748` contains `revision_only` inside a test NAME, not a call.
- Recommended correction: delete all three.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D3] — broad `except Exception` wrapping first-party imports (masks ImportError; the import failure degrades silently)
- Where: `console_provider_gateway.py:3409-3417` (`model_capabilities` → `logger.debug`), `console_runtime.py:492-497` (`load_settings` → stale snapshot, NO log), `:541-554` (`Chat_Dictionary_Lib` → `return text`), `:581-611` (`world_info_processor` → `return text`), `console_agent_bridge.py:4922-4990` (`Agents.canvas_tool_provider` inside the personal-context budget try → `ProfileContextSnapshot.empty()`), `:6230-6242` and `:6343-6370` (`Agents.automatic_work_runtime` → `ToolResult(ok=False, error=str(exc))` to the model)
- Evidence: `fbimports.py` → every target resolves today, so the guards are inert for imports and only widen the net around the call; no test pins any of them.
- Why it matters: a broken internal import turns into "no capabilities / stale config / no dictionary / no world-info / no personal context / tool error" with at most a debug line.
- Recommended correction: move each `import` above its `try`; keep the `except` for the call.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D3] — redundant function-body imports (target already bound at module scope, or already loaded before the importing module finishes); zero import cycles in all three files; the bridge's justification comment is false
- Where (module-scope duplicate): gateway `:583` (`custom_endpoint_registry`, at top :140-146), `:3779` (`local_reasoning`, at top :107-115); runtime `:493, :615, :1029, :2456, :3285` (`tldw_chatbook.config`, at top :159), `:3194` (`console_chat_models`, at top :132); bridge `:541, :581, :889, :905, :3006` (`tldw_chatbook.config`, at top :228-234 — and `:474-478` claims "this module has NO top-level dependency on `tldw_chatbook.config` (every config read in it is a function-local import)", which is false), `:577` (`Agents.agent_models`, at top :51), `:3003-3004` (`math`, `os`; `os` at top :17).
- Where (already in `sys.modules` before the file finishes importing; deferral buys nothing): gateway `Chat_Functions ×3 (:6463,:6505,:6519)`, `Utils.token_counter ×2 (:3426,:3738)`, `model_capabilities (:3410)`; runtime `Backup_Recovery.bootstrap ×3`, `Canvas.models`, `Character_Chat ×2`, `console_roleplay_identity`, `console_session_settings ×2`, `message_metadata`, `DB.ChaChaNotes_DB`, `Library.local_library_tool_service`, `Workspaces.models`, `collections.abc`; bridge `Agents.run_context`, `Agents.run_log ×3`, `Agents.run_log_search`, `console_scratch_space`, `runtime_policy.types ×2`.
- Evidence: `fbimports.py` → runtime 71 stmts/55 modules, gateway 29/12, bridge 51/24; **`cycle` column False for every row**; `loaded@import True` for the rows listed. Trace service (7 stmts): settlement↔service is the one genuine mutual dependency (`console_trace_settlement.py:201` imports the service in a function body); `console_provider_gateway`, `console_voice_trace_gateway`, `console_voice_trace_promotion` do NOT pull `console_trace_service` at import (subprocess probes), so trace_service :1683 and :6277 are deferrals, not cycle-breakers. The remaining ~90 statements across the three files target modules NOT yet loaded (console_chat_controller, console_chat_store, Canvas.*, voice.*, Agents.fleet_messages, Skills_Interop.*, UI.Screens.change_review_screen, console_context_window, console_send_diagnostics …) — those are the `097-boot-budget-ratchets.md` deferrals: **out of scope by ADR 097-boot-budget-ratchets.md**.
- Why it matters: each redundant statement is a per-call `IMPORT_NAME` and a false "must be a cycle" signal; the bridge comment already misleads.
- Recommended correction: delete the module-scope duplicates; fix bridge :474-478; leave the ADR-097 deferrals.
- Size: S · ADR: partly (097-boot-budget-ratchets.md covers the kept deferrals) · Confidence: verified
- Pinning test: `Tests/Chat/test_console_agent_bridge.py::test_bridge_default_budget_matches_config_defaults` pins the literal mirroring the false comment justifies (passes either way)
- Already covered: none

### P3 [D3] — private helpers imported across modules/packages
- Where: `console_provider_gateway.py:1296` (`Agents.agent_service._budget_weighted_tokens`), `:3779` (`.local_reasoning._LOCAL_FAMILIES`), `console_trace_service.py:6277` (`_DISPLAYABLE_THINKING_EXECUTION_KEYS`, `_HOSTED_THINKING_FINISH_POLICIES`, `_thinking_protocol` from the gateway), `console_agent_bridge.py:8569-8595` (`agent_service_module._coerce_max_live_subagents/_setting/_coerce_retained_transcripts/...` — documented as deliberate for monkeypatching at :8565-8568)
- Evidence: read; `fbimports.py` shows no cycle forces the function-body form for the first three.
- Recommended correction: promote the gateway/trace names to public (a `console_thinking_capability` leaf); leave the bridge's documented module-read.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D4b] — seven "UTC now" re-rolls in `Chat/` produce three incompatible timestamp formats; the trace repository's parser assumes one of them
- Where: (1) `isoformat().replace("+00:00","Z")` — variable precision (no fraction when `microsecond == 0`): `console_trace_runtime.py:48`, `console_runtime.py:341`, `chat_conversation_service.py:322`, `conversation_local_marks_service.py:95`; (2) `isoformat(timespec="microseconds").replace("+00:00","Z")`: `console_voice_trace_gateway.py:343`; (3) `isoformat()` offset form `+00:00`: `console_chat_store.py:1361`, `console_provider_gateway.py:1207` (`ExchangeCapture.created_at`)
- Evidence: `rg -n 'isoformat\(\)\.replace\("\+00:00", "Z"\)' tldw_chatbook/Chat/`; `rg -n "def (utc_now|now_utc|iso_now|…)" tldw_chatbook/Utils/ tldw_chatbook/DB/base_db.py` → **no canonical helper exists**. Consumer: `console_trace_repository.py:553-555 datetime.fromisoformat(previous_settled_at[:-1] + "+00:00")` strips the last character unconditionally (assumes form 1/2). `rg -n "ORDER BY[^;]*(dispatch_started_at|settled_at|response_started_at|provider_inactive_at)"` → none, so the variable precision is not currently load-bearing.
- Why it matters: forms 1/2 and 3 reach different tables today (trace ledger vs sessions/captures) — not a D1 yet — but a future form-3 writer on a trace column hits `ValueError` in the `[:-1]` parser, and form 1 breaks lexical ordering once in 10⁶.
- Recommended correction: one `utc_now_iso()` in `tldw_chatbook/Utils/` (fixed `timespec="microseconds"`, `Z`), used by all seven; replace the `[:-1]` parse with plain `fromisoformat` (3.11+ accepts `Z`).
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D4b] — `_mapping_value` verbatim ×2 plus a shape clone; no shared helper
- Where: `console_provider_gateway.py:7036`, `console_session_settings.py:1949` (byte-identical), `UI/Screens/chat_screen.py:7879 _config_section` (same body, `dict` for `Mapping`)
- Evidence: `sed` of the three bodies; `rg -n "def (_?mapping_value|mapping_section|config_section|…)" tldw_chatbook/Utils/ tldw_chatbook/config.py` → nothing canonical.
- Recommended correction: `config_section(mapping, key) -> Mapping` beside `coerce_bool_setting` in `tldw_chatbook/config.py` (already owns section coercion); three call sites swap.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D4b] — three lazy-delegate wrappers with an identical double-checked `_get_delegate`
- Where: `console_runtime.py:215-245 _LazyTraceCompatibilityMetrics`, `:248-276 _LazyConsoleActivityReceiptService`, `:279-309 _LazyTraceBoundaryFactory`
- Evidence: read; bodies differ only in the deferred import and constructor args.
- Recommended correction: one `_lazy(factory)` helper (lock + cached delegate).
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

## Candidate dispositions
| candidate (file:line pattern) | confirmed / retired (why) / unverified (check) |
|---|---|
| dup_shape `_require_ledger` bridge:2827 + 6 siblings (voice_attempts:319, chat_conversation_scope_service:105, hosted_chat:193, message.py:555/561/567) | **retired** — a 3-line require-or-raise guard with per-site exception class and message (`RuntimeError` vs `ValueError`, different copy); a shared helper would not be shorter than the inline form (`sed -n 2827,2831p; 319,324p; 105,110p`). |
| dup_shape/dup_verbatim `_thaw` ×3 + `_thaw_auxiliary_value` gateway:1745 + `thaw_json` | **confirmed** — P2 [D4a] (plus the drifted runtime:586 copy) |
| dup_shape/dup_verbatim `_mapping_value` gateway:7036 / session_settings:1949 / chat_screen:7879 | **confirmed** — P3 [D4b] |
| except_exception_pass runtime:4127, :4881 | **retired** — `setattr(view, "_console_runtime_attachment_generation", …)` on read-only view doubles, documented `# noqa: BLE001 -- read-only view doubles are supported` |
| except_exception_pass runtime:4421 | **confirmed** — P3 [D1] (with :4610 twin) |
| except_exception_return bridge (14) | **retired** — :545/:601/:607/:614/:913 config fallbacks documented "config must never break a run"; :2722 conservative `False` (payload "does not fit"); :6213/:6217/:6242/:6286/:6292/:6370/:10606 return `ToolResult(ok=False, error=…)` — the error is delivered to the model, not swallowed; :9606 stale run-log authority fails closed to `""` (documented). |
| except_exception_return gateway (1) | **retired** — :1434 `usage_json = None` inside `_flight_capture` (capture diagnostics, never-break-send). |
| except_exception_return runtime (5) | **retired** — :497 stale-snapshot fallback (flagged for silence under P3 [D3]), :554/:611 optional context never blocks send, :1342/:1379 `_terminal_receipt_outcome` returns None + debug log, retryable by design. |
| except_exception_return trace_service (4) | **retired** — :413/:463/:497 documented best-effort boundary methods ("cannot break a send"); the coordinator's `_settle_claimed` (settlement:313-330) re-enqueues on `Exception` and `_prepare` (settlement:403-470) degrades to `response_canonicalization_unavailable` rather than raising; :1955 returns `"unknown"` which becomes `_TraceDispatchWriteError` → `TraceCallPersistenceError` (not swallowed). |
| fetchall_no_limit trace_service:3279 | **retired** — actual statement (:3279-3284) is `SELECT sequence … ORDER BY sequence LIMIT 2`; the excerpt's query text belongs to :3221 (a `.fetchone()`). |
| fetchall_no_limit trace_service:3441 | **retired** — `… call_id IN (?, ?) ORDER BY sequence LIMIT 3` (:3441-3446). |
| fetchall_no_limit trace_service:3504 | **retired** — keyed by one `call_id`; `len(pin) != 1` raises on the next line (:3509-3512). Bounded by construction. |
| fetchall_no_limit trace_service:3643 | **retired** — `… BETWEEN ? AND ? ORDER BY n.sequence LIMIT 257` (:3643-3654). |
| fetchall_no_limit trace_service:5170 | **retired** — `IN ({placeholders})` over `artifact_keys[offset:offset+256]` (:5167-5168). |
| function_body_import trace_service (7) | **classified** — :1534/:1571/:1603/:1634 settlement ↔ service genuine cycle (settlement:201); :1683-1691 voice_trace_gateway/promotion and :6277 gateway are deferrals (probes: none pulls trace_service at import); :6277 also imports three PRIVATE names (P3 [D3]). |
| function_body_import gateway (27 → 29 by AST) | **classified** — 0 cycles; 8 redundant (P3 [D3]); 21 ADR-097 deferrals (voice_trace_gateway ×8, console_context_window ×3, send_diagnostics ×3, settlement ×2, Agents.* ×5). |
| function_body_import runtime (58 → 71 by AST) | **classified** — 0 cycles; 6 `config` + 1 `console_chat_models` module-scope duplicates; 12 already-loaded; ~50 ADR-097 deferrals. |
| function_body_import bridge (38 → 51 by AST) | **classified** — 0 cycles; 5 `config` + `agent_models` + `os`/`math` module-scope duplicates (comment :474-478 false); 9 already-loaded; ~30 ADR-097 deferrals (`stream_stall_watchdog` ×5, `fleet_messages` ×6, `Skills_Interop.*`, `UI.Screens.change_review_screen` …). |
| id_keyed_dict gateway:2734 `_provisional_voice_traces[id(attempt)]` | **retired** — `_ProvisionalVoiceTraceRecord.attempt` (:251) holds the key object; every read re-checks `record.attempt is not attempt` (:2751, :2776, :2798, :2822); every pop is `pop(id(attempt), None)` under the RLock (:2759, :2784, :2806, :2808, :2821). |
| id_keyed_dict trace_service:2369 `_parent_capabilities[id(child_state.parent)]` | **retired** — Verified-fine §A (id reuse impossible; KeyError structurally excluded) |
| id_keyed_dict trace_service:2716 `_parent_capabilities[id(checkpoint)]` | **retired** — §A |
| id_keyed_dict trace_service:4344 `_prepared_capabilities[id(projected)]` | **retired** — `_PreparedState.provenance` holds `projected` (§A) |
| id_keyed_dict trace_service:4472 `_parent_capabilities[id(prepared.parent)]` | **retired** — §A |
| id_keyed_dict trace_service:4499 / :4526 `_prepared_capabilities[id(provenance)] = replace(prepared, …)` | **retired** — in-place replace of an existing entry; `replace()` keeps `provenance` (§A) |
| id_keyed_dict trace_service:4607 `_child_capabilities[id(binding)]` | **retired** — `_ChildState.binding` holds `binding` (§A) |
| id_keyed_dict trace_service:4719 `_parent_capabilities[id(capability)]` | **retired** — `_ParentState.capability` holds `capability` (§A) |
| legacy_markers bridge (8) / gateway (6) / runtime (7) / trace_service (1) | not examined by line (no line numbers supplied; mechanical only) |
| try_import_guard bridge:540 `console_fallback_providers`, :576 `console_run_budget`, :904 `_console_tool_result_display_cap` | **retired (as import guards)** — `tldw_chatbook.config` is a hard module-scope dependency (:228-234), so the ImportError branch can never fire; the remaining `except Exception` guards `get_cli_setting` and is documented ("config must never break a run"). Listed under P3 [D3] redundant-imports for the dead import statements. |
| try_import_guard bridge:4922 `build_console_first_request_plan`, :6230 `install_skill_tool`, :6343 `run_skill_script_tool` | **confirmed** — P3 [D3] broad-except-around-internal-import |
| try_import_guard gateway:3409 `prepare_chat_request` | **confirmed** — P3 [D3] |
| try_import_guard runtime:492, :541, :581 | **confirmed** — P3 [D3] |
| try_import_guard runtime:2455 `_record_successful_first_send` | **retired** — `config` at module scope; the guard is for `save_setting_to_cli_config` I/O and logs a warning |
| try_import_guard runtime:3277 `run` handlers=['ImportError','Exception'] | **confirmed** — P2 [D3] |

## Verified-fine

### §A — the seven `id()`-keyed maps in `ConsoleTraceService` cannot alias a recycled CPython id
Every map value holds a strong reference to the object whose id is the key, so the key object is alive for exactly the entry's lifetime and no second live object can share its id:
- `_parent_capabilities[id(capability)] = _ParentState(capability, …)` (:4719 → field `capability` :1389)
- `_prepared_capabilities[id(projected)] = _PreparedState(projected, …)` (:4344 → field `provenance` :1425); :4499/:4526 are `replace(prepared, …)` on an existing entry and keep that field
- `_child_capabilities[id(binding)] = _ChildState(binding, …)` (:4607 → field `binding` :1440)
- `_pending_child_uses[id(child.binding)] = transaction_token` (:4661) does NOT hold the binding, but every pop of `_child_capabilities` also pops `_pending_child_uses` (:1877-1878, :2000-2001, :2830-2831, :4666-4668) and the entry is only created for a currently-registered child (:4652 reached from :2417 after `_validated_child_binding` :2022 succeeded), so the sibling map holds the binding for the whole lifetime.
Identity re-checks exist on every `.get(id(x))` read (:1318, :4008, :4404, :4433, :4520, :4549, :4561, :4637).
Unguarded `[]` reads: :2369 — `child_state` validated at :2022 and `_prune_unreferenced_parents` (:4681) retains `id(child.parent)` for every registered child; the segment sweeps at :1994-2014 can only pop a parent with `head_id != expected_predecessor`, but `child.head_id` is a copy of `parent.head_id` (:4591) and :2022 raises unless `child.head_id == expected_predecessor`. :2716 — non-None only if `current_surface_checkpoint` (:2454) found the parent, because `expected_route=route_identity` (a `str`) is always passed. :4472 — `prepared.parent` retained by :4684; the only other parent pops (:2002, :2014) are immediately followed by :2015-2017, which drops every prepared whose parent is gone, in the same synchronous call.
Evidence: `rg -n "_parent_capabilities\[|…\.pop" console_trace_service.py` (writes :4344, :4499, :4526, :4607, :4661, :4719; pops :1877-1878, :2000-2002, :2014, :2017, :2827, :2830-2831, :4506, :4608, :4666-4668, :4678, :4693). `Tests/Chat/test_console_trace_service.py:2607, :2715, :2788, :2876` poke these maps directly (pin the shapes, not id reuse). `_surface_ref_cache` (:1529) is unbounded per segment with a `ponytail:` comment — deliberate.

### §B — trace settlement swallow paths (item 6)
`ConsoleTraceCallBoundary.settle_response`/`mark_response_started`/`prepare_response_settlement` catch `Exception` and return False/None by documented contract. The coordinator's `submit` → `_submit_prepared` → `_settle_claimed` (settlement:235-330) enqueues a content-safe retry on `Exception` and `retry_pending` drains it; `_prepare_safely`/`_prepare` (:403-470) degrade malformed envelopes to `response_canonicalization_unavailable` rather than raising. Gateway `_settle_trace_response` (:2068-2069, :2084-2085) logs `type(exc).__name__` at WARNING and falls through to the coordinator. Not a silent drop. Residual (P3 [D2] above): the fallback runs on the calling loop.

### §C — agent-bridge → AgentService hand-off (item 3)
`console_chat_controller.py:26933` passes `self._agent_bridge.run_reply` to `_run_maintenance_agent_call`, which is `asyncio.to_thread(worker, **kwargs)` + `asyncio.shield` (:5595-5603) — the whole `run_reply` (AgentService construction, `run_turn`, all `self._db.*` AgentRunsDB calls, `self._store.*` appends) is on a worker thread. `rg -n "run_worker\(" console_agent_bridge.py` → none (only docstring mentions). Model calls are submitted to a per-run `_ModelCallLifeline` loop on its own thread (`run_coroutine_threadsafe`, :3849); fleet children own their own lifelines (:3261-3320). The two preview builders that call `asyncio.run` internally (`build_project_instruction_preview_request` :5486, `build_personal_context_preview_snapshot` :5620) are invoked under `asyncio.to_thread` (controller :20424, :20613). Every other `asyncio.run` in the bridge (:5063, :5909, :6140, :6163, :6238, :6288, :6353, :6362) sits in tool closures executed on the tool's per-call daemon thread or in `run_reply` itself. The cross-thread dicts (`_live`, `_historical_cache`, `_fleet_services`, `_live_primary_keys`, `_setup_started_at`) are unguarded by a documented GIL-atomicity decision (:5241-5282, :8994-8999) with an identity-checked teardown pinned by `test_fleet_teardown_pop_is_identity_checked_not_blind`.

### §D — credential handling (item 4)
Every API-key read in the gateway goes through `resolve_entry_credential` (ADR `146-console-custom-endpoint-registry.md`, `:583-585`) or `get_provider_readiness(...).api_key` (`:3720`, `:3959`, `:4210`), and `provider_readiness.py:37` binds `resolve_provider_api_key as _valid_api_key` (ADR `012-provider-credential-settings-boundary.md`). `rg -n "\.get\(\"api_key\"|\[\"api_key\"\]|getenv" console_provider_gateway.py` → none. `rg -n "logger\.(debug|info|warning|error|exception)" console_provider_gateway.py | rg -i "key|secret|token|credential"` → **none** (19 logger calls, all `type(exc).__name__` or fixed strings). Keys reach only `_authorization_headers` (:6910) and adapter kwargs; capture paths pass `known_credentials=(resolution.api_key,)` into `sanitize_capture_value_with_omission` (:5384, :5533, :6086). `console_run_budget` (bridge :633) reads `TLDW_AGENTS_DENIAL_CIRCUIT_BREAKER_LIMIT` from env — not a credential.

### §E — other smells that are not findings
- loguru + stdlib `logging` mix: `rg -n "^import logging|logging\.getLogger"` over the four files → none.
- `re.compile` in function bodies: none (bridge :1255/:1312/:1316/:2734 are module-scope).
- Mutable class attributes on `ConsoleRuntime`/`ConsoleAgentBridge`: none — every container is assigned in `__init__` (runtime :980-1117, bridge :5152-5424). `ConsoleProviderStreamSignals` is a `slots=True` dataclass with `default_factory` containers.
- Per-loop httpx clients (`_active_http_client` :3491-3564, `aclose` :3063) keyed in a `WeakKeyDictionary` under `_client_lock`; pinned by `Tests/Chat/test_console_provider_gateway.py::test_aclose_does_not_close_a_still_running_childs_client` (cited at :3127-3129).
- Provider work is off-loop: `_stream_generic_chat` runs `chat_api_call` in `asyncio.to_thread(worker)` (:6337); `complete_auxiliary` via `to_thread` (:4979); llama paths are native `httpx.AsyncClient`.
- `_ToolCallAccumulator` extra-key allow-list (:2311-2319, :2380-2382) and `_PRESERVED_FRAGMENT_EXTRAS` — deliberate (PR #662).
- `recover_open_calls` (settlement:364-374) `.fetchall()` without LIMIT is bounded by rows in non-terminal states (normally 0-1).
- `_LiveStepFeed` (bridge :2038) bounds the live step tail to 8 — the TASK-18604 fix, not a leak.
- bridge compile under `-W error::SyntaxWarning` → `no SyntaxWarning` (an earlier `<unknown>:32` warning during AST parsing was a stale artefact).

## Retired
- Mechanical `fetchall_no_limit` ×5 (trace_service) — the rows carried the wrong statement text; every real statement is LIMIT-bounded or keyed to one id (table above).
- Mechanical `id_keyed_dict` ×9 — all values hold their key objects (§A, gateway row).
- `_require_ledger` shape family — trivial idiom, no helper candidate.
- bridge try_import_guard :540/:576/:904 — the guarded import cannot fail (module-scope dependency); reclassified as redundant imports.
- Readiness inline cost as a P2 — measured 0.01 ms/call under the null keyring; kept as P3 inferred on the strength of commit 0077b872f1 only.
- `except_exception_return` rows in all four files — every site either logs, degrades by documented contract, or returns the error to the model (table above).
- "SyntaxWarning: invalid escape" in the bridge — not reproducible under `compile(..., -W error)`.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| first `ensure_chat_store` (recovery + media-reference retry) costs a measurable UI stall on a real ChaChaNotes DB | needs a populated user DB; the isolated profile is empty | `cd $WT && source $SCRATCH/env.sh && PYTHONPATH=$WT $PY - <<'EOF'`<br>`import time; from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB; from tldw_chatbook.Chat.console_runtime import recover_console_trace_calls`<br>`db = CharactersRAGDB("<copy of a real ChaChaNotes.db>", "review"); t=time.perf_counter(); recover_console_trace_calls(db); print("%.1f ms" % ((time.perf_counter()-t)*1000))`<br>`EOF` |
| `get_provider_readiness` on the llama path blocks the loop under a real keyring backend | isolated profile forces `PYTHON_KEYRING_BACKEND=null` (0.01 ms measured) | same timing loop as above with `PYTHON_KEYRING_BACKEND` unset and a real `~/.config/tldw_cli/config.toml` (do not run against the live profile from this worktree — ADR-126 `Recovery required`) |
| `settle_response` can drop a seal with no queue entry when `submit` raises before `_settle_claimed` | fault injection against a real DB needed; reading shows `_prepare_safely` degrades rather than raises | `cd $WT && source $SCRATCH/env.sh && $PY -m pytest Tests/Chat/test_console_trace_call_lifecycle.py Tests/Chat/test_console_trace_settlement.py -q` then inject `monkeypatch.setattr(coordinator, "_prepare_safely", raising)` in a copy of the lifecycle test |
| the `[:-1] + "+00:00"` parser at `console_trace_repository.py:553-555` ever sees a form-3 (`+00:00`) `settled_at` | no writer produces one today (all trace timestamps come from form 1/2 factories) | `rg -n "settled_at=" tldw_chatbook/ | rg -v "occurred_at|_utc_now"` — any hit that is not a `_utc_now()`/`occurred_at` source is a new writer to check |
