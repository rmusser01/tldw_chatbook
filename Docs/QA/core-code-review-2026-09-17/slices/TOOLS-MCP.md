# TOOLS-MCP — tldw_chatbook/Tools/ (25 files, 17,795 lines) + tldw_chatbook/MCP/ (32 files, 26,115 lines), 43,910 lines

Worktree: /Users/macbook-dev/Documents/GitHub/tldw-review @ origin/dev d8fb4053f9 (read-only). Report written incrementally (resilience rule) — a truncated tail means the run was cut.

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| Tools/tool_executor.py | 235 | read in full |
| Tools/_grep_worker.py | 181 | read in full |
| Tools/workspace_root_pin.py | 330 | read in full |
| Tools/local_tool_impls.py | 1092 | read in full |
| Tools/workspace_tool_worker.py | 162 | read in full |
| Tools/workspace_tool_dispatch.py | 256 | read in full |
| Tools/workspace_tool_executor.py | 801 | read in full |
| Tools/workspace_tool_protocol.py | 602 | read in full |
| Tools/patch_tool_impls.py | 506 | read in full |
| Tools/virtual_cli_impls.py | 278 | read in full |
| Tools/workspace_file_roots.py | 585 | read in full |
| Tools/file_operation_tools.py | 1811 | read in full |
| Tools/web_tool_impls.py | 2684 | read in full |
| Tools/raw_cli_executor.py | 1447 | sampled: symbol index + 820-870, 1070-1110, 1225-1260 (candidate rows) |
| Tools/git_tool_impls.py | 1360 | sampled: subprocess/allowlist/timeout grep (lines 11-25, 98-102, 173-185, 208-288) |
| Tools/watchlists_tool_service.py | 2216 | read in full |
| Tools/watchlists_command_service.py | 884 | mechanical only |
| Tools/code_audit_tool.py, document_expansion_tool.py, file_operation_hooks.py, note_management_tools.py, rag_search_tool.py, web_search_tool.py, __init__.py, repo2txt/__init__.py | 673+524+373+398+164+138+95+0 | mechanical only (candidate rows examined by line) |

(MCP coverage rows appended after the MCP pass — see "Coverage (MCP)" below.)

## Findings

### P1 [D1] — `CalculatorTool` (always-on built-in) evaluates model-supplied `**` and `str * int` with no bound: one call allocates a 200 MB string in 20 ms and bignum powers scale superlinearly (7**4e6 = 1.26 s, 9x the cost of 7**1e6)
- Where: `tldw_chatbook/Tools/tool_executor.py:181-220` (`allowed_operators` admits `ast.Pow`/`ast.Mult`; `ast.Constant` admits `str`, so `'ab' * 10**8` and `7**(4*10**6)` both evaluate)
- Evidence: `cd $WT && source env.sh && PYTHONPATH=$WT $PY - <<EOF ... asyncio.run(CalculatorTool().execute(expression="'ab' * 10**8")) ... EOF` → `str*int 200MB: 0.02s len=200000000 maxrss MB=227` (baseline 36 MB); `7**10**6: 0.14s bits=2807355`; `7**4e6: 1.26s bits=11229420`
- Why it matters: `tool_executor.py` docstring calls this one of "the two always-on built-in tools"; `risk_tags` is `()` so it inherits `allow` (`BUILTIN_DEFAULT_STATE`), and `Tool.timeout_seconds` documents that a timed-out worker THREAD is abandoned, not killed — so a prompt-injected `9**9**9` or `'a'*10**10` pins a core / exhausts RAM for the rest of the process lifetime with no user approval and no kill path.
- Recommended correction: in `safe_eval`, reject non-numeric `ast.Constant`; bound `Pow` (`abs(right) <= 1_000` and `left.bit_length()*right <= ~1<<20`) and refuse `Mult` on `str`; S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (`rg -n 'Pow|\*\*' Tests/Tools/test_tool_executor.py` → no hits)
- Already covered: none

### P1 [D1] — MCP permission store treats a transient read `OSError` as corruption: it renames the live `mcp_permissions.json` to `.bak` and resolves from fresh defaults — kill switch ON→OFF, a built-in set to Off resolves Allow
- Where: `tldw_chatbook/MCP/permission_store.py:747-757` (`except (OSError, ValueError, json.JSONDecodeError)` → `_backup_corrupt_file()` + `return _fresh_payload()`); same shape at `:892` (`_load_for_raw_getter`, no backup but same fresh-default fallback)
- Evidence: `cd $WT && source env.sh && PYTHONPATH=$WT $PY - <<EOF` (set kill_switch=True and `agent:builtin/calculator=deny`, then `p.chmod(0o000)` and `MCPPermissionStore(p).load()`) → `before: kill_switch= True calculator= deny` / `after chmod000 load(): kill_switch= False calculator= allow | store exists: False | .bak exists: True`; the WARNING logged says "unreadable/corrupt ([Errno 13] Permission denied ...); backing it up and resetting to defaults."
- Why it matters: EACCES/EIO/EMFILE/EINTR on read are not corruption, yet one such read moves the user's whole policy file aside; the verdict for that call is already the permissive default, and the next mutator (`_mutate_locked` → `load()` → `_save_locked`) persists the fresh payload, so every Off/Allow decision and the kill switch are lost for good. The module docstring's own contract ("uncertain native persistence failures propagate without resetting policy") is not honoured on the plain-file path. Trigger needs an I/O fault (wrong-owner file after a `sudo` run, fd exhaustion, a Windows sharing violation while the standalone `TldwMCPServer` — which opens the SAME path, `MCP/server.py:989` — writes it), so P1 not P0.
- Recommended correction: split `OSError` out of that except: re-raise (or return a deny-all payload) and only back up on `ValueError`/`JSONDecodeError`/shape mismatch, which is what spec §9 actually says ("unknown schema version -> back up"). S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: SEE "Left UNVERIFIED" (whether `Tests/MCP/test_permission_store*.py` pins the OSError branch as a requirement)
- Already covered: none

### P2 [D1] — `ReadFileTool` returns the ENTIRE file as `content` with no size cap, and both read families materialise the whole file before slicing
- Where: `tldw_chatbook/Tools/file_operation_tools.py:325-336` (`content = path.read_text(...)` → `{"content": content}`), `tldw_chatbook/Tools/local_tool_impls.py:345-356` (`text = target.read_text(...)` then `numbered[:MAX_READ_CHARS]`)
- Evidence: 150 MB single-line file: `ReadFileTool 150MB: 1.09s content_len=150000000 maxrss MB=396` (baseline 180); `local_tool_impls.read_file 150MB single-line: 2.81s returned=32782 chars maxrss MB=389`
- Why it matters: a model asking to read a build artefact or log in a bound workspace costs the app process (ReadFileTool runs in-process on the agent worker thread) file-size bytes of RSS and, for ReadFileTool, hands a 150 MB dict to `BuiltinToolProvider` before the runtime's 16,000-char result cap ever applies; the `fs_read` copy is bounded to the one-shot worker process but still pays 2.8 s per call.
- Recommended correction: `open(...).read(MAX_READ_BYTES + 1)` and stat-check before read in both families; S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found (`rg 'read_file.*(large|size|cap)' Tests/Tools` → no hits)
- Already covered: none

### P2 [D1] — `fs_edit`, `fs_patch` and the legacy `write_file` write in place (truncate-then-write, no fsync, symlink-following) while `fs_write` in the same module is O_EXCL-temp + fsync + `os.replace` via a pinned `dir_fd`
- Where: `tldw_chatbook/Tools/local_tool_impls.py:856` (`target.write_bytes(data)` in `_edit_relative_file`), `tldw_chatbook/Tools/patch_tool_impls.py:509` (`target.write_bytes(data)` in `_patch_relative_file`), `tldw_chatbook/Tools/file_operation_tools.py:744-751` (`open(path, "w"/"a")`); contrast `local_tool_impls.py:577-677` (`_atomic_write_target`)
- Evidence: `rg -n 'write_bytes\(|open\(path, "[wa]"' tldw_chatbook/Tools/` → the three sites above; `local_tool_impls.py:627,656` show the sibling's `os.fsync(temp_fd)` / `os.fsync(parent_fd)`
- Why it matters: the pinned worker is killed by its parent at the 300 s deadline (`workspace_tool_executor.py:215-227` → `_settle_after_spawn`) and `fs_edit`/`fs_patch` run inside it; a kill or crash between `O_TRUNC` and the write's completion leaves the user's file empty/partial with no recovery, and the edit path re-opens by name after `_relative_target_is_safe` (TOCTOU on a swapped symlink) where `fs_write` deliberately does not.
- Recommended correction: route `_edit_relative_file`/`_patch_relative_file` through `_atomic_write_target` (already in the module; pass `expected_sha256=None`); M (has to choose whether `fs_edit` keeps mode bits / CRLF — `_atomic_write_target` already `fchmod`s the live mode).
- Size: M · ADR: no · Confidence: verified (unconditional code path; the crash window itself was not reproduced)
- Pinning test: `Tests/Tools/test_local_tool_impls.py::test_fs_edit_unencodable_new_string_preserves_file` pins encode-before-open only; nothing pins in-place vs atomic
- Already covered: none

### P2 [D2] — `ListDirectoryTool` neither clamps `max_depth` (its own schema says `maximum: 5`) nor caps the scan; the 100-entry cap is applied after the full recursive walk
- Where: `tldw_chatbook/Tools/file_operation_tools.py:413` (`max_depth = kwargs.get("max_depth", 2)` unclamped), `:469-520` (recursive walk stat()s every entry), `:550` (`entries[:100]` after the fact); contrast `local_tool_impls.py:78,272-275` (`MAX_SCAN_ENTRIES = 10_000`)
- Evidence: probe `ListDirectoryTool().execute(directory_path=d, recursive=True, max_depth=50)` → `max_depth=50 accepted -> depths seen: [0..12] total_entries 24`
- Why it matters: a model can request `recursive=True, max_depth=50` on a bound workspace root and the tool `stat()`s and appends every entry of a node_modules/build tree in-process before truncating to 100 — the exact failure `MAX_SCAN_ENTRIES` was added to the sibling family for.
- Recommended correction: `max_depth = max(1, min(int(max_depth), 5))` and a scan counter that stops at `MAX_SCAN_ENTRIES` with a truncation notice; S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Tools/test_local_tool_impls.py::test_list_directory_caps_the_scan` pins the SIBLING; none for this class
- Already covered: none

### P2 [D1] — `MCP/tools.py::chat_with_character` is a shipped, advertised MCP tool that can never answer: it calls a module-level stub that unconditionally raises, while the unified dispatcher it says does not exist (`Chat_Functions.chat_api_call`) is imported 20 lines away in `server.py`
- Where: `tldw_chatbook/MCP/tools.py:47-58` (`save_conversation_from_messages`/`chat_with_provider` stubs raise `NotImplementedError`), `:135-143` (the only call site), `:32-44` (comment claims "no unified dispatcher ... verified: neither name exists"); contrast `tldw_chatbook/MCP/server.py:588` (`from ..Chat.Chat_Functions import chat_api_call`) and `:619-628` (`chat_with_llm` uses it); the tool is registered at `server.py:640-659`
- Evidence: `rg -n 'def chat_with_provider|raise NotImplementedError|chat_api_call' tldw_chatbook/MCP/tools.py tldw_chatbook/MCP/server.py` → stub at tools.py:54/55; `chat_api_call` imported+used only in server.py:588/620. Unconditional path: with a key the call raises → `except Exception` → `{"error": ...}`; without a key `{"error": "No API key configured"}` — no branch produces a chat response.
- Why it matters: every external MCP client that lists tools sees `chat_with_character` and gets a "dead upstream reference" error on every call; `MCPTools` also constructs `SimplifiedRAGSearchService` etc. for a tool that cannot work, and the stale comment misleads the next maintainer into believing the dispatcher is gone.
- Recommended correction: route through `chat_api_call` exactly as `chat_with_llm` does (or drop the registration); delete the two stubs and the stale comment; M (persistence of the new conversation needs a real `ChaChaNotes_DB` call — `add_conversation`/`add_message` exist).
- Size: M · ADR: no · Confidence: verified (unconditional stub; not executed)
- Pinning test: `Tests/MCP/test_tools_resources_prompts_real_methods.py::test_chat_with_character_uses_the_declared_api_key_accessor` pins only the accessor (AST), not the outcome — nothing states "always errors" as a requirement
- Already covered: none

### P2 [D2] — Every local-branch `UnifiedMCPControlPlaneService` call runs its `LocalMCPControlService` work synchronously on the event loop: `_maybe_await(sync_call())` evaluates the call before awaiting, and the section getters re-parse `server.py` with `ast` three times per call (6.3 ms measured) plus a full JSON-store load; mutations add a JSON temp-write+replace
- Where: `tldw_chatbook/MCP/unified_control_plane_service.py:568-571` (`_maybe_await`), `:551-565` (`load_section` local branch → `self.local_service.get_overview()/get_inventory()/...`), `:1293-1467` (`run_action` local branch: `save_external_profile`, `delete_external_profile`, `save_governance_rule`, ... all sync store writes), `:2326-2350` (`_record_local_attempt` → `store.save_profile_runtime_state` = full load + full save), `:2400-2422` (`save_local_profile`/`delete_local_profile`/`local_external_catalog` call the sync store directly); `tldw_chatbook/MCP/local_control_service.py:200-230` (`get_overview` → `get_inventory` → `manifest_provider()`), `tldw_chatbook/MCP/server.py:122-123,216-261` (`_load_server_module_ast()` = `Path(__file__).read_text()` + `ast.parse` per `_extract_registered_entries` call, ×3 per manifest, no memo)
- Evidence: `rg -c '^    async def ' MCP/local_control_service.py` → 13 vs `rg -c '^    def '` → 38 (the five section getters are sync); `PYTHONPATH=$WT $PY - <<EOF ... describe_local_mcp_capabilities() ... EOF` → `first 7.6 ms, steady 6.3 ms/call, tools=30`; `rg -n 'manifest_provider\(\)'` → 2 consumers (`local_control_service.py:224`, `local_runtime_delegate.py:673`, the latter on every `tools/list`)
- Why it matters: this service is the Hub screen's data source on the Textual loop; every section render blocks the loop for the JSON load + 3× AST parse, every profile save/delete/governance edit blocks it for a temp-file write + `replace`, and `_record_local_attempt` does that write twice per connect/test/refresh. The reference shape for the fix already exists in the tree (`Chat/chat_conversation_scope_service.py:152-215`, task-283: `if not inspect.iscoroutinefunction(fn): await asyncio.to_thread(fn, ...)`). Trace requested by the brief: `load_section("inventory")` → `local_service.get_inventory()` (sync) → `manifest_provider()` → `describe_local_mcp_capabilities()` → `_load_server_module_ast()` ×3; `load_section("external_servers")` → `get_external_servers()` → `store.get_external_catalog()` → `LocalMCPStore.load()` → `json.load(...)`. No sqlite on these paths (the only MCP sqlite is `tools.py`'s standalone-server calls; the in-process Library tools go through `asyncio.to_thread` in `local_runtime_delegate`).
- Recommended correction: (a) memoize `_load_server_module_ast()` on `(path, st_mtime_ns)` — S, removes 6.3 ms per manifest; (b) in `_maybe_await`'s callers, offload sync `local_service` methods with the task-283 shape (`to_thread` when not a coroutine function) — M.
- Size: S (a) / M (b) · ADR: no · Confidence: verified (cost measured; loop placement by reading)
- Pinning test: `Tests/MCP/test_local_control_service.py::test_local_control_service_uses_real_local_manifest_helper_by_default` pins that the real helper is used (would still pass with a memo); nothing pins the per-call re-parse
- Already covered: none

### P2 [D4b] — `_datetime_to_iso` is 4 verbatim copies inside MCP (7 counting `_iso_utc_now`, `_format_utc`, `runtime_policy/source_state.py:133`), the `replace("+00:00","Z")` idiom is re-rolled 44× in 38 files repo-wide with no Utils helper, and THREE timestamp formats reach the MCP stores/wire
- Where (copies): `MCP/unified_context_store.py:78`, `MCP/server_target_store.py:372`, `MCP/local_store.py:140`, `MCP/unified_control_models.py:15` (byte-identical), `MCP/permission_store.py:224` (`_iso_utc_now`), `Tools/watchlists_tool_service.py:2203` (`_format_utc`), `runtime_policy/source_state.py:133`
- Where (drift reaching storage/wire): `MCP/unified_control_plane_service.py:2332` writes `datetime.now(timezone.utc).isoformat()` (`+00:00` form) into `profile_runtime_state.last_attempt_at/last_ok_at` of `local_mcp_store.json`, stored raw (`local_store.py:933 dict(record)`) beside the store's own `Z`-form `updated_at`; `MCP/execution_log.py:144` writes `+00:00` form to `mcp_execution_log.jsonl`; `MCP/client.py:1253` writes NAIVE LOCAL `datetime.now().isoformat()` as `connected_at`, which `local_control_service._describe_profile` (`:894`) persists via `save_discovery_snapshot` and `describe_server` (`client.py:1612`) returns to callers; `MCP/server.py:731` returns naive local `created` to MCP clients; `local_control_service.py:1099` writes `+00:00` form but round-trips through `_iso_to_datetime`/`_datetime_to_iso` so it self-heals to `Z`.
- Evidence: `rg -n 'replace\("\+00:00", ?"Z"\)' tldw_chatbook -g '*.py' -g '!Third_Party/**' | wc -l` → 44 (38 files); `rg -n '^def .*(iso|utc_now|timestamp).*\(' tldw_chatbook/Utils/*.py` → no matches; format census above from `rg -n 'isoformat\(|_datetime_to_iso\(|_iso_utc_now\(' tldw_chatbook/MCP/*.py`
- Why it matters: `server_target_store` already imports `unified_control_models` and still carries its own copy; a consumer comparing/sorting `last_attempt_at` against `updated_at` as strings compares `+00:00` against `Z`, and `connected_at` carries no zone at all (wrong by the host's UTC offset when read on another machine or after a DST change).
- Recommended correction: one `Utils/time_format.py::utc_iso_z(dt: datetime | None) -> str | None` (new; the four MCP bodies are already identical) and use it at the three drifting write sites; S per site, the census is a follow-up sweep.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found for the format of `last_attempt_at`/`connected_at`
- Already covered: none

### P2 [D4b] — strict-JSON loading (`object_pairs_hook` duplicate-key reject + `parse_constant` NaN/Infinity reject) is re-rolled with 15 `_reject_*constant` functions and 40 `parse_constant=` call sites repo-wide, two of them in this slice
- Where (this slice): `tldw_chatbook/MCP/permission_store.py:388-399` (`_reject_duplicate_keys` + `_reject_json_constant`), `tldw_chatbook/Tools/workspace_tool_protocol.py:407-417` (`_reject_duplicate_keys` + `_reject_non_finite`), `tldw_chatbook/Tools/watchlists_tool_service.py:1257` (`_unique_json_object`, no constant hook); repo census: `Petdex/sources.py:68`, `Actor_Packs/export.py:718`, `LLM_Calls/hosted_chat.py:633`, `LLM_Calls/qwencloud.py:301`, `LLM_Calls/qwencloud_streaming.py:61`, `Persona_Visual/{validation:336,repository:1368,publication:1306,importer:1090}`, `DB/private_sqlite_protocol.py:351`, `Audio/voice_process_protocol.py:556`, `Character_Chat/visual_identity.py:1658`, `TTS/audio_cpp_contract.py:87`
- Evidence: `rg -n 'parse_constant=' tldw_chatbook -g '*.py' -g '!Third_Party/**' | wc -l` → 40; `rg -n 'def _reject_(json_constant|non_finite|nan|constant)'` → 15 defs; `rg -ln 'object_pairs_hook' tldw_chatbook/Utils` → none
- Why it matters: each copy raises a DIFFERENT exception type (`PermissionStoreSnapshotError("invalid_json")`, `WorkspaceProtocolError`, `ValueError`), so the same malformed input is classified differently per boundary, and `watchlists_tool_service` rejects duplicate keys but still accepts `NaN` cursors (`json.loads` default) — the drift the hook exists to prevent.
- Recommended correction: `Utils/strict_json.py::loads_strict(raw, *, error: type[Exception])` wrapping `json.loads(..., object_pairs_hook=..., parse_constant=...)`; S per site.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none for the shared shape
- Already covered: none

### P2 [D2] — `ServerUnifiedMCPService` re-runs the full access-context resolution (1 status probe + 1 bootstrap + 7 endpoint probes = 9 sequential client round-trips) and a sync JSON store write before EVERY server mutation and several plain reads — 65 call sites
- Where: `tldw_chatbook/MCP/server_unified_service.py:2823-2838` (`_revalidate_mutation_scope` → `resolve_access_context`), `:64-130` (`resolve_access_context`: `client.get_status()`, `_bootstrap_access_context`, `_probe_section_capabilities`), `:2852-2896` (seven awaited probes), `:111,3192-3216` (`_persist_target_status` → `target_store.update_target_status` → `mcp_sources.write_json`, synchronous file write on the event loop); reads that pay it too: `get_governance_pack_detail:2161`, `list_governance_pack_upgrade_history:2190`, `list_workspace_set_members:2567`
- Evidence: `rg -c '_revalidate_mutation_scope\(' tldw_chatbook/MCP/server_unified_service.py` → 65; the probe count is the seven `await _probe(...)` lines at `:2877-2885`
- Why it matters: one Hub click on a remote target costs 10 HTTP calls before its own request, serially; on a slow link each mutation stalls for seconds and the target-status JSON rewrite blocks the Textual loop each time. `_browse_cache` never short-circuits this path.
- Recommended correction: cache the resolved context per `(server_id, scope, scope_ref)` with a short TTL (or reuse the last `resolve_access_context` result when the scope selection is unchanged) and move `_persist_target_status` off-loop; M.
- Size: M · ADR: no (no ADR title matches `revalidat|mutation scope|unified mcp`) · Confidence: verified (unconditional code path; network cost not measured)
- Pinning test: `Tests/MCP/test_control_plane_permissions.py::test_session_approval_revalidates_profile_digest_under_fence` pins the PERMISSION revalidation, not this scope re-resolution
- Already covered: none

### P3 [D4a] — the `[:200] + "..."` note-preview truncation is inlined 3× in MCP while `Utils/Utils.py:253 truncate_content(content, max_length=200)` exists
- Where: `MCP/server.py:762-764`, `MCP/tools.py:252-255`, `MCP/local_runtime_delegate.py:635-637` (identical shape, identical 200); `Tools/code_audit_tool.py:286,429,487,531` use `[:100] + "..."` (same shape, different budget)
- Evidence: `rg -n '^def (truncate|_truncate|preview|truncate_text|clip_text|elide)' tldw_chatbook/Utils/*.py` → `Utils/Utils.py:253 truncate_content`; `rg -ln 'truncate_content' tldw_chatbook -g '*.py' -g '!Third_Party/**' | wc -l` → 1 (only its own module — a dead shared helper with 7 re-rolls in this slice alone; D3 "dead shared helper" applies too)
- Recommended correction: import `truncate_content`; S.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D4b] — `_identity()` (pid, thread, current task) is byte-identical in `MCP/activation.py:25`, `Agents/activation.py:24`, `RAG_Search/activation.py:141`; `_require_field`/`_require_payload_field` are identical in `MCP/unified_control_plane_service.py:2309` and `MCP/server_unified_service.py:3144`; `_text_or_none`/`_normalize_optional_text`/`_clean_text`/`_optional_text` (`str(v).strip() or None`) ×5 with `server_target_store.py:391` re-rolling a helper from a module it already imports
- Evidence: `diff <(sed -n 25,31p MCP/activation.py) <(sed -n 24,30p Agents/activation.py)` → identical; same vs RAG_Search → identical
- Recommended correction: one `Backup_Recovery`/`Utils` `activation_identity()`; one module-level `require_field` in `unified_control_models.py`; `server_target_store` imports `_text_or_none`. S.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D4b] — `web_tool_impls._format_size` is one of 15 human-readable-size formatters (no shared helper); the SSRF classifier `_is_public_ip`/`validate_outbound_url` is a second implementation beside `Utils/egress.py` (47 importers), which `egress.is_public_http_url`'s own docstring acknowledges
- Where: `Tools/web_tool_impls.py:463` (+14 copies listed by `rg -n 'def _?(format_size|format_bytes|human_size|human_readable_size|humanize_bytes|format_file_size|_fmt_size)\b' tldw_chatbook`); `Tools/web_tool_impls.py:77-130` vs `Utils/egress.py:127-206`
- Evidence: grep above (15 defs, `Widgets/Console/console_transcript.py:639` is the documented one); `rg -ln 'Utils\.egress|from .egress' tldw_chatbook` → 47; `backlog/tasks/task-609*` mentions neither `web_tool_impls` nor `validate_outbound_url`
- Why it matters: classification is equivalent today (verified: both block 100.64/10, 192.0.0/24, multicast, link-local; egress adds `METADATA_HOSTNAMES`/`_METADATA_IPS`, which resolve to link-local/CGNAT anyway), but two lists drift silently; `web_fetch`'s "no allowed_hosts bypass" requirement (ADR-032 addendum TASK-1354) is exactly what `egress.is_public_http_url` exists for.
- Recommended correction: fold `web_tool_impls` into task-609's consolidation scope (call `egress.is_public_http_url` for the verdict, keep the local reason strings); `Utils/format_size.py` for the 15-copy cluster. S each.
- Size: S · ADR: `032-local-agent-tool-permission-boundary.md` governs the behaviour (unchanged) · Confidence: verified · Already covered: task-609 (scope does not yet name this file)

### P3 [D1] — assorted small correctness gaps (each verified by reading; none reproduced)
- `Tools/web_tool_impls.py:252-269 _enforce_rate_limit`: `_domain_last_fetch` read-sleep-write is unlocked while the three caches got locks in task-3770; two worker threads fetching the same host both pass. Politeness only.
- `Tools/local_tool_impls.py:985-1039 _grep_relative_files`: no per-line search cap (the sibling `_grep_worker.py:144` caps at `max_line_search_chars=500`) and a 300 s helper timeout vs 18 s — a catastrophic pattern burns up to 300 s of one core per call, killable.
- `Tools/local_tool_impls.py:84-85 _WRITE_LOCKS`: one `threading.Lock` per distinct canonical path ever written, never evicted.
- `MCP/unified_control_plane_service.py:3604-3629 _owner`: 5 ms `asyncio.sleep` busy-poll on the UI loop for the whole Hub test (up to `effective_deadline`, 45-300 s) — `asyncio.wait({worker}, timeout=…)` would wake once.
- `MCP/server_unified_service.py:3199-3209 _persist_target_status`: `except Exception: return status` — a failed target-status write is silent (no log).
- `MCP/server.py:994-1000 _register_local_agent_tools`: `except Exception` prints a fixed sentence to stderr and drops the exception — an operator whose local tools vanish cannot see why; `MCP/__main__.py:25-27` does the same for the whole server ("MCP server failed." with no traceback).
- `MCP/server_unified_service.py:2898-2904 _client_cache`: keyed by `server_id` only; an auth/base_url edit that keeps the same normalized server id keeps the old client for the process lifetime (inferred; command to settle: `rg -n '_client_cache' tldw_chatbook/MCP/server_unified_service.py` → only `.get`/set, no invalidation).
- `Tools/web_tool_impls.py:55-63`: `defusedxml` absent → stdlib parser with a `logger.warning` at IMPORT time on every process start (noise, not a bug; `Subscriptions/security.py` does the same).

## Confirmed call sites for the sibling reviewer's ENTRY-config P1 (`config.get_api_key` returns placeholder/unstripped keys — not re-derived here)
- `tldw_chatbook/MCP/server.py:610` → `api_key = get_api_key(provider)`; `:619-628` passes it verbatim as `api_key=` to `chat_api_call` — a placeholder goes on the wire from the standalone MCP server's `chat_with_llm`.
- `tldw_chatbook/MCP/tools.py:108` → same accessor; moot in practice because the only consumer is the dead `chat_with_provider` stub (P2 above), but it would inherit the defect the moment that tool is repaired.

## Coverage (MCP)
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| MCP/unified_control_plane_service.py | 5599 | read in full |
| MCP/server_unified_service.py | 3401 | read in full |
| MCP/permission_store.py | 2555 | read in full |
| MCP/client.py | 1788 | read in full |
| MCP/local_store.py | 1271 | read in full |
| MCP/server.py | 1021 | read in full |
| MCP/tools.py | 483 | read in full |
| MCP/unified_control_models.py | 527 | read in full |
| MCP/server_target_store.py | 418 | read in full |
| MCP/local_server_tools.py | 449 | read in full |
| MCP/unified_context_store.py | 83 | read in full |
| MCP/__init__.py, __main__.py | 25+27 | read in full |
| MCP/local_control_service.py | 1193 | sampled: 120-260, 280-420, 690-760, 825-960 |
| MCP/local_runtime_delegate.py | 756 | sampled: 628-700 + symbol grep |
| MCP/execution_log.py | 463 | sampled: 130-150, 380-395, legacy-marker grep |
| MCP/prompts.py | 365 | sampled: 21-25 + the 5 except sites |
| MCP/activation.py | 242 | sampled: 18-32 |
| MCP/hub_test_execution.py | 383 | sampled: 140-160 |
| MCP/server_request_handlers.py | 270 | sampled: 160-190, 232-266 |
| MCP/gateway_runtime.py, readiness.py, recovery_activation.py, resources.py, redaction.py, mcp_import.py, permission_prompt_reducer.py, recovery.py, spawn_guard.py, tool_naming.py, live_server_request_wiring.py, hub_tool_catalog.py, local_server_tools (see above) | 1308+683+735+448+240+113+250+106+134+161+330+288 | mechanical only (no candidate rows or rows examined by line) |
| Tools/watchlists_tool_service.py | 2216 | read in full (row for the Tools table above) |

## Candidate dispositions
| candidate (file:line pattern) | confirmed / retired (why) / unverified (check) |
|---|---|
| dup_shape `_close_worker_queue@Tools/raw_cli_executor.py:841` (114-copy shape) | retired — 2-line queue close; shape match with 113 unrelated handlers, no shared behaviour |
| dup_shape `_unique_json_object@Tools/watchlists_tool_service.py:1257`, `_reject_duplicate_keys@Tools/workspace_tool_protocol.py:407`, `@MCP/permission_store.py:388` (+4 outside slice) | confirmed — strict-JSON cluster (P2 D4b) |
| dup_shape/verbatim `_datetime_to_iso` ×5 (4 in MCP) | confirmed — P2 D4b + storage drift |
| dup_shape `_clean_text/_optional_text/_normalize_optional_text@server_target_store:391/_text_or_none@unified_control_models:507` | confirmed — P3 D4b (server_target_store re-rolls a helper from a module it imports) |
| dup_verbatim `_maybe_await` ×3 (`MCP/unified_control_plane_service.py:568`) | confirmed — the MCP copy is the no-op-for-sync wrapper behind the P2 D2 loop-blocking finding; a shared `Utils` helper is not the fix, the task-283 offload shape is |
| dup_verbatim `_identity` ×3 (`MCP/activation.py:25`) | confirmed — byte-identical (diff run), P3 D4b |
| dup_shape `_default_unified_mcp_context_path/_default_server_targets_path/_default_local_mcp_store_path` | retired — each returns a different filename lazily (TASK-855); a 1-line `_user_data_file(name)` would remove 3 docstrings, not behaviour |
| dup_verbatim `_require_field@UCPS:2309` / `_require_payload_field@SUS:3144` | confirmed — P3 D4b |
| except_exception_pass `MCP/hub_test_execution.py:152` | retired — done-callback retrieving `Task.exception()` to silence "never retrieved"; nothing to act on |
| except_exception_pass `MCP/unified_control_plane_service.py:3420` | retired — `handle.close()` in a `finally` after the outcome is already sealed |
| except_exception_pass `MCP/unified_control_plane_service.py:3666` | retired — `await worker` in `finally` to reap the offloaded task after a `BaseException`; the outcome was sealed above |
| except_exception_pass `Tools/local_tool_impls.py:674,718` | retired — best-effort `portalocker.unlock` in cleanup; the primary error is preserved (`raise` at 721 / handle closed at 676) |
| except_exception_pass `Tools/raw_cli_executor.py:1242` | retired — admission thread; failure is surfaced through `admission_done` → `terminal_state="containment_unavailable"` (1256-1258); no log line, P3 nit only |
| except_exception_pass `Tools/web_tool_impls.py:1963` | retired — `HTMLParser.feed` over attacker HTML, keeps collected links (documented) |
| except_exception_pass `Tools/workspace_tool_executor.py:695,699,704,708,763` | retired — process terminate/kill/pipe-close best-effort inside `_settle_without_supervisor`/`_close_process_pipes_until`; the boolean result drives `cleanup_unproven` |
| except_exception_return `MCP/local_control_service.py` ×1 (`_disconnect_best_effort`) | retired — best-effort disconnect after a failed connect |
| except_exception_return `MCP/local_server_tools.py` ×1 (`_root_guard`) | retired — fail-closed |
| except_exception_return `MCP/permission_store.py` ×1 (`_canonical_args_json`) | retired — documented (`default=str` dead path; a miss only widens to a re-prompt) |
| except_exception_return `MCP/prompts.py` ×5 | retired — `_prompt_failure()` logs `logger.error` before returning the fixed fallback |
| except_exception_return `MCP/server_request_handlers.py` ×1 (`:265`) | retired — converts to a JSON-RPC error reply; a hang would be worse |
| except_exception_return `MCP/server_unified_service.py` ×1 (`_persist_target_status:3208`) | confirmed — P3 (silent) |
| except_exception_return `MCP/unified_control_plane_service.py` ×6 | retired — every site logs `logger.warning` with `exception_type` |
| except_exception_return `Tools/raw_cli_executor.py` ×2, `Tools/watchlists_command_service.py` ×9 | unverified — not read (`rg -n 'except Exception' -A2 tldw_chatbook/Tools/watchlists_command_service.py tldw_chatbook/Tools/raw_cli_executor.py`) |
| except_exception_return `Tools/workspace_file_roots.py` ×1 (`_binding_matches_frozen_authority`) | retired — fail-closed on any stat/attr error |
| except_exception_return `Tools/workspace_tool_executor.py` ×2 | retired — `settle()`/kill path, result drives `cleanup_unproven` |
| function_body_import `MCP/server_unified_service.py` ×20 | retired — every one is the documented task-285 phase-2 deferral of `tldw_api.mcp_unified_schemas` |
| function_body_import `MCP/local_server_tools.py` ×7 | retired — documented task-24458 pre-importer payload deferral |
| function_body_import `MCP/server.py` ×22, `MCP/unified_control_plane_service.py` ×20, `MCP/recovery_activation.py` ×19 | retired (server.py: config/DB/service imports inside `_init_databases`/`_register_*` — deferred by design for the ADR-126 admission fence; UCPS: `hub_test_execution`/`local_server_tools`/`Agents.*` are lazy to avoid import cycles, stated in `_ensure_hub_test_state`) / recovery_activation unverified (not read) |
| function_body_import `Tools/local_tool_impls.py` ×3 (`import re` ×2 in `grep_files`/`_grep_relative_files`, `portalocker`) | confirmed P3 nit — `re` is stdlib, hoist; `portalocker` is optional-shaped |
| function_body_import `Tools/tool_executor.py` ×3 (`zoneinfo`, `ast`, `operator`) | retired — stdlib, µs per call; not worth a diff |
| function_body_import `Tools/web_tool_impls.py` ×6 | retired — trafilatura/pymupdf/PIL are heavy optionals; `config` imports documented "keep module import cheap" |
| function_body_import remaining per-file counts (activation 3, client 2, local_control_service 4, local_runtime_delegate 4, local_store 5, mcp_import 1, permission_store 4, recovery 1, server_request_handlers 1, server_target_store 4, unified_context_store 4, unified_control_models 1, Tools/__init__ 1, document_expansion 4, file_operation_tools 1, note_management 1, workspace_file_roots 6, workspace_root_pin 3) | retired where read (`recovery_activation.select/readable/require_store_write` are the documented lazy recovery seam; `config` imports are the stated boot-ratchet deferral; `ctypes.wintypes` is Windows-only) / unverified for document_expansion_tool, note_management_tools, Tools/__init__ (not read) |
| inline_truncate `MCP/local_runtime_delegate.py:635`, `MCP/server.py:762`, `MCP/tools.py:252`, `Tools/code_audit_tool.py:286,429,487,531` | confirmed — P3 D4a (`Utils.truncate_content` exists, 0 external importers) |
| legacy_markers_per_file (15 files) | not examined (marker census only; no D-class attaches to a comment marker) |
| mutable_class_attr `MCP/server_unified_service.py:2846-2847 _FallbackBootstrap` | retired — fresh instance per `_bootstrap_access_context` call; only readers (`:87-88`, `:124-125` copy to tuples); `rg -n 'manageable_(team|org)_ids'` shows no `.append`/mutation |
| mutable_class_attr `Tools/workspace_root_pin.py:240 ByHandleFileInformation._fields_` | retired — ctypes `Structure._fields_` is the required idiom |
| os_replace_no_atomic `MCP/permission_store.py:956` | retired — `os.fsync(handle.fileno())` at 955 and `_fsync_parent_directory` at 966; stronger than `Utils/atomic_file_ops` (which fsyncs the file only — noted for the UTILS reviewer: `rg -n fsync Utils/atomic_file_ops.py` → 103,178 only) |
| os_replace_no_atomic `Tools/local_tool_impls.py:649` | retired — `os.fsync(temp_fd)` 627 + `os.fsync(parent_fd)` 656, `O_EXCL` temp via `dir_fd`; the D1 finding is the SIBLINGS that do not use it |
| raw_1024x1024 ×12 | retired — chunk sizes (local_tool_impls 518/777) and named cap constants (web 139-149,1112,1120,1409; protocol 18/23); no shared MiB constant exists to adopt |
| raw_mkdir `MCP/permission_store.py:942`, `Tools/file_operation_tools.py:109,731` | retired — idempotent `exist_ok=True`; 731 is guarded by `refuses_new_directory_chain` (TASK-849) |
| re_compile_in_def `Tools/_grep_worker.py:113` | retired — one-shot subprocess per batch, compile-once-per-process is inherent; the subprocess + `RLIMIT_CPU` IS the ReDoS guard |
| re_compile_in_def `Tools/file_operation_tools.py:1758` | retired — validation compile only; the search runs in `_grep_worker` |
| re_compile_in_def `Tools/local_tool_impls.py:965,997` | retired — 997 runs in the one-shot pinned worker; 965 is a validation compile on the (test-only) in-process path; per-line cap gap noted as P3 |
| seed `_datetime_to_iso` ×4 | confirmed (P2) |
| seed `_format_size@Tools/web_tool_impls.py:463` | confirmed (P3 cluster of 15) |
| seed `_identity@MCP/execution_log.py:387` | retired — a `lstat` fingerprint, unrelated to `activation._identity` (name collision) |
| seed `_identity@MCP/activation.py:25` | confirmed (P3) |
| seed `_maybe_await@UCPS:568` | confirmed (P2 D2 loop-blocking) |
| seed `_now@MCP/server_request_handlers.py:173` | retired — injectable monotonic clock seam for tests |
| seed `_reject_json_constant@MCP/permission_store.py:398` | confirmed (P2 cluster) |
| seed `_safe_text@Tools/watchlists_tool_service.py:2025` / `@UCPS:3828` | retired — same name, different functions (bounded stored text vs root+secret redaction) |
| strftime `Tools/tool_executor.py:144 %A` | retired — weekday name, correct |
| tempfile_no_secure `MCP/permission_store.py:943` | retired — `mkstemp` (0600) in the target's own directory |
| tempfile_no_secure `Tools/file_operation_tools.py:1237` | retired — `gettempdir()` used only as the grep worker's cwd, no file created |
| tempfile_no_secure `Tools/raw_cli_executor.py:1086,1093` | retired — `TemporaryFile` / `mkstemp`+`fchmod 0600`+immediate `unlink` |
| try_import_guard `MCP/__init__.py:20` | retired — availability probe |
| try_import_guard `MCP/__main__.py:18` | confirmed P3 — not an import guard; the entry point swallows the traceback ("MCP server failed.") |
| try_import_guard `MCP/server.py:70` | retired — genuine optional dep (`mcp_unified` absent in this venv; `MCP_AVAILABLE` consumed by `local_runtime_delegate`); cost-when-installed UNVERIFIED |
| try_import_guard `MCP/local_control_service.py:831`, `MCP/prompts.py:251`, `MCP/server.py:517` | retired — each logs (`opt(exception=True).warning` / `_prompt_failure` / `logger.bind(...).error` + re-raise) |
| try_import_guard `Tools/_grep_worker.py:70` (`resource`) | retired — POSIX-only stdlib, documented |
| try_import_guard `Tools/code_audit_tool.py:133`, `Tools/file_operation_hooks.py:283,296`, `Tools/rag_search_tool.py:83`, `Tools/watchlists_tool_service.py:1334` | retired where read (watchlists 1334 fails closed to both gates False) / unverified for code_audit_tool, file_operation_hooks, rag_search_tool (not read) |
| try_import_guard `Tools/local_tool_impls.py:670` | retired — see except_pass 674 |
| try_import_guard `Tools/web_tool_impls.py:55` (defusedxml) | confirmed P3 — warning at import time; `:602` trafilatura fallback and `:650` pymupdf `[missing-dep]` are correct optional handling (retired) |
| try_import_guard `Tools/workspace_file_roots.py:501` | retired — `allowed_file_roots` fails safe to sandbox-only with a warning |

## Verified-fine
- **Path checks in `Tools/` (brief item 1):** every model-named path goes through `Utils/path_validation.validate_path` — `local_tool_impls.resolve_workspace_path:176` (the one choke point for `fs_*`, `git_*` via `git_tool_impls`, `fs_patch` via `patch_tool_impls.patch_files:1023`), `workspace_tool_executor.WorkspaceToolExecutor.__init__:126` and `_build_request:387-470` (re-resolves every path/pathspec parent-side before serialising), `workspace_root_pin.pin_workspace_root:106`, `local_server_tools._build_hub_local_provider_handle:189`, and the legacy family through `validate_path_multi` (`file_operation_tools.py:294,420,644`). The inline `is_relative_to`/`.resolve()` at `local_tool_impls._relative_target_is_safe:1055-1058`, `_glob_relative_files:910`, `file_operation_tools.is_within:89-95` and `_is_hidden_within:954` are ADDITIVE per-candidate filters on entries the model never named (enumeration output), applied after the root passed the choke point — the module docstring (`local_tool_impls.py:26-46`) documents exactly this three-mechanism design. No bypass found. Hidden components are allowed only via `allow_hidden=True` per ADR-032 ("`validate_path` grows an `allow_hidden` parameter"). The worker-side `PinnedWorkspaceRoot.relative_path:37-49` rejects absolute/`..`/NUL/drive-anchored paths lexically; the worker `chdir`s to an `O_DIRECTORY|O_NOFOLLOW` fd whose identity chain is re-verified (`_pin_posix:144-149`) — ADR-032's "workspace authority is the strict canonical locator plus the root-first directory identity chain" holds.
- **Process boundary (ADR-033):** `git_tool_impls` uses `subprocess.Popen` with fixed argv (`:243-247`), `_ALLOWED_GIT_SUBCOMMANDS` (`:102,184`), `GIT_MAX_OUTPUT_BYTES=1_000_000` (`:98`) and a deadline (`:269-276`); `_grep_worker` spawns `sys.executable -S -P` with `cwd=tempdir`, minimal env, `communicate(timeout)` + `kill()`; the pinned worker spawns `sys.executable -I -m ...` with a scrubbed PATH that excludes the workspace root (`workspace_worker_environment:92-117`), `start_new_session`, and `ExecutorProcessTree` containment. No shell interpolation anywhere in the slice (`raw_cli_executor` is the ADR-094 user-invoked path, sampled only).
- **Web egress (brief item 2):** every `web_fetch`/`web_crawl`/robots fetch runs `_validate_hop` before each redirect hop (`web_tool_impls.py:1063,1595,777`), `trust_env=False` on every client (`:882,932,1838`), `follow_redirects=False` with a manual capped loop, per-hop robots check, bounded streaming reads with mid-stream kind resolution, PDF/binary refusal instead of truncation. `web_search` deliberately applies no public-target check (ADR-032 addendum TASK-1354: the engine is allowlisted). The classifier is equivalent to `egress._classify_ip` today (P3 above is about duplication, not a hole).
- **`MCP/permission_store.py` writer (brief item 3):** `mkstemp` in-directory, `json.dump` + `flush` + `fsync(file)`, `os.replace`, then `_fsync_parent_directory` with an explicit `_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS` tolerance; every mutator runs under the per-resolved-path `RLock` (`_resolved_path_lock`) with load-mutate-save inside the fence (`_mutate_locked`, `_mutate_profile_locked`); imported-profile CAS on `generation`/`revision`/`policy_digest`. Crash-safe. (The OSError-on-READ reset is the P1 above — the writer is not the problem.)
- **`_FallbackBootstrap` (brief item 5):** read-only, fresh per call — see dispositions.
- **fs_grep ReDoS (brief item 8):** bounded by the killable one-shot helper (`WORKSPACE_HELPER_TIMEOUT_SECONDS=300`, `_settle_after_spawn` → terminate/kill via `ExecutorProcessTree`); the legacy `grep_files` family is bounded by `_grep_worker` (18 s, `RLIMIT_CPU` 60 s, 500-char line slice, 200k lines). The in-process `VirtualCliRegistry` grep path is reached only when no executor is passed — `Agents/virtual_cli_provider.py:146-153,176-181` always passes one, so it is test-only.
- **Worker spawn cost:** `/usr/bin/time $PY -I -m tldw_chatbook.Tools.workspace_tool_worker </dev/null` → 0.09-0.10 s wall (×3); the package import is light enough that a process per `fs_*` call is not the cost I expected.
- **`client.py`:** bounded JSON copies with depth/byte caps and cycle detection, cursor-loop/page/item caps on catalogs, resource-URI validation (no userinfo, scheme regex), `spawn_guard` at both save and spawn time, ownership tokens on reconnect races, transport-failure cleanup as a tracked task — no finding beyond the naive-local `connected_at` timestamp.
- **`watchlists_tool_service.py`:** exact-argument dicts, canonical-id regexes, `_MAX_SQLITE_ROW_ID` bounds, fingerprinted keyset cursors with duplicate-key rejection, per-field byte budgets, URL sanitisation with IDNA + label checks, control-character stripping, fixed public error text (`_raise_unexpected`). Uses stdlib `logging` (`_LOGGER`) — the only file in the slice that does; not a mixed-logger file, so not a D3 violation, consistency nit only. Its `_default_operational_state` fails closed.
- **`local_server_tools.py` (ADR-053 external exposure):** `build_server_local_provider` filters to `CONSOLE_AND_EXTERNAL_MCP` descriptors by the code-owned exposure field (no name allowlist), `approval_callback=None` + `EXTERNAL_NO_CALLBACK_REFUSAL`, kill switch read fails closed, `_LazyWatchlistsDBResolver` opens `read_only=True` and asserts readiness — matches the ADR-032 TASK-22859-64 addendum.
- **Strict-JSON protocol (`workspace_tool_protocol.py`):** exact key sets, pydantic strict frames, byte ceilings, NUL/UTF-8 checks, intent↔operation cross-check, glob/path grammar; response frames are round-tripped through `from_bytes` before emission.
- **`_grep_worker` line cap / `_MAX_GREP_LINES_SCANNED` / `_MAX_CANDIDATES` / batched subprocess deadline (`_run_grep_search`)** — all as documented; the 20 s `timeout_seconds` vs 18 s inner deadline headroom is real.

## Retired
- "fs_grep runs the model's regex in-process with no timeout" (raised from `local_tool_impls.py:965/997`): symptom real in the code shape, cause wrong — `Agents/local_tool_provider.py:3098-3492` dispatches every `fs_*` through `WorkspaceToolExecutor.execute` (one-shot killable process, 300 s), and `virtual_cli_provider.py:146-153` always constructs an executor; the in-process branch is test-only (`Tests/Tools/test_virtual_cli_impls.py:25,126`).
- "server.py's module-scope `mcp_unified` import is a guarded-but-not-lazy cost paid by the app" — `mcp_unified` is not installed in this venv (`ModuleNotFoundError`), so the cost cannot be measured here; kept as UNVERIFIED, not a finding.
- "`_FallbackBootstrap` mutable class attrs shared across instances" — never mutated in place (grep), instance is per-call.
- "`os.replace` without fsync" at both candidate sites — both fsync file and parent.
- "`_safe_text` duplicated across Tools/MCP" — same name, different semantics.
- "`_default_*_path` triplicate" — three different filenames; shape-only.
- "`fs_edit` writes outside the root via a symlink" — `_request_mutation_path:200-214` resolves and checks in-root + denylist before the write; only the in-place/TOCTOU aspect survives as the P2.
- "`local_tool_impls.read_file` returns unbounded content" — it slices to `MAX_READ_CHARS` (32 KiB); only the whole-file materialisation (RSS) survives, folded into the P2 with `ReadFileTool`, which really does return everything.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| Whether `Tests/MCP/test_permission_store*.py` pins the OSError→backup+reset branch as a REQUIREMENT (would turn the P1 into a decision) | grep found only `corrupt`/`unknown-version` pins (`:3,201`) and a `PermissionError` raised inside a test double at `:149` (context not read) | `cd $WT && source $SCRATCH/env.sh && $PY -m pytest Tests/MCP/test_permission_store.py -q` then `rg -n 'PermissionError|OSError|chmod' Tests/MCP/test_permission_store*.py -B3 -A12` |
| Import cost of `tldw_chatbook.MCP.server` when `mcp_unified` IS installed (module-scope `from mcp_unified.gateway import serve_stdio` at `server.py:71`, pulled by `local_runtime_delegate.py:10` and `local_control_service.py:133`) | `mcp_unified` absent in the review venv | `cd $WT && source $SCRATCH/env.sh && $PY -X importtime -c 'import tldw_chatbook.MCP.server' 2>&1 \| rg 'mcp_unified\|MCP.server' \| tail -5` (after `uv pip install -e ".[mcp]"` in a scratch venv) |
| Wall-clock cost of `_revalidate_mutation_scope` per Hub mutation against a real tldw_server (the 10-round-trip count is verified; latency is not) | needs a live server | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify` → MCP Hub → Servers ▸ any mutation, with `[logging] level = DEBUG` and `rg 'Unified MCP endpoint probe' ~/.local/share/tldw_cli/logs/*.log \| wc -l` before/after one click |
| `fs_edit`/`fs_patch` truncate-window data loss under a mid-write kill | crash window not reproduced (unconditional code path only) | `cd $WT && source $SCRATCH/env.sh && PYTHONPATH=$WT $PY - <<'EOF'` — monkeypatch `Path.write_bytes` to raise after `open(...,"wb")` truncates, call `_edit_relative_file`, assert the file is empty `EOF` |
| `ServerUnifiedMCPService._client_cache` keeps a stale client after an auth-only target edit | needs a client_factory that reflects auth; not exercised | `rg -n '_client_cache' tldw_chatbook/MCP/server_unified_service.py` (expect only `.get`/assignment, no invalidation) + `rg -n 'invalidate_cache\|_client_cache' tldw_chatbook/UI/MCP_Modules/*.py` |
| The 9 `except Exception: return` sites in `Tools/watchlists_command_service.py` and 2 in `Tools/raw_cli_executor.py` | files not read (mechanical only) | `rg -n 'except Exception' -A3 tldw_chatbook/Tools/watchlists_command_service.py tldw_chatbook/Tools/raw_cli_executor.py` |
| ruff fatal baseline for the slice | not run | `cd $WT && .venv/bin/ruff check --select E9,F63,F7,F82 tldw_chatbook/Tools tldw_chatbook/MCP` |
