# UIM-nav-mcp-persona — tldw_chatbook/UI/{Navigation,MCP_Modules,Persona_Modules}, 26,845 lines

## Coverage
Mechanical sweeps applied to ALL 35 files: function-body import resolution (86 imports, `importlib`), `run_worker` kwarg audit (42 sites, `ast`), post-await `query_one` audit (`ast`), mutable-class-attribute scan (`ast`), and greps for egress/raw sqlite/module-level locks/path ops. "mechanical only" below means the file got those and nothing more.
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| UI/MCP_Modules/mcp_workbench.py | 6328 | read in full |
| UI/MCP_Modules/mcp_inspector.py | 3891 | read in full |
| UI/MCP_Modules/mcp_servers_mode.py | 1672 | read in full |
| UI/Persona_Modules/personas_conversations_controller.py | 1292 | read in full |
| UI/Navigation/main_navigation.py | 1254 | read in full |
| UI/MCP_Modules/mcp_permissions_mode.py | 1145 | read in full |
| UI/Navigation/buddy_management.py | 1031 | read in full |
| UI/Persona_Modules/personas_preview_controller.py | 850 | sampled (:99-200 controller state/reset, :490-850 `_run_reply` + the whole provider-reply path; full `def`-level outline) + all mechanical sweeps |
| UI/MCP_Modules/mcp_audit_mode.py | 820 | sampled (:265-295 timestamp/outcome renderers, the `_DECISION_LABELS`/`remediation_actions` export surface) + all mechanical sweeps |
| UI/Navigation/pending_handoff_store.py | 800 | sampled (:430-470 claim release/retain; the `AudioCppModelLibraryResult` construction at :780) + all mechanical sweeps |
| UI/MCP_Modules/mcp_tools_mode.py | 716 | sampled (:95-135 `_ellipsize`, :330-380 `update_tools`/`focus_server`) + all mechanical sweeps |
| UI/Navigation/buddy_conversation.py | 597 | mechanical only (run_worker/except/import sweeps; 6 function-body imports resolved) |
| UI/MCP_Modules/mcp_rail.py | 584 | sampled (:480-590 the Select blank-sentinel handling) + all mechanical sweeps |
| UI/Navigation/base_app_screen.py | 563 | sampled (:295-345 recompose/mouse-capture teardown) + all mechanical sweeps |
| UI/MCP_Modules/mcp_profile_form.py | 531 | sampled (:1-20 imports incl. the private `_looks_like_raw_secret_value`) + all mechanical sweeps |
| UI/Navigation/character_conversation_navigation.py | 526 | mechanical only (3 function-body imports resolved; private-name import pair with `_character_conversation_wire`) |
| UI/Navigation/audio_cpp_model_handoff.py | 467 | sampled (:1-110 the validators, :240-275 the lease-cancel path) + all mechanical sweeps |
| UI/Navigation/screen_registry.py | 457 | read in full |
| UI/Navigation/buddy_speech.py | 440 | sampled (:1-60 coordinator state, :434-440 `ensure_buddy_speech`) + all mechanical sweeps |
| UI/MCP_Modules/mcp_server_mutations.py | 429 | sampled (:42 blank-sentinel helper) + all mechanical sweeps |
| UI/MCP_Modules/mcp_schema_form.py | 426 | mechanical only |
| UI/Navigation/shell_destinations.py | 301 | mechanical only + every registered shell route id resolved through `resolve_screen_target()` (probe in Findings) |
| UI/Navigation/screen_state_store.py | 262 | mechanical only |
| UI/Navigation/conversation_settings_navigation.py | 258 | mechanical only |
| UI/Navigation/persona_buddy_overlay.py | 213 | sampled (:120-130 `start_scope_tracking` call site) + all mechanical sweeps |
| UI/Persona_Modules/buddy_conversion.py | 203 | mechanical only + the screen-private-access scan (12 distinct private members) |
| UI/Navigation/buddy_workspace.py | 182 | read in full |
| UI/Navigation/vllm_handoff.py | 171 | mechanical only |
| UI/Navigation/nav_overflow_menu.py | 157 | mechanical only |
| UI/Navigation/_character_conversation_wire.py | 119 | mechanical only |
| UI/Persona_Modules/personas_preview_coordinator.py | 70 | read in full |
| UI/Navigation/shortcut_context.py | 53 | mechanical only |
| UI/Navigation/__init__.py | 35 | read in full |
| UI/Persona_Modules/__init__.py | 1 | read in full (1 line) |
| UI/MCP_Modules/__init__.py | 1 | read in full (1 line) |

## Findings   (ordered P0→P3, then D1→D4)

### P0 [D1] — pressing "Hide advanced" while the MCP inspector's section load is in flight kills the whole app
- Where: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:3418` (`_load_advanced_section`, the two DOM reads after `await self._service.load_section(section)`: `query_one("#mcp-adv-content", Static)` and, on the next line, `self._refresh_advanced_actions()` which reads `#mcp-adv-action-select`/`#mcp-adv-payload`/`#mcp-adv-run`). Dispatched at `:3536` / `:3395` as `run_worker(partial(self._load_advanced_section, ...), group="mcp-adv-section", exclusive=True)` — Textual's `exit_on_error` defaults to **True**. `_hide_advanced()` (`:1639`) removes `#mcp-adv-collapsible` and every one of those widgets with it, and nothing cancels the in-flight worker.
- Evidence (real button-click path, fresh-install default of `advanced_visible=False`):
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
  import asyncio
  from textual.app import App, ComposeResult
  from textual.widgets import Button
  import tldw_chatbook.UI.MCP_Modules.mcp_inspector as mod
  from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
  mod.get_cli_setting = lambda s,k=None,d=None: False if k in ("advanced_open","advanced_visible") else d
  mod.save_setting_to_cli_config = lambda *a, **k: True
  class SlowService:
      def __init__(self): self.gate = asyncio.Event()
      async def load_section(self, section=None):
          await self.gate.wait(); return {"source":"local","section":section or "overview"}
      def available_actions(self): return []
  class Harness(App):
      def compose(self) -> ComposeResult: yield MCPInspector(id="insp")
  async def main():
      app = Harness()
      async with app.run_test() as pilot:
          insp = app.query_one(MCPInspector); svc = SlowService()
          insp.set_service_context(svc, [("Overview","overview")], source="local")
          await pilot.pause()
          await pilot.click("#mcp-inspector-advanced-reveal")      # "Advanced…"
          for _ in range(6): await pilot.pause()
          btn = app.query_one("#mcp-inspector-advanced-reveal", Button)
          print("after reveal: label=%r disabled=%r" % (str(btn.label), btn.disabled))
          await pilot.click("#mcp-inspector-advanced-reveal")      # "Hide advanced", load still in flight
          for _ in range(6): await pilot.pause()
          svc.gate.set()
          for _ in range(20): await pilot.pause()
      print("no crash")
  try: asyncio.run(main())
  except BaseException as e: print("TOP-LEVEL RAISE:", type(e).__name__, e)
  EOF
  ```
  ->
  ```
  ...
  /Users/macbook-dev/Documents/GitHub/tldw-review/tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:3418 in _load_advanced_section
    3417     payload = await self._service.load_section(section)
  ❱ 3418     self.query_one("#mcp-adv-content", Static).update(
  NoMatches: No nodes match '#mcp-adv-content' on MCPInspector(id='insp', classes='ds-inspector')
  after reveal: label='Hide advanced' disabled=False
  TOP-LEVEL RAISE: WorkerFailed Worker raised exception: NoMatches(...)
  ```
  Note the printed line: the toggle is re-enabled and relabelled "Hide advanced" BEFORE `_reveal_advanced()` calls `set_service_context()` (`mcp_inspector.py:1629-1631` vs `:1632`), so the second press is available to the user during the entire load.
- Why it matters: the whole Chatbook process exits. The window is not microseconds — `load_section` is the control-plane round trip (server source resolves its access context over multiple sequential client calls before answering), so any user who opts into Advanced and changes their mind mid-load loses the app. The same shape reaches `_refresh_advanced_actions()` on the line after.
- Recommended correction: guard the post-await DOM writes on `self._advanced_visible` (the flag `_hide_advanced` already clears synchronously at `:1668`, before its own `await`) and/or wrap the two reads in `except NoMatches: return`, matching how `_set_test_unavailable`/`show_test_preview`/`show_tool_result` in this same file already handle a panel that went away mid-flight. Dispatching the worker with `exit_on_error=False` (as `MCPWorkbench` does for every worker it owns) is the belt-and-braces half.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none. `Tests/UI/test_mcp_inspector.py` covers the reveal path and the rescheduled-reveal exclusivity, but never a hide racing a slow section load.
- Already covered: none

### P1 [D2] — every Permissions-matrix Space press blocks the event loop for ~56 ms in `tool_gate_breadcrumb()`, and the call site's comment claims it is "cheap"
- Where: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:2657-2662` (inside `_sync_permissions_mode`, comment: *"computed fresh every pass (cheap -- the same settings-time-enumeration cost ... already pays every pass)"*) and `:1899` (`_empty_tools_diagnosis`). `_sync_permissions_mode()` has NINE standalone callers besides the full `_sync_children()` pass: `:2139`, `:2152`, `:2165` (`select_tool_policy_profile`), `:3265`, `:3436` (`on_mcp_permissions_mode_state_cycle_requested` — the Space press), `:3494` (kill-switch toggle), `:4830`, `:4895`, `:4960` (re-allow / remove-arg-rule / revoke-approval).
- Evidence:
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
  import time, statistics
  from tldw_chatbook.Agents.builtin_tool_gate import tool_gate_breadcrumb, all_tool_gates
  for _ in range(3): tool_gate_breadcrumb()
  ts=[]
  for _ in range(10):
      t=time.perf_counter(); tool_gate_breadcrumb(); ts.append((time.perf_counter()-t)*1000)
  print("tool_gate_breadcrumb ms: median %.1f min %.1f max %.1f" % (statistics.median(ts),min(ts),max(ts)))
  ts=[]
  for _ in range(10):
      t=time.perf_counter(); all_tool_gates(); ts.append((time.perf_counter()-t)*1000)
  print("all_tool_gates ms: median %.1f min %.1f max %.1f" % (statistics.median(ts),min(ts),max(ts)))
  EOF
  ```
  ->
  ```
  tool_gate_breadcrumb ms: median 55.7  min 54.5  max 58.6
  all_tool_gates ms: median 55.2  min 54.4  max 60.9
  ```
  An instrumented count shows both make **11 `get_cli_setting` calls** (one per gate row), each ~4.8 ms (see the `_collect_snapshots` finding below for where that 4.8 ms goes: 482 `posix.open` syscalls per config read through `Backup_Recovery.storage_admission`).
  `all_tool_gates()` is also re-run on every Servers-mode detail repaint: `mcp_servers_mode.py:1322` (`_tool_gate_widgets`), reached from `show_detail()` -> `_rebuild_toggle_groups()` -> `_rebuild_tool_gate_buttons()` on EVERY `_sync_children()` pass whenever the selected row is the built-in — which is the fresh-install default (`_preselect_single_problem_on_load` lands on the lone built-in row, `mcp_workbench.py:1130-1136`).
  Aggregate for one `_sync_children()` pass on a fresh install with the built-in selected, measured piecewise: `_collect_snapshots` 20.6 ms + `_local_tools_config_values` 9.6 ms + `resolve_server_workspace_root` 4.7 ms + `tool_gate_breadcrumb` 55.7 ms + `all_tool_gates` 55.2 ms = **~146 ms of config reads alone**, on the loop, per rail click / lifecycle completion / gate toggle.
- Why it matters: Space-cycling a permission in the matrix is the mode's primary gesture (its own docstring says so) and each press stalls the Textual message pump for ~56 ms of config re-reads before the matrix repaints — held-Space repeat rate is capped at ~18/s by this alone, and it is paid on top of the store load and the server round-trips those handlers already make.
- Recommended correction: resolve the gate set ONCE per `_sync_permissions_mode()` pass and thread it, as this method already does for `effective`/`policy_inventory`/`_last_cascade`; better, memoize `all_tool_gates()` in `Agents/builtin_tool_gate.py` behind the same write path that already invalidates it (`_save_tool_gate` is the only writer in-process). Fix the comment either way — "cheap" is measurably false.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none found asserting per-pass recomputation
- Already covered: none

### P1 [D2] — `MCPWorkbench._collect_snapshots()` burns 20.6 ms of pure `get_cli_setting` on the event loop, and 14 call sites re-run it
- Where: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:1288-1295` (the 4 reads) — re-entered from `:1082` (`reload`), `:3771` (`_switch_source`), `:3836` (`_select_server_key`), `:4043`, `:4096`, `:4178`, `:5760` (`_save_builtin_flag`), `:5812` (`_save_tool_gate`), `:5861`, `:5956`, `:6041`, `:6199`, `:6309`
- Evidence:
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
  import asyncio, time
  import tldw_chatbook.UI.MCP_Modules.mcp_workbench as wb
  calls=[]; real=wb.get_cli_setting
  wb.get_cli_setting = lambda *a, **k: (calls.append(a[:2]), real(*a, **k))[1]
  w = wb.MCPWorkbench.__new__(wb.MCPWorkbench)
  w._source="local"; w._server_mutations_available=False; w._catalog_records={}; w._service=lambda: None
  asyncio.run(w._collect_snapshots()); calls.clear()
  t=time.perf_counter(); asyncio.run(w._collect_snapshots()); d=time.perf_counter()-t
  print(len(calls), calls); print(f"{d*1000:.1f} ms")
  EOF
  ```
  ->
  ```
  config reads per _collect_snapshots(): 4 [('mcp','enabled'),('mcp','expose_tools'),('mcp','expose_resources'),('mcp','expose_prompts')]
  wall time for one _collect_snapshots() (no service, no I/O): 20.6 ms
  ```
  Where the time goes (cProfile over 50 warm `get_cli_setting("mcp","enabled",False)` calls): every call runs `load_cli_config_and_ensure_existence` -> `Backup_Recovery/config_participants.py:400 wrapped` -> `storage_admission.acquire_storage` -> **24100 `posix.open` calls for 50 reads = 482 syscalls per config read** (~8 ms/call profiled, ~5 ms unprofiled).
- Why it matters: this 20.6 ms is paid synchronously on the asyncio loop by every rail click, every source/scope switch, every built-in-flag checkbox, every tool-gate button and every lifecycle completion in the Hub — before any of the server round-trips those paths also make. The "config reads are cache-backed" assumption does not hold at the call boundary: the cache lookup itself goes through the storage-admission scope.
- Recommended correction: hoist the 4 reads into one snapshot read per `_collect_snapshots()` (they are all `[mcp]` keys — one `get_cli_setting("mcp", ...)`-free section read, or a single cached tuple invalidated by `_save_builtin_flag`), or await them off-loop via `asyncio.to_thread` the way `_save_builtin_flag` already does for the write. The 482-syscall config read itself is `config.py`/`Backup_Recovery`'s to fix and out of this slice.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P2 [D1] — the Permissions matrix writes the permission store SYNCHRONOUSLY on the event loop (~13 ms per keypress), unlike every other write in the same file
- Where: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:3345`, `:3356`, `:3387`, `:3398`, `:3417` (`on_mcp_permissions_mode_state_cycle_requested` -> `_call_profile_scoped(service.set_global_default / set_server_default / set_tool_state, ...)`) and `:3477` (`on_mcp_permissions_mode_kill_switch_toggled` -> `set_kill_switch(event.value)`). Both are `async def` handlers running on the loop; neither offloads.
- Evidence:
  - Write path traced: `unified_control_plane_service.py:5304 set_tool_state` -> `permission_store.py:1501 MCPPermissionStore.set_tool_state` -> `_mutate_locked` -> `permission_store.py:909 save()` -> `json.dump(..., indent=2, sort_keys=True)` + `os.replace` (`:953`, `:956`).
  - Same shape at `:4816` (`on_mcp_inspector_reallow_requested` -> `set_tool_state`), `:4876` (`remove_tool_arg_rule`), `:4938` (`revoke_session_approval`).
  - Measured:
    ```
    cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
    import time, tempfile, pathlib, statistics
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore
    d = pathlib.Path(tempfile.mkdtemp()); s = MCPPermissionStore(d/"perm.json")
    s.set_kill_switch(True); s.set_kill_switch(False)
    ts=[]
    for i in range(20):
        t=time.perf_counter(); s.set_kill_switch(i%2==0); ts.append((time.perf_counter()-t)*1000)
    print("median %.2f min %.2f max %.2f" % (statistics.median(ts),min(ts),max(ts)))
    EOF
    ```
    -> `MCPPermissionStore.set_kill_switch ms: median 13.24 min 12.71 max 16.69`
- Why it matters: this file already offloads its blocking writes -- `_save_builtin_flag` (`:5748`) and `_save_tool_gate` (`:5802`) both wrap `save_setting_to_cli_config` in `asyncio.to_thread`, with a docstring explaining why. The permission setters, the *more* frequently exercised path (Space-cycling is the mode's primary gesture), do not. Combined with the `tool_gate_breadcrumb()` finding above, one Space press costs ~13 ms (write) + ~56 ms (breadcrumb) + the profile-inventory store reads `_capture_permission_render_state` makes 2-4 times, all on the loop before the matrix repaints.
- Recommended correction: wrap the five setter calls and `set_kill_switch` the same way `_save_builtin_flag` already wraps its write (`await asyncio.to_thread(...)`), keeping the existing resync-after shape.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: the store-side write itself is named by the sibling TOOLS-MCP P2 finding ("a sync JSON write before each of 65 Hub mutation/read sites"); these UI call sites are the amplifier, not the store.

### P2 [D4] — the Advanced runner renders un-redacted secrets and absolute paths, bypassing the two helpers the same file uses everywhere else (helper exists, ignored)
- Where: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:3882` (`if isinstance(result, dict): result = redact_mapping(result)` — a LIST result skips it), `:3878` (`result_widget.update(f"Action failed: {exc}")`), `:3873` (`f"{_ADVANCED_BLOCKED_HEADING}\n{exc}"`), `:3854` (`f"Invalid JSON payload: {exc}"`). The helpers that exist and are used by every OTHER result surface in this same file: `redact_mapping` (`MCP/redaction.py`, module docstring: *"Secret redaction applied at every MCP display and log boundary"*) and `_safe_exception_text`/`_safe_tool_test_text` (`mcp_inspector.py:192`/`:158`, which redact `api_key=`, bearer tokens, `sk-*`, and absolute paths).
- Evidence:
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY <repro_adv_redact.py>
  ```
  (harness mounts `MCPInspector`, binds a fake service, calls `_run_advanced_action()`) ->
  ```
  redact_mapping(dict) would give: {'name': 'docs', 'api_key': '***', 'env': {'TOKEN': '***'}}
  --- list result rendered into #mcp-adv-result ---
  [ { "name": "docs", "api_key": "sk-live-ABCDEF123456", "env": { "TOKEN": "t-secret-999" } } ]
  --- exception text rendered into #mcp-adv-result ---
  Action failed: connect failed: api_key=sk-live-ABCDEF123456 at /Users/rob/secret/path
  ```
  Script kept at `<SCRATCH>/repro_adv_redact.py`.
- Why it matters: the Advanced runner's whole point is dumping raw control-plane payloads (`external_servers` env/args included) — `mcp_workbench._redact_external_server_record()` exists precisely because that renderer leaked full raw records. That shim covers `load_section`; `run_action`'s own result and every exception path are not covered, so a list-shaped action result or any service exception puts credentials and absolute paths on screen in a pane users copy into bug reports.
- Recommended correction: route the result through `redact_mapping` for Mappings *inside* sequences too (or reuse `mcp_workbench._redact_external_servers_list`'s shape), and pass every `{exc}` through this module's own `_safe_exception_text()` — the same call `show_tool_result`, `show_test_unavailable`, and `_handle_test_run` already make four lines away.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P2 [D4] — the Select-blank predicate exists in THREE copies and the fourth site got it wrong (no helper home, copies drifted)
- Where: three identical copies —
  - `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:841-853` `_is_blank()` (`value is Select.BLANK or value is Select.NULL`, with a 12-line comment explaining the trap)
  - `tldw_chatbook/UI/MCP_Modules/mcp_server_mutations.py:42` (same one-liner)
  - `tldw_chatbook/UI/MCP_Modules/mcp_rail.py:582` (same expression, inlined)
  — and the divergent fourth: `tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py:1041` (`if event.value is Select.BLANK: return`), which checks only the sentinel that is NOT the blank marker.
- Evidence:
  ```
  $PY -c "from textual.widgets import Select; from textual.widget import Widget; print(repr(Select.BLANK), Select.BLANK is Widget.BLANK, repr(Select.NULL))"
  ```
  -> `False True Select.NULL` (Textual 8.2.8). The real no-selection sentinel is `Select.NULL`; `Select.BLANK` resolves through the MRO to `Widget.BLANK == False`, so `event.value is Select.BLANK` can never be true for a string profile id.
- Why it matters: the guard is dead. Today it is unreachable-harmless because `#mcp-perm-tool-profile` is constructed `allow_blank=False` (verified in Textual's `Select._setup_variables_for_options`: `NULL` is only inserted when `_allow_blank`), so the value is always a real profile id. If `allow_blank` ever flips, `ToolPolicyProfileSelected(str(Select.NULL))` posts the literal string `"Select.NULL"` as a profile id. `mcp_inspector._is_blank()` (`:841`) already documents this exact trap and checks BOTH sentinels; this sibling does not use it.
- Recommended correction: one canonical `select_is_blank(value)` — the natural home is `tldw_chatbook/UI/Widgets/` (or `MCP_Modules/__init__.py`, currently 1 line) since three of the four sites are MCP canvases and `mcp_inspector` is the wrong direction for `mcp_workbench` to import from anyway (see the private-import finding). All four sites call it.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P2 [D3] — `mcp_workbench.py` is a 6,328-line god module with no size ratchet
- Where: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` (one class, `MCPWorkbench`, 130 methods)
- Evidence: `wc -l` = 6328; `grep -rn "mcp_workbench|MCP_Modules" Tests/Architecture/test_screen_size_ratchet.py Tests/Architecture/test_library_modules_size_ratchet.py` -> no output (no ratchet row covers this package).
- Responsibilities carried by the one class: (1) triad assembly + deferred canvas mounting; (2) readiness snapshot collection and CHECKING overlay; (3) the local/server source switch and rail scope model; (4) the Tools catalog derivation (`_collect_hub_tools`, `_local_agent_hub_tools`, `_raw_shell_hub_tool`, `_empty_tools_diagnosis`); (5) the whole permission matrix derivation (`_tool_policy_inventory`, `_capture_permission_render_state`, `_build_permission_rows`, `_build_permission_preview`, `_builtin_permission_matrix_rows`, the cascade map); (6) the prepared Tool-Test admission/nonce/lease state machine (~600 lines); (7) profile CRUD + mcpServers import incl. path validation; (8) the server-mutation/credential-slot panel wiring; (9) audit log + findings; (10) the recovery-review dialog flow; (11) lifecycle dispatch and in-flight bookkeeping; (12) view-state save/restore.
- Why it matters: (5), (6) and (7) are each independently testable pure-ish derivations wedged into a widget; the file has no size governance, so it grows unchecked while `chat_screen.py`/`library_*` are ratcheted.
- Recommended correction: follow `backlog/docs/library-decomposition-recipe.md` §1 (per-subsystem PR series) — the Tool-Test admission machine and the permission-row derivation are the two clean first extractions — and add the package to the controller ratchet per §17 in the same PR that first moves code.
- Size: L · ADR: no (recipe exists) · Confidence: verified
- Pinning test: none (that is the finding)
- Already covered: none

### P3 [D3] — nine cross-module imports of underscore-private names, one of them a security helper from another package
- Where:
  - `tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py:15` — `from tldw_chatbook.MCP.local_store import _looks_like_raw_secret_value` (a private secret-lint predicate, imported across packages into the profile form's args warning)
  - `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:91` — `_safe_diagnostic_message`, `_safe_exception_text`, `_safe_tool_test_text` from `mcp_inspector`
  - `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:97` — `_PROFILE_HINT_TEXT`, `_undiscovered_servers_hint` from `mcp_permissions_mode`
  - `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:781` — `_DECISION_LABELS` from `mcp_audit_mode`
  - `tldw_chatbook/UI/Navigation/buddy_conversation.py:186` — `_ensure_launch_runtime` from `Chat.console_launch_wake` (cross-package)
  - `tldw_chatbook/UI/Navigation/_character_conversation_wire.py:7`, `character_conversation_navigation.py:340/383/419` (same-pair, less objectionable)
- Evidence: `ast` sweep over the three packages for `ImportFrom` names starting with `_` -> the nine rows above.
- Why it matters: the three `mcp_inspector._safe_*` redaction helpers are the sanitizers the P2 redaction finding above shows are *not* applied consistently; they are the subsystem's security primitives living behind a leading underscore in a widget module, which is why a call site can skip them without anything flagging it. `_looks_like_raw_secret_value` has the same shape one package over.
- Recommended correction: promote the sanitizer trio (`_safe_tool_test_text`/`_safe_exception_text`/`_safe_diagnostic_message`) and `_looks_like_raw_secret_value` to public names in `tldw_chatbook/MCP/redaction.py` — the module that already declares itself "applied at every MCP display and log boundary" — and import from there. The rest (`_DECISION_LABELS`, `_PROFILE_HINT_TEXT`, `_undiscovered_servers_hint`) are copy/format constants; drop the underscore where they are defined.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none (task-2542 covers only the `_toast` duplication, which is a different pair)

### P3 [D3] — the Persona controllers reach into 8-12 private members of `PersonasScreen` each, and write two of them
- Where: `tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py` (8 distinct: `_character_db`, `_edit_mode`, `_notify`, `_pending_character_conversation_link`, `_show_center`, `_show_character_link_recovery`, `_stage_handoff`, `_sync_title_and_console_actions`); `personas_preview_controller.py` (7); `buddy_conversion.py` (12). Writes: `screen._edit_mode = "view"` (`personas_conversations_controller.py:489`) and `screen._pending_character_conversation_link = None` (`:157`, `:447`).
- Evidence: `re.findall(r"(?:self\.screen|screen)\.(_[A-Za-z_]\w*)", …)` per file -> the counts above.
- Why it matters: the module docstring calls this "mirroring the `CCPCharacterHandler` pattern", but the controllers also MUTATE screen state, so the screen is not the single writer of its own mode/link fields — a rename or a lifecycle change on the screen breaks three controllers silently.
- Recommended correction: give `PersonasScreen` a small public surface for the ~8 operations these controllers actually need (`notify`, `show_center`, `stage_handoff`, `character_db`, and explicit setters for `edit_mode`/`pending_character_conversation_link`); leave the rest private.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D2] — the Buddy scope reconciler is an app-lifetime 0.5 s interval that is never stopped
- Where: `tldw_chatbook/UI/Navigation/buddy_management.py:929-933` (`start_scope_tracking` -> `self.app.set_interval(0.5, self.reconcile_scope)`), body at `:935-1004`
- Evidence: read only — `grep -n "_scope_timer" buddy_management.py` shows three sites (`:127` init, `:932` guard, `:933` assign) and no `.stop()`/`.pause()` anywhere in the tree (`grep -rn "_scope_timer" tldw_chatbook/ Tests/` returns only this file). The early-out at `:937` is `if not self._scope_configured` — and `_scope_configured` is only ever set True (`:905`), never back to False.
- Why it matters: once a user has ever configured the Buddy, this ticks twice a second for the rest of the app's life even with the Buddy disabled/closed, doing a function-body import of `buddy_speech`, a controller preference snapshot under a lock, a session enumeration, and two frozenset builds per tick. It is on the app (not a widget), so no screen teardown cancels it.
- Recommended correction: gate the early-out on the live preference (`controller.current_preferences().enabled`) and stop the timer when the Buddy is disabled, re-arming from `start_scope_tracking()` (already called on every apply).
- Size: S · ADR: no · Confidence: inferred (per-tick cost not measured; settle with `python -X importtime`-style timing around `BuddyManagementCoordinator.reconcile_scope` under a fake app)
- Pinning test: none
- Already covered: none

### P3 [D3] — `_SCREEN_ROUTES["customize"]` targets a module that no longer exists
- Where: `tldw_chatbook/UI/Navigation/screen_registry.py:182-187` (route) vs `:298` (`_SCREEN_ALIASES["customize"] = "settings"`)
- Evidence: import-probe over all 28 routes (see below) — 27 OK, `customize` -> `ModuleNotFoundError: No module named 'tldw_chatbook.UI.Screens.customize_screen'`. `_lookup_route()` checks the alias table FIRST, so the dead row is unreachable; `app.py:17739` skips alias-shadowed route ids in the pre-importer for exactly this reason (comment at `app.py:17729-17738` names it "unreachable dead metadata kept for history").
- Why it matters: `registered_screen_route_ids()` still advertises a route that cannot load; only the alias shadow keeps it from surfacing.
- Recommended correction: delete the `"customize"` `ScreenRoute` row; keep the alias.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found asserting the dead row exists
- Already covered: none (documented in app.py, not ticketed)

## Candidate dispositions
| candidate (file:line pattern) | disposition |
|---|---|
| dotted_section_setting mcp_inspector.py:1475 / :1515 `mcp.hub_state` | **retired** — brief says not to flag; also verified they resolve (TASK-1771). Both read once at compose/reveal, not in a loop. |
| get_cli_setting_hot mcp_inspector.py:1475 (compose) | **confirmed, downgraded to P3** — one `get_cli_setting` in `compose()` at ~4.8 ms (measured). `MCPInspector` never uses `recompose=True` (`grep -n "recompose" mcp_inspector.py` -> only a prose comment at :1816), so this is once per workbench mount, not per repaint. Not folded into a finding; the P1 config-read findings cover the repeated sites. |
| dup_shape `release@pending_handoff_store.py:445` (+ add_keyword/get_conversation_active_leaf/_provider_config) | **retired** — `release()` is a 3-line delegation to `self._release_claim(claim)`; shared only the "one-statement method returning a tuple element" shape with the other three. No behaviour in common. |
| dup_shape `update_findings@mcp_audit_mode.py:637` / `set_mutations_available@mcp_servers_mode.py:598` (+ chunk_preview / git_panel) | **retired** — both are two-line "store the argument, call one private repaint" setters against different state; nothing to share. |
| dup_shape `_tool_for_row_key@mcp_workbench.py:4213` (+ reference_by_id / get_section) | **retired** — a 4-line linear scan over `self._last_hub_tools`; `_tool_for()` right beside it is the deliberate field-based twin (documented, task-233). |
| dup_shape `focus_server@mcp_tools_mode.py:357` (+ two console action menus' `_show_page`) | **retired** — `focus_server` sets one field then `await self._rebuild_server_select(); self._apply_filter()`. The menus' `_show_page` is a paging routine; shape-only match. |
| except_exception_pass mcp_workbench.py:1036 | **retired** — `self.app.notify(...)` inside the load-failure handler; a failing notify must not mask the original error. |
| except_exception_pass mcp_workbench.py:1043 | **retired** — `watch_is_loading` before the canvas exists; documented. |
| except_exception_pass mcp_workbench.py:3628 | **retired** — `_focus_permissions_matrix`'s `canvas in focused.ancestors_with_self` probe; a raise there must not steal focus. |
| except_exception_pass audio_cpp_model_handoff.py:259 | **retired** — cleanup inside a `CancelledError` path that re-raises; the original cancellation stays authoritative (documented). |
| except_exception_pass base_app_screen.py:327 | **retired** — `ui_responsiveness_monitor.record_refresh` diagnostics during teardown, `# noqa: BLE001, S110` with rationale. |
| except_exception_pass main_navigation.py:1106 | **retired** — best-effort `scroll_to_widget` nudge that explicitly does not gate the ghost decision. |
| except_exception_pass personas_preview_coordinator.py:53 | **retired** — drains a shielded thread's result during cancellation before re-raising; the exception is intentionally discarded. |
| except_exception_return_per_file (9 files, 30 sites) | **retired** — every one inspected is a `query_one`/`self.screen`/`self.app` DOM or attribute guard returning a neutral value, not a data path. No swallowed writes found. |
| function_body_import_per_file (15 files, 86 imports) | **retired** — all 86 resolve. `ast` + `importlib.import_module` sweep under the isolated env: `function-body imports: 86 / bad count 0`. Several are documented circular-import breakers (`nav_overflow_menu`, `mcp_audit_mode.remediation_actions`, `buddy_speech`) or task-24458 pre-import-payload deferrals. |
| legacy_markers_per_file (8 files, 58 markers) | **retired as markers** — spot-read in `screen_registry.py` (21) and `shell_destinations.py` (4): every one is a retired-route comment explaining a live alias, not dead code. The one real dead artefact is the `customize` route (its own P3 finding). |
| plain_readback mcp_tools_mode.py:121 / :123 | **retired** — `_ellipsize` builds `Text(text)` from a plain string it was just handed (constructor, not `from_markup`), calls `truncate`, reads `.plain` back. No markup parse, so no un-escaping. |
| plain_readback main_navigation.py:89 | **retired** — `text.plain.find(" ")` on a `Text` the same function constructed from `nav_button_label(...)`; used only to locate the prefix boundary for `stylize("dim", …)`. `str(nav_button_label_text(d,l)) == nav_button_label(d,l)` is the stated contract. |
| raw_1024x1024 mcp_workbench.py:173 | **retired** — `MAX_MCP_IMPORT_FILE_BYTES = 1024 * 1024`, a documented 1 MB cap on an imported config JSON, not an image dimension. |
| run_worker_coroutine_per_file (8 files, 42 sites) | **confirmed in part** — all 42 inspected. Every one has `group=`, so the "exclusive without group" class is absent here. The coroutine bodies that block the loop are covered by the P1/P2 findings (`_collect_snapshots`, `_sync_permissions_mode`, the permission-store setters). The one that CRASHES is `mcp_inspector._load_advanced_section` (the P0). The rest either offload with `asyncio.to_thread` or touch only permanently-mounted children. |
| strftime mcp_audit_mode.py:277 | **retired** — `datetime.fromisoformat(...).astimezone().strftime("%Y-%m-%d %H:%M:%S")`; aware timestamps are converted to local first (TASK-294), naive ones deliberately left as-is. Correct. |
| try_import_guard mcp_workbench.py:1780 (`_local_agent_hub_tools`) | **retired** — `except Exception` around `VirtualCliProvider` construction, `# noqa: BLE001 -- catalog view must never break the hub`, logs `type(exc).__name__`. Not an optional-dep guard. |
| try_import_guard mcp_workbench.py:2270 (`_builtin_permission_rows`) | **retired** — same shape around `BuiltinToolProvider.list_catalog()`; documented fail-soft into `[]`. |
| try_import_guard buddy_management.py:821 (`apply_choice`) | **retired** — the `try` wraps `library.publish_review`, not an import; the `from …Petdex.review import drain_thread` inside it is a deliberate deferral. |
| try_import_guard buddy_workspace.py:149 (`_open_entry`) | **retired** — `except ValueError` -> user-facing warning, `except Exception` -> user-facing error, both notify. Not silent. |

## Verified-fine
- **`mcp_inspector` importing `Library.library_rag_state` / `library_rag_score_kinds`** (`:33-37`, used at `:718`/`:761`) looks like a cross-package reach but is duplication done RIGHT: `_ScoredRow` (`:659`) is a documented duck-typed shim so the Test Tool result summary uses the SAME canonical all-weak predicate the Library evidence list uses instead of copying its threshold logic.
- **`_validate_canonical_root`** (`audio_cpp_model_handoff.py:32-58`) re-rolls a canonicality check that `Utils/path_validation.validate_canonical_directory()` also performs — but deliberately: it runs in a frozen dataclass `__post_init__` (no filesystem I/O allowed) and accepts both POSIX and Windows absolute spellings for a value that crosses platforms. `validate_canonical_directory` does `resolve(strict=True)` + `lstat`. Different contracts; not a D4. (One widened acceptance worth knowing: a Windows-absolute spelling validates on POSIX and vice-versa. No `.`/`..` is ever accepted, so it is not a traversal hole.)
- **`_save_tools_mode_workspace_root`'s `validate_path(candidate, candidate.parent, …)`** (`mcp_workbench.py:4126-4131`) looks like a vacuous containment check (base == the candidate's own parent) — it is not useless: `validate_path` resolves symlinks, so a symlinked workspace root pointing outside its parent is rejected. For a root the user is explicitly choosing there is no other base to contain against.
- **Every `Static` in the MCP canvases that renders server/tool/finding text passes `markup=False`**, and the three DataTable cell builders wrap user text in `rich.text.Text(...)` (plain constructor). Spot-checked `mcp_servers_mode.py:270-277`, `mcp_permissions_mode.py:364-369`, `mcp_inspector.py:439-464`. No `.plain`/`str(label)` read-back re-escaping bug of the `console_display_state.py:93` class exists in this slice.
- **No mutable class attributes**: `ast` sweep for `list`/`dict`/`set` literals at class scope across all 35 files returned zero rows.
- **No module-level `threading.Lock`, no raw egress, one raw SQL statement** (`personas_conversations_controller.py:194`, parameterized, inside `db.transaction()`, and its only caller wraps it in `asyncio.to_thread` — `personas_screen.py:2130`).
- **`_toast` duplicated between `mcp_inspector.py:403` and `mcp_workbench.py:199`** — real, already filed as **task-2542**; both copies carry the documented "importing back the other way would be circular" rationale. Not re-recommended.
- **`screen_registry` routing table is healthy**: all 28 `ScreenRoute` module paths import and all 41 aliases / shell route ids / `ALL_TABS` entries resolve to a loadable class (probe output in the Findings section). The only failure is the alias-shadowed `customize` row.
- **`main_navigation.NavigationButton` is the LIVE class.** The same-named `Widgets/base_components.py:565 NavigationButton` has **zero production importers** (`grep -rn "base_components" tldw_chatbook/` returns only the file itself; only `Tests/UI/test_focus_token_parity.py`, `test_widget_css_consolidation.py` and `test_non_obscuring_focus_contract.py` reference it). Because Textual only folds a widget class's `DEFAULT_CSS` into the stylesheet when an instance is in the DOM, the dead twin's `NavigationButton { … }` type-selector rules never enter the live stylesheet either. No shadowing, Python-level or CSS-level.
- **Every `run_worker` in the slice passes `group=`** — the "exclusive=True without group" defect class is absent (42 call sites checked by `ast`).

## Retired
Everything in the Candidate-dispositions table marked **retired** was raised by the mechanical pass, read in context, and dropped with the evidence recorded there. Two raised by me and then retired:
- *"`_display_snapshot` pays a `get_cli_setting` per snapshot on every sync"* — **retired**: `_hub_lifecycle_timeout_seconds()` is called only inside `if snapshot.server_key in self._in_flight` (`mcp_workbench.py:1394-1396`), and `_in_flight` is empty except during a lifecycle op, where it holds one key. At most one extra read per sync, not one per row.
- *"`_run_advanced_action` writes `#mcp-adv-result` after an await, so hiding Advanced mid-run crashes like `_load_advanced_section`"* — **retired as a crash**: `result_widget` is resolved BEFORE the await and held as a reference; `Static.update()` on a detached widget does not raise. The stale write is invisible (the subtree is gone). Only `_load_advanced_section`'s post-await `query_one` is fatal.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| The P0's window is wide enough to hit by hand in the real app (my repro gates `load_section` on an `asyncio.Event`; the real width is however long the control-plane context resolution takes) | needs the running TUI against a real tldw_server target; the brief forbids running the app | `tmux -L verify new-session -d -s mcp '.venv/bin/python -m tldw_chatbook.app'` then navigate to MCP ▸ Servers, select a server-source target, press the inspector's "Advanced…" and immediately press it again; `tmux -L verify capture-pane -p -t mcp` (recipe: `.claude/skills/verify/SKILL.md`) |
| The ~146 ms per-`_sync_children()` aggregate is what a user actually feels on a rail click (measured piecewise against the isolated profile, not end-to-end in a mounted workbench) | needs a mounted `MCPWorkbench` with a real service; every piece is measured, the sum is arithmetic | `cd <worktree> && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY -m pytest Tests/UI/test_mcp_workbench.py -q` with a `time.perf_counter()` wrapper monkeypatched onto `MCPWorkbench._sync_children` |
| Per-tick cost of `BuddyManagementCoordinator.reconcile_scope` at 0.5 s | not measured; needs a fake app with a console runtime and a chat store | build a stub app with `console_runtime.chat_store.sessions()` returning N sessions and time `reconcile_scope()` over 1000 iterations |
| Whether the Servers-mode `all_tool_gates()` cost is paid on EVERY resync in practice (it is gated on `snapshot.source == "builtin"`, i.e. on the built-in row being the selected detail) | the selection state is runtime; the fresh-install preselect path is read from code (`_preselect_single_problem_on_load`), not observed | live: open MCP Hub on a fresh profile, confirm the built-in row is preselected, then toggle a gate and time the repaint (tmux recipe above) |
