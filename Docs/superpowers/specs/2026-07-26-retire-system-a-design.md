# Retire System A and make the tools config live (TASK-545 / P3)

**Date:** 2026-07-26
**Backlog:** TASK-545 Phase 3 (final), TASK-547, TASK-658.
**Branch:** `feat/tools-config-and-system-a` (worktree off `origin/dev` @ `5b1bda2d9`).
**Prerequisite:** P2 merged (PR #921, `5b1bda2d9`).

The last phase. P1 built the gate, P2 put real tools behind it; P3 removes the dead parallel system and makes the config that governs those tools actually reachable from the UI.

## What the investigation changed

**System A has no execution path.** Verified on merged `dev`:

- `ToolExecutor.execute_tool_calls` has **zero** production callers.
- `chat_streaming_events.py` — one of the two legacy handlers TASK-545 named — **no longer exists** (deleted in the task-577 campaign).
- `worker_events.py` — the other — has **zero** tool references.
- The only three production callers of `get_tool_executor()` are in `Tools_Settings_Window.py`, and all three call `get_available_tools()` only: a **listing**, never an execution.

So System A is a tool catalog with nothing behind it, rendering switches for tools that cannot be invoked from anywhere.

**"Remove `Tools/tool_executor.py` entirely" — one of the three options TASK-545's AC offers — is not achievable as written.** That 847-line file is about half load-bearing: it defines the `Tool` ABC (imported by `builtin_tool_gate`, `code_audit_tool`, and every tool module) and `CalculatorTool`/`DateTimeTool`, which System B actually runs.

**A deadlock, and a security-relevant one.** The Settings UI writes `tools_config[f"{tool_name}_enabled"] = switch.value` — precisely the keys P2 made live for the gated tools — but derives its switch list from `get_available_tools()`, which lists only *registered* tools. Registration is governed by the dead `get_cli_setting("tools", {})`. Measured:

```
System A registers: ['calculator', 'code_audit', 'get_current_datetime']
  read_file / write_file / create_note / update_note  → in the Settings list? False
```

A tool appears in the UI only if registered; it is registered only if its flag is on; the flag is settable only through that UI. Today the sole way to enable `write_file` is hand-editing `config.toml`.

## Design

### 1. Retire System A's execution machinery

Delete from `Tools/tool_executor.py`:

| Symbol | Why it goes |
|---|---|
| `ToolExecutor` (line 291) | zero production callers |
| `ToolResultCache` (line 86) | used only by `ToolExecutor` |
| `get_tool_executor()` / `reload_tool_executor()` (655, 838) | the dead `[tools]` read lives here |

Keep, untouched: `Tool` (20), `DateTimeTool` (520), `CalculatorTool` (564).

Drop the corresponding names from `Tools/__init__.py`'s imports and `__all__`. `Tests/Tools/test_tool_cache_json.py` tests the deleted cache and goes with it.

The module keeps its name. Renaming it to match its reduced role is churn across every importer for no behavioral gain; a docstring stating what it now holds is enough.

### 2. Repoint the Settings switches at System B

`Tools_Settings_Window.py`'s three call sites (compose ~3242, save ~4227, reset ~4324) stop calling `get_tool_executor().get_available_tools()`. The keys they write are unchanged — `{tool_name}_enabled` — because those are already exactly what `BuiltinToolProvider.__init__` reads.

**What they must not switch to is `BuiltinToolProvider().list_catalog()`.** That is the obvious move and it is wrong: a provider lists only the tools its gates *currently* permit, so a disabled tool is still absent from the list, and the deadlock reappears unchanged in a new place. Note also that the two shapes differ — the UI reads `tool_info["function"]["name"]`, while `list_catalog()` yields entries with a `.name` attribute — so this cannot be a drop-in substitution even mechanically.

The UI needs the **full set of gateable tools** with each one's current on/off state, which no provider instance can supply. Introduce a single source of truth in `tool_catalog.py` — the gate table the constructor already loops over, lifted to a module-level constant — and expose it as:

```python
def gateable_builtin_tools() -> tuple[tuple[str, str], ...]:
    """Return every config-gateable built-in as ``(gate_key, tool_name)``."""
```

The constructor loops over the same constant, so the UI and the runtime can never disagree about which tools exist. Unconditional tools (`calculator`, `get_current_datetime`) are listed with no switch — they cannot be turned off.

**Mutating tools appear in this list** (user decision). A switch makes a tool *reachable*; it does not grant silent execution — every `mutates`/`reads` tool still resolves to `ask` and raises an approval card per call. Each gated row shows its risk tags, so the switch states what enabling it grants.

### 3. Remove the orphaned controls

Six controls in that Settings section configure only the deleted executor: `tool-timeout-input`, `tool-max-workers-input`, and four cache controls (`tool-cache-enabled`, max size, TTL, persist). They are removed rather than left in place.

Leaving them would be worse than deleting them: they already control nothing, and after this change there is no executor for them to control even in principle. System B has no result cache and no worker pool.

The timeout does have a live successor — `RunBudget.max_tool_call_seconds` (300.0, currently a hardcoded constant) — but binding a Settings input to the agent run budget is TASK-659's agent-settings screen, not this task. Removing the dead input here does not remove a working feature; it stops the UI from claiming one.

### 4. TASK-547: satisfied by deletion, not by repair

TASK-547's AC says to fix the `get_cli_setting("tools", {})` call. That call disappears with `get_tool_executor()`. Its *intent* — "enabling a `[tools]` flag actually enables that tool" — is satisfied on the live path: `BuiltinToolProvider` already reads these keys with the working 3-argument form, and after §2 the Settings UI can set them.

The AC is reworded to match what shipped, as P2's was, rather than checked off against a call site that no longer exists.

### 5. TASK-658 and the wider bug class

`get_cli_setting("<bare section>", <non-string>)` returns the second argument **unconditionally** — `config.py`'s `if not isinstance(key, str) and default is None:` branch falls through to `return default` when the section has no dot. The same happens for a bare section with no key at all.

Fix `local_file_ingestion.py:1192` (`get_cli_setting("database", {})`), which makes `quick_ingest()` ignore a configured `media_db_path` and write to the hardcoded default.

TASK-658's AC also requires sweeping for further instances. An AST sweep plus runtime confirmation found **four more live ones**, all outside this task's subsystems:

| Site | Effect |
|---|---|
| `Widgets/splash_screen.py:196` | `[splash_screen]` ignored |
| `Widgets/settings_splash_screen_viewer.py:54` | same section, same bug |
| `Web_Server/serve.py:331` | `[web_server]` ignored (1-arg + keyword default) |
| `TTS/backends/openai.py:105,114` | `get_cli_setting("API")` / `("app_tts")` always `None` |

Confirmed at runtime: `[splash_screen]` and `[web_server]` **exist as populated dicts in a real config** and both call sites return their default regardless. The splash one matters most — `CLAUDE.md` documents `[splash_screen]` as a key configuration section, so every user who customized it has been silently ignored.

These are **filed, not fixed here.** Each lives in an unrelated subsystem and needs its own check of what honoring the config actually does — turning on four dormant config sections at once, inside a task about retiring a tool executor, is how a clean deletion becomes an unreviewable change.

Calls using the dotted form (`get_cli_setting("dictation.model", None)`) or an f-string/variable key are **not** affected: those are strings at runtime and resolve normally. A naive AST sweep flags them; the discriminator is the runtime type of the second argument.

### 6. `code_audit` loses its only registration

`CodeAuditTool` is registered solely at `tool_executor.py:807-810`. Deleting System A removes it from the catalog — but since System A cannot execute anything, this removes a listing that was never invocable, not a working feature. Porting it properly belongs to TASK-694, which already owns the four unported tools.

## Testing

- The Settings tool list contains every gateable built-in **including ones whose gate is currently off** — the deadlock regression test. It must fail against a `BuiltinToolProvider()`-derived list.
- Toggling a switch writes `{tool_name}_enabled`, and a provider built afterward registers that tool: the config → UI → runtime round trip.
- `gateable_builtin_tools()` and `BuiltinToolProvider.__init__` derive from the same constant — a test that adding an entry surfaces in both.
- Mutating tools listed by the UI still resolve to `ask` (a switch is not an approval).
- `quick_ingest()` honors a configured `[database] media_db_path`, and falls back only when genuinely absent.
- Nothing imports the deleted symbols: an import sweep, plus the suite.
- Default posture unchanged: no `[tools]` keys set → the runtime catalog is still exactly `calculator` + `get_current_datetime`.

## Acceptance criteria

- [ ] `ToolExecutor`, `ToolResultCache`, `get_tool_executor`, `reload_tool_executor` are deleted, along with their `Tools/__init__.py` exports and the cache's test file
- [ ] `Tool`, `CalculatorTool`, `DateTimeTool` remain importable from `Tools.tool_executor`; `builtin_tool_gate`, `tool_catalog`, and `code_audit_tool` are unaffected
- [ ] `gateable_builtin_tools()` exists, and `BuiltinToolProvider.__init__` iterates the same constant it derives from
- [ ] The Settings tool list shows every gateable built-in regardless of its current gate state, with unconditional tools shown as non-toggleable
- [ ] Toggling a switch and saving causes a subsequently-built `BuiltinToolProvider` to register (or drop) that tool
- [ ] Gated rows display their risk tags; enabling a mutating tool still produces an approval prompt on call
- [ ] The six orphaned timeout/worker/cache controls are gone from the Settings tools section
- [ ] `quick_ingest()` honors `[database] media_db_path`; a test pins both the configured and absent cases
- [ ] The four further `get_cli_setting` bug-class instances are filed with their evidence
- [ ] With no `[tools]` keys set, the runtime built-in catalog is exactly `calculator` + `get_current_datetime`
- [ ] TASK-547's and TASK-545's ACs are reworded to match what shipped; TASK-545 can then be marked Done

## Follow-ups to file

1. `[splash_screen]` config ignored at two call sites (highest impact — a documented config section).
2. `[web_server]` config ignored in `serve.py`.
3. TTS OpenAI backend's `get_cli_setting("API")` / `("app_tts")` always return `None`.
4. A lint/guard so `get_cli_setting("<bare section>", <non-string>)` cannot silently reappear — this bug class now has six known instances across five subsystems.
