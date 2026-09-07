# Substrate pre-merge review: fixes for findings 1-8

Branch: `feat/agent-runtime-substrate` (worktree `wt-agent-substrate`).
Scope: `tldw_chatbook/Utils/sensitive_paths.py`, `tldw_chatbook/Tools/file_operation_tools.py`,
`tldw_chatbook/Chat/console_agent_bridge.py`, `tldw_chatbook/Agents/agent_models.py`,
`tldw_chatbook/Tools/tool_executor.py`, plus matching test files.

## Finding 1 (CRITICAL) — permission-store path was a dead literal

**Root cause.** `sensitive_paths.py` hardcoded
`~/.config/tldw_cli/mcp_permissions.json`. The app never puts the permission
store there: `MCP/unified_control_plane_service.py`'s `permission_store`
property builds it as `Path(store.path).with_name("mcp_permissions.json")`,
where `store.path` is the `LocalMCPStore` path `app.py` constructs as
`get_user_data_dir() / "local_mcp_store.json"`. So the real path is
`get_user_data_dir() / "mcp_permissions.json"` — a directory the static
literal never pointed at.

**Reproduction (before).** With `HOME`/`XDG_DATA_HOME` isolated to a temp
dir, wrote `{"global_default": "ask"}` to the REAL path
(`get_user_data_dir() / "mcp_permissions.json"`), pointed the tool sandbox
root at `get_user_data_dir()`, then called `is_sensitive_path()` on it and
ran `WriteFileTool.execute(file_path="mcp_permissions.json", content=
'{"global_default": "allow"}')`:
- `is_sensitive_path(real_path)` → `False`
- `write_file` result: `{'action': 'overwritten', ...}`
- File on disk after the call: `{"global_default": "allow"}` — the
  one-step permission-gate bypass, live-reproduced.

**Fix.** Added `_sensitive_single_file_paths()` in `sensitive_paths.py`,
which resolves `config.get_user_data_dir() / "mcp_permissions.json"` (never
a literal) plus two companions built the identical
`Path(...).with_name(...)` way from the same base: `local_mcp_store.json`
(server definitions + env) and `mcp_execution_log.jsonl` (the execution
audit trail). `SensitivePathContext.files` now comes from this function.

**Reproduction (after).** Same script:
- `is_sensitive_path(real_path)` → `True`
- `write_file` result: `{'error': "Refused: 'mcp_permissions.json' is a
  protected path and cannot be written"}`
- File on disk unchanged: `{"global_default": "ask"}`.

**Regression tests** (derive the path the same way the app does, not a
literal): `Tests/Utils/test_sensitive_paths.py::
test_mcp_permission_store_is_refused_via_the_actually_used_path` (builds
the path via `MCP.local_store.LocalMCPStore` + `.with_name(...)`, exactly
mirroring `unified_control_plane_service.permission_store`) and
`test_mcp_permission_store_companions_are_refused`. Also updated
`Tests/Tools/test_file_tool_sandbox.py::
test_write_file_refuses_to_overwrite_sensitive_file` to target the real
path instead of the old fictional `~/.config/tldw_cli/` literal.

## Finding 2 (Important) — user-data-dir state files unprotected

**Root cause.** `_DB_PATH_ACCESSOR_NAMES` enumerates 11 `config.get_*_db_path`
accessors, but several databases/state files are created OUTSIDE
`config.py` (`app.py`, `RAG_Search/`, `UI/Views/RAGSearch/`, ...) and sit in
the same directory, unenumerated.

**Reproduction (before).** In an isolated `get_user_data_dir()`, wrote
`agent_runs.db`, `evals.db`, `local_mcp_store.json`, `tldw_cli_app.log` and
checked `is_sensitive_path()` on each: all four returned `False`.

**Rule chosen: refuse every FILE sitting directly (non-recursively) inside
`get_user_data_dir()`, checked by "is this an existing directory" — never by
name.** Rationale:
- A rule beats an enumeration here: new state files land in this directory
  constantly without ever touching `config.py`, so any accessor-name list
  permanently trails what the app actually creates next.
- Every legitimate use of this directory as a *container* creates a named
  SUBDIRECTORY instead of a loose file directly inside it —
  `tool_sandbox` (the default sandbox root itself), `chat_dicts`,
  `chromadb`, `exports`, `rag_profiles`, `skills` (confirmed by grepping
  every `get_user_data_dir() / "..."` call site in the repo). Excluding
  "is an existing directory" rather than hardcoding any of those names
  keeps every one of them reachable, including ones added later, with no
  dependency on the sandbox module's own default-name string.
- A path that does not exist yet (e.g. a `write_file` target for a
  brand-new file) is also "not a directory", so it still fails closed.
- Verified the one alternative considered — exempting by literal name
  (`"tool_sandbox"`) — would be brittle for the same trailing-enumeration
  reason and was rejected.

Implemented as a new `SensitivePathContext.user_data_dir` field and a final
check in `is_sensitive_path`: `resolved.parent == ctx.user_data_dir and not
resolved.is_dir()`.

**Reproduction (after).** Same four files → all `True`. Also confirmed the
existing per-file DB/sidecar checks and the (unmocked) default sandbox
subdirectory both still behave correctly (see "default configuration"
below).

**Regression tests:**
`test_arbitrary_direct_child_file_of_user_data_dir_is_now_refused` (a name
deliberately NOT one of the enumerated DBs/MCP files, proving it's a rule
not a list; checked both before and after the file exists),
`test_ordinary_file_inside_the_default_sandbox_subdirectory_is_still_allowed`
(nested one level deeper stays reachable). Updated
`test_db_sidecar_matching_is_exact_not_a_loose_prefix` to monkeypatch the DB
accessor to a location OUTSIDE `get_user_data_dir()` so it keeps testing
sidecar-exactness in isolation from this new rule. Updated
`Tests/Tools/test_file_tool_sandbox.py::
test_list_directory_filters_sensitive_entries_from_recursive_listing` to
place its `mcp_permissions.json` fixture at the real location.

## Finding 3 (Important) — `config.toml` denial ignored `TLDW_CONFIG_PATH`

**Root cause.** `sensitive_paths.py` hardcoded
`~/.config/tldw_cli/config.toml`, but `config._get_effective_config_path()`
honors a `TLDW_CONFIG_PATH` override — which this project's OWN test suite
sets on every single test via `Tests/conftest.py`'s autouse
`isolate_test_environment` fixture, to a path nowhere near the default.

**Reproduction (before).** Set `TLDW_CONFIG_PATH` to a temp path, wrote
`api_key = 'super-secret'` there, called
`is_sensitive_path(config._get_effective_config_path())` → `False`. (Also
observed directly: the pre-fix parametrized test
`test_credential_and_app_state_paths_are_refused[~/.config/tldw_cli/
config.toml]` only ever passed because it asserted the literal itself, not
the config file this project's test suite actually uses — proven when I
removed the two now-dynamic entries and the literal-based test failed for
an unrelated reason, confirming it was never testing the real behavior.)

**Fix.** `_sensitive_single_file_paths()` resolves
`config._get_effective_config_path()` (same accessor the DB paths already
use the override-aware pattern for) instead of a literal.

**Reproduction (after).** Same script → `is_sensitive_path(...)` → `True`.

**Regression tests:**
`test_config_toml_is_refused_via_the_actually_used_path` and
`test_config_toml_override_is_followed_when_retargeted` (retargets
`TLDW_CONFIG_PATH` again mid-test and confirms the NEW path is what's
protected, proving call-time resolution, not a value baked in at import).
Updated `test_read_file_refuses_sensitive_file_even_when_sandbox_root_
contains_it` to write to the real effective path instead of the literal.

## Finding 4 (Important) — dotted sandbox root inverted hidden-file protection

**Root cause.** `Utils.path_validation.validate_path` refuses every
candidate when `base_directory`'s own final component is dotted.
`ReadFileTool`/`WriteFileTool`/`ListDirectoryTool` all route through
`validate_path_multi` → `validate_path` against the sandbox root, so with
`[tools] file_sandbox_root = "~/.tldw_sandbox"` they refuse EVERYTHING
(over-broad, but the safe direction). `GlobFiles`/`GrepFiles` instead call
`_tool_sandbox_root()` and `root.glob(...)` directly, never passing through
`validate_path` at all — so a dotted root did not stop them.

**Reproduction (before).** Configured `file_sandbox_root` to a dotted
directory, put `secrets.txt` (a plain, non-hidden file) with
`API_KEY=sk-live-abc123` directly inside it:
- `read_file`: `{'error': "Failed to read file: Path 'secrets.txt' is
  outside every allowed root (.../.tldw_sandbox)."}` (refused, correctly
  but for the "over-broad" reason)
- `grep_files(pattern="API_KEY")`: returned
  `{'matches': [{'path': '.../.tldw_sandbox/secrets.txt', 'line': 'API_KEY=
  sk-live-abc123', ...}]}` — **leaked**.
- `glob_files(pattern="**/*")`: enumerated `secrets.txt` — disclosed the
  sandbox's structure through the same gap.

**Fix.** Added `_sandbox_root_is_hidden(root)` (`root.name.startswith(".")`)
in `file_operation_tools.py`, mirroring `validate_path`'s own check, called
in both `GlobFiles.execute` and `GrepFiles.execute` immediately after
resolving the sandbox root, returning the same
`"Access to hidden files/directories is not allowed"` error text
`validate_path` raises.

**Reproduction (after).** Same script:
- `grep_files`: `{'error': 'Access to hidden files/directories is not
  allowed'}` — no leak.
- `glob_files`: `{'error': 'Access to hidden files/directories is not
  allowed'}`.

**Regression tests** in `Tests/Tools/test_glob_grep_files.py`:
`test_glob_files_refuses_a_dotted_sandbox_root`,
`test_grep_files_refuses_a_dotted_sandbox_root` (asserts the secret string
itself is absent from the result, not just `"error" in result`),
`test_glob_files_consistent_with_read_file_on_a_dotted_root`.

## Finding 5 (Minor) — overstated token-ceiling comment

Corrected three near-identical claims (two in
`Chat/console_agent_bridge.py`'s `CONSOLE_MAX_TOTAL_TOKENS`-adjacent block
comment and constant docstring, one in `Agents/agent_models.py`'s
`clamp_child_budget` docstring) that said the ceiling "stops" the 90-turn
worst case. Actual behavior: `agent_runtime.run_agent_loop`'s
`total_tokens` is a per-run local, and `clamp_child_budget` passes
`max_total_tokens` through to each child UNCHANGED (not divided), so the
parent and each of up to `max_subagents=2` children can each independently
spend up to the full ceiling — the real worst-case aggregate is roughly 3x
the constant (~3M tokens at the Console's 1M setting), not bounded by it
directly. Reworded all three to say this; did not touch the constant or
`clamp_child_budget`'s clamping behavior.

## Finding 6 (Minor) — `GlobFiles`/`GrepFiles` missing outer `except Exception`

Both `execute()` methods now wrap their whole body (after the up-front
`pattern is required`/traversal-rejection guards) in one `try`, with the
existing narrow `except OSError` (sandbox root unusable) kept, plus a new
final `except Exception as exc: logger.error(...); return {"error": ...}` —
matching `ReadFileTool`/`WriteFileTool`/`ListDirectoryTool`'s own
catch-all. This also means a `RuntimeError` from `Path.expanduser()` inside
`_tool_sandbox_root()` (previously only caught by the narrow `except
OSError`, so it could escape) is now caught by the same outer handler,
since the narrow `except OSError` is nested inside the new outer `try`.

## Finding 7 (Minor) — `Tool.timeout_seconds` docstring named nonexistent tools

`run_command` and "ingestion, transcription" tools are not `Tool`
subclasses anywhere in this repo (confirmed: `grep -rn "def timeout_seconds"`
finds only the base-class definition itself, no override). Reworded the
docstring in `Tools/tool_executor.py` to describe the property generically
("a long-running external operation") instead of naming things that do not
exist here.

## Finding 8 (Minor) — double `list_catalog()` walk + per-turn re-warning

`_compose_run_registry_and_allowed` called both
`_non_colliding_mcp_names(...)` and `shadowed_mcp_names(...)`, each of which
independently calls `_partition_mcp_catalog_by_collision` — walking
`mcp_provider.list_catalog()` twice per run. Fixed to call
`_partition_mcp_catalog_by_collision` once directly and use both halves of
its result. The two public wrapper functions are unchanged (still used
elsewhere/tested directly).

The shadowed-name warning also fired unconditionally on every single
Console message (once per `run_reply`), re-logging the identical line every
turn of a long session. Added `_WARNED_SHADOWED_MCP_NAMES` (a
module-level set, private to `console_agent_bridge.py`) and
`_warn_shadowed_mcp_name_once`, mirroring `Internal_Prompts/resolver.py`'s
`_warn_once`/`_warned_ids` idiom exactly — including a dedicated reset,
since `Tests/Internal_Prompts/conftest.py` already established the pattern
of clearing that set around tests that assert on it. Added an autouse
fixture `_reset_shadowed_mcp_warning_dedup` in
`Tests/Chat/test_console_agent_bridge.py` that clears
`_WARNED_SHADOWED_MCP_NAMES` before and after every test in that file, so
the existing single-call warning tests stay order-independent.

**New tests:** `test_compose_run_registry_and_allowed_walks_mcp_catalog_
only_once` (asserts `mcp_provider.list_catalog_calls == 1`, a new counter
added to the `_FakeMCPProvider` test double),
`test_compose_run_registry_and_allowed_warns_about_a_shadowed_name_only_once`
(three simulated Console messages, one warning),
`test_compose_run_registry_and_allowed_warns_once_per_distinct_shadowed_name`
(pins the dedup is per-name, not global).

## Test runs

Focused, as each fix landed:
```
python -m pytest Tests/Utils/test_sensitive_paths.py Tests/Tools/test_file_tool_sandbox.py Tests/Tools/test_glob_grep_files.py -q
# 61 passed
python -m pytest Tests/Chat/test_console_agent_bridge.py -q
# 88 passed
```

Full required scope (foreground, venv python from
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv`, run from this
worktree):
```
python -m pytest Tests/Agents/ Tests/Tools/ Tests/Utils/ Tests/Chat/test_console_agent_bridge.py Tests/UI/test_tools_settings_window.py -q
# 6 failed, 972 passed, 16 skipped, 18 warnings in 175.69s
```
The 6 failures are exactly the pre-existing baseline named in the task
(`Tests/UI/test_tools_settings_window.py::test_chat_api_key_*`), unrelated
to this work. Re-ran a second time after the Finding 1-4 exploit
reproduction (which used `git stash`/`git stash pop` to compare pre- and
post-fix behavior) to confirm the working tree diff was untouched by that
detour (`git diff --stat` matched exactly, all 5 modules still parse).

## Default configuration proof (Finding 2 constraint)

`Tests/Tools/test_file_tool_sandbox.py::
test_default_sandbox_configuration_still_works_end_to_end` exercises
`WriteFileTool`/`ReadFileTool`/`ListDirectoryTool`/`GlobFiles`/`GrepFiles`
against the REAL, entirely unmocked default
(`get_user_data_dir() / "tool_sandbox"`) — no `_tool_sandbox_root` or
`_resolve_sandbox_config` monkeypatch anywhere in the test. All five calls
succeed. Independently re-confirmed with an ad hoc script calling
`fot._tool_sandbox_root()` for real and round-tripping write → read → list
→ glob → grep, all succeeding, with
`is_sensitive_path(real_default_sandbox_root)` returning `False`.

## Files changed

- `tldw_chatbook/Utils/sensitive_paths.py` — findings 1, 2, 3
- `tldw_chatbook/Tools/file_operation_tools.py` — findings 4, 6
- `tldw_chatbook/Chat/console_agent_bridge.py` — findings 5, 8
- `tldw_chatbook/Agents/agent_models.py` — finding 5
- `tldw_chatbook/Tools/tool_executor.py` — finding 7
- `Tests/Utils/test_sensitive_paths.py`
- `Tests/Tools/test_file_tool_sandbox.py`
- `Tests/Tools/test_glob_grep_files.py`
- `Tests/Chat/test_console_agent_bridge.py`
