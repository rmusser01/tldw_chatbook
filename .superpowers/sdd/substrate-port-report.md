# Agent-runtime substrate port report

Ported six changes from `wt-builtin-tool-packs` (`docs/builtin-tool-packs-spec`,
read-only reference, 418 commits behind dev) onto `wt-agent-substrate`
(`feat/agent-runtime-substrate`, cut fresh from `origin/dev`). Dev had already
independently landed related work (`BUILTIN_HIGH_RISK_TAGS`, `max_total_tokens`,
`max_tool_call_seconds`, the `CONSOLE_MAX_*` named-constants refactor at
20/64/1200) that the reference branch did not have in the same form, so each
change was re-derived against dev's actual code rather than copy-pasted.

## Commits (in order)

1. `6c0371d86` feat(agents): raise tool ceiling to 24 and disclosure threshold to 16
2. `0159741bd` feat(agents): cap tool results at the history-append seam
3. `8301ef2f8` feat(agents): per-tool timeout override resolved through the registry
4. `380323ac2` feat(mcp): float builtin egress tools to ask via BUILTIN_HIGH_RISK_TAGS
5. `d9049a7d6` feat(agents): surface MCP tools shadowed by built-in name collisions
6. `a86d319c8` feat(console): resize the agent run budget (20 turns -> 30)

All commits are on top of `origin/dev` HEAD (`963de5f79`), no other files touched.

## Per-change notes

### 1. Tool ceiling / disclosure threshold (`agent_models.py`)
Dev had `DIRECT_DISCLOSE_THRESHOLD = 8` / `RunBudget.max_active_tools = 8`,
unchanged from the reference's starting point — raised to 16/24 as specified.
The reference branch's own commits (`799180c35`, `41bc12795`) carried no
rationale comment matching the "one-way ratchet" wording given in the task, so
that comment block (above `DIRECT_DISCLOSE_THRESHOLD` and beside
`max_active_tools`) was newly authored here, matching the given rationale.
Test fixtures that hardcoded `range(8)`/`== 8` (`test_skill_tool_spawn.py`'s
`_nine_entry_names`, `test_console_agent_bridge.py`'s `_ManySkillsService`)
were updated to derive from the constant, mirroring the reference's own
follow-up commit `41bc12795` ("test(agents): derive disclosure-threshold test
fixtures from the constant").

**Adaptation:** `agent_models.py` is edited by three of the six changes (1, 2,
6) in tightly interleaved regions (a few lines apart in the same class body /
same docstring). Hunk-level `git diff -U0` analysis showed these could
technically be split into 9 zero-context hunks, but zero-context patches are
fragile to apply out of original order via separate `git apply` invocations
(no context to re-locate the hunk if line offsets drift), so — to avoid risking
the already-green working tree — `agent_models.py` and `test_agent_models.py`
were kept whole in commit 1. Commit 1's message documents this explicitly.

### 2. Tool-result cap at the history-append seam (`agent_runtime.py`)
Added `RunBudget.max_tool_result_chars: int = 16000` and
`_truncate_tool_result()`, called **unconditionally** right before
`add(STEP_TOOL_RESULT, ...)` — covering both the normal dispatch path and the
review-hook refusal path (`verdict != "proceed"` sets `content` earlier in the
same `if/else`). Matches dev's actual `run_agent_loop` structure exactly (verified
line-for-line against the reference's two-commit history, `1c74d9ea3` then
`5025a88f7`, which fixed the same "refusal path uncapped" gap the task called out).
Ported both integration tests that drive `run_agent_loop` end-to-end
(`test_run_agent_loop_truncates_oversized_tool_result_in_history`,
`test_run_agent_loop_truncates_review_hook_refusal_in_history`) plus the three
pure-helper unit tests. `clamp_child_budget` now passes `max_tool_result_chars`
through (ported test `test_clamp_child_budget_preserves_max_tool_result_chars`,
bundled into commit 1's `agent_models.py` for the reason above).

### 3. Per-tool timeout override
`Tool.timeout_seconds` (default `0.0`) added to the ABC in
`Tools/tool_executor.py` — dev never relocated this ABC to `Tools/base.py` (that
relocation was explicitly out of scope), so the property landed in its
original, still-current home. `BuiltinToolProvider.timeout_for()` and
`ToolCatalogRegistry.timeout_for()` added to `tool_catalog.py`, matching dev's
current class structure hunk-for-hunk. `agent_service.py`'s `invoke_tool` now
does `self.registry.timeout_for(call.name) or config.budget.max_tool_call_seconds`
— `0`/`None` from a tool with no override falls through to the existing budget
default, never a literal zero-second timeout. Ported
`test_registry_timeout_for_reports_a_tools_own_ceiling`.

### 4. `network` risk tag
Confirmed dev already has `BUILTIN_HIGH_RISK_TAGS = HIGH_RISK_TAGS | frozenset({"reads"})`
in `MCP/permission_store.py` with the documented "MCP deliberately keeps
HIGH_RISK_TAGS" comment. Added `"network"` to `BUILTIN_HIGH_RISK_TAGS` only,
extended that comment in its existing style, and updated the two places that
echo the tag vocabulary (`resolve_builtin_state`'s docstring,
`Tool.risk_tags`'s docstring). **Did not** touch `HIGH_RISK_TAGS` itself (the
old reference behaviour this port explicitly avoids). Added
`test_network_tag_floors_inherited_allow_to_ask` to
`Tests/Agents/test_builtin_tool_gate.py` (a `_Networked(Tool)` fixture +
`resolve_builtin_state` assertion, adapted from the reference's version which
used the same helper against the now-wrong `HIGH_RISK_TAGS`). Extended the
existing `test_builtin_risk_set_is_a_strict_superset_of_the_mcp_set` pin in
`Tests/MCP/test_permission_store.py` to also assert `"network"` is in
`BUILTIN_HIGH_RISK_TAGS` and not in `HIGH_RISK_TAGS`.

### 5. MCP name-shadowing visibility
`console_agent_bridge.py`'s `_non_colliding_mcp_names`/`_compose_run_registry_and_allowed`
matched the reference's pre-change state byte-for-byte, so the port was a
direct application: added `shadowed_mcp_names()` and the shared
`_partition_mcp_catalog_by_collision()` helper (both `_non_colliding_mcp_names`
and `shadowed_mcp_names` are now thin views over one partition), and a
`logger.warning(...)` per dropped name at the composition site. Ported all
three tests (`test_shadowed_mcp_names_reports_what_the_filter_drops`,
`test_compose_run_registry_and_allowed_warns_when_mcp_tool_is_shadowed`,
`test_compose_run_registry_and_allowed_no_warning_without_mcp_collisions`) using
the existing `_FakeMCPProvider`/loguru-sink pattern already present in dev's
test file.

### 6. Console budget resize
Dev already had the PR #869 "name the constants" refactor
(`CONSOLE_MAX_MODEL_TURNS`/`CONSOLE_MAX_STEPS`/`CONSOLE_MAX_WALL_SECONDS` as
named constants at 20/64/1200, no `CONSOLE_MAX_TOTAL_TOKENS`) — an
intermediate state between the reference's own history. Applied the
subsequent raise (20→30 / 64→96 / 1200→1800, `CONSOLE_MAX_TOTAL_TOKENS =
1_000_000` wired into `CONSOLE_RUN_BUDGET`) and rewrote the explanatory
comment block in place (old derivations removed, not left stale beside new
numbers). `DEFAULT_MAX_MODEL_TURNS` raised 20→30 in `agent_models.py`
(bundled into commit 1, see above). `clamp_child_budget`'s docstring refreshed
to the "90 at 30/2" worst case and `max_tool_result_chars` pass-through added
(this was the fix the reference branch itself needed a follow-up commit,
`c96a6a718`, to catch — ported directly here as part of the initial edit, plus
its test `test_clamp_child_budget_preserves_max_tool_result_chars`). Existing
dev tests `test_console_budget_step_cap_admits_a_full_model_turn_run` and
`test_console_budget_reaches_its_model_turn_cap_before_step_cap` already
derived from `CONSOLE_RUN_BUDGET.max_model_turns`/`max_steps` rather than
literals, so they needed no changes and pass unmodified against the new values.
Added `test_console_budget_bounds_spend_not_only_time`.

## Not ported (out of scope, confirmed absent from dev and left alone)

- `builtin_packs/` package, `builtin_services.py`, `builtin_pack_config.py`
- `Utils/sensitive_paths.py`, `glob_files`/`grep_files`
- The `Tools/base.py` relocation of the `Tool` ABC (dev's `tool_executor.py`
  still hosts it — confirmed this is where the per-tool-timeout reference
  commit itself was made too, before the later relocation)
- `ToolExecutor`/`code_audit_tool` deletions
- Any `Tools_Settings_Window.py` change (dev's `_GATEABLE_BUILTINS` table
  approach is untouched)
- Reference-branch-only test files with no dev equivalent (`test_builtin_pack_config.py`,
  `test_builtin_packs.py`) — dev's own replacements
  (`test_builtin_gate_live_tools.py`, `test_builtin_provider_workspace_binding.py`,
  `test_builtin_tool_risk_tags.py`, `test_gateable_builtin_tools.py`) were left as-is.

## Test commands run (foreground, venv from the main checkout since the
worktree has no venv of its own — confirmed via `sys.path` that the editable
install resolves `tldw_chatbook` from the worktree's own source when invoked
with cwd inside the worktree, not the main checkout)

```
source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate
cd /Users/macbook-dev/Documents/GitHub/wt-agent-substrate
python -m pytest Tests/Agents/ Tests/Chat/test_console_agent_bridge.py Tests/MCP/ -q
```

Result (run twice: once mid-port, once against the final committed state):
**774 passed**, 3 pre-existing warnings (unrelated `RuntimeWarning: coroutine
... was never awaited` in `test_mcp_tool_provider.py`, and a `requests`
dependency-version warning — both present before this port).

Breakdown: `Tests/Agents/` 348 passed, `Tests/Chat/test_console_agent_bridge.py`
85 passed, `Tests/MCP/` 341 passed.

Did not run the full suite (per instructions — exceeds 4 hours here). Did not
hit the documented baselines (`Tests/Chat/test_chat_functions.py::TestChatApiCall`
pytest-mock error, `Tests/TTS`/`Tests/Transcription` collection failures) since
those paths were out of scope for the required command.

## Commit-splitting method (for anyone re-deriving these commits)

Where a file's edits for two changes lived in cleanly separable hunks
(`console_agent_bridge.py`, `test_console_agent_bridge.py`,
`test_agent_runtime.py`'s single append hunk manually split at the function
boundary), commits were built via `git apply --cached <hunk-subset>` against
the real index rather than staging the whole file, so each commit's diff is
exactly the intended change. `agent_models.py`/`test_agent_models.py` were the
one exception (see change 1 above) — kept whole in commit 1 for safety rather
than risk a fragile zero-context patch application.
