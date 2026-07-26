# Port mutating tools behind the gate (TASK-545 / P2)

**Date:** 2026-07-25
**Backlog:** TASK-545 Phase 2. Prerequisites TASK-628 (child-run stamp scoping) and TASK-656 (permissions UI, née 627) are both **Done**.
**Branch:** `feat/port-mutating-tools` (worktree off `origin/dev`).

Phase 2 of three. Puts real mutating tools behind the gate P1 built, and closes a live silent-read gap found while designing this. P3 remains: `[tools]` config and the legacy System A decision.

## What this phase actually turns on

The built-in approval path is **currently unreachable in production by design**. `resolve_builtin_state` floors an inherited `allow` to `ask` only when a tool's tags intersect the high-risk set, and **no shipped tool overrides `risk_tags`** — so `BuiltinToolGate.check()`'s `ask` branch, the review hook's built-in rows, the approval card, and TASK-628's `stamp_scope` are all exercised solely by tests today.

Tagging tools is what makes that machinery live for real users for the first time. That, not the mechanical registration, is the risk this spec is organized around.

## Scope

**Port and tag `mutates`:** `write_file`, `create_note`, `update_note`.

**Tag only (already registered):** `read_file`, `list_directory`. **task-584** already wired these into `BuiltinToolProvider.__init__` behind the pre-existing `[tools] read_file_enabled` / `list_directory_enabled` gates (default-off). They carry **no** `risk_tags`, so a user who enables either today gets a filesystem-read tool that resolves to the `allow` floor and runs **with no prompt at all**. That gap is live now, independent of this port, and is closed here.

**Not in scope:** `rag_search`, `web_search`, `search_notes`, `code_audit`.

## Design

### 1. A built-in-only risk vocabulary

`HIGH_RISK_TAGS = frozenset({"mutates", "process"})` is consulted at exactly two call sites — `permission_store.py:657` (`resolve_effective_state`, MCP) and `:726` (`resolve_builtin_state`, built-ins). Extending that shared set would make **MCP** tools carrying the new tag start prompting too — a behavior change to a subsystem this phase must not touch.

So add a separate set and change only the built-in call site:

```python
#: Risk tags that floor an INHERITED allow to ask for in-process built-ins.
#: A superset of HIGH_RISK_TAGS: built-ins additionally treat filesystem
#: reads as prompt-worthy, because an agent reading arbitrary sandbox files
#: is a disclosure risk even though it mutates nothing. MCP keeps
#: HIGH_RISK_TAGS unchanged -- widening the shared set would make remote
#: tools carrying "reads" start prompting, which is not this phase's call.
BUILTIN_HIGH_RISK_TAGS = HIGH_RISK_TAGS | frozenset({"reads"})
```

`resolve_effective_state` is unmodified. Only `resolve_builtin_state`'s flooring condition swaps to the new set.

`Tool.risk_tags`'s docstring must be updated in the same change — it currently states "The vocabulary is the permission store's `HIGH_RISK_TAGS` (`mutates`/`process`)", which this design makes false for built-ins. A stale docstring on the exact property implementers read to decide how to tag a tool is worse than none.

### 2. Tagging

`risk_tags` is a concrete property on the `Tool` ABC defaulting to `()`; no shipped tool overrides it yet. Override on five classes:

| Tool | Tag | Why |
|---|---|---|
| `WriteFileTool` | `("mutates",)` | writes/overwrites/appends files |
| `CreateNoteTool` | `("mutates",)` | inserts a note row |
| `UpdateNoteTool` | `("mutates",)` | mutates an existing note under optimistic locking |
| `ReadFileTool` | `("reads",)` | reads arbitrary sandbox file content |
| `ListDirectoryTool` | `("reads",)` | enumerates sandbox structure |

### 3. Registration follows the shipped pattern

`BuiltinToolProvider.__init__` already has the shape: a loop over `(gate_key, factory_name)` pairs importing lazily inside a `try`, skipping on `get_cli_setting("tools", gate_key, False)`, and swallowing an unavailable tool as merely absent. The three mutating tools join that loop, **default-off**, matching the read tools.

The gate keys are **not new**. `write_file_enabled` (`tool_executor.py:735`), `create_note_enabled` (`:763`) and `update_note_enabled` (`:789`) already govern these same tools on the legacy path. Reusing them is task-584's precedent verbatim — "they stay behind the SAME config gates that already govern them" — and carries one implication worth stating: a user who already set one of these keys gets the tool on **both** paths after this change. On the agent path it arrives gated to `ask`, so it prompts rather than acting silently.

The note tools import `NotesInteropService`, so their registration must tolerate an import failure the same way — the existing `try`/`continue` already does.

Note this uses the **three-argument** `get_cli_setting` form, which works — unlike the section-dict form that TASK-547 documents as permanently returning `{}`. So these gates are genuinely reachable by a user editing `config.toml`, and P2 does not depend on P3.

### 4. Note tools: resolve the real user

Both note tools hardcode `user_id="default_user"` (with a literal `# Would be actual user in production` comment) while the app assigns `self.notes_user_id = settings.get("USERS_NAME", …)` **once** at init and never reassigns it. Anyone who set `[general] users_name` would have agent-created notes land in a bucket their Notes UI never reads.

Resolve at execute time from **`load_settings()["USERS_NAME"]`** — the identical source `app.notes_user_id` comes from.

**Do not use `get_cli_setting("general", "users_name", …)`.** That reads only the TOML, whereas the real value is `os.getenv("USERS_NAME", toml_value)` resolved inside `load_settings()`. With the env var set, a config read would diverge from `notes_user_id` and land notes in a *third* bucket — reintroducing the same bug in a subtler form.

Execute-time resolution (rather than threading through constructors) matches the existing pattern in the same tool family — `file_operation_tools` resolves its sandbox root lazily the same way — and avoids threading an argument through **four** production `BuiltinToolProvider()` construction sites (`console_chat_controller.py:3733`, `console_agent_bridge.py:822` and `:931`, `builtin_tool_gate.py:352`), two of which have no app access.

### 5. Register the new names with the existing shadow guard

`ToolCatalogRegistry` resolves same-named entries **first-registrant-wins** (an explicit `setdefault` in `_build_owner_cache`), and `_compose_run_registry_and_allowed` registers built-ins (`:823`) before skills (`:827`) before MCP (`:834`). A user-authored skill named `write_file` or `create_note` is therefore permanently unreachable — and those are generic names.

The risk is **skills only**: MCP names are always minted `mcp__<server>__<tool>`, so they can never equal a bare built-in name.

**This is already solved.** `_SHADOWED_BUILTIN_NAMES` in `Library/library_skills_state.py` backs `skill_name_shadows_builtin()`, which `library_skills_canvas.py:388` calls on the **live Name field** — so a skill author is warned as they type, before the skill exists. That is a better seam than a compose-time log, and no new mechanism is warranted.

What P2 owes is **membership**: add `write_file`, `create_note`, `update_note` to that set. This is not optional bookkeeping. The set has a drift guard (`test_shadow_name_set_stays_in_sync_with_real_sources`) that constructs a `BuiltinToolProvider` **with default config** — so a config-gated tool is structurally invisible to it. task-584 hit exactly this and listed `read_file`/`list_directory` explicitly with a comment saying so. Miss this and the guard passes while the names go unwarned.

Mirror task-584's `test_gated_tool_names_are_covered_by_the_shadow_guard`, which enables the gates and asserts the visible names are a subset of the set — the test that *does* catch gated names.

## Testing

The tests that matter are not "is it registered" but whether the newly-live machinery behaves:

- A `mutates` tool resolves to `ask` and produces an approval row; approving executes it; **`Off` refuses absolutely** (a stamp must not override a resolved deny — the property Qodo caught in P1).
- A `reads` tool likewise prompts rather than running silently.
- A **sub-agent's** gated call reaches the approval card and the parent's stamps survive — the path TASK-628's `stamp_scope` exists to protect, and the first time it carries real tools.
- A denial returns `ToolResult(ok=False, …)`, never an exception into the pure loop.
- Note tools write under the resolved `USERS_NAME`, including when it is set via env var.
- `create_note`/`update_note` execute correctly on the **agent's worker thread** — they have zero tests anywhere today and have never run there. `CharactersRAGDB` is built for it (`threading.local`, `check_same_thread=False`), but that is unverified on this path.
- Default posture unchanged: with no `[tools]` flags set, the catalog is exactly what it is today.

## Known limitations carried, not fixed

- **`write_file` is confined to `<user data dir>/tool_sandbox`.** Safe, but an agent that can only write where the user never looks is of limited use. The root *is* configurable via `[tools] file_sandbox_root` (working 3-arg read), but nothing surfaces that. Worth stating plainly rather than letting users discover it.
- **`UpdateNoteTool.expected_version` defaults to `1`**, so an LLM calling it on any note edited more than once hits a spurious version conflict.
- **Every note call constructs a fresh `CharactersRAGDB`** rather than reusing the app's singleton, paying a full DB-open and schema check per call. Pre-existing; now on an agent-driven path.
- **Disclosure threshold:** `DIRECT_DISCLOSE_THRESHOLD = 8`. Today's baseline is 2 tools; all five gates on would make 7, still under. But with skills and MCP in the same catalog the total can cross 8 and flip disclosure to `find_tools`/`load_tools`. Not a defect — worth knowing.

## Acceptance criteria

- [ ] `BUILTIN_HIGH_RISK_TAGS` exists and is consulted **only** by `resolve_builtin_state`; `resolve_effective_state` is unmodified and MCP's existing tests pass untouched. `Tool.risk_tags`'s docstring names the built-in vocabulary correctly.
- [ ] `write_file`, `create_note`, `update_note` are registered behind default-off `[tools]` gates and tagged `("mutates",)`.
- [ ] `read_file`, `list_directory` are tagged `("reads",)`; enabling either produces a prompt rather than a silent execution.
- [ ] A tagged tool resolves to `ask`; approval executes it; a resolved `deny` refuses regardless of any stamp.
- [ ] A sub-agent's gated call reaches an approval route and the parent's stamps survive the nested run.
- [ ] Note tools resolve `user_id` from `load_settings()["USERS_NAME"]`, matching `app.notes_user_id`, including under an env-var override.
- [ ] `create_note`/`update_note` are covered by tests that execute them on a worker thread.
- [ ] `write_file`, `create_note`, `update_note` are members of `_SHADOWED_BUILTIN_NAMES`, covered by a gates-enabled test in the shape of task-584's, so the Library skill-name warning fires on them.
- [ ] With no `[tools]` flags set, the built-in catalog and disclosure behavior are unchanged from today.

## Follow-ups to file

1. `UpdateNoteTool.expected_version` default-of-1 spurious conflicts.
2. Per-call `CharactersRAGDB` construction in the note tools.
3. Surfacing the sandbox root (and its configurability) to the user, so `write_file` is discoverably useful.
