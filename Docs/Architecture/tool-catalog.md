# Tool catalog and providers

This document describes how tools become visible to agents and get dispatched: the `ToolCatalogRegistry` provider seam, the provider families (builtin, local `fs_*`/git, skills, MCP, canvas/library/profile), namespacing and shadowing, per-run composition, and the runtime tools the loop owns itself.

## Authoritative files

| File | Role |
| --- | --- |
| `Agents/tool_catalog.py` | `ToolProvider` Protocol, `ToolCatalogRegistry`, `BuiltinToolProvider`, `SkillToolProvider`, `ToolExecutionPolicy`, all runtime-tool schemas |
| `Agents/local_tool_provider.py` | `LocalToolProvider`, `LocalToolSpec`, admitted workspace roots model |
| `Agents/mcp_tool_provider.py` | `MCPToolProvider`, `MCPPendingCall` |
| `Agents/library_tool_provider.py`, `library_rag_tool_provider.py`, `profile_tool_provider.py`, `canvas_tool_provider.py`, `raw_shell_tool_provider.py`, `virtual_cli_provider.py` | The remaining provider families |
| `Tools/tool_executor.py` | `Tool` ABC plus the two always-on builtins: `DateTimeTool` (`get_current_datetime`) and `CalculatorTool` (`calculator`, AST-whitelisted eval) |
| `Tools/local_tool_impls.py`, `git_tool_impls.py`, `patch_tool_impls.py`, `web_tool_impls.py`, `virtual_cli_impls.py` | Synchronous tool implementations |

There is no `AVAILABLE_TOOLS` global anymore (the old batch executor was retired). The source of truth is `_GATEABLE_BUILTINS` plus `ALWAYS_ON_BUILTIN_NAMES` in `tool_catalog.py`, gated by `[tools]` config keys.

## The provider seam

`ToolProvider` is the minimal protocol: `list_catalog` / `load_schema` / `invoke`. Optional protocols extend it: `ToolRecordProjectionProvider` (content-boundary projection for run records) and `PathAwareToolProvider.path_targets` (validated scopes for project-instruction activation — see [console-file-authority.md](./console-file-authority.md)).

### Registry mechanics

- `register_provider` appends and invalidates a snapshot cache; **registration order defines shadowing** — the first registrant wins on id/name collisions.
- One provider sweep builds an immutable catalog snapshot (`by_id`, `by_name`, `entries`). `AgentService` resets the cache at run start (per-run freshness doctrine — registries are never reused across runs).
- `invoke_by_name` is **the** dispatch choke point, in order: canvas authority checks → JSON-string argument coercion/repair → persona call-cap check (`RunToolPolicy`, refuses before dispatch) → ephemeral-session gate (unaudited sources refused) → `provider.invoke(tool_id, args)`. It never raises — it returns a `ToolResult`.
- `timeout_for(name)` honors duck-typed per-provider overrides; `execution_policy_for(name)` fails closed to `BOUNDED_ABANDONABLE`.
- `probe_initial_catalog` discloses full schemas only if every cumulative prefix fits `max_schema_tokens`, otherwise the agent discovers tools progressively via `find_tools`/`load_tools` (ADR-104 token-budgeted disclosure).
- Authenticated registration seams (`register_builtin_library_provider`, `register_canvas_provider`) are fail-closed against lookalikes: they require exact provider types plus exact authority objects.

### Namespacing

| Family | Tool id | LLM-facing name |
| --- | --- | --- |
| Builtin | `builtin:<name>` | bare name |
| Local | `local:<name>` | bare name; permissions resolve under the synthetic hub server `local:__local__` |
| Skill | `skill:<name>` | bare name |
| MCP | — | `mcp__<server>__<tool>` (deduplicated; can never equal a bare builtin name) |
| Runtime (loop-owned) | `runtime:<name>` | bare name |

## Per-run composition

The Console bridge composes a fresh registry per `run_reply`, in a fixed order: Builtin → Canvas → Local → virtual CLI → raw shell → Library (authenticated) → profile → Skills → MCP. MCP registers last and is therefore shadowed by every earlier provider. The composition returns the registry plus the run's `allowed_tools` set. Ephemeral sessions drop skill and MCP providers entirely.

## Local tools and file authority

`LocalToolProvider` exposes the `fs_*` family (`fs_list`, `fs_read`, `fs_write`, `fs_edit`, `fs_glob`, `fs_grep`), git tools, patch tools, and web tools. Local tools deliberately reuse the MCP permission model through the `local:__local__` hub server key (ADR-032), so they appear in the same permission store and approval cards. Their root model is the admitted-roots contract described in [console-file-authority.md](./console-file-authority.md): `None` = legacy single `workspace_root`; `()` = path tools stripped entirely; ≥1 root = per-alias specs with an injected `root_alias` parameter when multiple. Read-only composition filters every mutating spec — `fs_write`/`fs_edit`/`fs_patch`/todo tools are simply not advertised.

Tool results that exceed budgets spill into the chat's private scratch space (`tool-spill`, per-result cap 32 KiB, aggregate inline budget 256 KiB, 0600 files in a 0700 dir); the preview names the read-back path so `fs_read` can page it back with no new grant. The opaque absolute scratch locator is stripped from every model/log-bound result via the redaction-root seam.

`fs_write` is CAS-capable (`expected_sha256`/`expected_absent`, portalocker inode lock, `O_NOFOLLOW` descriptor writes) and the provider injects stale-write guards from the per-run ReadLedger — a write to a file the run has not (re)read refuses with "Re-read the file and retry".

## Skills as tools

`SkillToolProvider` lists skills from a per-run snapshot (`{name, description, argument_hint}`; the schema is a single free-form `args` string). Its `invoke()` **raises `RuntimeError` by design** — skill tools never dispatch through the registry. The Console bridge intercepts skill names before registry dispatch and runs them through the run-scoped skill runner, which:

1. re-verifies trust at render time (fingerprint digest comparison; mismatch → `SkillTrustBlockedError`),
2. renders the skill (`skills_service.execute_skill`, mode `"local"`),
3. intersects the skill's declared `allowed_tools` against the run's builtin+local names — narrow-only, a skill can never grant a tool the run lacks,
4. spawns the rendered prompt as a child run.

Script skills (`run_skill_script`) add three independent gates: the runtime policy launch rule, per-run trust re-verification against the trusted manifest, and a human confirm card whose "Always allow" grant is pinned to the fingerprint digest. Script execution is POSIX-only (sandbox support gate).

## Runtime tools

The loop owns several tools itself (ids `runtime:<name>`, schemas in `tool_catalog.py`): `spawn_subagent`, the fleet tools (`wait_agents`, `check_agents`, `send_to_agent`, message/steering tools, worktree merge/discard), `find_tools`/`load_tools`, `skill_file` / `install_skill` / `run_skill_script`, `fork_chat` / `new_chat` (ADR-150), and the run-log search tools. Fleet waits dispatch in-loop rather than through `invoke_tool` so they are not killed by the per-call timeout (see [agent-runtime.md](./agent-runtime.md)).

## MCP tools

`MCPToolProvider.compose_catalog()` builds the agent-visible MCP tools once per registration: kill switch → empty catalog; otherwise local external records plus the builtin inventory, one batched `effective_tool_states` resolution, `deny` states dropped, persona maximums enforced, names deduplicated. Invocation serializes on a per-provider lock (the stdio session is not proven multiplex-safe), peeks per-run approval stamps, applies the persona floor, and routes execution onto the main loop bounded by the configured tool timeout. The full permission decision lifecycle is in [mcp-hub.md](./mcp-hub.md).

## Config keys

`[tools]`: `read_file_enabled`, `list_directory_enabled`, `write_file_enabled`, `create_note_enabled`, `update_note_enabled`, `glob_files_enabled`, `grep_files_enabled`, `expand_document_enabled` (all default **false**), `web_deep_search_enabled` (double opt-in), `ask_user_enabled` (default on). Every gate is read through `coerce_bool_setting` — a quoted `"false"` in TOML is truthy and must not fail a gate open.

## Boundaries

- `Agents/` owns the loop, the provider Protocol, per-run composition, and the approval seams.
- `Tools/` owns synchronous implementations and workspace file-root context; it has no knowledge of the registry.
- `MCP/` owns the permission store, hub catalog normalization, and the external client/server (see [mcp-hub.md](./mcp-hub.md)).

## Verified gotchas

1. Skill tools must never go through `provider.invoke` — it raises by design; the bridge dispatches them before registry dispatch.
2. Registration order is shadowing order; MCP is deliberately last.
3. `local:__local__` resolver consults only `HIGH_RISK_TAGS = {"mutates","process"}` — the local specs' `reads`/`network` tags are deliberately inert markings; the real protection is the default-ask global plus in-workspace hard refusals.
4. The default `workspace_root` is the app cwd — launching from `$HOME` makes `$HOME` the confinement root, which is why the sensitive-path denylist is a mandatory second layer.
5. Registry composition is never cached across runs; resetting the catalog cache at the wrong moment would disarm the run's persona call caps.
