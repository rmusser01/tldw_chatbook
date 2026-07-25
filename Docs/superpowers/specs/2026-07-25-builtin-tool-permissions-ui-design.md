# Built-in tool permissions UI (TASK-627)

**Date:** 2026-07-25
**Backlog:** TASK-627, blocking TASK-545/P2's ability to offer persistent decisions.
**Branch:** `feat/builtin-permissions-ui` (worktree off `origin/dev`).

Surfaces the `agent:builtin` namespace in the MCP workbench's Permissions mode so a user can see each built-in tool's effective state and set or clear a **persistent** allow/deny. TASK-545/P1 shipped session-scoped decisions only, precisely because a persistent decision could not be undone in-app.

## Informed by a comparative spike

Two third-party agent frameworks were read at the code level (not README level) before designing: **Pi** (`earendil-works/pi`, TypeScript) and **CheetahClaws** (`SafeRL-Lab/cheetahclaws`, a Python reimplementation of Claude Code's loop with ~28 built-in tools plus MCP). The spike changed this design; the findings are recorded because they justify choices that otherwise look like extra work.

**CheetahClaws has the bug this design must avoid.** Its `_check_permission()` decides by **hardcoded built-in tool-name matching** (`("Read","Glob","Grep",…)` auto-approve; `("Write","Edit","NotebookEdit")` prompt). MCP tools are named `mcp__<server>__<tool>`, match none of those literals, and therefore inherit whatever each mode's fall-through happens to do:

- `auto`/`accept-edits` → falls to `return False`: every MCP call prompts, even one whose server declared `readOnlyHint: true`.
- `plan` mode → falls to a trailing `return True  # reads are fine`: **any MCP tool auto-approves silently, including mutating ones.** Plan mode's stated guarantee is not enforced for external tools.

Notably it *has* the metadata to do better — `ToolDef.read_only`, populated from MCP's `readOnlyHint` — but consumes it only for result caching, never for the permission decision.

**Consequences for this design:**

1. **Keep the per-namespace resolver.** `resolve_builtin_state` for `agent:builtin` and `resolve_effective_state` for MCP means neither namespace can fall through logic written for the other. An earlier draft of this task considered collapsing them into one path "to reduce plumbing"; the spike shows that collapse is exactly what produces CheetahClaws' plan-mode hole.
2. **Decide from risk metadata, not name lists.** Our `risk_tags` → `HIGH_RISK_TAGS` → floor-to-ask is the mechanism CheetahClaws lacks.
3. **The hash question has a clean answer.** CheetahClaws has no definition hash at all (`register_tool()` silently overwrites on reload — an undetected rug-pull vector). Our hash is a genuine strength for *remote* tools, and confirms P1's decision to skip it for in-process code. So persistent `allow` for `agent:builtin` is unblocked by **relaxing the guard for hash-free namespaces**, not by synthesizing a meaningless hash.

**Pi** contributes one negative data point: it has no per-call permission check at all (`bash` passes the model's command string straight to `spawn()`), no risk metadata on tool definitions, and no namespacing — extension tools and built-ins share one flat `Map` keyed by bare name, so an extension tool named `bash` **shadows the builtin**. That is the collision our `agent:builtin` vs `builtin:tldw_chatbook` separation already prevents.

## The three real gaps (verified)

1. **Rows come from the live MCP catalog only.** `MCPWorkbench._build_permission_rows()` iterates `tools_by_server` built from `_collect_hub_tools()`; the store payload is consulted only for a `server_key` that already came from that catalog. A key present only in the permission store is never a candidate — there is no filter to relax, it simply never enters the loop.
2. **The resolver is hardcoded.** `_resolve_effective_states()` calls `service.effective_tool_states(tools)`, which calls `resolve_effective_state` for every tool. Feeding built-ins through it would resolve them with MCP semantics (ask-floor + hash check) — wrong, not merely absent.
3. **Persistent `allow` is blocked at two layers.** `MCPPermissionStore.set_tool_state` raises `"definition_hash is required when state is 'allow'"`, and `UnifiedMCPControlPlaneService.set_tool_state` raises `"tool is required to set state 'allow'"` and computes a hash. `deny`/`ask`/inherit already work unchanged for any `server_key`; only `allow` is blocked.

The render layer is **not** a gap: `format_tool_state_label()` and `PermRow` take an already-built `EffectiveToolState` and are resolver-agnostic.

## Design

### 1. Hash-free namespaces

Add to `MCP/permission_store.py`:

```python
#: Server keys whose tools carry no meaningful definition hash.
HASH_FREE_SERVER_KEYS = frozenset({BUILTIN_TOOL_SERVER_KEY})
```

`MCPPermissionStore.set_tool_state` requires `definition_hash` for `state == "allow"` **unless** `server_key` is in that set. `UnifiedMCPControlPlaneService.set_tool_state` likewise skips the `tool is required` guard and passes `definition_hash=None` for those keys.

Rationale, and why this is not a weakening: the hash exists to detect a **remote** server mutating a tool after you trusted it. `agent:builtin` tools are in-process code shipped with the app — an attacker who can change them already has code execution, so the check protects nothing while guaranteeing a re-prompt on every release that edits a docstring. `resolve_builtin_state` already never reads the hash, so a stored value would be inert as well as meaningless. MCP's behavior is unchanged: any key not in the set keeps the guard exactly as today.

### 2. Enumerating built-in tools without a run

`BuiltinToolProvider` is per-run and has no module-level registry, and `list_catalog()` omits `input_schema`/`risk_tags`. The composition (new, in a helper — not in the UI):

```
BuiltinToolProvider()            # cheap: the gate is lazy, built only on invoke()
  .list_catalog()                # -> entries with id/name/one_line_description
  .tool_for(entry.name)          # -> the real Tool (description, parameters, risk_tags)
tool_ref(tool)                   # -> GatedToolRef  (existing, Agents/builtin_tool_gate.py)
resolve_builtin_state(payload, ref)  -> EffectiveToolState
```

### 3. Merging into the rows without touching the MCP path

`MCPWorkbench` gains a built-in section built from the above and merged into the `effective` map **before** `_build_permission_rows()` runs, and a corresponding entry in the server ordering. `effective_tool_states()` and `resolve_effective_state` are **not** modified — MCP resolution stays byte-identical.

**Do not reuse `HubTool` for these rows.** Its `source` field is documented `local|builtin|server` and `builtin` already means the built-in MCP *server*; reusing it would create exactly the conflation AC#3 forbids. The built-in section carries its own row-source type.

### 4. Fail closed on an unrecognized namespace

The merge helper resolves `agent:builtin` via `resolve_builtin_state` and everything else via the MCP path. A `server_key` matching neither must **fail closed** — rendered as `deny`/unknown, never silently inheriting either branch. This is the direct lesson from CheetahClaws' plan-mode hole: a fall-through decided its behavior, and the unsafe direction was the one that shipped.

### 5. Presentation

The built-in section is labelled distinctly from the built-in MCP server (AC#3). Suggested: `Built-in (agent runtime)` for `agent:builtin` versus the MCP server's existing `tldw_chatbook` label, with a one-line description naming these as in-process tools.

**Origin rendering needs no change.** Permissions mode does not render origin sentences — `format_tool_state_label()` renders only a *marker*, and its precedence is `config_changed ⚠` → `risk_floored ⚑` → `tool_override •` → plain label. All three behave correctly for a built-in row already: `config_changed` is always `False` from `resolve_builtin_state`, `risk_floored` is exactly the flag we want surfaced for a `"mutates"` tool, and `tool_override` marks a user-set persistent decision. `_ORIGIN_SENTENCES` lives only in `mcp_inspector.py` (which P1 already extended with `builtin_default`); this task does not touch it.

### 6. Bulk-action visual flag

`_set_all_batch_decisions` skips a row whose narrowed `options` cannot take the bulk value, leaving it visually identical to an untouched row. Add a `needs-decision` marker class applied to rows a bulk action skipped, cleared when the row's `Select` changes. Note this path is **not currently reachable** — today's only narrowed rows (`approve_once`/`approve_session`/`deny`) accept a candidate from both bulk buttons — so this is a guard for future narrowings, and its test must construct a row that genuinely excludes both.

### 7. Audit-log note

`_record_cancelled_approval_decisions` is owner-agnostic and already writes `agent:builtin` rows into the MCP execution log. That is desirable audit parity, but any audit filtering by server in this UI must not present `agent:builtin` as an MCP server.

## Acceptance criteria

- [ ] `agent:builtin` and its tools appear in Permissions mode with effective state and origin, resolved via `resolve_builtin_state` (never `resolve_effective_state`).
- [ ] A user can set and clear a persistent allow **and** deny for an individual built-in tool; the change is visible immediately without restart.
- [ ] `HASH_FREE_SERVER_KEYS` gates the relaxation; a **non**-listed `server_key` still raises when setting `allow` without a hash — pinned by a test asserting MCP's guard is unchanged.
- [ ] `resolve_effective_state` and `effective_tool_states` are unmodified; the existing MCP permissions tests pass untouched.
- [ ] The UI labels `agent:builtin` distinctly from `builtin:tldw_chatbook`; a test asserts the two are never presented as one section.
- [ ] A `server_key` in neither namespace fails closed rather than inheriting a branch.
- [ ] A bulk action that cannot apply to a row leaves that row visibly flagged; test constructs a row excluding both bulk candidates.
- [ ] Built-in tools are enumerated without starting an agent run.

## Out of scope

- Changing MCP's resolver, hash guard for MCP keys, or `effective_tool_states`.
- Permission **modes** (CheetahClaws' `auto`/`accept-edits`/`plan` axis) — a larger idea worth its own task; see below.
- Porting tools (TASK-545/P2), agent budget settings (TASK-634).

## Follow-ups to file

1. **Hard floor / never-allowable category.** CheetahClaws puts its destructive-command denylist *inside* the execution primitive, below the mode branch, so no permission level can bypass it. We have "Off is absolute" per-tool but no category that can never be enabled at all.
2. **Permission modes.** A mode axis (`auto`/`accept-edits`/`manual`/`plan`) is better UX than per-tool toggles and a natural home for TASK-634. If ever built: it must be driven by `risk_tags`, not tool-name lists, and must fail closed for unrecognized tools — the two things CheetahClaws got wrong.
3. **Never let the model change its own permission posture.** CheetahClaws' `EnterPlanMode`/`ExitPlanMode` are always-auto-approved and mutate `permission_mode` directly, letting the LLM widen its own authority. Record this as a standing constraint before any mode work.
