---
id: TASK-662
title: 'Permission modes: an auto/accept-edits/manual/plan axis'
status: To Do
assignee: []
created_date: '2026-07-25'
labels: [tools, security, agents, ux]
dependencies: [TASK-545, TASK-627, TASK-659]
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the comparative spike behind TASK-627 (`Docs/superpowers/specs/2026-07-25-builtin-tool-permissions-ui-design.md`). Today's permission surface is entirely per-tool: a user sets `allow`/`ask`/`deny` per tool (or per server, or a global default) via the MCP workbench's Permissions mode, one row at a time. CheetahClaws (`SafeRL-Lab/cheetahclaws`) instead exposes a small **mode** axis — `auto`, `accept-edits`, `manual`, `plan` — that a user switches as one unit, changing how *every* tool call is decided for the duration of that mode. That is a better UX shape for the common case ("I trust this session, stop asking me") than clicking through N per-tool toggles, and it is a natural home for TASK-659's agent-settings surface, which currently has no single "how cautious should this run be" control.

**This must not be built by copying CheetahClaws' actual decision mechanism** — the spike exists specifically because that mechanism is broken in a way directly relevant to this repo:

`_check_permission()` decides by **hardcoded built-in tool-name matching**: `("Read","Glob","Grep",…)` auto-approve, `("Write","Edit","NotebookEdit")` prompt. MCP tools are named `mcp__<server>__<tool>` and match none of those literals, so they fall through to whatever a given mode's fall-through happens to do:
- `auto`/`accept-edits` fall to `return False` — every MCP call prompts, even a read-only one whose server declared `readOnlyHint: true`.
- `plan` mode falls to a trailing `return True  # reads are fine` — **any MCP tool auto-approves silently, including a mutating one.** Plan mode's stated guarantee ("nothing changes state while planning") is not enforced for external tools at all.

CheetahClaws has the metadata to do this correctly (`ToolDef.read_only`, populated from MCP's `readOnlyHint`) but only ever consumes it for result caching, never for the permission decision — the risk signal exists and is simply not wired to the mode branch.

**If this is ever built here, it must be driven by `risk_tags` (this repo's existing `HIGH_RISK_TAGS` vocabulary, already resolver-agnostic across `agent:builtin` and MCP), never a tool-name list**, and **must fail closed for a tool/namespace the mode logic doesn't recognize** — the exact two properties CheetahClaws lacks. TASK-627's design already establishes the pattern to extend: a per-namespace resolver (`resolve_builtin_state` for `agent:builtin`, `resolve_effective_state` for MCP) so neither namespace's logic can fall through into the other's. A mode axis is a decision layer *above* that resolution, not a replacement for it, and must compose with (not bypass) TASK-661's hard-floor mechanism if that lands first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] A mode setting (at minimum an auto/manual-equivalent pair; `accept-edits`/`plan` may be later increments) governs the default per-call decision for a run, without removing the existing per-tool override mechanism
- [ ] Mode decisions are computed from `risk_tags` (or an equivalent risk classification already present on the tool/`GatedToolRef`), never from a tool-name or tool-name-pattern list
- [ ] A tool/namespace the mode logic does not recognize fails closed (denies or requires approval), never auto-approves via an unhandled fall-through branch — a test constructs such a tool and pins the closed behavior for every mode
- [ ] A "plan"-equivalent mode (if implemented) is proven, by test, to never auto-approve a tool carrying a mutating risk tag, for both `agent:builtin` and MCP-sourced tools
- [ ] The mode axis composes with the existing per-namespace resolvers (`resolve_builtin_state`, `resolve_effective_state`) as a layer above them rather than a third parallel decision path that could disagree with either
- [ ] If TASK-661's hard-floor mechanism exists by the time this ships, no mode can bypass it — a test pins this composition
<!-- AC:END -->
