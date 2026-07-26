---
id: TASK-545
title: Wire built-in tool executor into MCP permission gate
status: To Do
assignee: []
created_date: '2026-07-24 12:00'
updated_date: '2026-07-25'
labels: [tools, security, agents]
dependencies: [TASK-331]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Rescoped 2026-07-25 — see `Docs/superpowers/specs/2026-07-25-builtin-tool-permission-gate-design.md` for the full investigation.** The original text described wiring `Tools/tool_executor.py`'s `ToolExecutor` and gating `write_file`/`create_note`/`update_note`. That premise does not hold: `ToolExecutor` (System A — reached only from the deprecated legacy/enhanced chat path, `Event_Handlers/worker_events.py` and `chat_streaming_events.py`, on the main loop) and `Agents/tool_catalog.py`'s `BuiltinToolProvider` (System B — the path `run_agent_loop` actually calls on the worker thread, today only `calculator`/`datetime`) are two disjoint systems, not two call sites into one path. The Console never reaches System A at all — `console_chat_controller.py`/`console_agent_bridge.py` contain zero references to `execute_tool_call`/`get_tool_executor`/`tool_executor` — while System A's fs/note tools (`write_file`, `create_note`, `update_note`, etc.) live there and nowhere else, and its own `[tools]` config is separately dead (TASK-547). Gating System A would harden code slated for removal while leaving the live path (System B) unprotected; gating System B needs a permission gate that doesn't exist yet.

**Decision (user-approved): port the fs/note tools to the agent runtime (System B) and gate them there.** Work is split into three phases, all tracked under this one task:

- **P1 — the gate seam (done, this pass).** Built on branch `feat/builtin-tool-permission-gate`: a concrete `Tool.risk_tags` property (default `()`, vocabulary = existing `HIGH_RISK_TAGS`); `resolve_builtin_state` in `MCP/permission_store.py`, a sibling of `resolve_effective_state` (not a modification of it) under a namespace deliberately distinct from the built-in MCP *server*'s own live key — `server_key = "agent:builtin"`, never `builtin:tldw_chatbook` (`MCP/readiness.py` `BUILTIN_SERVER_KEY`) — with precedence tool override → server default → built-in `allow` floor, no definition-hash comparison, and the existing risk-flooring pass unchanged; `Agents/builtin_tool_gate.py`'s `BuiltinToolGate` (one permission-store load per turn, fail-closed on a missing service, never raises); enforcement inside `BuiltinToolProvider.invoke` as defense-in-depth; a run-level `build_tool_review_hook` wired **unconditionally** (previously the review hook existed only when MCP was configured, so a user with zero MCP servers had no gate at all); and per-row `options` on `ChatApprovalCard` so built-in rows can be narrowed (excluding only `always_allow`, the sole persistent write) without lying about what a bulk action will do. Session-scoped decisions only — nothing persists under `agent:builtin` in P1. No tool is ported or tagged yet, so this phase changes no behavior for `calculator`/`datetime` (both untagged, both still executing silently) **except one deliberate change**: the kill switch is now global (spec §7), so a user with the MCP kill switch on now has built-in tool calls refused where they previously executed — this is intentional and is what the "kill switch blocks built-in tools" acceptance criterion pins.
- **P2 — port the tools (done, this pass).** **Rescoped again 2026-07-25 — see `Docs/superpowers/specs/2026-07-25-port-mutating-tools-design.md`.** Moved `write_file`, `create_note`, `update_note` into `BuiltinToolProvider` (tagged `("mutates",)`) and tagged the already-registered `read_file`/`list_directory` (`("reads",)`, closing a live silent-read gap: enabling either previously ran with no prompt at all). `rag_search`, `web_search`, `search_notes`, `code_audit` are explicitly **out of scope** for this pass per that spec — not implemented, not tagged, not gated. ~~Blocked on the child-run approval routing follow-up~~ — **UNBLOCKED 2026-07-25 by TASK-628** (`BuiltinToolGate.stamp_scope` + `_combine_state_scopes`), which fixed the nested-sub-agent stamp clobber and proved a child's gated call resolves through the shared approval route rather than failing closed. ~~Blocked on the persistent-decision UI prerequisite~~ — **UNBLOCKED 2026-07-25 by TASK-627** (`agent:builtin` Permissions-mode section + `HASH_FREE_SERVER_KEYS`), which made a persistent allow/deny for a built-in tool visible and reversible in-app. Live end-to-end coverage (real tools, not the synthetic P1 test double) lives in `Tests/Agents/test_builtin_gate_live_tools.py`.
- **P3 — config and the legacy path (not started).** Fix the dead `[tools]` config (TASK-547) and decide System A's fate (port its remaining behavior, gate it in place, or remove `Tools/tool_executor.py` entirely) so no tool anywhere ever executes ungated.

TASK-331's sandbox fix made the fs tools functional-within-a-sandbox; this task's P1+P2 are the intended protection layer on top of that. This task stays open (not Done) until P2 and P3 both land — see the unchecked criteria below.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
P1 — the gate seam (complete):
- [x] `Tool` ABC exposes a concrete `risk_tags` property defaulting to `()`; no existing subclass breaks
- [x] A built-in tool resolves through `resolve_builtin_state` via a `GatedToolRef` using `server_key="agent:builtin"`, with precedence tool override → server default → built-in `allow` floor, and no definition-hash comparison
- [x] No built-in permission entry is ever written or read under `builtin:tldw_chatbook`; a test asserts the two namespaces stay disjoint
- [x] With no `unified_mcp_service` available, untagged tools still execute and `"mutates"` tools fail closed — a missing service is never treated as allow-everything
- [x] Turn-level resolution: a run executing N built-in calls in one turn performs at most one permission-store load
- [x] An untagged built-in (calculator, datetime) resolves to `allow` and executes with no prompt — verified as a no-behavior-change test
- [x] A tool tagged `"mutates"` has its inherited allow floored to `ask` (`risk_floored=True`)
- [x] `BuiltinToolProvider.invoke` blocks execution when the kill switch is on, when the resolved state is `deny`, and when the state is `ask` with no approval route — each returning `ToolResult(ok=False, ...)`, never raising
- [x] A run-level `build_tool_review_hook` is wired whether or not MCP is configured, and folds MCP's gate in when a provider exists; built-in and MCP approvals appear in one card per turn
- [x] `resolve_builtin_state` exists as a sibling of `resolve_effective_state`; the MCP resolver's behavior is unchanged (its existing tests still pass untouched)
- [x] `BuiltinToolProvider(gate=None)` is gated by default — a test asserts a bare instance refuses a `deny`-resolved tool
- [x] The review hook routes calls by owner: MCP-claimed names to the MCP gate, registry-resolved built-ins to the built-in gate, and everything else (skills, native spawn) unreviewed
- [x] Built-in approvals are session-scoped only; no persistent allow/deny is written for `agent:builtin` in P1
- [x] The approval card offers built-in rows `approve_once`/`approve_session`/`deny` and NOT `always_allow`; MCP rows keep all four, verified byte-identical
- [x] The kill switch blocks built-in tools, and its MCP workbench label/echo no longer describe it as MCP-only
- [x] Tests inject a temporary permission-store path; no test touches the real user store

P2 — port the tools (complete; rescoped 2026-07-25, see `Docs/superpowers/specs/2026-07-25-port-mutating-tools-design.md`):
- [x] `read_file`, `list_directory`, `write_file`, `create_note`, `update_note` are implemented as `BuiltinToolProvider` tools, reachable from the Console's agent-runtime path. (`rag_search`, `search_notes`, `web_search`, `code_audit` are explicitly out of scope for this pass per the design spec's Scope section — not ported.)
- [x] Each mutating tool carries `risk_tags` and is floored to `ask` by default; a live end-to-end test drives at least one tool through allow/ask/deny — `Tests/Agents/test_builtin_gate_live_tools.py`, using the real registered tools rather than P1's synthetic `_Mutating` double
- [x] The child-run approval routing follow-up is resolved before any mutating tool ships gated (no unreviewable nested-subagent execution path) — TASK-628, Done 2026-07-25
- [x] A UI surface exists that can list and reverse a persistent allow/deny for a built-in tool before P2 offers persistent decisions for any mutating tool — TASK-627, Done 2026-07-25

P3 — config and legacy path (not started):
- [ ] TASK-547's dead `[tools]` config is fixed and reachable
- [ ] System A's (`Tools/tool_executor.py`) fate is decided and implemented — ported, gated in place, or removed — with no tool left executing ungated in either system
<!-- AC:END -->

## Implementation Notes

P2 landed on branch `feat/port-mutating-tools`, split into five tasks (design spec: `Docs/superpowers/specs/2026-07-25-port-mutating-tools-design.md`; plan: `Docs/superpowers/plans/2026-07-25-port-mutating-tools.md`):

1. Added `BUILTIN_HIGH_RISK_TAGS = HIGH_RISK_TAGS | {"reads"}`, consulted only by `resolve_builtin_state` — `resolve_effective_state` (MCP) is untouched, so MCP tools tagged `"reads"` do not start prompting.
2. Tagged five `Tool` subclasses: `WriteFileTool`/`CreateNoteTool`/`UpdateNoteTool` with `("mutates",)`, `ReadFileTool`/`ListDirectoryTool` with `("reads",)`.
3. Registered `write_file`, `create_note`, `update_note` in `BuiltinToolProvider.__init__` behind default-off `[tools]` gate keys (`write_file_enabled`/`create_note_enabled`/`update_note_enabled`), same pattern task-584 used for `read_file`/`list_directory`.
4. Fixed `note_management_tools.py` to resolve the writing user from `load_settings()["USERS_NAME"]` (matching `app.notes_user_id`) instead of a hardcoded `"default_user"`.
5. **This task.** Added `Tests/Agents/test_builtin_gate_live_tools.py` — the first coverage of `BuiltinToolGate`/`BuiltinToolProvider` against the real ported tools rather than P1's synthetic `_Mutating`/`CalculatorTool` doubles. 9 tests: refusal without approval (mutating and reads-tagged), an untagged tool still running unprompted, a stamped approval reaching real execution (`write_file` actually writing through a monkeypatched sandbox root), a resolved `deny` beating a permitting stamp, refusals surfacing as `ToolResult` never an exception, `create_note`/`update_note` executing correctly off the main thread (parametrized), and a parent's approval surviving a nested sub-agent run via `stamp_scope` without the child's verdict leaking back.

**Scope note:** `rag_search`, `web_search`, `search_notes`, `code_audit` were never in scope for this pass (design spec's Scope section says so explicitly) — AC #17 above is worded against that decision, not the original (broader) P1-era text. Narrowing the AC to match what shipped would otherwise have dropped those four tools off the board entirely, so they are now tracked by **TASK-690**, which also owns the risk-tag decision each of them needs (`web_search` in particular has no existing tag describing outbound network access).

**Sabotage-verification of `test_a_resolved_deny_beats_a_permitting_stamp`:** temporarily moved the `state.state == "deny"` check in `BuiltinToolGate.check()` to after the stamp checks — the test FAILed as expected (the sabotaged code let a stamped `approve_once` write through a resolved `deny`, and the write actually completed). Reverted (`git diff` against the pre-sabotage version was empty) and re-ran to confirm PASS. No production file was left modified — only the new test file was added.

**Follow-ups filed** (see spec's "Follow-ups to file"): TASK-687 (`UpdateNoteTool.expected_version` default-of-1 causes spurious version conflicts), TASK-688 (note tools construct a fresh `CharactersRAGDB` per call instead of reusing the app singleton), TASK-689 (surface the configurable `[tools] file_sandbox_root` to the user so `write_file` is discoverably useful).

Full affected-suite run (`Tests/Agents/ Tests/MCP/ Tests/Tools/ Tests/Library/test_library_skills_state.py`): 707 passed, 0 failed.

P3 (`[tools]` config fix, System A's fate) remains open; this task is not marked Done.
