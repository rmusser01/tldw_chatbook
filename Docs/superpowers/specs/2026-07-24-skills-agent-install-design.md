# Agent-Callable Skill Install — Design

**Status:** Design (awaiting user spec-review gate)
**Branch:** `worktree-skills-agent-install` off `origin/dev`
**Program:** skills-install (north star: "a user asks an agent to install a skill/pack from a GitHub link"). Prior layers merged: #762 trust foundation, #784 bundle fidelity, #801 `$`-mention invocation, #814 fork reachability, #831 remote URL fetch. This layer closes the north star: the *agent itself* can install from a link, on your behalf, with a human gate.

## Goal

Give the top-level Console agent a runtime tool, `install_skill(url)`, so that when you ask it to "install the skill at `<github link>`," it can — pausing to **confirm with you in-chat before it fetches anything**, and always landing the skill **trust-pending** so it cannot run until you review and approve it in Library ▸ Skills.

## Non-Goals

- No new network, SSRF, classifier, or archive code — this layer composes the merged `install_skill_from_url` seam unchanged.
- No skill *execution* (script running) — a later layer.
- No agent ability to *approve trust* — that is, and remains, a human-only action.
- No subagent/fork access to the tool.
- No `overwrite` capability for the agent (see §4).

## Architecture

`install_skill` is the **fifth runtime tool**, joining `spawn_subagent`, `find_tools`, `load_tools`, and `skill_file`. It follows the runtime-tool wiring pattern end to end:

1. A name constant `INSTALL_SKILL_TOOL_NAME = "install_skill"` in `agent_models.py`, added to `RUNTIME_TOOL_NAMES`.
2. A `ToolSchema` `INSTALL_SKILL_TOOL_SCHEMA` in `tool_catalog.py`.
3. A dispatch branch in `run_agent_loop` (`agent_runtime.py`) that reads `url` off `call.args` and calls `deps.install_skill(url)`.
4. An optional `LoopDeps.install_skill: Callable[[str], ToolResult] | None` field.
5. Conditional wiring + schema pin in `AgentService._run_one`, **gated on `agent_kind == AGENT_KIND_PRIMARY`** (see §3).
6. Bridge wiring in `ConsoleAgentBridge.run_reply`, which builds the install closure from `self._skills_service` and a controller-supplied confirm callback.
7. Registration in every collision guard that lists `RUNTIME_TOOL_NAMES`.

Three independent gates protect the tool. All three must pass before an installed skill can ever run:

1. **Runtime policy** — the existing fail-closed `skills.install_remote.launch.local` gate, checked via `SkillsScopeService.enforce_install_remote()`.
2. **In-chat human confirm** — a new pause-and-ask, *before any network I/O* (§2).
3. **Trust-pending landing** — `import_skill_file(..., trust_approved=False)`, which makes the skill inert until a human approves it (§5).

Chosen decisions (user, 2026-07-24):
- Consent model: **in-chat confirm before install** (two human gates: confirm now, trust review later).
- Scope: **top-level Console agent only**, not spawned subagents/forks; **per-install confirm, no "allow all this session"** bypass.
- Tool result: **name + trust-pending state + review pointer**, with multi-skill candidate lists and error messages passed through verbatim.

## 1. The runtime tool

**Schema.** `install_skill(url: string)` — a single required `url`. No `overwrite` parameter (see §4). Always-pinned (never disclosure-gated behind `find_tools`/`load_tools`), so a model asked to install a link can always see the tool.

**Dispatch closure** (`deps.install_skill`, built in the bridge, mirroring `skill_file`'s reader), executed on the agent worker thread during tool dispatch:

1. `scope_service.enforce_install_remote()` — synchronous policy check **first**. On `PolicyDeniedError`, return `ToolResult(ok=False, …)` immediately, with **no prompt** (don't interrupt the human for a disabled feature).
2. `allowed = request_skill_install_confirm(url)` — the in-chat confirm (§2), a plain blocking call returning a bool. On deny/timeout/cancel, return `ToolResult(ok=False, "The user declined to install this skill.")` — **`install_skill_from_url` is never called**, so no fetch occurs.
3. `result = asyncio.run(install_skill_from_url(url, scope_service=self._skills_service))` — the merged seam (which re-runs `enforce_install_remote()` internally; the double check is side-effect-free and idempotent).
4. Wrap `result` into a `ToolResult` (§4). Catch `RemoteSkillError`/`PolicyDeniedError`/`OSError` at the closure boundary and convert to `ToolResult(ok=False, …)` — nothing throws into the loop.

Ordering is load-bearing: the confirm's blocking wait is a plain `threading.Event` wait, run **before and outside** the `asyncio.run(...)` in step 3, so it never blocks an event loop.

## 2. The in-chat confirm

Modeled directly on the existing MCP approval flow (`ConsoleChatController.request_mcp_approvals`), which already solves "block a sync tool call on the worker thread and resume from a UI-thread widget." A **new, parallel** single-item mechanism (not a reuse of the MCP batch machinery, whose 5-way decision vocabulary and `MCPPendingCall` shape do not generalize):

- **Controller owns the HITL.** `ConsoleChatController` gains `request_skill_install_confirm(url) -> bool`, a `_pending_skill_install` state pair, and `resolve_pending_skill_install(allow: bool)`. It passes `request_skill_install_confirm` **down into `bridge.run_reply` as a new kwarg** (`request_skill_install_confirm: Callable[[str], bool] | None = None`), exactly as it already passes `should_cancel` and `review_tool_calls`.
- **Worker-thread block.** `request_skill_install_confirm` runs on the agent worker thread: it builds a `threading.Event` + a decision box, marshals a payload to the UI thread via `self.app.call_from_thread(self.set_pending_skill_install, payload)`, then polls `event.wait(poll)` while re-checking cooperative Stop (`_stop_requested`/`_active_cancel_event`) and a wall-clock timeout. **Cancel or timeout → deny** (fail-closed). When `self.app` is `None` (headless/tests), it resolves to deny via the same cancel/timeout path.
- **Bridge threads the callback into the closure.** `run_reply` captures the new kwarg and builds the `deps.install_skill` closure over it + `self._skills_service`, then hands the closure to `AgentService` as a kwarg (e.g. `install_skill_tool: Callable | None`). This threads the controller's callback into the dispatch closure directly — it does **not** go through `review_tool_calls` (a generic pre-dispatch batch gate, wrong shape for a per-install confirm).
- **UI card.** A new `SkillInstallConfirmCard` (mirroring `ChatApprovalCard`'s round trip) shows the URL with **Allow / Deny** buttons only — no session-wide bypass. Its button posts `InstallDecided(allow: bool)`; the `ChatScreen` `@on(SkillInstallConfirmCard.InstallDecided)` handler calls `controller.resolve_pending_skill_install(allow)`, which sets the decision box and the event, releasing the worker thread.
- **Markup safety.** The card renders the agent/attacker-influenced URL with `markup=False` (or `rich.markup.escape()`), following the batch-row discipline in `ChatApprovalCard` — never the legacy `set_approval` path, which interprets Rich markup.

## 3. Scope: top-level agent only

Spawning recurses `_run_one` on the **same `AgentService` instance**, so a closure gated on an instance field (e.g. the way `skill_file` is gated on `self.skill_file_bindings`) is reachable at every depth — that is how `skill_file` deliberately reaches depth-1 subagents. `install_skill` must **not** use that shape.

Instead, both the schema pin and the `deps.install_skill` wiring are gated on the **`agent_kind` per-call parameter** of `_run_one`:

```python
# runtime_schemas build (AgentService._run_one)
if agent_kind == AGENT_KIND_PRIMARY and self._install_skill_tool is not None:
    runtime_schemas.append(INSTALL_SKILL_TOOL_SCHEMA)

# LoopDeps construction
install_skill=(
    self._install_skill_tool
    if agent_kind == AGENT_KIND_PRIMARY and self._install_skill_tool is not None
    else None
),
```

Because `agent_kind` is a genuine per-call argument (`AGENT_KIND_SUBAGENT` at every spawn call site, `agent_service.py:502-511`) and is never mutated after the fact by anything a skill or subagent controls, a subagent receives **neither the schema nor the closure** — it structurally cannot call `install_skill`. Spawning is hard-capped at one level (`clamp_child_budget` zeroes `max_subagents`), so `agent_kind == AGENT_KIND_PRIMARY` is the exact and total "top-level only" signal.

## 4. Tool contract

**Success** returns structured content the agent can relay accurately:
- the installed skill **name**,
- that it is **trust-pending** and cannot run yet,
- the pointer to **review and approve it in Library ▸ Skills**.

Example agent-facing content: `Installed "foo" — it is pending your review and cannot run until you approve it in Library ▸ Skills.`

**Multi-skill repositories.** `install_skill_from_url` raises `RemoteSkillError` with a ≤20-candidate listing when a repo root holds multiple skills. That message passes through to the agent verbatim (as `ToolResult(ok=False, …)`), so the agent can re-call `install_skill` with a subdirectory URL. No extra tool parameter is needed — subdir selection is expressed in the URL and handled by the existing classifier.

**Errors.** `RemoteSkillError` (bad URL, SSRF rejection, download failure, corrupt/ambiguous archive), `PolicyDeniedError`, user-declined, and already-exists (`import_skill_file` with `overwrite=False` raises for a name collision) all become `ToolResult(ok=False, <user-presentable message>)`. Nothing escapes into the loop.

**No `overwrite`.** Overwriting an existing skill is a downgrade vector (replace trusted bytes, re-quarantine, hope for a rubber-stamp re-approval) and is unnecessary — a human can delete/replace in Library. The tool omits the parameter entirely; a name collision is a plain error the agent relays.

## 5. Trust landing — what "trust-pending" denies

`import_skill_file(..., trust_approved=False)` (which `install_skill_from_url` always passes) leaves the skill inert across every invocation surface until a human approves it in Library ▸ Skills:

- **`$`-mention / user invocation refuses** (`SKILL_UNTRUSTED_REFUSE` via `SkillTrustBlockedError`).
- **`skill_file` reads refuse** (`_require_trusted_skill` → `SkillTrustBlockedError`, surfaced as `ToolResult(ok=False, …)`).
- **Agent implicit use excludes it** — `get_context` places a `trust_blocked` skill in `blocked_skills`, never `available_skills`, so it is never registered as a callable tool for the model this run.
- **Defense-in-depth** — `execute_skill` itself re-checks `_require_trusted_skill`, so even a hallucinated direct call fails closed.

It remains visible only in needs-review listings and direct detail lookups, which do not enforce trust. So: the agent can *install* (deposit quarantined bytes bounded by the import caps), but it can never make an installed skill *runnable* — that gate is the human trust review.

## 6. Availability

`install_skill` is reachable only when the Console agent loop actually runs — the four-way condition in `console_chat_controller.py`: `_agent_runtime_enabled` (config `[console] agent_runtime`, default true) **and** a durable (non-`:memory:`) DB agent bridge exists **and** not a prefill turn **and** not a plain/character turn (`force_plain`). It is **orthogonal to the native-tools flag**, which only selects tool-call *formatting*, not whether the loop runs. Outside those conditions the turn is a plain completion with no tools at all.

*Possible refinement (out of scope for MVP):* suppress the schema pin when `skills.install_remote` policy is disabled, so a policy-off install never even offers the tool. MVP pins whenever the closure is wired and lets `enforce_install_remote()` return the denial at call time.

## 7. Error handling summary

Fail-closed throughout: policy denial short-circuits before prompting; confirm timeout/cancel/headless → deny; unknown-name calls fall through to the generic "not permitted" path (the closure is absent for subagents); every seam exception → `ToolResult(ok=False)`. Collision guards (`RUNTIME_TOOL_NAMES`, the `_SHADOWED_BUILTIN_NAMES` superset test, `_non_colliding_skill_entries`/`_non_colliding_mcp_names`) gain the new name so a skill or MCP tool named `install_skill` can never shadow it.

## 8. Testing

- **Pure loop dispatch** — `run_agent_loop` routes the `install_skill` name to `deps.install_skill`; a `None` dep falls through to the generic path.
- **Closure behavior** — allow → `install_skill_from_url` is called and its result is wrapped; **deny → `install_skill_from_url` is asserted *not* called**; policy denial → no prompt, error result; `RemoteSkillError` (incl. multi-skill listing) → passed-through error result. Uses a fake confirm callback + fake/real scope service.
- **Confirm mechanism** — `request_skill_install_confirm` blocks on the event and returns the resolved decision; `resolve_pending_skill_install(True/False)` releases it; cancel/timeout/headless → deny. Mirrors the MCP approval tests.
- **Scope** — a spawned run (`agent_kind == AGENT_KIND_SUBAGENT`) has `LoopDeps.install_skill is None` and no `INSTALL_SKILL_TOOL_SCHEMA` in its `runtime_schemas`; the top-level run has both.
- **Card safety** — the confirm card renders a URL containing Rich-markup-like text without interpreting it (`markup=False`/escaped).
- **Collision guards** — `RUNTIME_TOOL_NAMES` includes `install_skill`; the `_SHADOWED_BUILTIN_NAMES` superset test passes; a skill named `install_skill` is excluded from the eligible set.
- **E2E** — a model turn emits `install_skill(url)` → a fake confirm auto-allows → `MockTransport` serves a zipball → **real** `SkillsScopeService.import_skill_file` lands the skill **trust-pending on disk**; plus a denied variant proving no install occurred.

## Accepted residuals / follow-ups

- The in-chat confirm shows the URL but cannot show the archive contents before download (the fetch happens after confirm) — the human's second gate (trust review) is where bundle contents are inspected.
- The DNS-rebinding TOCTOU residual from the remote-fetch layer (no transport-level IP pinning) still applies to `install_skill_from_url`; unchanged here (tracked as task-524/525).
- Policy-off schema suppression (§6) is deferred.
