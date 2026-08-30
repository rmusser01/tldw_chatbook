# MCP Hub local-tool execution design

**Task:** TASK-3605

**Status:** Proposed

**Date:** 2026-08-30

## Decision summary

The MCP Hub may execute a deliberately narrow subset of the tools displayed under the synthetic `local:__local__` principal. Eligibility comes from the existing `LocalToolSpec.exposure` descriptor: only tools marked `console_and_external_mcp` are executable from the Hub. Console-only, session-owned, and otherwise omitted tools stay inspectable but have no Test Tool action.

Every click rebuilds the local provider and workspace authority, obtains the current tool definition, re-resolves the shared permission gate, and fails closed before dispatch if any step is unavailable or has changed. Allow is labelled **Run**. Ask is labelled **Approve & run once** and performs the one-time approval and execution in that single click. Off, an unresolved gate, or an ineligible descriptor remains blocked. A one-time approval is not written to the permission store.

No MCP transport route is added. The Hub calls an application service, which invokes the local provider directly off the Textual UI loop and records the result through the existing bounded, redacted MCP execution log.

## Context

TASK-2838 shipped the local catalog into Hub Tools and Permissions as non-executable rows. It deliberately stopped before execution because the existing Hub service routes `local:<profile>` to external MCP profiles; treating `local:__local__` that way would either fail or create an unsafe alias with the reserved local principal.

The repository already has the required authorities:

- ADR-032 defines `local:__local__`, the shared permission store, descriptor exposure, definition-hash revalidation, workspace confinement, and code-owned execution policy.
- The MCP Hub design defines Test Tool as an operator-initiated management action with structured arguments, bounded/redacted results, permission awareness, and an execution-log record.
- `LocalToolProvider.invoke()` independently checks availability, permission, approval, current root authority, and execution policy.
- `MCP/local_server_tools.py` already composes a descriptor-filtered provider for external-safe local tools without `todo_*` session state.

The missing piece is a typed in-process adapter that joins those authorities without making the Hub a second tool runtime or opening raw MCP dispatch.

## Goals

1. Let an operator test eligible local tools from the existing Hub inspector.
2. Preserve one source of truth for eligibility, permission, root confinement, and execution policy.
3. Make Ask a genuine one-click, one-invocation approval rather than a redundant arm-then-run sequence.
4. Keep every terminal outcome bounded, redacted, and auditable.
5. Preserve definitive-after-start ownership so the UI cannot abandon a mutation that may still commit.

## Non-goals

- Exposing `local:__local__` through MCP `tools/call`.
- Making Console-only or session-owned tools executable from the Hub.
- Allowing Hub tests to persist permission changes; the Permissions mode remains the explicit persistent-policy surface.
- Adding a parallel tool-name allowlist, permission store, workspace-root setting, or execution engine.
- Turning the Hub into an agent run or supplying model-controlled approval callbacks.

## Executable projection

The Hub continues to collect the full local catalog for inspection and permission editing. It also derives a fresh set of executable names from the same provider composition used for descriptor-approved shared exposure. A catalog row is executable only when all of the following hold:

1. `[console] local_tools_enabled` permits provider composition.
2. Fresh workspace-root resolution succeeds.
3. Provider construction succeeds.
4. The tool is registered in the fresh provider.
5. Its `LocalToolSpec.exposure` is `console_and_external_mcp`.

The workbench marks only those rows `executable=True`. There is no hard-coded name list. `todo_*`, article-body reads, Watchlists authoring commands, and any future `console_only` descriptor remain visible but non-executable automatically.

Catalog projection is fail-soft: an unavailable local provider removes the local execution affordance without damaging external-profile or built-in catalog rows. Execution itself is fail-closed: a stale visible affordance never authorizes a call.

## Permission and one-click Ask flow

The inspector presents the current gate as follows:

| Effective state | Primary action | Result |
| --- | --- | --- |
| Allow | **Run** | Revalidate and dispatch |
| Ask | **Approve & run once** | Revalidate, grant one invocation, and dispatch in the same click |
| Off | **Blocked** | No dispatch |
| Unresolved/error | **Unavailable** | No dispatch |
| Not executable | No action | No dispatch |

Opening the panel may resolve state for copy and button labelling, but that result is advisory. The click path must resolve the selected row, current definition hash, arguments, provider, root, and permission again. A stored Allow whose definition changed naturally resolves back to Ask under the existing rug-pull rule.

For Ask, the service creates an in-memory one-shot approval callback bound to one invocation record containing the server key, tool name, current definition hash, and normalized arguments. The provider consumes it once. It is never persisted and cannot approve another click, a changed definition, or changed arguments. The button therefore does not need a second press and does not silently change the Permissions matrix.

## Service boundary and dispatch

`UnifiedMCPControlPlaneService` gains a typed local-Hub execution entry point rather than teaching the external MCP profile router that `local:__local__` is a transport profile. The entry point:

1. Validates the synthetic server key and requested catalog identity.
2. Resolves a fresh workspace root and builds a fresh descriptor-filtered provider.
3. Obtains the current `HubTool` definition and confirms it remains eligible.
4. Resolves the shared gate and, for Ask, requires the explicit one-shot confirmation carried by this click.
5. Invokes the provider off the UI loop.
6. Converts the provider `ToolResult` into the existing Test Tool result envelope.
7. Records one execution-log entry for every allowed, approved, denied, unavailable, failed, timed-out, or completed attempt.

The existing external-profile execution route remains unchanged. The raw local runtime delegate continues to reject `tools/call` for `local:__local__`.

## Execution ownership and cancellation

Local handlers are synchronous and must not block Textual. Ordinary bounded tools run in a worker thread under the existing Hub test timeout. Their result is returned only if the call completes within the bound; timeout and failure are audited with scrubbed details.

Tools whose descriptor declares `definitive_after_start` need stronger ownership. Once provider dispatch begins, the application service owns the completion task and waits shielded from inspector-worker cancellation. Navigation, remount, or caller cancellation may detach result presentation, but cannot emit a cancellation/timeout outcome while a mutation may still commit. The owned task writes the single terminal audit record before release. Pre-dispatch permission review remains cancellable and an already-cancelled request never starts the handler.

This reuses ADR-032's execution-policy meaning; the Hub does not infer mutability from names, tags, or form values.

## Confinement and configuration races

The workspace root is resolved at execution time, not retained from catalog assembly. The provider performs its normal path validation again at invocation. If configuration disables local tools, root resolution fails, or a binding changes between display and click, the run fails closed before handler dispatch.

Absolute workspace paths are operational authority, not user-facing output. Result and error formatting must scrub or relativize them before display or audit storage.

## Results and audit

The existing Test Tool result panel remains the presentation surface. Local results use the same status, duration, and bounded output envelope as other Hub tests. Registered argument names drive audit capture; secret-shaped values and unexpected exception strings are redacted through the existing control-plane logging seam.

Every attempted click creates one terminal audit outcome. Audit records distinguish persistent Allow, one-time Ask approval, configured Off, unresolved permission, eligibility/configuration failure, timeout, provider refusal, crash, and success without storing an approval callback or absolute root.

## UI behavior

The structured form remains unchanged. Only the action state and explanatory copy change for eligible local tools:

- **Run** for Allow.
- **Approve & run once** for Ask, with concise copy that the choice affects only this invocation.
- **Blocked** for Off, with a jump to Permissions rather than an execution override.
- No Test Tool action for non-executable rows.

The click handler disables duplicate submission while the invocation is admitted. If the inspector remounts, service ownership and audit continue independently; stale widgets do not receive a late update.

## Failure handling

- Catalog provider/root failure: omit the execution affordance and log a warning; keep the rest of the Hub intact.
- Click-time provider/root/definition/gate failure: do not dispatch; show a bounded unavailable/blocked result and audit it.
- Approval mismatch or reuse: refuse before dispatch and audit the refusal.
- Handler exception: return a scrubbed failure and audit once.
- Bounded timeout: return a bounded timeout and audit once.
- Definitive handler after dispatch: await owned completion and audit its actual terminal result.
- Audit write failure: log locally without replacing or duplicating the tool outcome.

## Verification strategy

Automated coverage will prove:

1. Descriptor-derived executable projection and automatic exclusion of Console-only/session tools.
2. Allow dispatch with a fresh provider/root/definition.
3. One-click Ask dispatch, single consumption, no persistence, argument/definition binding, and click-time revalidation.
4. Off and gate-error refusal with zero handler calls.
5. Definition-hash change downgrading a stored Allow to Ask.
6. Disabled configuration, provider failure, and root failure.
7. Workspace confinement and a display-to-click root/configuration race.
8. Bounded-tool timeout and definitive-after-start completion/audit under caller cancellation.
9. Bounded/redacted success and failure audit records.
10. The unchanged raw `tools/call` refusal for `local:__local__`.
11. Textual button labels and one-click Ask behavior through a mounted inspector flow.

Verification will use focused Agent, MCP control-plane, local-server, and MCP workbench test files plus scoped Ruff, compilation, diagnostic inventory if touched, and `git diff --check`. A full repository sweep is not required unless requested or demanded by the eventual merge gate.

## ADR check

**ADR required:** no new ADR.

**ADR path:** `backlog/decisions/032-local-agent-tool-permission-boundary.md`

**Reason:** ADR-032 already decides the principal, descriptors, permission and definition-hash authority, confinement, and execution-policy ownership. The accepted MCP Hub design already decides operator Test Tool behavior and auditing. This design is the adapter between them and deliberately adds no new transport, storage, principal, or policy.
