# MCP Hub local-tool execution design

**Task:** TASK-3605

**Status:** Proposed

**Date:** 2026-08-30

## Decision summary

The MCP Hub may execute a deliberately narrow subset of the tools displayed under the synthetic `local:__local__` principal. Eligibility comes from the existing `LocalToolSpec.exposure` descriptor: only tools marked `console_and_external_mcp` are executable from the Hub. Catalogued Console-only tools stay inspectable without a Test Tool action. Session-owned tools such as `todo_*` remain absent because the Hub never composes their session store.

Opening Test Tool creates an immutable admission preview from the full tool identity, definition hash, authority fingerprint, displayed gate state, and a panel nonce. Allow is labelled **Run**. Ask is labelled **Approve & run once** and performs the one-time approval and execution in that single click. The click also carries explicit `run` or `approve_once` intent. The service rebuilds the provider and authority, then refuses and refreshes if the definition or authority changed or if a rendered **Run** now resolves to Ask. Off, unresolved state, and ineligible descriptors remain blocked. A one-time approval is not written to the permission store. The one-click Ask affordance replaces the Hub's existing arm-then-confirm behavior for every executable Test Tool row rather than adding a local-only exception.

No MCP transport route is added. The Hub calls an application service, which invokes the local provider directly off the Textual UI loop and records the result through the existing bounded, redacted MCP execution log.

## Context

TASK-2838 shipped the local catalog into Hub Tools and Permissions as non-executable rows. It deliberately stopped before execution because the existing Hub service routes `local:<profile>` to external MCP profiles; treating `local:__local__` that way would either fail or create an unsafe alias with the reserved local principal.

The repository already has the required authorities:

- ADR-032 defines `local:__local__`, the shared permission store, descriptor exposure, definition-hash revalidation, workspace confinement, and code-owned execution policy. TASK-3605 adds an operator-Hub addendum for Off and kill-switch semantics.
- The MCP Hub design defines Test Tool as an operator-initiated management action with structured arguments, bounded/redacted results, permission awareness, and an execution-log record.
- `LocalToolProvider.invoke()` independently checks availability, permission, approval, and current root authority; its descriptor exposes the execution policy to the calling runtime.
- `MCP/local_server_tools.py` already composes a descriptor-filtered provider for external-safe local tools without `todo_*` session state.

The missing piece is a typed in-process adapter that joins those authorities without making the Hub a second tool runtime or opening raw MCP dispatch.

## Goals

1. Let an operator test eligible local tools from the existing Hub inspector.
2. Preserve one source of truth for eligibility, permission, root confinement, and execution policy.
3. Make Ask a genuine one-click, one-invocation approval for every executable Hub test rather than retaining two UI mechanisms.
4. Keep every terminal outcome bounded, redacted, and auditable.
5. Preserve definitive-after-start ownership so the UI cannot abandon a mutation that may still commit.

## Non-goals

- Exposing `local:__local__` through MCP `tools/call`.
- Making Console-only or session-owned tools executable from the Hub.
- Allowing Hub tests to persist permission changes; the Permissions mode remains the explicit persistent-policy surface.
- Adding a parallel tool-name allowlist, permission store, workspace-root setting, or execution engine.
- Turning the Hub into an agent run or supplying model-controlled approval callbacks.
- Claiming crash-safe exactly-once audit durability; the existing JSONL execution log is intentionally best-effort.

## Executable projection

The Hub continues to collect the full local catalog for inspection and permission editing. It also derives a fresh set of executable identities from the same provider composition used for descriptor-approved shared exposure. Eligibility is keyed by exact `(server_key, tool_name)` plus the current definition hash, never by a bare name. A catalog row is executable only when all of the following hold:

1. `[console] local_tools_enabled` permits provider composition. `[mcp] expose_local_tools` is irrelevant because it controls external stdio publication, not an in-app operator diagnostic.
2. Fresh workspace-root resolution succeeds.
3. Provider construction succeeds.
4. The tool is registered in the fresh provider.
5. Its `LocalToolSpec.exposure` is `console_and_external_mcp`.

The workbench marks only those rows `executable=True`. There is no hard-coded name list. Article-body reads, Watchlists authoring commands, and any future catalogued `console_only` descriptor remain visible but non-executable automatically. `todo_*` remains absent, not merely disabled, because no Console session store is present. The separately catalogued `local:__virtual_cli__` principal is outside TASK-3605 and remains non-executable.

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

Opening the panel calls `prepare_hub_test()` on the control-plane service. The service issues and registers a `ToolTestAdmissionPreview` containing the exact server key and tool name, the rendered definition hash, a non-secret fingerprint of the resolved workspace authority when applicable, the displayed gate state, and a fresh opaque panel nonce. The nonce is minted by the service and retained in a bounded, short-lived preview registry; it is not merely a client-provided token. The UI shows a safe workspace label but does not expose the absolute root. The preview is immutable and single-use. Switching tools, closing, expiry, or remounting calls `revoke_hub_test_preview()` and removes the registered nonce; service expiry remains the backstop when a client disappears without cleanup.

The authority fingerprint is process-local admission data, not a hash of a path string. The service captures the same `DirectoryChain` authority used by workspace root pinning: the strict canonical root locator plus the root-first `(device, inode, mode, reparse)` identity chain. It hashes a canonical encoding of both values for comparison, retains the unhashed authority only inside the preview record, and never displays or persists either form. A directory replacement at the same textual path therefore changes authority and blocks dispatch. Click admission captures a new chain, compares it with the registered preview, and passes that same chain to the provider's root guard and pinned workspace executor.

The button event carries the issued preview nonce and explicit intent derived from the visible label: `run` for **Run**, `approve_once` for **Approve & run once**. The service atomically removes and consumes the registered preview before admission, so concurrent clicks cannot reuse it, canonicalizes the current form arguments, then resolves the selected identity, current definition, provider, root, and permission again. An unknown, revoked, expired, or already-consumed nonce fails before dispatch. The service dispatches only when all registered preview fields still match and the intent is compatible with the fresh gate:

- rendered Allow + fresh Allow + `run`: dispatch;
- rendered Allow + fresh Ask/Off/error: refuse and refresh, never reinterpret `run` as approval;
- rendered Ask + fresh Ask + `approve_once`: consume one approval and dispatch;
- rendered Ask + fresh Allow + `approve_once`: dispatch under Allow without persisting an approval;
- any definition, authority, identity, or nonce mismatch: refuse and refresh;
- any fresh Off or unresolved state: refuse.

A stored Allow whose definition changed therefore refreshes as Ask; the click that discovered the new definition cannot approve it. The generic Hub Ask arm, its mutable armed state, and its second-press tests are removed; source-specific execution remains behind the same click-time control-plane gate.

For a local Ask, the service canonicalizes the exact handler argument object before admission: only JSON objects with string keys, JSON scalar/list/object values, and finite numbers are accepted; canonical encoding uses sorted keys, compact separators, and `allow_nan=False`. The deep-copied object decoded from those bytes is the object passed to `invoke()`. Raw-form values that cannot satisfy that contract fail before approval.

The service then creates a fresh provider with an in-memory one-shot approval callback bound to one invocation ID, the preview identity and definition hash, the authority fingerprint, and a digest of those exact canonical bytes. `LocalToolSpec.approval_arguments` remains presentation-only and is never used as the security binding. The callback is private to that provider invocation, is consumed at most once, and returns only `approve_once`; no session or persistent approval callback is supplied. It is never persisted and cannot approve another click, a changed definition, changed authority, or changed arguments.

## Service boundary and dispatch

`UnifiedMCPControlPlaneService` gains a gated `execute_prepared_hub_test(preview_nonce, intent, arguments)` entry point for every Hub Test Tool source. This service method, rather than the widget, owns preview consumption and click-time permission enforcement. Existing external/built-in tests delegate to the established low-level execution method only after admission. Local tests use a typed local-Hub adapter rather than teaching the external MCP profile router that `local:__local__` is a transport profile.

A dedicated Hub provider factory reuses the existing descriptor-filtered spec composition but does not reuse the external server's approval or kill-switch callbacks: external MCP must fail closed on Ask and honor the chat/runtime kill switch, while an operator-initiated Hub diagnostic offers one-time approval and deliberately ignores that switch. The Hub factory injects `kill_switch=lambda: False`, `result_redaction_root=fresh_root`, and no provider-level decision recorder; permission-store and definition checks remain mandatory. Any lazily opened Watchlists read dependency has an explicit close owned by the per-invocation factory rather than being left to garbage collection. The prepared entry point:

1. Validates the immutable preview, explicit intent, panel nonce, and full synthetic catalog identity.
2. Canonicalizes and deep-copies the exact arguments.
3. Resolves a fresh workspace root and builds a fresh descriptor-filtered provider.
4. Obtains the current `HubTool`, confirms eligibility, and compares its identity, definition hash, and authority fingerprint with the preview.
5. Resolves the shared gate and proves it is compatible with the rendered state and explicit click intent.
6. Resolves the effective timeout from the Hub lifecycle default and any longer provider-owned floor, preserving tools such as `web_deep_search` that require time to produce their own bounded partial result.
7. Invokes the provider through the service-owned coordinator off the UI loop.
8. Calls `LocalToolProvider.invoke_detailed()`, a narrow provider-specific seam shared with normal `invoke()` internals. It returns `LocalToolInvocationResult(result, final_gate, approval_consumed, reason_code, dispatch_started, provider_terminal)`. `reason_code` is a closed enum covering permission Off, unresolved gate, approval refusal/timeout, root change, authority unavailable, handler failure, and success; `provider_terminal` is limited to `not_started`, `returned`, or `raised`. Ordinary `invoke()` calls the same private implementation and still exposes only the unchanged `ToolResult` protocol. Root/gate races and provider-owned terminals therefore remain structured without widening every provider or parsing refusal strings.
9. The coordinator produces one internal `LocalHubExecutionOutcome` containing the decision, terminal status, safe error category, `dispatch_started` fact, duration, and root-redacted result. Only the coordinator may synthesize `timed_out` or detached-cancellation terminals around the off-loop provider worker; the synchronous provider never claims that its still-running thread timed out.
10. Uses that one typed outcome for both the Test Tool envelope and the single best-effort audit finalizer.

The existing external-profile execution route remains unchanged. The raw local runtime delegate continues to reject `tools/call` for `local:__local__`.

## Execution ownership and cancellation

The control-plane service owns a `LocalHubExecutionCoordinator`, not the workbench. Admission creates an unguessable in-memory invocation ID, registers a strong task reference, and reserves the exact `(server_key, tool_name)` active key before dispatch. A remounted inspector queries this registry and cannot admit the same tool while its prior invocation remains active. The internal task owns `BaseException`-safe terminal construction, the at-most-one audit finalizer, dependency cleanup, and registry release; UI workers only await it through a shield and may detach without owning completion.

Local handlers are synchronous and must not block Textual. The coherent rebuild → preview comparison → gate → invoke transaction runs off the UI loop. Ordinary bounded tools run in a worker thread under the effective timeout: the Hub lifecycle default unless the provider declares a longer tool-specific floor. Their result is returned only if the call completes within the bound; timeout, cancellation, and failure are audited with scrubbed details. `bounded_abandonable` retains its existing meaning: after a timeout the caller may stop waiting even though Python cannot terminate an already-running worker thread. The coordinator seals the timeout outcome and audit before detaching; any later worker return or exception is consumed only for cleanup and cannot replace the sealed outcome, append another terminal audit row, or update a remounted inspector.

Tools whose descriptor declares `definitive_after_start` need stronger ownership. Once provider dispatch begins, the coordinator waits for the actual handler result without applying an abandonable timeout. Navigation, remount, or caller cancellation may detach result presentation, but cannot emit a cancellation/timeout outcome while a mutation may still commit. Pre-dispatch permission review remains cancellable and an already-cancelled request never starts the handler. All currently eligible shared descriptors are bounded; definitive coverage uses an injected eligible test descriptor so the proof is not vacuous and protects a future descriptor change.

This reuses ADR-032's execution-policy meaning; the Hub does not infer mutability from names, tags, or form values.

## Confinement and configuration races

The workspace root is resolved and captured as a strict `DirectoryChain` for the immutable panel preview and again during click admission. The service compares the process-local canonical-locator-plus-identity fingerprint defined above, and the provider receives that same admitted chain through the root guard and pinned workspace executor for immediate pre-dispatch revalidation. If configuration disables local tools, root resolution fails, an ancestor identity changes, or the root is replaced at the same locator after display, the run fails closed and the panel refreshes before handler dispatch.

Absolute workspace paths are operational authority, not user-facing output. The Hub provider receives the fresh root as `result_redaction_root`; `redact_root_locator()` runs before `LocalHubExecutionOutcome` construction. UI and audit derive from that same sanitized `ToolResult`, so generic secret-key redaction is not incorrectly treated as path redaction.

## Results and audit

The existing Test Tool result panel remains the presentation surface. Local results use the same status, duration, and bounded output envelope as other Hub tests. Registered argument names drive audit capture; secret-shaped values and unexpected exception strings are redacted through the existing control-plane logging seam.

The control-plane service is the sole audit owner for Hub-local tests. The provider's optional refusal recorder is not wired for this composition, avoiding a provider refusal record followed by a duplicate service terminal. One in-process finalizer attempts at most one terminal JSONL row per admitted invocation. The log remains best-effort: an append failure or process loss may omit the row, and no crash-safe exactly-once claim is made. The finalizer does not retry an ambiguous append. Invalid client-side forms rejected before service admission do not create execution records.

Audit records distinguish persistent Allow, one-time Ask approval, configured Off, unresolved permission, eligibility/configuration mismatch, cancellation, timeout, provider refusal, crash, and success without storing the one-shot callback, canonical argument bytes, authority locator, or absolute root.

## UI behavior

The structured form remains unchanged. The action state and explanatory copy become consistent for every executable Hub tool:

- **Run** for Allow.
- **Approve & run once** for Ask, with concise copy that the choice affects only this invocation and no separate armed-confirm press.
- **Blocked** for Off, with a jump to Permissions rather than an execution override.
- No Test Tool action for non-executable rows.

The click handler disables duplicate submission while the invocation is admitted. The service registry, not the widget boolean, is the authoritative duplicate guard. If the inspector remounts, it renders the active state from that registry; service ownership and finalization continue independently, and stale widgets do not receive a late update. Removing the generic Ask arm also removes its edit/switch/disarm state machine rather than leaving dead compatibility branches.

## Failure handling

- Catalog provider/root failure: omit the execution affordance and log a warning; keep the rest of the Hub intact.
- Stale nonce, intent/gate transition, or identity/root/definition mismatch: do not dispatch; invalidate and refresh the preview, show a bounded unavailable/blocked result, and audit only if service admission already occurred.
- Click-time provider/root/definition/gate failure: do not dispatch; show a bounded unavailable/blocked result and finalize best-effort audit.
- Approval mismatch or reuse: refuse before dispatch and audit the refusal.
- Handler exception: classify the typed outcome, return a scrubbed failure, and finalize audit once in process.
- Bounded cancellation or timeout: return the policy-appropriate terminal and finalize audit once in process; never rely on cancelling a worker thread.
- Definitive handler after dispatch: retain the coordinator task, await owned completion, and finalize its actual terminal result.
- Audit write failure: log locally without retrying, replacing, or duplicating the tool outcome.

## Verification strategy

Automated coverage will prove:

1. Descriptor-derived executable projection, visible-but-disabled Console-only tools, absent session tools, and unchanged Virtual CLI non-executability.
2. Allow dispatch with a fresh provider/root/definition.
3. Generic one-click Ask UX plus local Ask dispatch, single consumption, no persistence, canonical exact-argument binding, and click-time revalidation.
4. Rendered Allow → fresh Ask, rendered Ask → fresh Off, revoked/expired/reused nonce, concurrent double-click, and definition/root/ancestor-identity mismatch refusing with zero handler calls and a refreshed preview.
5. Definition-hash change downgrading a stored Allow to Ask without approving the new definition on that click.
6. Disabled configuration, provider failure, and root failure.
7. Workspace confinement and a display-to-click root/configuration race.
8. Tool-specific timeout precedence, bounded-tool cancellation/timeout, service-registry duplicate refusal/remount behavior, and injected definitive-after-start completion under caller cancellation.
9. `invoke_detailed()` compatibility with ordinary `invoke()`, typed gate/approval/root/provider-terminal categories, coordinator-owned timeout/detached-cancellation terminals, late-worker cleanup without outcome/audit replacement, common root-redacted UI/audit results, at-most-one in-process audit finalization, audit-write failure, and no audit for client-side invalid form input.
10. The unchanged raw `tools/call` refusal for `local:__local__`.
11. Textual button labels, removal of the armed-confirm state, and one-click Ask behavior through mounted local and existing non-local inspector flows.

Verification will use focused Agent, MCP control-plane, local-server, and MCP workbench test files plus scoped Ruff, compilation, diagnostic inventory if touched, and `git diff --check`. A full repository sweep is not required unless requested or demanded by the eventual merge gate.

## ADR check

**ADR required:** yes, by amendment of an existing decision.

**ADR path:** `backlog/decisions/032-local-agent-tool-permission-boundary.md`

**Reason:** ADR-032 already decides the principal, descriptors, permission and definition-hash authority, confinement, and execution-policy ownership. TASK-3605 amends it to make the operator-only Hub policy explicit: configured Off blocks, Ask requires an intent-bound rendered preview, and the chat/runtime kill switch does not block the diagnostic. The older MCP Hub design's Off-confirmation sentence is corrected to match the accepted fail-closed UI and this amendment. No new transport, storage, or principal is added.
