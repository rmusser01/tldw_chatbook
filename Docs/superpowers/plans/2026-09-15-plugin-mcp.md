# Direct MCP and plugin ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide typed MCP results, qualified direct transports and stable credential/scoped connection ownership for plugin tools.

**Architecture:** Extend the existing MCP client/control/provider seams with a typed result path and explicit protocol profiles. Keep transport and credential ownership outside plugin discovery, with per-request workspace authority and conservative uncertain outcomes.

**Tech Stack:** Python 3.12+, Textual 8.2.8, Pydantic 2, private SQLite, httpx, portalocker and existing trust/credential primitives.

**Spec:** [Plugin spec](../specs/2026-09-15-managed-plugins-design.md) and [hook spec](../specs/2026-09-15-expanded-hook-runtime-design.md). Read both; this plan covers its assigned subsystem within the complete [delivery plan](2026-09-15-managed-plugins-delivery.md).

## Global Constraints

- Python >=3.12; current checkout pins Textual 8.2.8, Pydantic >=2.4,<3 and portalocker 3.2.0. Preserve these pins; use the existing SQLite/httpx/crypto/keyring seams.
- One installation and selected revision exist per user-data directory.
- Global default activation starts disabled. Importing a catalog does not install or enable its entries.
- All approved hook additions are in scope. Delivery stages are ordering, not deferral.
- No parallel agent/permission runtime, package build/install execution, vendor grants or per-workspace package versions.
- Required constraints never disappear because parsing, configuration, hooks or persistence fail. Native/foreign instructions remain attributed untrusted context.
- Package activation grants no filesystem binding, tool permission, network credential or trusted project status. Console local tools retain scratch/explicit-binding authority.
- Apply current authority before injection/launch/dispatch/result acceptance. Workspace disable preserves other authorized scopes; namespace marker changes alone do not cancel all work.
- Full-suite runs require explicit user opt-in. Every task runs its exact feature/regression files and a successful control on the same production entry.
- Implementation uses an isolated execution worktree and profile; do not repoint a shared editable environment. Verify child interpreter package provenance as well as pytest cwd.
- Every To Do task must move In Progress and receive its Implementation Plan via Backlog CLI before code changes. Add Implementation Notes and mark Done only after its acceptance criteria, review, targeted tests and static checks pass.

ADR required: yes
ADR paths: [ADR-162](../../../backlog/decisions/162-managed-agent-plugins.md); [ADR-163](../../../backlog/decisions/163-expanded-console-hook-runtime.md)
Reason: Implements the accepted storage, trust, runtime and UI contracts. No additional ADR is needed unless implementation changes one of those decisions.

---

## Execution and evidence

This is an implementation plan, not implemented code or passing runtime evidence.
The code blocks below are small invariant/RED-test sketches, not a complete
implementation to paste blindly. Each task must also exercise its named production
entry and failure/control matrix. Preserve the stated interfaces across tasks;
when current library behavior contradicts a sketch, establish the real RED failure
and correct the sketch/test before implementation, as the repository's
[testing-evidence lesson](../../../backlog/docs/lessons-testing-evidence.md) requires.

Read [live verification](../../../backlog/docs/lessons-live-verification.md) before
running the app. Isolate config, data, credential and child-process roots before
importing runtime code, and verify the isolation. Use sys.executable for controlled
children. A disabled path failing from an unrelated event-loop error is not evidence.

Each task is one independently reviewable deliverable. Its checklist is the
sequence of small test/implementation increments; repeat the RED/GREEN cycle for
each listed failure/control case. Do not implement the entire subsystem before
running its first integration test. File roles and public contracts below define
the decomposition; no unrelated broad refactoring is part of these plans.

## File ownership map

| Task | New implementation units | Existing integration boundaries |
| --- | --- | --- |
| M1 | `tldw_chatbook/MCP/tool_results.py` | `tldw_chatbook/MCP/client.py`, `tldw_chatbook/MCP/local_control_service.py`, `tldw_chatbook/Agents/mcp_tool_provider.py` |
| M2 | `tldw_chatbook/MCP/streamable_http.py`, `tldw_chatbook/MCP/protocol_profiles.py` | `tldw_chatbook/MCP/client.py`, `tldw_chatbook/MCP/local_store.py`, `tldw_chatbook/MCP/local_control_service.py`, `tldw_chatbook/Utils/optional_deps.py` |
| M3 | `tldw_chatbook/MCP/credential_bindings.py` | `tldw_chatbook/MCP/local_store.py`, `tldw_chatbook/MCP/local_control_service.py`, `tldw_chatbook/MCP/streamable_http.py`, `tldw_chatbook/config.py` |
| M4 | `tldw_chatbook/Plugins/mcp_provider.py`, `tldw_chatbook/MCP/connection_ownership.py` | `tldw_chatbook/MCP/local_store.py`, `tldw_chatbook/MCP/local_control_service.py`, `tldw_chatbook/Agents/mcp_tool_provider.py`, `tldw_chatbook/Agents/tool_catalog.py`, `tldw_chatbook/Plugins/admission.py`, `tldw_chatbook/Plugins/runtime_owner.py` |

## M1: Preserve typed MCP tool results through client services

**Backlog:** [TASK-32681](../../../backlog/tasks/task-32681%20-%20Preserve-typed-MCP-tool-results-through-client-services.md). **Requires:** [TASK-32645](../../../backlog/tasks/task-32645%20-%20Design-managed-plugins-and-expanded-hook-runtime.md).

**Deliverable:** Retain the protocol information needed to distinguish tool errors from successful structured results while keeping ordinary display compatible.

**Files:**

- Create: `tldw_chatbook/MCP/tool_results.py`
- Modify: `tldw_chatbook/MCP/client.py`
- Modify: `tldw_chatbook/MCP/local_control_service.py`
- Modify: `tldw_chatbook/Agents/mcp_tool_provider.py`
- Test: `Tests/MCP/test_typed_tool_results.py`
- Test: `Tests/MCP/test_client_catalog_pagination.py`
- Test: `Tests/MCP/test_control_plane_tool_execute.py`

**Interfaces**

- Consumes: MCPClient.call_tool currently projects result.content or string into a dict; _StdioJSONRPCConnection.call_tool is the underlying raw result seam.
- Produces: MCPToolResult is a frozen Pydantic model: content: tuple[dict, ...], structured_content: dict | None, is_error: bool, metadata: dict, transport_error: str | None. parse_tool_result(payload: object) -> MCPToolResult accepts raw protocol/SDK forms with strict boolean validation and bounded serialization. MCPClient.call_tool_result(server_id: str, tool_name: str, arguments: dict) -> Awaitable[MCPToolResult]. project_tool_result(result: MCPToolResult) -> dict preserves existing UI shapes; legacy call_tool delegates through typed handling before projection.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_structured_error_survives_client_normalization():
    from tldw_chatbook.MCP.tool_results import parse_tool_result
    raw = {"content": [], "structuredContent": {"version": 2, "decision": "pass"}, "isError": True, "_meta": {"trace": "t"}}
    result = parse_tool_result(raw)
    assert result.is_error
    assert result.structured_content == raw["structuredContent"]
    assert result.metadata == raw["_meta"]
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/MCP/test_typed_tool_results.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def strict_error_flag(payload: dict) -> bool:
    value = payload.get("isError", False)
    if type(value) is not bool:
        raise ValueError("mcp_error_flag_invalid")
    return value
```

  - [ ] 3.1. Add the typed model/parser and explicit old-display projection. Preserve complete content blocks and metadata without interpolating error strings into success values.
  - [ ] 3.2. Route both SDK and built-in stdio result paths through the typed boundary, and retain a distinct sanitized transport failure. Use the typed service path for hooks/providers; display consumers get only the explicit projection.
  - [ ] 3.3. Bound the original encoded payload before hook normalization, including metadata. Keep protocol framing bounds independent of the hook 16 KiB limit, so ordinary larger non-hook results follow their existing presentation policy.
  - [ ] 3.4. Add controlled stdio client/service tests that return structured-only, mirrored text, error-with-pass and malformed flags; verify existing non-hook rendering/tool-log consumers still see compatible data.

**Failure and successful-control matrix:** Missing versus false isError, non-boolean flags, transport failure, structured-only results, resource/image blocks for non-hook display, oversized metadata and error sentinels. Do not flatten before hook interpretation.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/MCP/test_typed_tool_results.py Tests/MCP/test_client_catalog_pagination.py Tests/MCP/test_control_plane_tool_execute.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32681 --plain
git diff --check
```

## M2: Add qualified direct Streamable HTTP MCP transport

**Backlog:** [TASK-32682](../../../backlog/tasks/task-32682%20-%20Add-qualified-direct-Streamable-HTTP-MCP-transport.md). **Requires:** [TASK-32681](../../../backlog/tasks/task-32681%20-%20Preserve-typed-MCP-tool-results-through-client-services.md).

**Deliverable:** Connect directly to generic MCP servers using explicit protocol profiles instead of treating the tldw_server wrapper as equivalent support.

**Files:**

- Create: `tldw_chatbook/MCP/streamable_http.py`
- Create: `tldw_chatbook/MCP/protocol_profiles.py`
- Modify: `tldw_chatbook/MCP/client.py`
- Modify: `tldw_chatbook/MCP/local_store.py`
- Modify: `tldw_chatbook/MCP/local_control_service.py`
- Modify: `tldw_chatbook/Utils/optional_deps.py`
- Test: `Tests/MCP/test_streamable_http.py`
- Test: `Tests/MCP/test_protocol_profiles.py`
- Test: `Tests/MCP/test_local_store.py`
- Test: `Tests/MCP/test_control_plane_lifecycle.py`

**Interfaces**

- Consumes: M1 typed tool results; existing httpx stack; LocalExternalMCPProfile currently has only command/args/env. Do not reuse tldw_api/mcp_unified_client.py as direct transport evidence.
- Produces: TransportProfile is a closed discriminated stdio/streamable_http record owned by MCP/local_store.py; legacy command records migrate to stdio. MCPClient.connect_profile(profile: TransportProfile) -> Awaitable[bool]. StreamableHTTPConnection.request(method: str, params: dict, *, timeout_seconds: float) -> Awaitable[dict], notify(method: str, params: dict) -> Awaitable[None], close() -> Awaitable[None]. protocol_profile(version: str) returns only the three qualified named protocol profiles.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_unknown_protocol_does_not_fall_back_to_old_handshake():
    import pytest
    from tldw_chatbook.MCP.protocol_profiles import protocol_profile
    with pytest.raises(ValueError):
        protocol_profile("2099-01-01")
    assert protocol_profile("2025-03-26").version == "2025-03-26"
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/MCP/test_streamable_http.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
SUPPORTED_VERSIONS = frozenset({"2026-07-28", "2025-11-25", "2025-03-26"})

def require_supported_version(version: str) -> str:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError("mcp_protocol_unsupported")
    return version
```

  - [ ] 3.1. Pin official protocol fixture revisions and expected exchanges per the three spec profiles; read their primary versioning/transport sources before implementing wire details. Add controlled JSON/SSE servers, pagination, session headers and per-request metadata cases, with no tools/call detection probe.
  - [ ] 3.2. Extend closed external-profile storage with transport discrimination and an explicit schema migration/reopen test. Keep old stdio fields accepted through migration; unsupported transport/auth remains an unready diagnostic.
  - [ ] 3.3. Implement Streamable HTTP with bounded responses, explicit cancellation/close, declared version negotiation and appropriate initialize/session versus per-request flow. Retain host readiness only after discovery succeeds.
  - [ ] 3.4. Enforce selected origin, HTTPS or explicit loopback development mode, and no credential forwarding on cross-origin redirects. Reconnect refreshes definitions/permissions without replaying uncertain invocation; never downgrade to legacy HTTP+SSE or weaker TLS.

**Failure and successful-control matrix:** Both JSON and SSE responses, malformed/oversized frames, cursor loops, session expiry, disconnect during call, response loss, unsupported versions, pagination changes, cancelled connect and old stdio profile successful controls.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/MCP/test_streamable_http.py Tests/MCP/test_protocol_profiles.py Tests/MCP/test_local_store.py Tests/MCP/test_control_plane_lifecycle.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32682 --plain
git diff --check
```

## M3: Bind MCP credentials to stable reviewed authority

**Backlog:** [TASK-32683](../../../backlog/tasks/task-32683%20-%20Bind-MCP-credentials-to-stable-reviewed-authority.md). **Requires:** [TASK-32682](../../../backlog/tasks/task-32682%20-%20Add-qualified-direct-Streamable-HTTP-MCP-transport.md), [TASK-32670](../../../backlog/tasks/task-32670%20-%20Authenticate-complete-plugin-authority-snapshots.md).

**Deliverable:** Keep normal token renewal usable while ensuring account, endpoint or scope changes cannot inherit stale plugin authority.

**Files:**

- Create: `tldw_chatbook/MCP/credential_bindings.py`
- Modify: `tldw_chatbook/MCP/local_store.py`
- Modify: `tldw_chatbook/MCP/local_control_service.py`
- Modify: `tldw_chatbook/MCP/streamable_http.py`
- Modify: `tldw_chatbook/config.py`
- Test: `Tests/MCP/test_credential_bindings.py`
- Test: `Tests/MCP/test_transport_auth.py`
- Test: `Tests/Plugins/test_credential_authority.py`

**Interfaces**

- Consumes: M2 selected transport/origin and F3 authenticated references. Use existing host config/encryption/keyring boundaries; do not reinterpret runtime_policy/server_credentials.py credentials for arbitrary MCP origins.
- Produces: CredentialBinding is frozen and includes reference_id, authority_generation, method, issuer, audience, endpoint_origin, principal and scopes; opaque unverified fields are explicit None. same_authority(old: CredentialBinding, new: CredentialBinding) -> bool ignores storage/token revision. CredentialBindingService.resolve(reference_id: str, expected_generation: int, endpoint_origin: str) -> dict returns current secret material only to transport; renew(reference_id: str) -> Awaitable[CredentialBinding] never proves an uncertain tool outcome. Supported OAuth adapters remain owned by this service, not Plugins.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_token_rotation_does_not_change_binding_generation(binding_case):
    case = binding_case
    old = case.binding()
    case.rotate_token_with_same_identity()
    assert case.binding().authority_generation == old.authority_generation
    case.switch_account()
    assert case.binding().authority_generation > old.authority_generation
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/MCP/test_credential_bindings.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
AUTHORITY_FIELDS = ("reference_id", "authority_generation", "method", "issuer", "audience", "endpoint_origin", "principal", "scopes")

def same_authority(old, new) -> bool:
    return all(getattr(old, field) == getattr(new, field) for field in AUTHORITY_FIELDS)
```

  - [ ] 3.1. Build binding_case with a real CredentialBindingService and an isolated fake credential backend: binding returns the service record; rotation/switch methods simulate verified host-auth callbacks with distinct token storage revisions.
  - [ ] 3.2. Separate stable reviewed identity/scope from token bytes, expiry and storage revision. Resolve current usable secrets at dispatch and authenticate only references/generations in plugin snapshots.
  - [ ] 3.3. Wire explicit header/token mappings and available host OAuth flows. Inventory which OAuth flow is actually supported before exposing it; missing generic OAuth support returns unsupported_authentication rather than implementing a new OAuth framework or claiming imported vendor access.
  - [ ] 3.4. Invalidate captured mappings on account/issuer/audience/scope/reference/revocation changes. Test unknown continuity, opaque credentials and store migration; sanitize every status/error/receipt path with credential sentinels.

**Failure and successful-control matrix:** Valid renewal during a pending review/live run, expired token, failed refresh, identity changes, unauthorized redirect, new scope, unknown issuer/principal and no imported connector grant. Package trust alone never makes expired credentials ready.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/MCP/test_credential_bindings.py Tests/MCP/test_transport_auth.py Tests/Plugins/test_credential_authority.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32683 --plain
git diff --check
```

## M4: Expose owned plugin MCP tools with scoped connection leases

**Backlog:** [TASK-32684](../../../backlog/tasks/task-32684%20-%20Expose-owned-plugin-MCP-tools-with-scoped-connection-leases.md). **Requires:** [TASK-32683](../../../backlog/tasks/task-32683%20-%20Bind-MCP-credentials-to-stable-reviewed-authority.md), [TASK-32672](../../../backlog/tasks/task-32672%20-%20Admit-native-plugin-skills-through-existing-Console-authority.md), [TASK-32673](../../../backlog/tasks/task-32673%20-%20Stop-and-revoke-plugin-work-within-the-requested-scope.md), [TASK-32675](../../../backlog/tasks/task-32675%20-%20Delete-plugin-data-only-after-exact-root-users-drain.md), [TASK-32678](../../../backlog/tasks/task-32678%20-%20Integrate-hook-input-transformations-and-post-event-barriers.md).

**Deliverable:** Let plugin skills invoke reviewed MCP tools through normal authority while shared connections retain per-workspace ownership.

**Files:**

- Create: `tldw_chatbook/Plugins/mcp_provider.py`
- Create: `tldw_chatbook/MCP/connection_ownership.py`
- Modify: `tldw_chatbook/MCP/local_store.py`
- Modify: `tldw_chatbook/MCP/local_control_service.py`
- Modify: `tldw_chatbook/Agents/mcp_tool_provider.py`
- Modify: `tldw_chatbook/Agents/tool_catalog.py`
- Modify: `tldw_chatbook/Plugins/admission.py`
- Modify: `tldw_chatbook/Plugins/runtime_owner.py`
- Test: `Tests/Plugins/test_owned_mcp_tools.py`
- Test: `Tests/MCP/test_connection_ownership.py`
- Test: `Tests/Agents/test_mcp_tool_provider.py`

**Interfaces**

- Consumes: F5 admission, F6 scoped revocation, F8 data-root usage, M1-M3 qualified transport/credentials and H3 final tool guards.
- Produces: ConnectionAuthorityKey is frozen: installation/revision, executable/environment/cwd or endpoint, effective config digest, credential-binding generation and session-isolation qualification. ConnectionOwnership.attach(key: ConnectionAuthorityKey, owner_id: str) -> str; async detach(connection_id: str, owner_id: str) -> None; bind_request(connection_id: str, request_id: str, snapshot: RunPluginSnapshot) -> None. PluginMCPProvider implements existing ToolProvider and always dispatches typed results through the normal permission seam.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
import pytest

@pytest.mark.asyncio
async def test_detaching_a_keeps_b_transport_alive(shared_connection_case):
    case = shared_connection_case
    await case.disable_a()
    assert case.transport_connected()
    assert await case.invoke_b() == "ok"
    assert not case.accept_late_a_result()
    assert case.unresolved_a_is_recorded()
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_owned_mcp_tools.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def may_close_connection(owners: frozenset[str], affected: frozenset[str]) -> bool:
    return owners <= affected
```

  - [ ] 3.1. Create shared_connection_case with a controlled multiplexed MCP server and two real scope snapshots. Its methods call production coordinator/provider/ownership paths; hold A response while B completes and observe actual transport close.
  - [ ] 3.2. Register owned profiles and namespace-safe tool definitions through existing catalog services. Recheck exact definition hashes, stable mappings, plugin dependencies, permission profile and parent/workspace restrictions at dispatch.
  - [ ] 3.3. Expand only portable args/env/cwd fields, set host-controlled variables last and reject unresolved required config. Configuration save is data-only; explicit connection/test invokes normal review and records data-root ownership before launch.
  - [ ] 3.4. Share only identical reviewed effective authority with qualified isolated session state. Detach/cancel A at request granularity; unknown A outcome retains request/root ownership, while B remains authorized. Release shared processes only after all owners drain or are affected, and guard package-owned edit/delete at MCP service entry.

**Failure and successful-control matrix:** Different credential/config/cwd bindings, two workspaces, default inheritors, late A response, uncancellable remote call, idle writer-capable server, global disable, reserved variable override, stale approval and standalone MCP successful control.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_owned_mcp_tools.py Tests/MCP/test_connection_ownership.py Tests/Agents/test_mcp_tool_provider.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32684 --plain
git diff --check
```

## Shared package limits

These exact approved limits apply wherever this plan handles the corresponding resource.

| Resource | V1 limit | Exhaustion behavior |
| --- | --- | --- |
| Manifest or hooks definition JSON | 256 KiB each; depth 32 | Reject that document with a bounded diagnostic. |
| Catalog snapshot | 5 MiB; 10,000 entries; 50 sources | Reject oversized refresh; retain previous catalog. |
| Package snapshot | 100 MiB expanded; 10,000 files; 10 MiB/file; path depth 32 | Abort materialization; preserve current installation. |
| Normalized component inventory | 512/package | Reject inspection; do not drop arbitrary tail components. |
| Concurrent acquisitions | 2; 120 s overall per acquisition | Queue visibly or cancel with timeout; no plugin execution. |
| Git staging including object data | 500 MiB/operation | Terminate fetch at quota check; discard staging. This is host acquisition control, not an OS disk quota. |
| Managed package/cache storage | 2 GiB total; require estimated new bytes plus 100 MiB free reserve | Prune eligible cache or refuse before commit. |
| Inactive revisions | 2 most recent per installation; 30-day age target | Prune only unleased/unreferenced revisions; current and recovery records are protected. |
| Abandoned staging | 24 hours | Remove only after journal reconciliation and ownership checks. |
| Plugin instruction blocks | 8 KiB/block, 32 KiB combined per send, also bounded by remaining model context | Reject oversized selected material whole; explain affected components. |
| Listing page | 50 rows | Paginate; search remains over cached metadata. |
| Display metadata | 256 characters/name; 2,000/summary; 64 KiB README preview | Sanitize and mark display truncation; preserve immutable source for explicit file review. |
| Operation receipts | 1,000 terminal receipts or 30 days | Drop oldest eligible terminal receipts; never delete recovery authority. |
