# Direct MCP and plugin ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide typed MCP results, qualified direct transports and stable credential/scoped connection ownership for plugin tools.

**Architecture:** Extend the existing MCP client/control/provider seams with a typed result path and explicit protocol profiles. Keep transport and credential ownership outside plugin discovery, with per-request workspace authority and conservative uncertain outcomes.

**Tech Stack:** Python 3.12+, Textual 8.2.8, Pydantic 2, private SQLite, httpx, portalocker and existing trust/credential primitives.

**Spec:** [Plugin spec](../specs/2026-09-15-managed-plugins-design.md) and [hook spec](../specs/2026-09-15-expanded-hook-runtime-design.md). Read both; this plan covers its assigned subsystem within the complete [delivery plan](2026-09-15-managed-plugins-delivery.md).

## Transport ownership reconciliation (R4)

[ADR-162](../../../backlog/decisions/162-managed-agent-plugins.md) partially
supersedes [ADR-111](../../../backlog/decisions/111-mcp-remote-transport-and-client-dependency.md)
only for direct generic Streamable HTTP transport. The reciprocal amendments
place the connection under existing `MCPClient`/`httpx` and retain the existing
permission, registry and audit owners. M2 qualifies the three named protocol
profiles; M3 separately resolves supported credential bindings. No server-package
publication prerequisite, deprecated HTTP+SSE fallback or generic OAuth
qualification is implied.

R60 resolves the cross-era version-offer ambiguity: recognized modern errors
remain in the modern era; with no different qualified modern revision, surface
unsupported version. Explicit legacy profiles may counteroffer only the two
qualified legacy revisions. Qualify these cases through actual transport tests,
including rejected/same-version offers and successful legacy counteroffers.
Some dual-era servers may require explicit legacy profile selection; no automatic
initialize, retry loop or uncertain invocation replay is permitted. See ADR162
and the plugin specification for the binding contract.

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

- [x] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_structured_error_survives_client_normalization():
    from tldw_chatbook.MCP.tool_results import parse_tool_result
    raw = {"content": [], "structuredContent": {"version": 2, "decision": "pass"}, "isError": True, "_meta": {"trace": "t"}}
    result = parse_tool_result(raw)
    assert result.is_error
    assert result.structured_content == raw["structuredContent"]
    assert result.metadata == raw["_meta"]
```

- [x] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/MCP/test_typed_tool_results.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [x] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def strict_error_flag(payload: dict) -> bool:
    value = payload.get("isError", False)
    if type(value) is not bool:
        raise ValueError("mcp_error_flag_invalid")
    return value
```

  - [x] 3.1. Add the typed model/parser and explicit old-display projection. Preserve complete content blocks and metadata without interpolating error strings into success values.
  - [x] 3.2. Route both SDK and built-in stdio result paths through the typed boundary, and retain a distinct sanitized transport failure. Use the typed service path for hooks/providers; display consumers get only the explicit projection.
  - [x] 3.3. Bound the original encoded payload before hook normalization, including metadata. Keep protocol framing bounds independent of the hook 16 KiB limit, so ordinary larger non-hook results follow their existing presentation policy.
  - [x] 3.4. Add controlled stdio client/service tests that return structured-only, mirrored text, error-with-pass and malformed flags; verify existing non-hook rendering/tool-log consumers still see compatible data.

**Failure and successful-control matrix:** Missing versus false isError, non-boolean flags, transport failure, structured-only results, resource/image blocks for non-hook display, oversized metadata and error sentinels. Do not flatten before hook interpretation.

- [x] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/MCP/test_typed_tool_results.py Tests/MCP/test_client_catalog_pagination.py Tests/MCP/test_control_plane_tool_execute.py -q
```

- [x] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

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

- [x] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_unknown_protocol_does_not_fall_back_to_old_handshake():
    import pytest
    from tldw_chatbook.MCP.protocol_profiles import protocol_profile
    with pytest.raises(ValueError):
        protocol_profile("2099-01-01")
    assert protocol_profile("2025-03-26").version == "2025-03-26"
```

- [x] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/MCP/test_streamable_http.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [x] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
SUPPORTED_VERSIONS = frozenset({"2026-07-28", "2025-11-25", "2025-03-26"})

def require_supported_version(version: str) -> str:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError("mcp_protocol_unsupported")
    return version
```

  - [x] 3.1. Pin official protocol fixture revisions and expected exchanges per the three spec profiles; read their primary versioning/transport sources before implementing wire details. Add controlled JSON/SSE servers, pagination, session headers and per-request metadata cases, with no tools/call detection probe.
  - [x] 3.2. Extend closed external-profile storage with transport discrimination and an explicit schema migration/reopen test. Keep old stdio fields accepted through migration; unsupported transport/auth remains an unready diagnostic.
  - [x] 3.3. Implement Streamable HTTP with bounded responses, explicit cancellation/close, declared version negotiation and appropriate initialize/session versus per-request flow. Retain host readiness only after discovery succeeds.
  - [x] 3.4. Enforce selected origin, HTTPS or explicit loopback development mode, and no credential forwarding on cross-origin redirects. Reconnect refreshes definitions/permissions without replaying uncertain invocation; never downgrade to legacy HTTP+SSE or weaker TLS.

**Failure and successful-control matrix:** Both JSON and SSE responses, malformed/oversized frames, cursor loops, session expiry, disconnect during call, response loss, unsupported versions, pagination changes, cancelled connect and old stdio profile successful controls.

- [x] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/MCP/test_streamable_http.py Tests/MCP/test_protocol_profiles.py Tests/MCP/test_local_store.py Tests/MCP/test_control_plane_lifecycle.py -q
```

- [x] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

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

- [x] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_token_rotation_does_not_change_binding_generation(binding_case):
    case = binding_case
    old = case.binding()
    case.rotate_token_with_same_identity()
    assert case.binding().authority_generation == old.authority_generation
    case.switch_account()
    assert case.binding().authority_generation > old.authority_generation
```

- [x] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/MCP/test_credential_bindings.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [x] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
AUTHORITY_FIELDS = ("reference_id", "authority_generation", "method", "issuer", "audience", "endpoint_origin", "principal", "scopes")

def same_authority(old, new) -> bool:
    return all(getattr(old, field) == getattr(new, field) for field in AUTHORITY_FIELDS)
```

  - [x] 3.1. Build binding_case with a real CredentialBindingService and an isolated fake credential backend: binding returns the service record; rotation/switch methods simulate verified host-auth callbacks with distinct token storage revisions.
  - [x] 3.2. Separate stable reviewed identity/scope from token bytes, expiry and storage revision. Resolve current usable secrets at dispatch and authenticate only references/generations in plugin snapshots.
  - [x] 3.3. Wire explicit header/token mappings and available host OAuth flows. Inventory which OAuth flow is actually supported before exposing it; missing generic OAuth support returns unsupported_authentication rather than implementing a new OAuth framework or claiming imported vendor access.
  - [x] 3.4. Invalidate captured mappings on account/issuer/audience/scope/reference/revocation changes. Test unknown continuity, opaque credentials and store migration; sanitize every status/error/receipt path with credential sentinels.

**Failure and successful-control matrix:** Valid renewal during a pending review/live run, expired token, failed refresh, identity changes, unauthorized redirect, new scope, unknown issuer/principal and no imported connector grant. Package trust alone never makes expired credentials ready.

- [x] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/MCP/test_credential_bindings.py Tests/MCP/test_transport_auth.py Tests/Plugins/test_credential_authority.py -q
```

- [x] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

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

- [x] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

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

- [x] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_owned_mcp_tools.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [x] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def may_close_connection(owners: frozenset[str], affected: frozenset[str]) -> bool:
    return owners <= affected
```

  - [x] 3.1. Create shared_connection_case with a controlled multiplexed MCP server and two real scope snapshots. Its methods call production coordinator/provider/ownership paths; hold A response while B completes and observe actual transport close.
  - [x] 3.2. Register owned profiles and namespace-safe tool definitions through existing catalog services. Recheck exact definition hashes, stable mappings, plugin dependencies, permission profile and parent/workspace restrictions at dispatch.
  - [x] 3.3. Expand only portable args/env/cwd fields, set host-controlled variables last and reject unresolved required config. Configuration save is data-only; explicit connection/test invokes normal review and records data-root ownership before launch.
  - [x] 3.4. Share only identical reviewed effective authority with qualified isolated session state. Detach/cancel A at request granularity; unknown A outcome retains request/root ownership, while B remains authorized. Release shared processes only after all owners drain or are affected, and guard package-owned edit/delete at MCP service entry.

**Failure and successful-control matrix:** Different credential/config/cwd bindings, two workspaces, default inheritors, late A response, uncancellable remote call, idle writer-capable server, global disable, reserved variable override, stale approval and standalone MCP successful control.

- [x] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_owned_mcp_tools.py Tests/MCP/test_connection_ownership.py Tests/Agents/test_mcp_tool_provider.py -q
```

- [x] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

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

### M2 implementation boundary (TASK-32682)

The resolved `MCP.local_store.TransportProfile` is a frozen discriminated
`stdio`/`streamable_http` record consumed by `MCPClient.connect_profile`.
It preserves literal argv and an owned, read-only environment mapping. Existing
manual `LocalExternalMCPProfile` records retain their historical argv/env
normalization. Recognized legacy JSON stores migrate atomically to schema 2
with explicit stdio/version fields; later opens do not rewrite the file.
Unknown versions and malformed authoritative sections refuse migration without
replacing their bytes. Reserved profile IDs retain the existing quarantine.
Changing transport, endpoint or protocol invalidates persisted discovery.

`StreamableHTTPConnection` uses core httpx below the existing client cleanup,
discovery and typed-result methods. It neither creates a federation manager nor
owns tool permissions or audit. `LocalMCPControlService` resolves only stdio
launch environments and delegates both transports to the client. Explicit
profiles discover advertised catalogs only; unavailable optional catalogs and
invalid modern header annotations have fixed diagnostics. The legacy
`connect_to_server` entry retains its exact-version compatibility contract.

HTTP admission closes before resource teardown. Request scopes drain independently
of arbitrary caller finalization. The existing five-second client cleanup bound
still applies: a cancelled wait leaves the same cleanup task and unready session
owned for a later close attempt, with its live catalog withdrawn. A pending close
may finish and then ordinary disconnect/reconnect can release/reuse the profile.
An actual lower close failure remains incomplete and cannot be automatically
replaced: HTTPX marks its client closed before closing the pool, so that flag or
a second no-op client close cannot establish resource closure. Local cleanup
never settles an uncertain remote outcome or authorizes invocation replay.

Wire qualification uses original, repository-owned stdio peers and loopback
HTTP peers: the three explicit versions each exercise actual stdio plus HTTP
JSON and SSE, with full M1 raw-result/dispatch provenance. These are controlled
interoperability fixtures, not an external certification suite. The optional
mcp-unified server remains unqualified when its extra is unavailable.

Transport policy: HTTPS with normal certificate verification, or an explicitly
selected numeric loopback HTTP development endpoint; URL userinfo and fragments
are refused. Literal queries are preserved without host credential insertion or
expansion (R61). No automatic redirects or environment proxy inheritance. Request `Accept-Encoding: identity` and reject other content
encodings before raw byte iteration. Wire body budget is 1,048,576 bytes per
exchange including resumed streams, 1,024 SSE events per stream, at most three
GET resumptions, and the existing result/depth/catalog bounds. Legacy resumptions
respect `retry` within the caller's deadline and retain the same request ID;
they never repost the invocation. A session 404 or malformed/lost transport
retires readiness; reconnect performs fresh discovery without replay.

Modern request metadata and routing headers use the selected profile, including
UTF-8 Base64 sentinel encoding and properties-only `x-mcp-header` extraction.
Only the declared primitive types and safe-range integers are mirrored; invalid
tool annotations exclude that tool. HTTP literal credential/custom-header
mapping, Unicode-to-octet policy for those arbitrary headers, credential
refresh, and OAuth remain M3/M4 work. Auth challenges are explicitly unsupported
at this boundary; profile input cannot silently discard configured auth fields.
Legacy server requests support ping and method-not-supported replies, with no
advertised sampling/elicitation/roots capability. Modern input-required/MRTR
results are unsupported. Standalone legacy push listeners and modern
subscriptions are not activated; observed list-change notifications invalidate
readiness until reconnect. This is distinct from request-scoped SSE and legacy
GET response resumption, which are supported.

### Literal MCP endpoint queries (R61)

Preserve literal routing query parameters in an MCP endpoint URL, including
duplicates and empty values. They are visible configuration, not credential
references, and changing the full endpoint invalidates its discovery. Do not
expand placeholders/environment values or insert host credentials into URLs.
Host authorization uses its selected-origin credential service. Existing
HTTPS/explicit-loopback, userinfo/fragment and redirect restrictions still apply.
This follows the [Agent Plugins endpoint contract](https://agent-plugins.org/specification)
without adding a blanket query-string restriction.

### M3 credential reference recovery boundary (R62)

M3 captures and validates complete connection mappings through the actual local MCP and credential owners: saved profile target, retained component definition, effective configuration and stable credential binding must all match. Authenticated recovery may reconstruct only those supported current references. M3 recovery fixtures may seed the existing protected snapshot/transaction boundary, but this does not qualify a public mapping-edit or launch workflow. M4 supplies reviewed publication and registration before plugin connections launch. No mapping, successful recovery or credential binding creates tool permission or vendor grants.

### Credential reference identity after record loss (R63)

New MCP credential records receive immutable UUID reference IDs from the host credential service. A supplied missing reference is unready and cannot be recreated at generation one. Normal replacement, renewal and revocation reread the protected record under the existing owner lock; retained tombstones and monotonically advancing signed-64-bit authority generations prevent ordinary reuse, and generation exhaustion refuses. After record loss, the user must create and review a fresh reference before rebinding a plugin mapping. No credential creation restores prior tool permission or proves a prior remote invocation completed.

### Credential I/O and async transport deadlines (R64)

Blocking credential-store and file-lock operations run outside the shared MCP event loop. The existing credential service retains at most one worker operation; other async callers wait within their applicable deadlines before performing a fresh operation, without a queued worker backlog or secret-result cache. Cancellation ends the wait and cannot trigger later HTTP dispatch; it does not terminate an OS keychain call. A stalled backend may retain one daemon worker until completion or process exit and cause authenticated requests to time out, while anonymous connections remain responsive. Capacity releases only after actual completion, and local waiting or cleanup never proves remote invocation completion or permits replay.

### M3 implementation and operation contract

`MCP.credential_bindings.CredentialBindingService` owns protected references.
`create_opaque(endpoint_origin=..., headers=..., method="headers"|"bearer")`
mints a new host reference; `set_opaque(reference_id, ...)` replaces an existing
record and always advances its authority generation. `create(adapter)` and
`authorize(adapter, reference_id)` call explicitly registered host adapters;
`renew(reference_id)` reuses that adapter with a storage-revision check. The
production factory registers no generic MCP OAuth adapters, so unavailable flows
return `unsupported_authentication`. Server-account/provider OAuth and imported
vendor connector claims do not establish MCP authority.

Complete records live only in a secure OS keyring namespace scoped by the
canonical data root. Existing portalocker plus a shared process lock serializes
read/modify/write across instances. No plaintext/config fallback exists. A failed
write returns a fixed failure; later resolution rereads actual protected state.
Revocation keeps a generation tombstone. Deleted records require fresh UUID
references and explicit rebinding. Signed-64-bit generation exhaustion refuses.
`config.create_mcp_credential_service` is a lazy factory, not a new TOML secret
format. JSON MCP profile schema 3 stores only `credential_reference` and
`credential_generation`; schema 1/2 migrate without inventing bindings, and
malformed/older-schema credential authority is not rewritten.

Only selected-origin HTTP dispatch receives current headers through
`resolve_async`; synchronous `resolve` is for existing worker-owned callers.
The async storage adapter retains at most one daemon worker, waits for real
completion before admitting another operation, and never queues thread jobs.
Timeout/cancellation stops waiting, not an OS operation; authenticated callers
can remain unready while anonymous peers continue. No late request or uncertain
invocation replay follows credential renewal or cancellation.

Header names are case-insensitively unique. Host protocol/routing, hop-by-hop,
proxy and cookie headers are reserved. Explicit bearer/custom header values
support literal Latin-1 octets (including obs-text), empty values and interior
HTAB; leading/trailing whitespace, other controls and wider Unicode are refused.
This is the host's wire-encoding policy, not a claim that portable Unicode
inventory is RFC-invalid. Credentials never enter endpoint queries. Redirects
remain refused. Legacy notification/DELETE authentication refusal still closes
local HTTP resources.

`LocalMCPControlService.capture_connection_mapping` captures the actual saved
HTTP profile, exact retained MCP definition/configuration digests and stable
credential metadata in the existing `connection` mapping shape.
`validate_connection_mapping` checks the complete reference before coordinator
recovery publishes or reconstructs it. Unsupported owners/kinds still refuse.
M3's guarded authenticated recovery fixtures qualify reconstruction only;
M4 must publish/register reviewed mappings and enforce live plugin admission.
No successful recovery creates tool permission, account grants or runtime proof.

Qualification uses memory credential backends, real portalocker with a fake
keyring API, isolated profiles and controlled local HTTP peers. Real OS keychain
interoperability, generic OAuth, other platforms and external production MCP
servers remain unqualified by this task.

### Shared MCP session qualification and uncertain request custody (R65/R66)

Owned MCP profiles default to separate scoped connections. Sharing requires an explicit host-controlled `request_independent` qualification bound by the same immutable configure review and authenticated mapping as the exact execution, definition, effective configuration and credential authority. Package metadata, a server claim or transport multiplexing cannot supply it. Unknown qualification stays isolated. Every request retains its workspace/parent/permission and current-authority checks; changed qualification or binding requires review. The host attestation can be mistaken and does not prove arbitrary external-server or original-host isolation.

A same-session request with an uncertain outcome retains its published active owner, durable host identity/outcome and complete root joins while exact connection custody remains. Local waiter cancellation does not call recovery settlement merely to mark uncertainty, release a request, or block unrelated authorized B work globally. Existing unresolved and foreign-session records are never promoted; lost custody and restart follow the existing recovery/dirty-checkpoint gates. Request completion cannot settle idle writer-capable server lifetime, and local transport closure cannot prove uncertain remote completion or permit replay. Uncertain requests can continue blocking revision drain and data deletion until positive terminal evidence exists.

### Initial owned MCP setup and discovered definitions (R67)

Saving owned configuration is data-only. Publish its exact connection mapping through the existing immutable configure review/commit before explicit connect or test. That authorized discovery can produce tool definitions; publish their exact reviewed tool mappings through the same configuration owner before plugin advertisement. Unchanged already-reviewed discovery may be reused. First setup can therefore require a connection review followed by a discovered-tool review. Neither step grants ordinary tool permission, starts execution implicitly or creates another approval owner; all per-call checks and no-tools/call probing rules still apply.


### Portable literal HTTP headers and credential separation (R68)

Agent Plugins 1.0.0 section 7.2.1 defines remote headers as visible package data,
with client-generated HTTP/MCP/authorization headers taking precedence by
case-insensitive name. Only the authenticated retained package definition may
supply an owned profile's literal header map; arbitrary save-time raw overrides
and foreign per-install header import remain forbidden. The reviewed exact
configuration digest covers that public declaration. Runtime credentials stay
in the protected credential owner and are resolved fresh for the selected origin;
no resolved secret goes into an authority snapshot, profile, audit or diagnostic.
Header spelling cannot prove a value is non-secret.

Compose one case-insensitive map from package literals, then current credential
headers, then authoritative host HTTP/MCP/routing/session fields. Preserve host
framing, hop-by-hop, proxy and MCP namespace controls, including headers normally
generated by the HTTP client. Never expand placeholders in URL/header names or
values, forward across origins, or follow redirects. Preserve the existing exact
Latin-1 wire policy for empty/interior-HTAB/obs-text values; wider Unicode remains
explicitly unsupported at runtime. Package literals are not a credential mechanism,
and standalone raw secret fields remain refused. Actual controlled-peer observation,
case-collision precedence and no-launch save tests qualify this boundary; they do
not qualify external services, arbitrary Unicode or foreign app behavior.

Source: [Agent Plugins specification, remote MCP configuration](https://agent-plugins.org/specification#streamable-http-and-legacy-httpsse).


### Component readiness and immutable MCP capture ceilings (R69)

A missing, changed or unusable MCP owner mapping makes its own component and
declared dependents unready. It does not discard a valid independent sibling.
Global authenticated authority, retained interpretation and current installation,
scope and root generations remain exact; unknown/missing required dependencies
cannot be treated as independent. Recovery still validates every supported
reference in the complete authenticated snapshot.

The existing MCP capture API may take an explicit component ceiling, checked
against current authenticated selection with the complete prerequisite closure.
Unavailable explicitly requested components refuse instead of silently shrinking
the request. Mappings, dependency records and advertised tools respect that
immutable ceiling. Default capture uses the currently eligible set. An already
admitted snapshot containing a newly failed component still refuses; a fresh
narrower independent capture is required. A B-only snapshot need not fail merely
because unrelated A becomes unready, provided B's captured mappings, complete
requirements and generations still match. No narrowing restores an old approval,
replays an invocation or weakens whole-snapshot recovery.


### M4 implementation verification handoff

The owned provider and connection adapter use the actual protected configuration review/commit owner and existing catalog/permission/typed transport entries. The permanent qualification lives in `Tests/Plugins/test_owned_mcp_tools.py`, `Tests/MCP/test_connection_ownership.py` and the existing `Tests/Agents/test_mcp_tool_provider.py` regression. Controls cover real multiplexed stdio workspace A/B, component A/B prerequisite ceilings, permission changes during approval, literal argv/env/cwd, native root deletion, pending/idle/global/default lifecycle, actual public revision drain, controlled HTTP loss/retained cleanup, package header octets/precedence and credential changes. Schema 4 migration/source preservation is covered in the existing local-store and credential tests. The delivery task report records exact targeted covering/static commands and limits; counts from intermediate runs are not additive.

### Current native owner integration (M4)

Retained MCP launch/request tasks acquire their own recovery admission through the existing worker-isolation seam; inherited task state is never treated as a transferable storage lease. Direct profile admission refusal returns false, and owned launch requires an actual true result plus the exact retained session.

For an exact host-qualified request-independent stdio session with another attached owner, a request deadline/cancel retains its original native producer/source admission until its original validated terminal reply or actual child exit. It never replays or terminates the shared peer to settle one request. Ordinary/separate/last-owner cleanup retains native kill-and-reap custody. Revoked scopes still refuse late results.

Portable MCP expansion recognizes only ${PLUGIN_ROOT} and ${PLUGIN_DATA} in args, env values and cwd, once. Unknown placeholder text remains literal. Stdio configuration requires an explicitly created persistent plugin data-root binding before publication/launch; saving configuration never creates a root or launches a peer.
