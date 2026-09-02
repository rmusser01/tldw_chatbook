# Portable Tool-use Packs V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add deterministic, policy-only `.tldw-tool-pack` export/import with exact review, unbound installation, confirmed first binding, and fail-closed profile removal.

**Architecture:** `MCPPermissionStore` remains runtime policy authority and gains strict snapshots, lifecycle validation, and serialized complete-profile mutations. A new dependency-light `Tool_Packs` package owns portable contracts, inventory flattening, receipts, import/export, activation, binding, removal, and presentation orchestration; the workspace registry remains the only binding writer and invokes an attached guard. Settings manages profiles, while the existing MCP Permissions surface remains the only policy editor.

**Tech Stack:** Python 3.11+, standard-library `json`/`zipfile`/`hashlib`/`os`/`pathlib`/`threading`/`sqlite3`, Textual 8.x, pytest, real temporary SQLite and filesystem fixtures.

**Spec:** `Docs/superpowers/specs/2026-08-31-tool-use-pack-design.md`

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/107-portable-tool-use-packs.md`

**Reason:** Accepted ADR-107 already fixes the storage, runtime, binding, removal, privacy, and future-plugin boundaries implemented by this plan.

## Global Constraints

- V1 archives contain exactly `tool-pack.json` and `profile/profile.json`, use `ZIP_STORED`, and obey every deterministic ZIP/JSON field in spec §1.
- Archive/profile limits are 5 MiB/4 MiB; manifest limit is 256 KiB; maximum inventory is 2,000 tools, 256 source servers, and 257 fallbacks.
- Import-admission limits are 128 permission profiles and 8 MiB canonical permission-store bytes; receipts are at most 4 MiB each and 32 MiB total.
- Permission-store schema remains `1`; Tool Pack review paths never call the mutating recovery behavior of `MCPPermissionStore.load()`.
- Packs contain no tools, skills, plugins, commands, arguments, environment, endpoints, credentials, workspace/Persona bindings, session grants, or executable/install instructions.
- Imported MCP-global, builtin, and server fallbacks are Ask or Deny only. Exact Allow requires an exact portable contract match and a freshly computed destination runtime hash where that namespace uses hashes.
- Code-owned local/Virtual CLI inventory uses the unbound fallback root with no admitted-root aliases. A later contextual schema mismatch downgrades Allow to Ask through the existing runtime guard.
- Lock order is lifecycle coordinator → permission-store path fence → workspace SQLite transaction. Cross-process writers remain unsupported.
- Native Windows picker/publication support is not claimed by this plan. Unsupported secure publication fails with `publication_unsupported`.
- Operations over 100 ms run in Textual workers; canonicalization/import never blocks the event loop.
- Run targeted suites after each task. Do not run the full repository suite unless the user explicitly requests it.
- Before each implementation task, create/read one atomic Backlog task, set it In Progress, add that task's plan plus the ADR check above, and close it only after its tests, notes, review, and ACs are complete.

## File and ownership map

- `tldw_chatbook/MCP/permission_store.py`: strict snapshots, lifecycle-aware resolution, profile-scoped raw accessors, shared path fencing, and complete-profile mutations.
- `tldw_chatbook/MCP/unified_control_plane_service.py`: profile-scoped session approvals and by-key/Test Tool gates.
- `tldw_chatbook/Agents/{builtin_tool_gate,mcp_tool_provider}.py` and `tldw_chatbook/Chat/console_chat_controller.py`: capture one run profile and propagate it through every included provider.
- `tldw_chatbook/Tool_Packs/contracts.py`: exact schemas, canonical JSON, bounds, identifiers, digests, and stable errors.
- `tldw_chatbook/Tool_Packs/catalog_snapshot.py`: code-owned provider registry, complete inventory, portable contract fingerprints, and flattening inputs.
- `tldw_chatbook/Tool_Packs/export.py`: immutable export review/snapshot and deterministic archive bytes.
- `tldw_chatbook/Tool_Packs/publication.py`: captured-destination, no-follow atomic publication and reconciliation.
- `tldw_chatbook/Tool_Packs/importer.py`: bounded, side-effect-free archive inspection and exact/manual mappings.
- `tldw_chatbook/Tool_Packs/activation.py`: destination revalidation and safe runtime-profile compilation/install.
- `tldw_chatbook/Tool_Packs/receipt_store.py`: private bounded receipts, reservations, compaction, and orphan reconciliation.
- `tldw_chatbook/Tool_Packs/binding.py`: lifecycle coordinator, leases, bind reviews/tokens, registry guard, and tombstone removal.
- `tldw_chatbook/Tool_Packs/service.py`: app/UI-facing orchestration and stable outcomes.
- `tldw_chatbook/Widgets/Settings_Widgets/tool_profiles_panel.py`: modular Settings profile management UI.
- `tldw_chatbook/UI/MCP_Modules/{mcp_permissions_mode,mcp_workbench}.py`: captured-profile selector and policy editing/testing.

---

### Task 1: Strict snapshots and authoritative lifecycle resolution

**Files:**
- Modify: `tldw_chatbook/MCP/permission_store.py`
- Modify: `Tests/MCP/test_permission_store.py`
- Modify: `Tests/MCP/test_permission_resolution.py`

**Interfaces:**
- Produces: `PermissionStoreSnapshot`, `PermissionStoreSnapshotError`, `MCPPermissionStore.read_snapshot_strict()`, `profile_lifecycle_disposition()`, and lifecycle/tombstone-aware resolver behavior.
- Preserves: legacy `load()` recovery behavior and unknown-profile inheritance for profiles with neither lifecycle field.

- [ ] **Step 1: Write strict-read failure tests**

```python
@pytest.mark.parametrize("raw", [b"{", b'{"schema_version":99}', b'{"schema_version":1,"profiles":null}'])
def test_strict_snapshot_rejects_without_touching_bytes(tmp_path, raw):
    path = tmp_path / "mcp_permissions.json"
    path.write_bytes(raw)
    with pytest.raises(PermissionStoreSnapshotError):
        MCPPermissionStore(path).read_snapshot_strict()
    assert path.read_bytes() == raw
    assert not path.with_suffix(".json.bak").exists()
```

- [ ] **Step 2: Run the strict-read tests and confirm the missing API failure**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_permission_store.py -k strict_snapshot -q`

Expected: FAIL because `PermissionStoreSnapshotError` and `read_snapshot_strict` do not exist.

- [ ] **Step 3: Add immutable snapshot types and a byte-preserving parser**

```python
@dataclass(frozen=True)
class PermissionStoreSnapshot:
    payload: Mapping[str, Any]
    generation: str
    file_exists: bool

class PermissionStoreSnapshotError(ValueError):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category

def read_snapshot_strict(self) -> PermissionStoreSnapshot:
    """Read schema 1 once without creating, renaming, normalizing, or saving."""
```

Use strict UTF-8 JSON decoding with duplicate-key rejection, validate `schema_version == 1`, `kill_switch: bool`, `profiles: dict`, every profile/servers/tools container, states, hashes, and lifecycle shapes, then recursively freeze a deep copy. For a missing file return the frozen fresh payload and `generation="missing:<sha256>"`; for a present file use `generation="sha256:<raw-byte-digest>"`.

- [ ] **Step 4: Write lifecycle-disposition and resolver tests**

```python
@pytest.mark.parametrize(
    ("profile", "origin"),
    [
        ({"profile_kind": "tool_pack_imported", "servers": {}}, "lifecycle_invalid"),
        ({"tool_pack_lifecycle": {"schema": "tldw.tool-pack-lifecycle/v1"}, "servers": {}}, "lifecycle_invalid"),
        ({"profile_kind": "unknown", "tool_pack_lifecycle": {}, "servers": {}}, "lifecycle_invalid"),
    ],
)
def test_invalid_lifecycle_resolves_deny(profile, origin):
    payload = _named_payload("portable", profile)
    assert resolve_effective_state(payload, _tool(), profile_id="portable") == EffectiveToolState("deny", origin)

def test_tombstone_short_circuits_named_inheritance():
    payload = _tombstone_payload(default_global="allow")
    assert resolve_builtin_state(payload, _builtin(), profile_id="portable").state == "deny"
```

- [ ] **Step 5: Implement exact imported/tombstone lifecycle validation**

```python
ProfileLifecycleDisposition = Literal["legacy", "imported", "tombstone", "invalid"]

def profile_lifecycle_disposition(profile: Mapping[str, Any]) -> ProfileLifecycleDisposition:
    """Require kind+lifecycle together and reject missing, extra, or mismatched fields."""
```

Validate the imported and tombstone variants from spec §4.2, including SHA-256 fields, positive revision, exact count keys, receipt link, boolean marker, and origin/kind agreement. Make `resolve_effective_state`, `resolve_builtin_state`, and `resolve_effective_state_by_key` return Deny with origin `lifecycle_invalid` or `tombstone` before `_profile_chain()` inheritance.

- [ ] **Step 6: Add profile-aware raw getter tests and implementation**

```python
assert store.get_global_default(profile_id="portable") == "deny"
assert store.get_server_entry("local:docs", profile_id="portable")["default"] == "ask"
assert store.get_tool_entry("local:docs", "search", profile_id="portable")["state"] == "allow"
```

Change the three getter signatures to require/accept `profile_id: str = "default"` and ensure they never seed or mutate data while reading.

- [ ] **Step 7: Run focused permission-store and resolver suites**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_permission_store.py Tests/MCP/test_permission_resolution.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the strict authority seam**

```bash
git add tldw_chatbook/MCP/permission_store.py Tests/MCP/test_permission_store.py Tests/MCP/test_permission_resolution.py
git commit -m "feat: add strict lifecycle-aware permission snapshots"
```

---

### Task 2: Shared mutation fencing and complete-profile operations

**Files:**
- Create: `tldw_chatbook/Tool_Packs/__init__.py`
- Create: `tldw_chatbook/Tool_Packs/binding.py`
- Modify: `tldw_chatbook/MCP/permission_store.py`
- Create: `Tests/Tool_Packs/__init__.py`
- Create: `Tests/Tool_Packs/test_permission_profile_authority.py`

**Interfaces:**
- Consumes: `PermissionStoreSnapshot` and lifecycle validation from Task 1.
- Produces: `ToolProfileLifecycleCoordinator`, `profile_policy_digest()`, `MCPPermissionStore.mutation_fence()`, digest-aware field mutators, `install_profile_if_absent()`, `update_imported_profile()`, and `replace_profile_with_tombstone()`.

- [ ] **Step 1: Write a controlled multi-instance lost-update test**

```python
def test_two_store_instances_share_one_path_fence(tmp_path, monkeypatch):
    first = MCPPermissionStore(tmp_path / "permissions.json")
    second = MCPPermissionStore(tmp_path / "permissions.json")
    # Pause first after its locked read, start second, then release first.
    # Assert both distinct server-default writes survive in the final payload.
```

Use `threading.Event` barriers around the internal locked mutation hook; do not use sleeps.

- [ ] **Step 2: Run the concurrency test and verify the lost-update failure**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_permission_profile_authority.py::test_two_store_instances_share_one_path_fence -q`

Expected: FAIL because store instances do not share a path fence.

- [ ] **Step 3: Implement the resolved-path reentrant fence**

```python
@contextmanager
def mutation_fence(self) -> Iterator[None]:
    """Hold the process-wide RLock for this resolved store path."""

def _mutate_locked(self, change: Callable[[dict[str, Any]], bool]) -> bool:
    """Load, validate, mutate, and atomically save under the shared fence."""
```

Refactor every existing field mutator to perform its load/change/save inside `_mutate_locked`; no mutator may keep the current `load()`-then-public-`save()` gap. Wrap the low-level full-replacement `save()` itself in the shared fence and add optional expected-generation comparison for callers that derived a payload from a prior snapshot. Keep callbacks and foreign service calls outside the fence. Preserve the existing path and schema; write one private sibling temporary, flush/fsync it, `os.replace`, then fsync the parent on supported POSIX hosts.

- [ ] **Step 4: Write complete-profile CAS tests**

```python
result = store.install_profile_if_absent(
    "research",
    imported_profile,
    expected_generation=snapshot.generation,
    max_profiles=128,
    max_store_bytes=8 * 1024 * 1024,
)
assert result.profile_id == "research" and result.revision == 1
with pytest.raises(ProfileMutationError, match="profile_exists"):
    store.install_profile_if_absent("research", imported_profile, expected_generation=result.store_generation)
```

Cover exact/case-folded collisions, stale generation/revision, invalid lifecycle, projected profile/byte caps, imported policy edits updating digest/revision while keeping the first-bind marker, and tombstone replacement containing no Allow/Ask rows.
Also cover a legacy/default profile field mutation with `expected_profile_digest`: a concurrent edit to that same profile must raise `stale_profile`, while an edit to a different profile may coexist.

- [ ] **Step 5: Implement complete-profile mutation result and errors**

```python
@dataclass(frozen=True)
class ProfileMutationResult:
    profile_id: str
    revision: int
    policy_digest: str
    store_generation: str

class ProfileMutationError(ValueError):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category
```

`profile_policy_digest()` hashes canonical normalized policy fields plus `profile_kind`, excluding the lifecycle object and top-level store timestamp, for imported and ordinary profiles alike. All complete-profile methods strictly reload inside `mutation_fence`, compare expected values, validate exact lifecycle shape, compute that digest, and save once. Existing field mutators accept optional `expected_profile_digest`/`expected_revision`; imported-profile changes update lifecycle revision/digest atomically and all invalid lifecycle pairs are refused.

- [ ] **Step 6: Implement the lifecycle coordinator and lease accounting**

```python
class ToolProfileLifecycleCoordinator:
    @contextmanager
    def mutation(self) -> Iterator[None]: ...

    @contextmanager
    def lease(self, profile_id: str) -> Iterator[None]: ...

    def active_lease_count(self, profile_id: str) -> int: ...
```

Use one process-wide `RLock` for lifecycle mutations and a condition-protected exact-profile lease counter. Mutation callers acquire this coordinator before the permission-store fence; lease acquisition never calls the permission store.

- [ ] **Step 7: Run authority tests and the existing permission-store suite**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_permission_profile_authority.py Tests/MCP/test_permission_store.py -q`

Expected: PASS.

- [ ] **Step 8: Commit serialized profile authority**

```bash
git add tldw_chatbook/Tool_Packs tldw_chatbook/MCP/permission_store.py Tests/Tool_Packs Tests/MCP/test_permission_store.py
git commit -m "feat: serialize complete tool profile mutations"
```

---

### Task 3: Named-profile propagation across every included runtime

**Files:**
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Modify: `tldw_chatbook/Agents/builtin_tool_gate.py`
- Modify: `tldw_chatbook/Agents/mcp_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `Tests/MCP/test_control_plane_permissions.py`
- Modify: `Tests/Agents/test_builtin_tool_gate.py`
- Modify: `Tests/Agents/test_local_tool_provider.py`
- Modify: `Tests/Agents/test_virtual_cli_provider.py`
- Modify: `Tests/Agents/test_raw_shell_tool_provider.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`

**Interfaces:**
- Consumes: profile-aware resolvers/getters from Task 1.
- Produces: profile-scoped session approval triples and one captured `profile_id` across builtin, MCP, local, Virtual CLI, and raw-shell resolution/persistence.

- [ ] **Step 1: Write session-isolation tests**

```python
service.approve_for_session("local:docs", "search", profile_id="research")
assert service.is_session_approved("local:docs", "search", profile_id="research")
assert not service.is_session_approved("local:docs", "search", profile_id="default")
```

Also assert `clear_session_approvals(profile_id="research")` preserves another profile and the no-argument form clears all for backward compatibility.

- [ ] **Step 2: Change session approval storage to exact triples**

```python
self._session_approvals: set[tuple[str, str, str]] = set()

def approve_for_session(self, server_key: str, tool_name: str, *, profile_id: str = "default") -> None: ...
def is_session_approved(self, server_key: str, tool_name: str, *, profile_id: str = "default") -> bool: ...
def clear_session_approvals(self, *, profile_id: str | None = None) -> None: ...
```

- [ ] **Step 3: Write provider propagation regressions**

```python
def test_named_run_approval_never_mutates_default(store, controller):
    controller._compose_local_provider(turn_context=_turn(profile_id="research"))
    controller.local_persist_callback(_hub("fs_read"), "always_allow")
    assert store.get_tool_entry("local:__local__", "fs_read", profile_id="research")
    assert store.get_tool_entry("local:__local__", "fs_read", profile_id="default") is None
```

Repeat for MCP `approve_session`/`always_allow`, Virtual CLI, raw-shell resolution, and `BuiltinToolGate.resolve()`/session approval. Use signature-recording doubles to prove every call carries the exact captured id.

- [ ] **Step 4: Thread profile id through control-plane by-key gates**

```python
def gate_tool_test_by_key(self, server_key: str, tool_name: str, *, profile_id: str = "default") -> EffectiveToolState:
    payload = store.load()
    return resolve_effective_state_by_key(payload, server_key, tool_name, profile_id=profile_id)
```

Keep global kill-switch methods profile-neutral.

- [ ] **Step 5: Make builtin and MCP providers use the captured profile for session paths**

```python
def build_builtin_gate(service: Any | None, *, profile_id: str = "default") -> BuiltinToolGate: ...

def _profile_kwargs(self) -> dict[str, str]:
    profile_id = self._profile_id()
    return {} if profile_id == "default" else {"profile_id": profile_id}
```

Apply `_profile_kwargs()` to MCP provider session reads/writes as well as persistent writes. `BuiltinToolGate` stores one constructor profile id and passes it to `resolve_builtin_state`, `approve_for_session`, and `is_session_approved`.

- [ ] **Step 6: Fix Console local/Virtual CLI/raw-shell closures**

Capture `turn_context.tool_policy_profile_id` once. Pass it to every `gate_tool_test`, `gate_tool_test_for_profile`, `approve_for_session`, `is_session_approved`, and `set_tool_state` call. Build the builtin gate with that same id. Raw shell keeps its permanent Allow→Ask floor and runtime-owned temporary session authority; only its permission-store resolution becomes profile-aware.

- [ ] **Step 7: Run the provider and controller matrix**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_control_plane_permissions.py Tests/Agents/test_builtin_tool_gate.py Tests/Agents/test_local_tool_provider.py Tests/Agents/test_virtual_cli_provider.py Tests/Agents/test_raw_shell_tool_provider.py Tests/Chat/test_console_chat_controller.py -q`

Expected: PASS.

- [ ] **Step 8: Commit profile propagation**

```bash
git add tldw_chatbook/MCP/unified_control_plane_service.py tldw_chatbook/Agents/builtin_tool_gate.py tldw_chatbook/Agents/mcp_tool_provider.py tldw_chatbook/Chat/console_chat_controller.py Tests/MCP/test_control_plane_permissions.py Tests/Agents Tests/Chat/test_console_chat_controller.py
git commit -m "fix: keep tool approvals scoped to the active profile"
```

---
### Task 4: Portable contracts and canonical JSON

**Files:**
- Create: `tldw_chatbook/Tool_Packs/contracts.py`
- Create: `Tests/Tool_Packs/test_contracts.py`

**Interfaces:**
- Produces: `ToolPackError`, `PortableFallback`, `PortableToolRule`, `ToolProfilePayload`, `ToolPackManifest`, `ToolPackDocument`, `canonical_json_bytes()`, `strict_json_object()`, `portable_contract_sha256()`, and exact validators used by all later tasks.

- [ ] **Step 1: Write canonical JSON rejection tests**

```python
@pytest.mark.parametrize(
    ("raw", "category"),
    [
        (b'{"a":1,"a":2}', "payload_invalid"),
        (b'{"a":NaN}', "payload_invalid"),
        (b'\xff', "payload_invalid"),
        ('{"a":"\\ud800"}'.encode(), "payload_invalid"),
    ],
)
def test_strict_json_rejects_noncanonical_inputs(raw, category):
    with pytest.raises(ToolPackError) as error:
        strict_json_object(raw, category=category, max_bytes=1024)
    assert error.value.category == category
```

Add NFC identity, maximum depth/node, exact-key, string-byte, state, id grammar, and case-fold collision tests.

- [ ] **Step 2: Run the contract tests and verify the module is missing**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_contracts.py -q`

Expected: FAIL with `ModuleNotFoundError: tldw_chatbook.Tool_Packs.contracts`.

- [ ] **Step 3: Implement stable errors and immutable contract dataclasses**

```python
class ToolPackError(ValueError):
    def __init__(self, operation: str, category: str) -> None:
        self.operation = operation
        self.category = category
        super().__init__(f"tool_pack.{operation}.{category}")

@dataclass(frozen=True)
class PortableToolRule:
    authority: Literal["mcp", "builtin"]
    server_key: str
    tool_name: str
    state: Literal["allow", "ask", "deny"]
    contract_sha256: str | None
```

Define all manifest/profile fields explicitly. Reject unknown/missing fields rather than ignoring them.

- [ ] **Step 4: Implement canonical encoding and strict decoding**

```python
def canonical_json_bytes(value: object) -> bytes:
    normalized = _require_nfc_tree(value)
    return (json.dumps(normalized, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

def strict_json_object(data: bytes, *, category: str, max_bytes: int) -> dict[str, Any]: ...
```

Use `object_pairs_hook` to reject duplicate keys and `parse_constant` to reject non-finite numbers. Catch `UnicodeDecodeError`, `RecursionError`, and JSON errors and expose only the supplied stable category.

- [ ] **Step 5: Write portable fingerprint tests**

```python
base = _hub(name="search", tags=("network", "mutates"))
assert portable_contract_sha256(base) == portable_contract_sha256(_hub(tags=("mutates", "network", "network")))
assert portable_contract_sha256(base) != portable_contract_sha256(_hub(tags=("network",)))
assert portable_contract_sha256(base) != definition_hash(base.description, base.input_schema)
```

- [ ] **Step 6: Implement the portable fingerprint**

```python
def portable_contract_sha256(tool: HubTool, *, risk_tags: Iterable[str] | None = None) -> str:
    preimage = {
        "tool_name": tool.name,
        "description": _lf_nfc(tool.description),
        "input_schema": tool.input_schema,
        "policy_risk_tags": sorted(set(risk_tags if risk_tags is not None else tool.tags)),
    }
    return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()
```

- [ ] **Step 7: Run contract tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_contracts.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the portable contract**

```bash
git add tldw_chatbook/Tool_Packs/contracts.py Tests/Tool_Packs/test_contracts.py
git commit -m "feat: define canonical tool pack contracts"
```

---

### Task 5: Durable bounded receipt store

**Files:**
- Create: `tldw_chatbook/Tool_Packs/receipt_store.py`
- Create: `Tests/Tool_Packs/test_receipt_store.py`

**Interfaces:**
- Consumes: canonical JSON and `ToolPackError` from Task 4.
- Produces: `ToolPackReceipt`, `VerifiedToolPackReceipt`, `ReceiptHandle`, `ReceiptReservation`, and `ToolPackReceiptStore`.

- [ ] **Step 1: Write capacity, privacy-mode, and crash-residue tests**

```python
def test_receipt_is_private_and_reserved_before_commit(tmp_path):
    store = ToolPackReceiptStore(tmp_path, max_receipt_bytes=4096, max_total_bytes=8192)
    with store.reserve(1024) as reservation:
        handle = reservation.commit(_receipt_bytes())
    assert stat.S_IMODE(handle.path.stat().st_mode) == 0o600
    assert store.read(handle.receipt_id, expected_digest=handle.digest).digest == handle.digest
```

Inject failure before temp replace, after replace, and before authority linking; assert idempotent release, no truncation, and only authenticated receipt filenames are touched.

- [ ] **Step 2: Run receipt tests and verify the missing implementation**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_receipt_store.py -q`

Expected: FAIL because `ToolPackReceiptStore` is undefined.

- [ ] **Step 3: Implement exact receipt and reservation types**

```python
@dataclass(frozen=True)
class ReceiptHandle:
    receipt_id: str
    digest: str
    path: Path
    size: int

class ReceiptReservation:
    def commit(self, data: bytes) -> ReceiptHandle: ...
    def release(self) -> None: ...

class ToolPackReceiptStore:
    def reserve(self, projected_bytes: int) -> ReceiptReservation: ...
    def read(self, receipt_id: str, *, expected_digest: str) -> VerifiedToolPackReceipt: ...
    def reconcile_orphans(self, referenced_ids: AbstractSet[str], live_ids: AbstractSet[str], *, now: datetime) -> tuple[str, ...]: ...
    def write_compact_tombstone(self, source: ReceiptHandle, *, profile_id: str) -> ReceiptHandle: ...
```

Define a strict local receipt union: import receipts carry schema/kind, destination profile id, pack/archive digests, producer, import time, reviewed mappings, and the safe identities in matched/changed/missing/pending-Deny/omitted groups; compact tombstone receipts carry schema/kind, profile id, preserved pack digest, removal time, and prior receipt digest. Neither variant stores tool descriptions/schemas, configuration, secrets, workspace/Persona data, or authority. Reserve capacity under one process lock. Commit canonical bytes through a private mode-`0700` directory and new mode-`0600` temp, file fsync, atomic replace, and parent fsync. Use random `tp-<32 lowercase hex>` ids, retry authenticated-name collisions, and require the lifecycle's exact digest when reading.

- [ ] **Step 4: Write reconciliation and compaction tests**

```python
removed = store.reconcile_orphans({"tp-linked"}, {"tp-live"}, now=after_grace)
assert "tp-old-orphan" in removed
assert store.exists("tp-linked") and store.exists("tp-live")

compact = store.write_compact_tombstone(source_handle, profile_id="research")
assert compact.size < source_handle.size
```

Pin the grace interval to 24 hours, skip unknown/nonregular/symlink entries, and preserve a corrupt but referenced receipt for explicit recovery.

- [ ] **Step 5: Run receipt-store tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_receipt_store.py -q`

Expected: PASS.

- [ ] **Step 6: Commit durable receipts**

```bash
git add tldw_chatbook/Tool_Packs/receipt_store.py Tests/Tool_Packs/test_receipt_store.py
git commit -m "feat: add durable bounded tool pack receipts"
```

---

### Task 6: Complete provider inventory, flattening, and deterministic export

**Files:**
- Create: `tldw_chatbook/Tool_Packs/catalog_snapshot.py`
- Create: `tldw_chatbook/Tool_Packs/export.py`
- Create: `Tests/Tool_Packs/test_catalog_snapshot.py`
- Create: `Tests/Tool_Packs/test_export.py`
- Create: `Tests/Tool_Packs/fixtures/minimal-tool-pack.sha256`
- Create: `Tests/Tool_Packs/fixtures/minimal-tool-pack.bytes`

**Interfaces:**
- Consumes: strict permission snapshots (Task 1), provider profile correctness (Task 3), and contract types/fingerprints (Task 4).
- Produces: `PermissionInventoryRegistry`, `PermissionInventorySnapshot`, `ToolPackExportReview`, `ToolPackExportSnapshot`, `ToolPackExportService.capture()`, and `write_tool_pack_archive()`.

- [ ] **Step 1: Write the portability-registry tripwire test**

```python
def test_unclassified_permission_namespace_blocks_export():
    registry = PermissionInventoryRegistry(current_permission_namespaces=lambda: {"agent:builtin", "local:new"})
    registry.register(_builtin_adapter())
    with pytest.raises(ToolPackError, match="inventory_incomplete"):
        registry.capture()
```

Also assert excluded display-only server-source, skills, managed-skill approvals, runtime orchestration, and Library capability tools appear as category counts in review.

- [ ] **Step 2: Implement immutable inventory types and registry**

```python
@dataclass(frozen=True)
class PermissionInventoryTool:
    authority: Literal["mcp", "builtin"]
    tool: HubTool
    contract_sha256: str

@dataclass(frozen=True)
class PermissionInventorySnapshot:
    tools: tuple[PermissionInventoryTool, ...]
    excluded_counts: tuple[tuple[str, int], ...]
    digest: str

class PermissionInventoryAdapter(Protocol):
    namespace: str
    def snapshot(self) -> tuple[HubTool, ...]: ...
```

Reject duplicate exact/case-folded identities and incomplete adapters. Sort by `(authority, server_key, tool_name)`.

- [ ] **Step 3: Add concrete provider adapter tests**

Cover `agent:builtin`, `builtin:tldw_chatbook`, `local:__local__` including `RawShellToolProvider.hub_tool()`, `local:__virtual_cli__`, and each `local:<external-profile>`. Build local/Virtual CLI inventory using configured fallback root/app cwd and `admitted_roots=None`; assert no root path or alias occurs in review/archive bytes.

- [ ] **Step 4: Write flattening tests**

```python
review = exporter.capture(profile_id="research", display_name="Research", suggested_id="research")
assert review.payload.fallbacks[0].state in {"ask", "deny"}
assert _rule(review, "local:docs", "search").state == "allow"
assert _rule(review, "local:__local__", "shell_exec").state == "ask"
assert review.omitted_allow_ask == (("local:missing", "gone"),)
```

Pin named inheritance, builtin fallback, high-risk floor, raw-shell floor, `config_changed`, pending Deny, and omission of definitionless Ask/Allow.
Reject invalid-lifecycle profiles and tombstones. For `default` or a `ws-` source, require/derive a nonreserved suggestion that contains no workspace id. Re-export an imported profile from its current strict policy only; receipt omissions and historical rules must not reappear.

- [ ] **Step 5: Implement one-snapshot flattening**

```python
class ToolPackExportService:
    def capture(self, *, profile_id: str, display_name: str, suggested_id: str) -> ToolPackExportSnapshot:
        store = self._permission_store.read_snapshot_strict()
        inventory = self._inventory.capture()
        return _flatten(store, inventory, profile_id, display_name, suggested_id)
```

Load each authority once, use the provider's pure resolver, clamp unseen fallback Allow to Ask, and omit lifecycle/receipt/store timestamps and runtime gates.

- [ ] **Step 6: Write deterministic ZIP golden tests**

```python
first = archive_bytes(_minimal_snapshot())
second = archive_bytes(_minimal_snapshot())
assert first == second
assert hashlib.sha256(first).hexdigest() == fixture_digest
assert zip_header_projection(first) == EXPECTED_PINNED_HEADERS
```

Assert member order, `ZIP_STORED`, fixed timestamp, create/extract version 20, POSIX system, `0644` regular-file mode, zero flags/attrs/comments/extras, no data descriptor, exact canonical payload hash, and no forbidden privacy keys.

- [ ] **Step 7: Implement the canonical archive writer**

```python
def write_tool_pack_archive(snapshot: ToolPackExportSnapshot, sink: BinaryIO) -> str:
    """Write the two canonical members and return the archive SHA-256."""
```

Construct `ZipInfo` explicitly for both ASCII member names and never inherit host/library defaults. Generate the manifest after canonical payload bytes, then stream the final archive to `sink` without export timestamps.

- [ ] **Step 8: Run inventory/export tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_catalog_snapshot.py Tests/Tool_Packs/test_export.py -q`

Expected: PASS.

- [ ] **Step 9: Commit inventory and export**

```bash
git add tldw_chatbook/Tool_Packs/catalog_snapshot.py tldw_chatbook/Tool_Packs/export.py Tests/Tool_Packs
git commit -m "feat: export deterministic tool policy packs"
```

---

### Task 7: Captured-destination safe publication

**Files:**
- Create: `tldw_chatbook/Tool_Packs/publication.py`
- Create: `Tests/Tool_Packs/test_publication.py`

**Interfaces:**
- Consumes: `ToolPackExportSnapshot` and `write_tool_pack_archive()` from Task 6.
- Produces: `CapturedToolPackDestination`, `ToolPackPublicationResult`, and `publish_tool_pack()`.

- [ ] **Step 1: Write destination-race and unsupported-host tests**

```python
captured = CapturedToolPackDestination.capture(tmp_path / "research.tldw-tool-pack")
(tmp_path / "research.tldw-tool-pack").write_bytes(b"appeared")
with pytest.raises(ToolPackError, match="destination_changed"):
    publish_tool_pack(snapshot, captured, overwrite=False)

with pytest.raises(ToolPackError, match="publication_unsupported"):
    publish_tool_pack(snapshot, captured, primitives=_without_nofollow_replace())
```

Cover symlink/nonregular target, parent substitution, exact overwrite token, cancellation, and pre-replace cleanup touching only the authenticated private temp.

- [ ] **Step 2: Run publication tests and verify the missing seam**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_publication.py -q`

Expected: FAIL because `CapturedToolPackDestination` is undefined.

- [ ] **Step 3: Implement captured identity and atomic publication**

```python
@dataclass(frozen=True)
class CapturedToolPackDestination:
    path: Path
    parent_identity: tuple[int, int]
    target_identity: tuple[int, int] | None

@dataclass(frozen=True)
class ToolPackPublicationResult:
    archive_sha256: str
    committed: bool
    durability_uncertain: bool
```

Capture parent descriptor identity and target identity at picker acceptance. Write the complete archive to a mode-`0600` same-parent temp, flush/fsync, revalidate, perform the supported no-follow atomic replace, and fsync the parent.

- [ ] **Step 4: Add post-replace reconciliation tests**

Inject parent-fsync failure after `os.replace`. If destination identity and archive digest match, return `committed=True, durability_uncertain=True`; if state is neither exact new nor exact old, raise `tool_pack.export.durability_uncertain`. Never report the old file as preserved after a possible commit.

- [ ] **Step 5: Run publication tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_publication.py -q`

Expected: PASS.

- [ ] **Step 6: Commit publication safety**

```bash
git add tldw_chatbook/Tool_Packs/publication.py Tests/Tool_Packs/test_publication.py
git commit -m "feat: publish tool packs atomically"
```

---

### Task 8: Side-effect-free import review and exact/manual mapping

**Files:**
- Create: `tldw_chatbook/Tool_Packs/importer.py`
- Create: `Tests/Tool_Packs/test_importer.py`
- Create: `Tests/Tool_Packs/test_import_safety.py`

**Interfaces:**
- Consumes: strict contracts (Task 4), destination inventory (Task 6), and `MCPPermissionStore.read_snapshot_strict()` (Task 1).
- Produces: `ServerMapping`, `MappedToolRule`, `ToolPackImportReview`, and `ToolPackImportService.inspect_archive()`.

- [ ] **Step 1: Write hostile-archive and byte-preservation tests**

```python
before = permission_path.read_bytes()
with pytest.raises(ToolPackError) as error:
    importer.inspect_archive(_archive_with_duplicate_member(), destination_id="research")
assert error.value.category == "archive_invalid"
assert permission_path.read_bytes() == before
assert not list(permission_path.parent.glob("*.bak"))
```

Cover traversal/absolute/backslash/NUL/dot-segment/Windows-device names, exact and case-folded duplicate or extra members, symlinks/hard-links/nonregular members, encrypted/data-descriptor/compressed/nested entries, duplicate JSON keys, digest mismatch, member/archive/depth/node bounds, corrupt live store, and unknown store schema. Monkeypatch `MCPPermissionStore.load` to raise so inspection proves it never enters the recovery path.

- [ ] **Step 2: Run the importer tests and confirm the missing module**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_importer.py Tests/Tool_Packs/test_import_safety.py -q`

Expected: FAIL with `ModuleNotFoundError: tldw_chatbook.Tool_Packs.importer`.

- [ ] **Step 3: Implement bounded in-memory ZIP admission**

```python
@dataclass(frozen=True)
class ServerMapping:
    source_server_key: str
    destination_server_key: str

@dataclass(frozen=True)
class ToolPackImportReview:
    archive_path: Path
    archive_sha256: str
    manifest_sha256: str
    payload_sha256: str
    destination_id: str
    store_generation: str
    inventory_digest: str
    mappings: tuple[ServerMapping, ...]
    expires_at: datetime
    matched: tuple[MappedToolRule, ...]
    changed: tuple[PortableToolRule, ...]
    missing: tuple[PortableToolRule, ...]
    pending_denies: tuple[PortableToolRule, ...]
    omitted_allow_ask: tuple[PortableToolRule, ...]
```

Read the regular archive file through a no-follow descriptor, hash its bounded bytes, and parse without extraction. Accept exactly the pinned two `ZIP_STORED` members and headers from spec §1.1; validate manifest before payload and verify its exact payload digest.

- [ ] **Step 4: Write identity, collision, and mapping tests**

```python
review = importer.inspect_archive(
    pack,
    destination_id="research",
    mappings=(ServerMapping("source:mcp", "local:mcp"),),
)
assert review.matched[0].destination_identity == ("mcp", "local:mcp", "search")
assert review.pending_denies[0].state == "deny"
assert all(rule.state == "deny" for rule in review.pending_denies)
```

Assert automatic matching needs exact authority/server/raw name/portable hash; a risk-tag-only change is `changed`; labels and projected names are ignored; mappings are external-MCP-only, capped at 256, and one-to-one; duplicate destination identities fail; validated disconnected cache can match; reserved/invalid destination ids, exact/case-fold profile collisions, or any active/archived dangling reference fail.

- [ ] **Step 5: Implement immutable 15-minute reviews**

```python
class ToolPackImportService:
    def inspect_archive(
        self,
        archive_path: Path,
        *,
        destination_id: str,
        mappings: Sequence[ServerMapping] = (),
    ) -> ToolPackImportReview: ...
```

Capture the destination inventory and strict store exactly once, normalize the requested id without silently suffixing it, and classify exact matches, changed contracts, missing tools, pending Denies, and omitted Ask/Allow. The review object is process-local evidence only and contains no mutation callback.

- [ ] **Step 6: Run importer tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_importer.py Tests/Tool_Packs/test_import_safety.py -q`

Expected: PASS.

- [ ] **Step 7: Commit import inspection**

```bash
git add tldw_chatbook/Tool_Packs/importer.py Tests/Tool_Packs/test_importer.py Tests/Tool_Packs/test_import_safety.py
git commit -m "feat: inspect tool packs without side effects"
```

---

### Task 9: Safe unbound activation and strict outcome reconciliation

**Files:**
- Create: `tldw_chatbook/Tool_Packs/activation.py`
- Create: `Tests/Tool_Packs/test_activation.py`
- Modify: `Tests/MCP/test_permission_resolution.py`
- Modify: `tldw_chatbook/Tool_Packs/receipt_store.py`
- Modify: `Tests/Tool_Packs/test_receipt_store.py`

**Interfaces:**
- Consumes: reviews (Task 8), lifecycle coordinator/store operations (Task 2), receipts (Task 5), and inventory (Task 6).
- Produces: `InstalledToolProfile`, `ToolPackActivationResult`, `compile_imported_profile()`, and `ToolPackActivationService.install()`.

- [ ] **Step 1: Correct empty reviewed-mapping receipt admission, then write safe-compilation tests**

An exact automatic match may require no manual server mapping, and Task 8's review
categories intentionally overlap one diagnostic dimension (`changed`/`missing`) with
one action dimension (`pending_deny`/`omitted`). Add failing receipt regressions for
`reviewed_mappings=[]` and these exact-identity overlaps, then admit them while
rejecting case-fold aliases and incompatible overlaps (`matched` with anything,
`changed` with `missing`, or `pending_deny` with `omitted`). Apply the 2,000-tool cap
to distinct identities and retain the existing mapping maximum, sort, per-group
uniqueness, strict-field, and privacy checks. Run the focused receipt-store tests
before continuing.

```python
compiled = compile_imported_profile(review, destination_inventory)
assert compiled["global"] in {"ask", "deny"}
assert compiled["builtins"]["agent:builtin"] in {"ask", "deny"}
assert all(server["default"] in {"ask", "deny"} for server in compiled["servers"].values())
assert compiled["servers"]["local:docs"]["tools"]["search"]["definition_hash"] == definition_hash(
    destination_tool.description,
    destination_tool.input_schema,
)
assert portable_hash not in json.dumps(compiled)
```

Cover matched Allow/Ask, omitted changed/missing Allow/Ask, retained unresolved Deny, unmapped source server fallback behavior, fallback-equal exception elision, hash-free code-owned namespaces, and explicit unseen-server protection.

- [ ] **Step 2: Run activation tests and confirm the missing module**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_activation.py -q`

Expected: FAIL with `ModuleNotFoundError: tldw_chatbook.Tool_Packs.activation`.

- [ ] **Step 3: Implement the activation compiler and lifecycle sentinel**

```python
@dataclass(frozen=True)
class InstalledToolProfile:
    profile_id: str
    policy_digest: str
    revision: int
    receipt_id: str

def compile_imported_profile(review: ToolPackImportReview, inventory: PermissionInventorySnapshot) -> dict[str, Any]: ...
```

Write `profile_kind="tool_pack_imported"` and a complete `tldw.tool-pack-lifecycle/v1` object with revision `1`, authoritative first-bind marker, canonical policy digest, receipt link/digest, and compact counts. Do not place unresolved identity details in the hot permission file.

- [ ] **Step 4: Write stale-review, ordering, and ambiguity tests**

```python
with lifecycle.events() as events:
    result = activation.install(review)
assert events == ["receipt_durable", "coordinator", "store_fence", "reference_check", "install"]
assert result.installed.profile_id == review.destination_id
assert workspace_registry.references_profile(review.destination_id, include_archived=True) is False
```

Change the archive, mappings, inventory, store generation, destination id availability, reference set, review expiry, profile/byte caps, and receipt capacity between review and commit. Each must fail stale/admission before authority write. Inject failure before replace, after replace, and during strict reconciliation; assert exact installed state succeeds idempotently, absence is failure, and a third state is `activation_uncertain` without retry.

- [ ] **Step 5: Implement receipt-first, locked, install-if-absent activation**

```python
class ToolPackActivationService:
    def install(self, review: ToolPackImportReview) -> ToolPackActivationResult:
        """Revalidate and install exactly the reviewed id as an unbound profile."""
```

Re-read and revalidate the archive, capture fresh inventory, then reserve and durably write the detailed receipt while marking it live-owned by this commit. Next enter lifecycle coordinator → store fence; under both, refresh strict store state and active/archived references, compile exact destination definitions, and call `install_profile_if_absent`. On a stale/failing commit, release live ownership and leave the receipt to orphan-grace reconciliation. Reconcile with strict bytes on any ambiguous authority exception; never overwrite, suffix, bind, or mutate a workspace.

- [ ] **Step 6: Prove import alone is unbound**

Add an integration test with two existing workspaces, including one using `default`; snapshot their assistant defaults and effective policy before install and assert both remain byte/effectively identical afterward. Assert the installed id has zero references and `first_bind_confirmation_required=True`.

- [ ] **Step 7: Run activation and resolver tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_activation.py Tests/MCP/test_permission_resolution.py -q`

Expected: PASS.

- [ ] **Step 8: Commit safe activation**

```bash
git add tldw_chatbook/Tool_Packs/activation.py Tests/Tool_Packs/test_activation.py Tests/MCP/test_permission_resolution.py
git commit -m "feat: install tool packs as unbound safe profiles"
```

---

### Task 10: First-bind guard at the workspace authority boundary

**Files:**
- Modify: `tldw_chatbook/Tool_Packs/binding.py`
- Modify: `tldw_chatbook/Workspaces/registry_service.py`
- Modify: `tldw_chatbook/Workspaces/agent_provisioning.py`
- Create: `Tests/Tool_Packs/test_binding_guard.py`
- Modify: `Tests/Workspaces/test_workspace_assistant_defaults.py`
- Modify: `Tests/Workspaces/test_agent_provisioning.py`

**Interfaces:**
- Consumes: lifecycle/store coordination (Task 2) and authoritative imported lifecycle (Task 9).
- Produces: dependency-inverted `WorkspaceToolProfileGuard`, `ToolProfileBindingReview`, one-use confirmation tokens, and registry-wide guarded assistant-default mutations.

- [ ] **Step 1: Write direct-service bypass and race tests**

```python
with pytest.raises(ToolProfileConfirmationRequired):
    registry.set_assistant_defaults("w-1", _defaults("research"))

review = guard.review("w-1", _defaults("research"), action="set")
token = guard.confirm(review)
registry.set_assistant_defaults("w-1", _defaults("research"), tool_profile_confirmation_token=token)
assert registry.get_workspace("w-1").assistant_defaults.tool_policy_profile_id == "research"
```

Repeat confirmation-bypass tests for inline `create_workspace` and replacement with an imported profile. Assert clear, provisioning, and backfill traverse lifecycle serialization but do not require a Tool-Pack token unless their intended non-null profile is imported. Mutate policy after review, replay/expire token, change Persona/memory/full defaults/action/workspace, and race removal/binding; all confirmation attempts must fail closed. Existing local and `ws-` profiles must retain behavior while traversing the guard.

- [ ] **Step 2: Run binding tests and verify the bypass exists**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_binding_guard.py Tests/Workspaces/test_workspace_assistant_defaults.py Tests/Workspaces/test_agent_provisioning.py -q`

Expected: FAIL because the registry accepts imported defaults without a Tool-profile token.

- [ ] **Step 3: Define the registry-side protocol without importing `Tool_Packs`**

```python
class WorkspaceToolProfileGuard(Protocol):
    @contextmanager
    def mutation_scope(
        self,
        *,
        action: str,
        workspace_id: str,
        current_defaults: WorkspaceAssistantDefaults | None,
        intended_defaults: WorkspaceAssistantDefaults | None,
        confirmation_token: str | None,
    ) -> Iterator[None]: ...
```

Add `attach_tool_profile_guard()` to `LocalWorkspaceRegistryService`. Route every assistant-default transaction—including `create_workspace(..., assistant_defaults=...)`, `set_assistant_defaults`, `clear_assistant_defaults`, provisioning, and backfill—through `mutation_scope`. Keep `confirm_read_write` as a separate argument/check.

- [ ] **Step 4: Implement immutable review and token types**

```python
@dataclass(frozen=True)
class ToolProfileBindingReview:
    workspace_id: str
    action: Literal["create", "set", "replace"]
    intended_defaults_digest: str
    profile_id: str
    policy_digest: str
    revision: int
    expires_at: datetime
    summary: ToolProfileBindingSummary

class ToolProfileBindingGuard:
    def review(self, workspace_id: str, intended_defaults: WorkspaceAssistantDefaults, *, action: str) -> ToolProfileBindingReview: ...
    def confirm(self, review: ToolProfileBindingReview) -> str: ...
```

The summary recomputes current global/builtin/server fallback posture, stored exact Allows, effective Allows, unavailable/rug-pull-downgraded Allows, and high-risk status from strict authority plus current inventory; it never trusts the receipt.

- [ ] **Step 5: Implement one-use commit fencing and reconciliation**

Issue random process-local tokens with a 10-minute TTL. `mutation_scope` acquires coordinator → store fence, strictly revalidates the lifecycle marker, policy digest/revision, action, workspace, full intended-default digest, and token, consumes it, and holds both outer locks through the registry SQLite commit. After a known-successful bind, clear the first-bind marker with expected revision while still fenced. On uncertain DB commit, reread exact defaults and return `binding_uncertain`; never clear the marker without confirmed binding.

- [ ] **Step 6: Verify independent memory acknowledgement**

Add four cases: neither acknowledgement, memory only, Tool-profile only, and both. Only both may bind a read-write memory configuration to a first-bind imported profile.

- [ ] **Step 7: Run workspace/binding tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_binding_guard.py Tests/Workspaces/test_workspace_assistant_defaults.py Tests/Workspaces/test_agent_provisioning.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the authority-boundary guard**

```bash
git add tldw_chatbook/Tool_Packs/binding.py tldw_chatbook/Workspaces/registry_service.py tldw_chatbook/Workspaces/agent_provisioning.py Tests/Tool_Packs/test_binding_guard.py Tests/Workspaces/test_workspace_assistant_defaults.py Tests/Workspaces/test_agent_provisioning.py
git commit -m "feat: confirm the first imported tool profile binding"
```

---

### Task 11: Runtime leases, deny tombstones, and bounded removal

**Files:**
- Modify: `tldw_chatbook/Tool_Packs/binding.py`
- Create: `tldw_chatbook/Tool_Packs/removal.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`
- Create: `Tests/Tool_Packs/test_removal.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: coordinator/leases (Task 2), receipts (Task 5), and workspace reference authority (Task 10).
- Produces: `ToolProfileLease`, `ToolProfileRemovalService.remove()`, permanent tombstones, and Console/Test Tool runtime-use fencing.

- [ ] **Step 1: Write removal-eligibility and tombstone tests**

```python
@pytest.mark.parametrize("profile_id", ["default", "ws-w-1", "legacy", "invalid", "already-removed"])
def test_nonimported_profile_is_not_removable(profile_id, remover):
    with pytest.raises(ToolPackError, match="profile_not_removable"):
        remover.remove(profile_id)

result = remover.remove("research", expected_revision=3)
assert result.tombstone.profile_kind == "tool_pack_tombstone"
assert resolve_effective_state_by_key(store.load(), "any:new", "future", profile_id="research").state == "deny"
```

Assert the tombstone has MCP-global Deny, explicit `agent:builtin` Deny, no Allow/Ask entry, a new compact receipt link, preserved provenance, incremented revision/policy digest, hidden/reserved id, and permanent cap accounting.

- [ ] **Step 2: Write reference and lease race tests**

Block removal for active and archived references, dangling references, and leases. Race bind vs removal and lease acquisition vs removal under the coordinator: exactly one side may win, and the loser sees a current reference/tombstone. Prove separate profile ids do not interfere.

- [ ] **Step 3: Implement exact-profile runtime leases**

```python
@dataclass(frozen=True)
class ToolProfileLease:
    profile_id: str
    lease_id: str

@contextmanager
def lease(self, profile_id: str) -> Iterator[ToolProfileLease]: ...
```

The Console captures its selected profile once and holds the lease from run admission until the last governed invocation finishes. MCP Test Tool holds it from captured-profile gate through final result/error. Release in `finally`; do not claim revocation after dispatch.

- [ ] **Step 4: Implement compact-receipt-first removal**

```python
class ToolProfileRemovalService:
    def remove(self, profile_id: str, *, expected_revision: int) -> ToolProfileRemovalResult: ...
```

Acquire coordinator → store fence, revalidate exact imported lifecycle/revision, zero leases, and zero active/archived references. Reserve and durably stage a compact receipt under a new id, then call `replace_profile_with_tombstone`. Strictly reconcile post-replace: exact tombstone succeeds, exact prior state fails, anything else returns `outcome_uncertain` and retains both receipts. The old receipt becomes only an orphan-grace candidate after confirmed replacement.

- [ ] **Step 5: Run removal and runtime tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_removal.py Tests/Chat/test_console_chat_controller.py Tests/UI/test_mcp_workbench.py Tests/MCP/test_permission_resolution.py -q`

Expected: PASS.

- [ ] **Step 6: Commit fail-closed removal**

```bash
git add tldw_chatbook/Tool_Packs/binding.py tldw_chatbook/Tool_Packs/removal.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/UI/MCP_Modules/mcp_workbench.py Tests/Tool_Packs/test_removal.py Tests/Chat/test_console_chat_controller.py Tests/UI/test_mcp_workbench.py Tests/MCP/test_permission_resolution.py
git commit -m "feat: replace removed tool profiles with deny tombstones"
```

---

### Task 12: App-facing service, deferred wiring, and receipt reconciliation

**Files:**
- Modify: `tldw_chatbook/Tool_Packs/__init__.py`
- Create: `tldw_chatbook/Tool_Packs/service.py`
- Modify: `tldw_chatbook/app.py`
- Create: `Tests/Tool_Packs/test_service.py`
- Modify: `Tests/App/test_startup_init_hygiene.py`
- Modify: `Tests/Performance/test_app_startup_performance.py`

**Interfaces:**
- Consumes: export/publication/import/activation/binding/removal services (Tasks 6–11).
- Produces: `ToolPackService`, stable presentation outcomes, app-owned singleton composition, attached workspace guard, and bounded receipt recovery.

- [ ] **Step 1: Write orchestration and import-boundary tests**

```python
def test_tool_pack_service_exposes_review_then_explicit_commit(service, pack):
    review = service.inspect_import(pack, destination_id="research")
    assert review.unbound is True
    assert service.list_profiles().by_id("research") is None
    result = service.import_unbound(review)
    assert result.installed.profile_id == "research"
    assert service.list_profiles().by_id("research").reference_counts == (0, 0)
```

Assert the service exposes separate `capture_export`/`publish_export`, `inspect_import`/`import_unbound`, `review_first_bind`/`confirm_first_bind`, and `remove_profile` operations. Verify stable user-facing categories contain no exception text, paths, credentials, commands, environment values, or archive payload excerpts.

- [ ] **Step 2: Run service tests and confirm the missing facade**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_service.py -q`

Expected: FAIL with `ModuleNotFoundError: tldw_chatbook.Tool_Packs.service`.

- [ ] **Step 3: Implement the narrow application facade**

```python
class ToolPackService:
    def list_profiles(self) -> ToolProfileListing: ...
    def capture_export(self, profile_id: str, *, display_name: str, suggested_id: str) -> ToolPackExportSnapshot: ...
    def publish_export(self, snapshot: ToolPackExportSnapshot, destination: CapturedToolPackDestination, *, overwrite_token: str | None = None) -> ToolPackPublicationResult: ...
    def inspect_import(self, archive_path: Path, *, destination_id: str, mappings: Sequence[ServerMapping] = ()) -> ToolPackImportReview: ...
    def import_unbound(self, review: ToolPackImportReview) -> ToolPackActivationResult: ...
    def remove_profile(self, profile_id: str, *, expected_revision: int) -> ToolProfileRemovalResult: ...
```

Return immutable presentation models for origin, lifecycle validity, binding state, active/archived reference counts, compact counts, receipt health, and removal eligibility. Keep policy mutations out of this facade; Settings must deep-link to MCP Permissions for editing.

- [ ] **Step 4: Write app composition and reconciliation tests**

Assert `app.py` creates the service only after the unified MCP service and workspace registry exist, attaches exactly one binding guard, points receipts under the user data directory, and calls bounded reconciliation only after startup-critical composition. Feed referenced, live-owned, fresh orphan, expired orphan, symlink, and corrupt receipt entries; only the expired unreferenced authenticated regular file may be removed.

- [ ] **Step 5: Wire deferred composition and recovery**

```python
def _wire_tool_pack_service(self) -> None:
    service = ToolPackService.compose(
        permission_store=self.unified_mcp_service.permission_store,
        tool_catalog=self.unified_mcp_service.tool_catalog,
        workspace_registry=self.workspace_registry_service,
        receipt_root=get_user_data_dir() / "tool_pack_receipts",
    )
    self.workspace_registry_service.attach_tool_profile_guard(service.binding_guard)
    self.tool_pack_service = service
```

Schedule receipt reconciliation as post-ready/first-use bounded work rather than adding filesystem traversal to import time. If required composition is unavailable, expose a stable unavailable state and do not attach a partial guard.

- [ ] **Step 6: Verify startup/import budgets**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_service.py Tests/App/test_startup_init_hygiene.py Tests/Performance/test_app_startup_performance.py -q`

Expected: PASS with no eager `Tool_Packs` import before the deferred wiring point and no startup-budget regression.

- [ ] **Step 7: Commit the application seam**

```bash
git add tldw_chatbook/Tool_Packs/__init__.py tldw_chatbook/Tool_Packs/service.py tldw_chatbook/app.py Tests/Tool_Packs/test_service.py Tests/App/test_startup_init_hygiene.py Tests/Performance/test_app_startup_performance.py
git commit -m "feat: wire portable tool policy services"
```

---

### Task 13: Captured Tool policy profile in MCP Permissions and Test Tool

**Files:**
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`
- Modify: `Tests/UI/test_mcp_permissions_mode.py`
- Modify: `Tests/UI/test_mcp_workbench.py`
- Modify: `Tests/MCP/test_control_plane_permissions.py`
- Modify: `Tests/MCP/test_control_plane_tool_execute.py`

**Interfaces:**
- Consumes: profile-aware control-plane calls (Task 3), lifecycle validity (Task 1), and runtime leases (Task 11).
- Produces: `PermissionProfileContext`, a local Tool policy profile selector, and profile-captured matrix/edit/re-allow/Test Tool events.

- [ ] **Step 1: Write selector rendering and stale-event tests**

```python
old = PermissionProfileContext("research", selector_generation=4, policy_digest="a" * 64, revision=2)
await workbench.select_tool_policy_profile("default")
await workbench.on_mcp_permissions_mode_state_cycle_requested(_cycle(context=old))
service.set_tool_state.assert_not_called()
assert "changed" in app.notifications[-1].message.lower()
```

Assert the selector lists default, valid local, imported, and workspace-managed profiles; hides tombstones; marks invalid lifecycle profiles as unavailable; distinguishes this local Tool policy selector from server-side governance profiles; and never retargets an event after selection/revision changes.

- [ ] **Step 2: Add captured context to the permissions canvas contract**

```python
@dataclass(frozen=True)
class PermissionProfileContext:
    profile_id: str
    selector_generation: int
    policy_digest: str
    revision: int | None

async def update_matrix(
    self,
    *,
    global_state: str,
    servers: Sequence[PermissionServerRow],
    profile_context: PermissionProfileContext,
) -> None: ...
```

Include `profile_context` in `StateCycleRequested` and `RowSelected`; include it in inspector jump/re-allow and Test Tool request models. Keep `KillSwitchToggled` profile-neutral because the kill switch remains global.

- [ ] **Step 3: Make every workbench action validate captured context**

Before read, mutation, re-allow, preview, persistent/session approval, or Test Tool execution, compare profile id, selector generation, strict policy digest, and imported revision with the current selection. Persistent mutations pass the captured digest/revision into the store's locked compare-and-set field mutator. Session approval uses a control-plane helper that enters the store fence, rechecks that digest/revision, and only then inserts the exact `(profile_id, server_key, tool_name)` tuple. Reject mismatches as stale and refresh; never substitute the current profile id into an old event.

- [ ] **Step 4: Write profile-specific edit and Test Tool tests**

```python
await workbench.select_tool_policy_profile("research")
await workbench.cycle_permission("local:docs", "search")
service.set_tool_state.assert_called_once_with(
    "local:docs",
    "search",
    expected_state="ask",
    new_state="deny",
    profile_id="research",
    expected_profile_digest=context.policy_digest,
    expected_revision=context.revision,
)
assert store.get_tool_entry("local:docs", "search", profile_id="default") is None
```

Repeat for global/server/builtin rows, re-allow definition hash, by-key Test Tool gate, Test Tool persistent/session approvals, session isolation, profile switch while confirmation is armed, and edit while Test Tool is in flight. Assert a Tool Test acquires the exact profile lease until final outcome.

- [ ] **Step 5: Implement profile selection and explicit service calls**

Read the selected profile through strict lifecycle-aware listing, render its own global/server/builtin matrix, and pass the captured id/digest/revision to every profile-aware control-plane method. Control-plane setters and session approvals expose those expected-value arguments and surface `stale_profile` from their under-fence comparison. Invalidate armed confirmation and child panels on selection generation change. Invalid/tombstoned profiles remain non-editable and non-testable.

- [ ] **Step 6: Run MCP Permissions UI/runtime tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_mcp_permissions_mode.py Tests/UI/test_mcp_workbench.py Tests/MCP/test_control_plane_permissions.py Tests/MCP/test_control_plane_tool_execute.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the captured profile editor**

```bash
git add tldw_chatbook/MCP/unified_control_plane_service.py tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py tldw_chatbook/UI/MCP_Modules/mcp_workbench.py Tests/UI/test_mcp_permissions_mode.py Tests/UI/test_mcp_workbench.py Tests/MCP/test_control_plane_permissions.py Tests/MCP/test_control_plane_tool_execute.py
git commit -m "feat: edit the selected tool policy profile"
```

---

### Task 14: Settings Tool Profiles panel and first-bind review UX

**Files:**
- Create: `tldw_chatbook/Widgets/Settings_Widgets/tool_profiles_panel.py`
- Create: `tldw_chatbook/Widgets/Settings_Widgets/tool_pack_import_review.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Create: `Tests/UI/test_settings_tool_profiles.py`
- Modify: `Tests/UI/test_settings_workspace_assistant_defaults.py`
- Modify: `Tests/UI/test_settings_category_sweep.py`
- Modify: `Tests/UI/test_settings_search_index.py`

**Interfaces:**
- Consumes: `ToolPackService` (Task 12), MCP profile deep-link (Task 13), and binding reviews/tokens (Task 10).
- Produces: canonical Settings profile management, import/export review workflows, removal confirmation, and current-state first-bind modal.

- [ ] **Step 1: Write panel ownership and profile-list tests**

```python
assert settings.query_one(ToolProfilesPanel).profile_ids == ("default", "research", "ws-w-1")
assert settings.query_one(ToolProfilesPanel).row("research").origin == "Imported Tool Pack"
assert settings.query_one(ToolProfilesPanel).row("research").reference_counts == (0, 0)
assert not settings.query_one(ToolProfilesPanel).has_policy_editor
```

Assert tombstones are hidden, invalid lifecycles are visible but disabled for bind/edit/export/remove, receipt-unavailable degrades provenance only, removal eligibility/reference counts are current, and deprecated settings surfaces remain untouched.

- [ ] **Step 2: Build a modular panel with explicit events**

```python
class ToolProfilesPanel(Vertical):
    class ImportRequested(Message): ...
    class ExportRequested(Message): ...
    class EditPolicyRequested(Message): ...
    class RemoveRequested(Message): ...
```

Keep file-picking and service orchestration in `settings_screen.py`; the panel renders immutable listing rows and never mutates authority. `Edit policy` opens/deep-links MCP Permissions with the exact profile selected.

- [ ] **Step 3: Write import review interaction tests**

The modal must show producer/content digest/proposed id, source fallback and counts, exact/changed/missing/pending-Deny/omitted counts, connected versus disconnected-cached destinations, explicit mappings, the unbound notice, and “does not install tools.” Assert only `Import unbound profile` commits; no import-and-bind control exists. Changing id/mapping re-runs inspection and invalidates the old review.

- [ ] **Step 4: Implement worker-backed import/export/removal flows**

Run inspection, activation, export capture/publication, and removal outside the event loop with exclusive per-operation workers. Capture export source profile and destination identity at review/picker acceptance. Show overwrite confirmation only for the exact captured destination token. Surface stable categories, cancellation, uncertain outcomes, and Windows `publication_unsupported` without claiming native Windows support.

- [ ] **Step 5: Write first-bind modal tests from current state**

```python
await settings.choose_workspace_tool_profile("research")
modal = settings.query_one(ToolProfileFirstBindModal)
assert modal.policy_digest == current_lifecycle.policy_digest
assert modal.global_fallback in {"ask", "deny"}
assert modal.stored_exact_allows
assert modal.unavailable_allows
assert modal.high_risk_allows
```

Assert modal review includes the current global/builtin/server Allow posture, every stored exact Allow, effective Allows, unavailable/rug-pull-downgraded rows, risk status, Persona/memory/full intended defaults, and expandable Ask/Deny detail. A profile edit, selection change, workspace change, modal timeout, or failed save invalidates the token. The read-write memory acknowledgement remains independently required.

- [ ] **Step 6: Route workspace apply through review/confirm/registry commit**

On `ToolProfileConfirmationRequired`, request a current binding review from the service, show it, exchange explicit confirmation for a token, and retry the exact registry call with both separate acknowledgement arguments. Never bind from the import modal. If bind succeeds but marker clear fails, show the saved binding plus a warning that future use may ask again.

- [ ] **Step 7: Verify navigation, search, and keybinding conventions**

Add Tool Profiles to the canonical Settings category/search index. Ensure action bindings do not use terminal-convention or globally reserved keys, footer hints advertise only implemented actions, narrow layouts remain usable, and all controls have focus/disabled states.

- [ ] **Step 8: Run Settings UI tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_settings_tool_profiles.py Tests/UI/test_settings_workspace_assistant_defaults.py Tests/UI/test_settings_category_sweep.py Tests/UI/test_settings_search_index.py Tests/UI/test_settings_narrow_layout.py Tests/UI/test_settings_footer_hints.py -q`

Expected: PASS.

- [ ] **Step 9: Commit canonical management UX**

```bash
git add tldw_chatbook/Widgets/Settings_Widgets/tool_profiles_panel.py tldw_chatbook/Widgets/Settings_Widgets/tool_pack_import_review.py tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_tool_profiles.py Tests/UI/test_settings_workspace_assistant_defaults.py Tests/UI/test_settings_category_sweep.py Tests/UI/test_settings_search_index.py
git commit -m "feat: manage portable tool profiles in settings"
```

---

### Task 15: Security, performance, documentation, and implementation closeout

**Files:**
- Create: `Tests/Tool_Packs/test_privacy_contract.py`
- Create: `Tests/Tool_Packs/test_limits_and_performance.py`
- Create: `Tests/Tool_Packs/test_architecture_boundaries.py`
- Modify: `Docs/Design/User_Settings.md`
- Modify: `Docs/Design/MCP.md`

**Interfaces:**
- Consumes: the complete V1 implementation.
- Produces: recursive privacy evidence, pinned bounds/performance evidence, ownership tests, user documentation, and Definition-of-Done records.

- [ ] **Step 1: Add recursive forbidden-data contract tests**

```python
FORBIDDEN_KEYS = {
    "command", "args", "env", "endpoint", "url", "credential", "secret",
    "api_key", "workspace_id", "persona_id", "session_grant", "approval_history",
    "description", "input_schema", "inputSchema",
}
assert not recursive_key_intersection(manifest_and_profile, FORBIDDEN_KEYS)
assert b"/Users/" not in archive_bytes
assert b"C:\\\\" not in archive_bytes
```

Use sentinels injected into server configuration, environment, workspace/Persona bindings, admitted-root aliases, receipts, session approvals, and display labels; assert none appears in review models, canonical archive bytes, logs, notifications, or stable errors. Assert executable/plugin/skill/runtime-install fields are rejected as unknown keys.

- [ ] **Step 2: Add maximum-bound and structural performance tests**

Build exactly 2,000 tools/256 servers/257 fallbacks and near-limit JSON/ZIP/store/receipt payloads; assert they succeed and record inspect/activation elapsed time plus peak memory as non-gating benchmark output. Add one-over cases for every bound and prove early deterministic failure. Instrument authority/inventory/reference calls so export/import review does one snapshot per source and never performs per-tool disk/network I/O.

- [ ] **Step 3: Add architecture and ownership tripwires**

Assert `Workspaces/registry_service.py` imports only its guard protocol, not `Tool_Packs`; Settings owns profile management but contains no direct permission-store writes; MCP Permissions owns rule edits but cannot import/bind; Actor Pack internals are not imported; and newly permission-addressable namespaces must register or explicitly exclude an inventory adapter.

- [ ] **Step 4: Document the user-visible contract**

Document deterministic policy-only packs, what is excluded, export review, manual exact server mapping, unbound import, first-bind confirmation, MCP Permissions editing, tombstone removal constraints, receipt degradation behavior, stable uncertain outcomes, and the separate Windows-publication limitation. State explicitly that Tools+Skills/plugin installation is outside V1 and requires a future schema/ADR/trust design.

- [ ] **Step 5: Run the complete targeted Tool Pack matrix**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs Tests/MCP/test_permission_store.py Tests/MCP/test_permission_resolution.py Tests/MCP/test_control_plane_permissions.py Tests/MCP/test_control_plane_tool_execute.py Tests/Agents/test_builtin_tool_gate.py Tests/Agents/test_local_tool_provider.py Tests/Agents/test_virtual_cli_provider.py Tests/Agents/test_raw_shell_tool_provider.py Tests/Chat/test_console_chat_controller.py Tests/Workspaces/test_workspace_assistant_defaults.py Tests/Workspaces/test_agent_provisioning.py Tests/UI/test_mcp_permissions_mode.py Tests/UI/test_mcp_workbench.py Tests/UI/test_settings_tool_profiles.py Tests/UI/test_settings_workspace_assistant_defaults.py -q
```

Expected: PASS. Do not run the full repository suite without the user's explicit opt-in.

- [ ] **Step 6: Run static and hygiene checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/Tool_Packs tldw_chatbook/MCP/permission_store.py tldw_chatbook/MCP/unified_control_plane_service.py tldw_chatbook/Agents/builtin_tool_gate.py tldw_chatbook/Agents/mcp_tool_provider.py tldw_chatbook/Workspaces/registry_service.py tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py tldw_chatbook/UI/MCP_Modules/mcp_workbench.py tldw_chatbook/Widgets/Settings_Widgets/tool_profiles_panel.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 7: Perform final code/security review**

Use `superpowers:requesting-code-review`; resolve actionable findings with `superpowers:receiving-code-review`. Re-run every affected targeted test. Confirm every permission resolution/mutation carries the captured profile, every review path uses strict reads, all authority writes respect lock order, every ambiguous replace reconciles exact state, and no V1 path installs executable content.

- [ ] **Step 8: Close the Backlog task and implementation branch**

Check every acceptance criterion, add concise implementation notes with ADR-107 and test evidence, document any genuine lesson in the appropriate `backlog/docs/lessons-*.md`, and set the task Done only after the repository Definition of Done is satisfied. Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch` to present integration options.

- [ ] **Step 9: Commit closeout evidence**

```bash
git add Tests/Tool_Packs Docs/Design/User_Settings.md Docs/Design/MCP.md backlog/tasks
git commit -m "docs: close portable tool pack v1"
```

---

## Spec coverage checklist

- Tasks 4, 6, and 7 implement spec §§1–2: canonical envelope, payload, hard bounds, complete inventory, flattening, and safe publication.
- Tasks 8 and 9 implement §3: side-effect-free review, exact/manual mapping, revalidation, safe compilation, receipt-first unbound installation, and uncertainty reconciliation.
- Tasks 1, 2, and 5 implement §4: strict reads, lifecycle authority, path-wide mutation fencing, caps, receipts, and fixed lock order.
- Tasks 10 and 11 implement §§5–6: first-bind token authority, independent memory acknowledgement, runtime leases, reference checks, and permanent Deny tombstones.
- Tasks 12–14 implement §§7–9: canonical surfaces, profile-captured actions, application ownership, stable outcomes, workers, and deferred composition.
- Task 15 implements §§10–11: privacy, contract/bound/performance/race/architecture verification, Windows claim boundaries, and user documentation.
- The non-goals and §13 boundary remain explicit: V1 never packages skills/plugins/tools or performs runtime installation; Windows publication is separate.
