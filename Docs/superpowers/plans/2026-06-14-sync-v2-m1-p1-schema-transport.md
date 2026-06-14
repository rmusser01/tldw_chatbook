# Sync v2 M1 — Phase 1: Schema + Transport Conformance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Chatbook Sync v2 client parse the live server's `capabilities` response and complete `/profile/bootstrap` + `/profile` against the M1 server, unblocking everything downstream.

**Architecture:** Conform only the *response-parsing* and *profile/bootstrap transport* in `sync_schemas.py` + `client.py`. Domain/operation/encryption-policy fields on the capability/profile models are typed loosely (`list[str]`/`dict`) so this phase does **not** touch the `SyncV2Envelope` literal vocabulary or the envelope builder/applier — those change atomically in P2. Every commit here stays green.

**Tech Stack:** Python 3.11+, Pydantic v2, httpx (via `TLDWAPIClient`), pytest. Run pytest only via the project venv: `.venv/bin/python -m pytest` (system python breaks test collection).

**Spec:** `Docs/superpowers/specs/2026-06-12-sync-v2-client-m1-conformance-design.md`
**ADR:** `backlog/decisions/008-sync-v2-client-m1-contract-alignment.md`
**Server contract:** `tldw_server2` `Docs/API/Sync_V2_M1.md` @ `992e89a03`

**Scope note / spec deviation (intentional):** The spec lists "persist server-assigned
`device_id`/`dataset_id`" under P1's exit. That persistence couples to
`sync_state_repository` (a 2198-line SQLite store) and belongs with the per-object mirror,
so it moves to **P2**. P1's exit is: client parses real server capabilities and
`bootstrap_sync_v2_profile` returns a parsed response exposing `device_id`/`dataset_id`,
verified live against `:8076` (not fail-closed on attestation).

---

## Pre-flight: start the M1 server (needed only for Task 6 live check)

The live server is the pinned `tldw_server2` worktree. To (re)launch it on `:8076`:

```bash
d=~/Documents/GitHub/tldw_server2/.worktrees/sync-v2-m1-next
PY=~/Documents/GitHub/tldw_server2/.venv/bin/python
cd "$d" && PYTHONPATH="$d" AUTH_MODE=single_user \
  SINGLE_USER_API_KEY='THIS-IS-A-SECURE-KEY-123-FAKE-KEY' TLDW_API_PORT=8076 \
  nohup "$PY" -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 8076 \
  > /tmp/syncqa-v2/server.log 2>&1 &
```

Wait for readiness:
```bash
curl -s -m 5 -H "X-API-KEY: THIS-IS-A-SECURE-KEY-123-FAKE-KEY" \
  http://127.0.0.1:8076/api/v1/sync/capabilities | head -c 80
```
Expected: a JSON body starting `{"protocol_version": "sync-v2-m1"...`.

---

## File Structure

- `tldw_chatbook/tldw_api/sync_schemas.py` — replace `SyncV2CapabilitiesResponse`; add `SyncV2Profile*` + bootstrap models. (No envelope/literal changes in P1.)
- `tldw_chatbook/tldw_api/client.py` — add `get_sync_v2_profile`, `bootstrap_sync_v2_profile`; import new models. (`get_sync_v2_capabilities` is unchanged — its parsing is fixed by the schema.)
- `tldw_chatbook/tldw_api/__init__.py` — export the new models.
- `Tests/tldw_api/test_sync_client.py` — update the protocol-version pin; add profile/bootstrap tests.
- `Tests/Sync_Interop/test_server_sync_service.py`, `Tests/MCP/test_unified_control_plane_service.py`, `Tests/MCP/test_local_control_service.py` — fix any capability-shape assertions broken by the rename.

---

## Task 1: Conform `SyncV2CapabilitiesResponse` to the server shape

**Files:**
- Modify: `tldw_chatbook/tldw_api/sync_schemas.py` (replace the `SyncV2CapabilitiesResponse` class, currently at lines ~162-178)
- Test: `Tests/tldw_api/test_sync_schemas_m1.py` (create)

- [ ] **Step 1: Write the failing test**

Create `Tests/tldw_api/test_sync_schemas_m1.py`:

```python
"""P1: client parses the live M1 server's capability/profile payloads."""

from tldw_chatbook.tldw_api import SyncV2CapabilitiesResponse

# Captured verbatim from the live codex/sync-v2-m1-next server @ 992e89a03.
LIVE_CAPABILITIES = {
    "protocol_version": "sync-v2-m1",
    "min_supported_protocol_version": "sync-v2-m1",
    "domains": [
        "notes.note", "chat.conversation", "chat.message", "attachment.ref",
        "workspaces.workspace", "workspaces.source_ref", "source_cache.entry",
        "media.item", "media.keyword", "media.keyword_link",
    ],
    "operations": {
        "notes.note": ["upsert", "tombstone"],
        "chat.conversation": ["upsert", "tombstone"],
        "chat.message": ["append", "tombstone"],
        "attachment.ref": ["upsert", "tombstone"],
    },
    "encryption": {"policy": "server_trusted_v1", "ready": True},
    "encryption_policies": ["server_trusted_v1"],
    "blob_transfer": {"supported": False},
    "max_batch_size": 100,
    "max_envelope_payload_bytes": 262144,
    "max_attachment_bytes": 1048576,
    "supports_restore_manifest": True,
    "supports_conflicts": True,
    "supports_attachments": False,
    "compatibility_flags": {},
    "quota": {},
    "server_time": "2026-06-14T00:00:00Z",
    "warnings": [],
}


def test_capabilities_parses_live_m1_payload():
    caps = SyncV2CapabilitiesResponse.model_validate(LIVE_CAPABILITIES)
    assert caps.protocol_version == "sync-v2-m1"
    assert caps.min_supported_protocol_version == "sync-v2-m1"
    assert "notes.note" in caps.domains
    assert caps.operations["chat.message"] == ["append", "tombstone"]
    assert caps.encryption_policies == ["server_trusted_v1"]
    assert caps.supports_attachments is False
    assert caps.encryption["policy"] == "server_trusted_v1"


def test_capabilities_back_compat_properties():
    caps = SyncV2CapabilitiesResponse.model_validate(LIVE_CAPABILITIES)
    # Legacy readers used .supported_domains / .supported_operations.
    assert "notes.note" in caps.supported_domains
    assert "append" in caps.supported_operations


def test_capabilities_coerces_legacy_int_protocol_version():
    caps = SyncV2CapabilitiesResponse.model_validate(
        {"protocol_version": 2, "min_supported_protocol_version": 2}
    )
    assert caps.protocol_version == "sync-v2-m1"
    assert caps.min_supported_protocol_version == "sync-v2-m1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/tldw_api/test_sync_schemas_m1.py -q`
Expected: FAIL — `test_capabilities_parses_live_m1_payload` raises `ValidationError`
("Input should be a valid integer" for `protocol_version`), and the back-compat test
errors with `AttributeError`/`ValidationError`.

- [ ] **Step 3: Replace the `SyncV2CapabilitiesResponse` class**

In `tldw_chatbook/tldw_api/sync_schemas.py`, replace the existing
`class SyncV2CapabilitiesResponse(BaseModel): ...` block (≈ lines 162-178) with:

```python
class SyncV2CapabilitiesResponse(BaseModel):
    """Server-supported Sync v2 protocol capabilities (M1 shape).

    Domain/operation/policy fields are typed loosely so the client can read whatever
    the server advertises without coupling to the envelope vocabulary (which is
    conformed in P2).
    """

    protocol_version: str = "sync-v2-m1"
    min_supported_protocol_version: str = "sync-v2-m1"
    domains: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("domains", "supported_domains"),
    )
    operations: dict[str, list[str]] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("operations", "supported_operations"),
    )
    encryption: dict[str, Any] = Field(default_factory=dict)
    encryption_policies: list[str] = Field(default_factory=list)
    blob_transfer: dict[str, Any] = Field(default_factory=dict)
    quota: dict[str, Any] = Field(default_factory=dict)
    max_batch_size: int = Field(100, ge=1)
    max_envelope_payload_bytes: int = Field(262_144, ge=1)
    max_attachment_bytes: int = Field(1_048_576, ge=1)
    supports_restore_manifest: bool = True
    supports_conflicts: bool = True
    supports_attachments: bool = False
    compatibility_flags: dict[str, bool] = Field(default_factory=dict)
    server_time: str | None = None
    warnings: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("protocol_version", "min_supported_protocol_version", mode="before")
    @classmethod
    def _coerce_protocol_version(cls, value: Any) -> str:
        if value in (None, 2, "2"):
            return "sync-v2-m1"
        return str(value)

    @property
    def supported_domains(self) -> list[str]:
        """Back-compat alias for pre-M1 readers."""
        return self.domains

    @property
    def supported_operations(self) -> list[str]:
        """Back-compat: flattened, de-duplicated operation names across all domains."""
        return sorted({op for ops in self.operations.values() for op in ops})

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/tldw_api/test_sync_schemas_m1.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/sync_schemas.py Tests/tldw_api/test_sync_schemas_m1.py
git commit -m "Sync v2 P1: conform SyncV2CapabilitiesResponse to M1 server shape"
```

---

## Task 2: Add profile + bootstrap models

**Files:**
- Modify: `tldw_chatbook/tldw_api/sync_schemas.py` (add new classes immediately after `SyncV2CapabilitiesResponse`)
- Test: `Tests/tldw_api/test_sync_schemas_m1.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `Tests/tldw_api/test_sync_schemas_m1.py`:

```python
from tldw_chatbook.tldw_api import (
    SyncV2ProfileBootstrapRequest,
    SyncV2ProfileBootstrapResponse,
    SyncV2ProfileResponse,
)

# Shape mirrors tldw_server2 Sync_V2_M1.md POST /profile/bootstrap response.
LIVE_BOOTSTRAP_RESPONSE = {
    "created": True,
    "profile_bootstrapped": True,
    "user_id": "user_123",
    "active_dataset_id": "ds_personal_01HZZ0",
    "device": {
        "device_id": "dev_chatbook_laptop",
        "registered": True,
        "client_profile_id": "chatbook_profile_main",
        "last_seen_at": "2026-06-14T00:00:00Z",
    },
    "dataset": {
        "dataset_id": "ds_personal_01HZZ0",
        "scope": "personal",
        "default_personal": True,
        "client_family": "chatbook",
        "domains": ["notes.note", "chat.conversation", "chat.message", "attachment.ref"],
    },
    "server_cursor": 0,
    "capabilities": LIVE_CAPABILITIES,
    "domain_status": [],
    "warnings": [],
}


def test_bootstrap_response_parses_and_exposes_identity():
    resp = SyncV2ProfileBootstrapResponse.model_validate(LIVE_BOOTSTRAP_RESPONSE)
    assert resp.created is True
    assert resp.profile_bootstrapped is True
    assert resp.device.device_id == "dev_chatbook_laptop"
    assert resp.dataset.dataset_id == "ds_personal_01HZZ0"
    assert resp.active_dataset_id == "ds_personal_01HZZ0"
    assert resp.capabilities.protocol_version == "sync-v2-m1"


def test_profile_response_handles_unbootstrapped():
    resp = SyncV2ProfileResponse.model_validate(
        {"profile_bootstrapped": False, "user_id": "user_123", "server_cursor": 0}
    )
    assert resp.profile_bootstrapped is False
    assert resp.dataset is None
    assert resp.device is None


def test_bootstrap_request_defaults_to_m1_domains_and_offline_mode():
    req = SyncV2ProfileBootstrapRequest(device_name="Riley's MacBook")
    dumped = req.model_dump(mode="json")
    assert dumped["mode"] == "offline_sync"
    assert dumped["client_family"] == "chatbook"
    assert dumped["requested_domains"] == [
        "notes.note", "chat.conversation", "chat.message", "attachment.ref",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/tldw_api/test_sync_schemas_m1.py -q`
Expected: FAIL — `ImportError` (the new model names are not exported yet).

- [ ] **Step 3: Add the models**

In `tldw_chatbook/tldw_api/sync_schemas.py`, directly below the
`SyncV2CapabilitiesResponse` class, add:

```python
class SyncV2ProfileDeviceStatus(BaseModel):
    """Device registration status in a Sync v2 profile response."""

    device_id: str | None = None
    registered: bool = False
    client_profile_id: str | None = None
    last_seen_at: str | None = None
    mode: str | None = None
    client_type: str | None = None
    client_version: str | None = None

    model_config = ConfigDict(extra="ignore")


class SyncV2ProfileDatasetStatus(BaseModel):
    """Default personal dataset metadata in a Sync v2 profile response."""

    dataset_id: str
    scope: str = "personal"
    default_personal: bool = False
    client_family: str | None = None
    domains: list[str] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    encryption_policy: str = "server_trusted_v1"

    model_config = ConfigDict(extra="ignore")


class SyncV2ProfileDomainStatus(BaseModel):
    """Per-domain Sync v2 status summary."""

    domain: str
    last_server_cursor: int = Field(0, ge=0)
    envelope_count: int = Field(0, ge=0)
    pending_apply_count: int = Field(0, ge=0)
    unresolved_conflicts: int = Field(0, ge=0)
    last_apply_status: str | None = None

    model_config = ConfigDict(extra="ignore")


class SyncV2ProfileResponse(BaseModel):
    """Read-only Sync v2 M1 profile/status response."""

    protocol_version: str = "sync-v2-m1"
    min_supported_protocol_version: str = "sync-v2-m1"
    profile_bootstrapped: bool = False
    user_id: str | None = None
    active_dataset_id: str | None = None
    device: SyncV2ProfileDeviceStatus | None = None
    dataset: SyncV2ProfileDatasetStatus | None = None
    server_cursor: int = Field(0, ge=0)
    capabilities: SyncV2CapabilitiesResponse = Field(default_factory=SyncV2CapabilitiesResponse)
    domain_status: list[SyncV2ProfileDomainStatus] = Field(default_factory=list)
    warnings: list[dict[str, str]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class SyncV2ProfileBootstrapRequest(BaseModel):
    """Request to bootstrap a server-connected Chatbook profile (POST /profile/bootstrap)."""

    client_family: str = "chatbook"
    mode: Literal["server_frontend", "offline_sync"] = "offline_sync"
    device_id: str | None = None
    device_name: str | None = None
    client_profile_id: str | None = None
    client_instance: dict[str, Any] = Field(default_factory=dict)
    requested_domains: list[str] = Field(
        default_factory=lambda: [
            "notes.note", "chat.conversation", "chat.message", "attachment.ref",
        ]
    )


class SyncV2ProfileBootstrapResponse(SyncV2ProfileResponse):
    """Response from explicit profile bootstrap."""

    created: bool = False
```

- [ ] **Step 4: Run test to verify it passes (after Task 3 exports)**

These tests import from `tldw_chatbook.tldw_api`, so they pass only once Task 3 wires the
exports. Proceed to Task 3, then run:
`.venv/bin/python -m pytest Tests/tldw_api/test_sync_schemas_m1.py -q`
Expected: PASS.

- [ ] **Step 5: Commit (with Task 3)**

Commit together with Task 3 (the models are unusable until exported).

---

## Task 3: Export the new models

**Files:**
- Modify: `tldw_chatbook/tldw_api/__init__.py` (the `from .sync_schemas import (...)` block at ~line 1158 and the `__all__` list at ~line 1662)

- [ ] **Step 1: Add to the import block**

In `tldw_chatbook/tldw_api/__init__.py`, inside the `from .sync_schemas import (` block
(alphabetically near the other `SyncV2*` names), add:

```python
    SyncV2ProfileBootstrapRequest,
    SyncV2ProfileBootstrapResponse,
    SyncV2ProfileDatasetStatus,
    SyncV2ProfileDeviceStatus,
    SyncV2ProfileDomainStatus,
    SyncV2ProfileResponse,
```

- [ ] **Step 2: Add to `__all__`**

In the `__all__` list, alongside the other `SyncV2*` string entries, add:

```python
    "SyncV2ProfileBootstrapRequest", "SyncV2ProfileBootstrapResponse",
    "SyncV2ProfileDatasetStatus", "SyncV2ProfileDeviceStatus",
    "SyncV2ProfileDomainStatus", "SyncV2ProfileResponse",
```

- [ ] **Step 3: Run the schema tests**

Run: `.venv/bin/python -m pytest Tests/tldw_api/test_sync_schemas_m1.py -q`
Expected: PASS (all tests from Tasks 1 + 2).

- [ ] **Step 4: Verify the package imports cleanly**

Run: `.venv/bin/python -c "import tldw_chatbook.tldw_api as t; print(t.SyncV2ProfileBootstrapResponse.__name__)"`
Expected: prints `SyncV2ProfileBootstrapResponse`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/sync_schemas.py tldw_chatbook/tldw_api/__init__.py Tests/tldw_api/test_sync_schemas_m1.py
git commit -m "Sync v2 P1: add + export profile/bootstrap models"
```

---

## Task 4: Add client transport methods for profile + bootstrap

**Files:**
- Modify: `tldw_chatbook/tldw_api/client.py` (the `from .sync_schemas import (` block at ~line 403; add methods next to `get_sync_v2_capabilities` at ~line 14776)
- Test: `Tests/tldw_api/test_sync_client.py` (add tests; fix the legacy protocol-version pin)

- [ ] **Step 1: Write the failing test**

Append to `Tests/tldw_api/test_sync_client.py`:

```python
@pytest.mark.asyncio
async def test_sync_v2_client_bootstraps_profile(monkeypatch):
    from tldw_chatbook.tldw_api import (
        SyncV2ProfileBootstrapRequest,
        SyncV2ProfileBootstrapResponse,
    )

    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "created": True,
            "profile_bootstrapped": True,
            "user_id": "user_123",
            "active_dataset_id": "ds_1",
            "device": {"device_id": "dev_1", "registered": True},
            "dataset": {"dataset_id": "ds_1", "scope": "personal", "default_personal": True},
            "server_cursor": 0,
            "capabilities": {"protocol_version": "sync-v2-m1"},
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    resp = await client.bootstrap_sync_v2_profile(
        SyncV2ProfileBootstrapRequest(device_name="Test Device")
    )

    assert isinstance(resp, SyncV2ProfileBootstrapResponse)
    assert resp.device.device_id == "dev_1"
    assert resp.dataset.dataset_id == "ds_1"
    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/sync/profile/bootstrap")
    assert mocked.await_args_list[0].kwargs["json_data"]["mode"] == "offline_sync"


@pytest.mark.asyncio
async def test_sync_v2_client_gets_profile(monkeypatch):
    from tldw_chatbook.tldw_api import SyncV2ProfileResponse

    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={"profile_bootstrapped": False, "user_id": "user_123", "server_cursor": 0}
    )
    monkeypatch.setattr(client, "_request", mocked)

    resp = await client.get_sync_v2_profile(device_id="dev_1")

    assert isinstance(resp, SyncV2ProfileResponse)
    assert resp.profile_bootstrapped is False
    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/sync/profile")
    assert mocked.await_args_list[0].kwargs["params"]["device_id"] == "dev_1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/tldw_api/test_sync_client.py -k "bootstraps_profile or gets_profile" -q`
Expected: FAIL — `AttributeError: 'TLDWAPIClient' object has no attribute 'bootstrap_sync_v2_profile'`.

- [ ] **Step 3: Import the new models in client.py**

In `tldw_chatbook/tldw_api/client.py`, inside the `from .sync_schemas import (` block at
~line 403, add:

```python
    SyncV2ProfileBootstrapRequest,
    SyncV2ProfileBootstrapResponse,
    SyncV2ProfileResponse,
```

- [ ] **Step 4: Add the methods**

In `tldw_chatbook/tldw_api/client.py`, immediately after the `get_sync_v2_capabilities`
method (ends ~line 14787), add:

```python
    async def get_sync_v2_profile(
        self,
        *,
        device_id: str | None = None,
    ) -> SyncV2ProfileResponse:
        """Fetch the current Sync v2 profile/status without creating sync state.

        Args:
            device_id: Existing client device ID, when known.

        Returns:
            Parsed profile response (capabilities, device/dataset status, cursor).

        Raises:
            Exception: Propagates request failures and response validation errors.
        """

        response = await self._request(
            "GET",
            "/api/v1/sync/profile",
            params={"device_id": device_id},
        )
        return SyncV2ProfileResponse.model_validate(response)

    async def bootstrap_sync_v2_profile(
        self,
        request_data: SyncV2ProfileBootstrapRequest,
    ) -> SyncV2ProfileBootstrapResponse:
        """Idempotently bootstrap server-connected Chatbook sync for the user.

        Registers/refreshes the device and creates/returns the default personal dataset.
        With an omitted device_id, the server assigns one; callers must persist the
        returned device_id/dataset_id before pushing (persisted in P2).

        Args:
            request_data: Bootstrap request (mode, device name, requested domains).

        Returns:
            Parsed bootstrap response including assigned device_id and dataset_id.

        Raises:
            Exception: Propagates request failures and response validation errors.
        """

        response = await self._request(
            "POST",
            "/api/v1/sync/profile/bootstrap",
            json_data=request_data.model_dump(mode="json"),
        )
        return SyncV2ProfileBootstrapResponse.model_validate(response)
```

- [ ] **Step 5: Fix the legacy protocol-version pin in the existing test**

In `Tests/tldw_api/test_sync_client.py`, in `test_sync_v2_client_routes_protocol_endpoints`
(~line 139), the mocked capabilities payload uses integer protocol versions. Update both
occurrences in that test's first mocked response:

Change:
```python
                "protocol_version": 2,
                "min_supported_protocol_version": 2,
```
to:
```python
                "protocol_version": "sync-v2-m1",
                "min_supported_protocol_version": "sync-v2-m1",
```
And update the corresponding assertion in that test if it asserts
`capabilities.protocol_version == 2` — change it to
`capabilities.protocol_version == "sync-v2-m1"`.

- [ ] **Step 6: Run the client tests**

Run: `.venv/bin/python -m pytest Tests/tldw_api/test_sync_client.py -q`
Expected: PASS (existing + 2 new tests).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/tldw_api/client.py Tests/tldw_api/test_sync_client.py
git commit -m "Sync v2 P1: add profile + bootstrap transport methods"
```

---

## Task 5: Fix downstream capability-shape readers

**Files:**
- Inspect/Modify: `Tests/Sync_Interop/test_server_sync_service.py`, `Tests/MCP/test_unified_control_plane_service.py`, `Tests/MCP/test_local_control_service.py`, and any production reader of `.supported_domains`/`.supported_operations` or `capabilities.protocol_version`.

- [ ] **Step 1: Find readers that may break**

Run:
```bash
grep -rnE 'supported_domains|supported_operations|protocol_version\b|SyncV2CapabilitiesResponse' \
  tldw_chatbook Tests | grep -v test_sync_schemas_m1
```
Inspect each hit. Production readers of `.supported_domains`/`.supported_operations` keep
working via the back-compat properties added in Task 1. Tests that assert
`protocol_version == 2` or construct the capabilities model with integer versions or the
old `supported_domains=[...]`/`supported_operations=[...]` field names must be updated.

- [ ] **Step 2: Run the affected suites to surface real breakage**

Run:
```bash
.venv/bin/python -m pytest Tests/Sync_Interop/test_server_sync_service.py \
  Tests/MCP/test_unified_control_plane_service.py Tests/MCP/test_local_control_service.py -q
```
Expected before fixes: any failures are shape mismatches (integer protocol version, old
field names). Note each failing assertion.

- [ ] **Step 3: Update each broken assertion to the M1 shape**

For each failure, change integer `protocol_version` expectations to `"sync-v2-m1"`, and
replace any `SyncV2CapabilitiesResponse(supported_domains=[...], supported_operations=[...])`
construction with `domains=[...]`, `operations={...}` (or rely on defaults). Make only the
edits needed to match the new shape; do not change behavior.

- [ ] **Step 4: Re-run the suites**

Run:
```bash
.venv/bin/python -m pytest Tests/Sync_Interop/test_server_sync_service.py \
  Tests/MCP/test_unified_control_plane_service.py Tests/MCP/test_local_control_service.py -q
```
Expected: PASS.

- [ ] **Step 5: Run the full sync/tldw_api test scope**

Run:
```bash
.venv/bin/python -m pytest Tests/tldw_api Tests/Sync_Interop -q
```
Expected: PASS (no regressions from the schema change).

- [ ] **Step 6: Commit**

```bash
git add Tests/
git commit -m "Sync v2 P1: update capability-shape readers to M1 response"
```

---

## Task 6: Live verification against the M1 server (P1 exit gate)

**Files:**
- Create: `/tmp/syncqa-v2/p1_check.py` (a verification script, not committed)

- [ ] **Step 1: Ensure the server is running**

Use the Pre-flight launch + readiness check above. Confirm the capabilities curl returns
`{"protocol_version": "sync-v2-m1"...`.

- [ ] **Step 2: Write the live check script**

Create `/tmp/syncqa-v2/p1_check.py`:

```python
import asyncio
import sys
from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api import SyncV2ProfileBootstrapRequest

BASE = "http://127.0.0.1:8076"
KEY = "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"


async def main():
    client = TLDWAPIClient(base_url=BASE, token=KEY)
    try:
        client.bearer_token = KEY
    except Exception:
        pass

    caps = await client.get_sync_v2_capabilities()
    print("CAPABILITIES protocol:", caps.protocol_version, "| domains:", caps.domains[:3])
    assert caps.protocol_version == "sync-v2-m1"

    resp = await client.bootstrap_sync_v2_profile(
        SyncV2ProfileBootstrapRequest(device_name="p1-check-device", mode="offline_sync")
    )
    print("BOOTSTRAP created:", resp.created, "| bootstrapped:", resp.profile_bootstrapped)
    print("  device_id:", resp.device and resp.device.device_id)
    print("  dataset_id:", resp.dataset and resp.dataset.dataset_id)
    print("  warnings:", resp.warnings)

    await client.close()

    ok = (
        resp.profile_bootstrapped
        and resp.device is not None and resp.device.device_id
        and resp.dataset is not None and resp.dataset.dataset_id
    )
    print("\nP1_EXIT_OK:", ok)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Run the live check**

Run:
```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /tmp/syncqa-v2/p1_check.py
```
Expected: prints capabilities + bootstrap details and `P1_EXIT_OK: True`.

If `profile_bootstrapped` is `False` with a `sync_encryption_attestation_required`
warning, the server bootstrap is fail-closed on attestation. Resolve by launching the
server with attestation configured (re-check `tldw_Server_API` `.env`/settings for the
at-rest-encryption attestation flag for the user database directory), then re-run. This is
the attestation gate called out in the spec; do not proceed to P2 until it is green.

- [ ] **Step 4: Record the result**

Append the script output to the QA notes and update backlog task #24 with "P1 green:
client parses M1 capabilities + bootstraps a profile live against :8076".

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
backlog task edit 24 --notes "P1 complete: client parses M1 capabilities and bootstraps a profile live vs :8076 (codex/sync-v2-m1-next @ 992e89a03)."
```

---

## Self-Review (completed at authoring time)

- **Spec coverage:** P1 spec items — widen capabilities parsing (Task 1), add
  bootstrap/profile transport (Tasks 2-4), P1 exit "parses capabilities + bootstraps
  profile, not fail-closed" (Task 6). The `device_id`/`dataset_id` *persistence* is
  intentionally deferred to P2 (documented in the Scope note); envelope/literal/builder/
  applier are P2 by design.
- **Placeholder scan:** none — every code/edit step shows concrete content.
- **Type consistency:** new names used identically across tasks
  (`SyncV2ProfileBootstrapRequest/Response`, `SyncV2ProfileResponse`,
  `get_sync_v2_profile`, `bootstrap_sync_v2_profile`); endpoints
  `/api/v1/sync/profile` and `/api/v1/sync/profile/bootstrap` match the server.
