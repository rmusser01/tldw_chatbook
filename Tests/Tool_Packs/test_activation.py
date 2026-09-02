"""Activation contract tests for portable Tool Pack imports."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Callable, Iterator

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import (
    MCPPermissionStore,
    PermissionStoreSnapshot,
    definition_hash,
    resolve_effective_state,
)
from tldw_chatbook.Tool_Packs import activation as activation_module
from tldw_chatbook.Tool_Packs.activation import (
    InstalledToolProfile,
    ToolPackActivationResult,
    ToolPackActivationService,
    compile_imported_profile,
)
from tldw_chatbook.Tool_Packs.binding import (
    ProfileMutationError,
    ProfileMutationResult,
    profile_policy_digest,
)
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventorySnapshot,
    PermissionInventoryTool,
)
from tldw_chatbook.Tool_Packs.contracts import (
    PortableFallback,
    PortableToolRule,
    ToolPackError,
    portable_contract_sha256,
)
from tldw_chatbook.Tool_Packs.importer import (
    MappedToolRule,
    ServerMapping,
    ToolPackImportReview,
)
from tldw_chatbook.Tool_Packs.receipt_store import (
    RECEIPT_SCHEMA,
    ReceiptHandle,
    ToolPackReceipt,
    ToolPackReceiptStore,
)
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


_NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
_RECEIPT_ID = "tp-" + "1" * 32
_RECEIPT_DIGEST = "2" * 64


def _tool(
    *,
    server_key: str = "local:docs",
    name: str = "search",
    description: str | None = None,
) -> HubTool:
    return HubTool(
        server_key=server_key,
        server_label="ignored",
        source="local",
        name=name,
        description=description or f"{name} description",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        tags=(),
        stale=False,
        executable=True,
    )


def _inventory(
    *tools: HubTool, digest: str = "inventory-digest"
) -> PermissionInventorySnapshot:
    entries = tuple(
        PermissionInventoryTool(
            "builtin" if tool.server_key == "agent:builtin" else "mcp",
            tool,
            portable_contract_sha256(tool),
        )
        for tool in tools
    )
    namespaces = tuple(
        sorted({(entry.authority, entry.tool.server_key) for entry in entries})
    )
    return PermissionInventorySnapshot(entries, namespaces, (), digest)


def _review(
    *,
    matched: tuple[MappedToolRule, ...] = (),
    fallbacks: tuple[PortableFallback, ...] = (
        PortableFallback("mcp", "*", "ask"),
        PortableFallback("builtin", "agent:builtin", "deny"),
    ),
    pending_denies: tuple[PortableToolRule, ...] = (),
    omitted_allow_ask: tuple[PortableToolRule, ...] = (),
    changed: tuple[PortableToolRule, ...] = (),
    missing: tuple[PortableToolRule, ...] = (),
    inventory_digest: str = "inventory-digest",
) -> ToolPackImportReview:
    return ToolPackImportReview(
        archive_path=Path("reviewed.tldw-tool-pack"),
        archive_sha256="a" * 64,
        manifest_sha256="b" * 64,
        payload_sha256="c" * 64,
        destination_id="research",
        store_generation="sha256:" + "d" * 64,
        inventory_digest=inventory_digest,
        mappings=(),
        expires_at=_NOW + timedelta(minutes=15),
        matched=matched,
        changed=changed,
        missing=missing,
        pending_denies=pending_denies,
        omitted_allow_ask=omitted_allow_ask,
        content_digest="e" * 64,
        display_name="Research",
        producer=("test-producer", "1"),
        fallbacks=fallbacks,
    )


def _receipt() -> ReceiptHandle:
    return ReceiptHandle(_RECEIPT_ID, _RECEIPT_DIGEST, Path(_RECEIPT_ID), 123)


class _Importer:
    def __init__(
        self,
        review: ToolPackImportReview,
        events: list[str],
        *,
        expected_request: ToolPackImportReview | None = None,
    ) -> None:
        self.review = review
        self.events = events
        self.expected_request = expected_request or review

    def inspect_archive(
        self,
        archive_path: Path,
        *,
        destination_id: str,
        mappings: tuple[object, ...],
    ) -> ToolPackImportReview:
        self.events.append("reinspect")
        assert archive_path == self.expected_request.archive_path
        assert destination_id == self.expected_request.destination_id
        assert mappings == self.expected_request.mappings
        return self.review


class _Lifecycle:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    @contextmanager
    def mutation(self) -> Iterator[None]:
        self.events.append("coordinator")
        yield


class _References:
    def __init__(self, events: list[str], *, referenced: bool = False) -> None:
        self.events = events
        self.referenced = referenced

    def references_profile(self, profile_id: str, *, include_archived: bool) -> bool:
        self.events.append("reference_check")
        assert profile_id == "research"
        assert include_archived is True
        return self.referenced


class _Store:
    def __init__(
        self,
        events: list[str],
        review: ToolPackImportReview,
        *,
        install_outcome: str = "success",
        fail_reconciliation_read: bool = False,
        fail_strict_read_at: int | None = None,
    ) -> None:
        self.events = events
        self.generation = review.store_generation
        self.profiles: dict[str, object] = {
            "default": {"global_default": "allow", "servers": {}}
        }
        self.install_outcome = install_outcome
        self.fail_reconciliation_read = fail_reconciliation_read
        self.fail_strict_read_at = fail_strict_read_at
        self.strict_reads = 0
        self.install_calls = 0

    def read_snapshot_strict(self) -> PermissionStoreSnapshot:
        self.strict_reads += 1
        if self.fail_strict_read_at == self.strict_reads or (
            self.fail_reconciliation_read and self.strict_reads > 1
        ):
            raise OSError("strict reconciliation unavailable")
        return PermissionStoreSnapshot(
            {
                "schema_version": 1,
                "kill_switch": False,
                "profiles": self.profiles,
            },
            self.generation,
            True,
        )

    @contextmanager
    def mutation_fence(self) -> Iterator[None]:
        self.events.append("store_fence")
        yield

    def install_profile_if_absent(
        self,
        profile_id: str,
        profile: dict[str, object],
        *,
        expected_generation: str,
        max_profiles: int,
        max_store_bytes: int,
    ) -> ProfileMutationResult:
        self.events.append("install")
        self.install_calls += 1
        assert expected_generation == self.generation
        assert max_profiles == 128
        assert max_store_bytes == 8 * 1024 * 1024
        if self.install_outcome == "before":
            raise OSError("before replace")
        if self.install_outcome == "profile_limit":
            raise ProfileMutationError("profile_limit")
        if self.install_outcome == "store_bytes_limit":
            raise ProfileMutationError("store_bytes_limit")
        if self.install_outcome == "third":
            self.profiles[profile_id] = {"global_default": "deny", "servers": {}}
            self.generation = "sha256:" + "f" * 64
            raise OSError("third state")
        self.profiles[profile_id] = profile
        self.generation = "sha256:" + "f" * 64
        if self.install_outcome == "after":
            raise OSError("after replace")
        lifecycle = profile["tool_pack_lifecycle"]
        assert isinstance(lifecycle, dict)
        return ProfileMutationResult(
            profile_id,
            lifecycle["revision"],  # type: ignore[arg-type]
            lifecycle["policy_digest"],  # type: ignore[arg-type]
            self.generation,
        )


class _RecordingReservation:
    def __init__(self, reservation: object, events: list[str]) -> None:
        self.reservation = reservation
        self.events = events

    def __enter__(self) -> "_RecordingReservation":
        self.reservation.__enter__()  # type: ignore[attr-defined]
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.reservation.__exit__(exc_type, exc, traceback)  # type: ignore[attr-defined]

    def commit(self, data: bytes) -> ReceiptHandle:
        handle = self.reservation.commit(data)  # type: ignore[attr-defined]
        self.events.append("receipt_durable")
        return handle


class _RecordingReceiptStore:
    def __init__(self, store: ToolPackReceiptStore, events: list[str]) -> None:
        self.store = store
        self.events = events

    def reserve(self, projected_bytes: int) -> _RecordingReservation:
        return _RecordingReservation(self.store.reserve(projected_bytes), self.events)

    def read(self, receipt_id: str, *, expected_digest: str):
        return self.store.read(receipt_id, expected_digest=expected_digest)


def _service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    review: ToolPackImportReview,
    fresh_review: ToolPackImportReview | None = None,
    inventory: PermissionInventorySnapshot | None = None,
    store: _Store | None = None,
    references: _References | None = None,
    receipt_bytes: int = 4 * 1024 * 1024,
    now: datetime | Callable[[], datetime] = _NOW,
) -> tuple[
    ToolPackActivationService,
    _Store,
    _References,
    _RecordingReceiptStore,
    list[str],
]:
    events = store.events if store is not None else []
    inventory = inventory or _inventory()

    def capture(_registry: object) -> PermissionInventorySnapshot:
        events.append("inventory")
        return inventory

    monkeypatch.setattr(activation_module, "capture_v1_inventory", capture)
    store = store or _Store(events, review)
    references = references or _References(events)
    receipts = _RecordingReceiptStore(
        ToolPackReceiptStore(
            tmp_path / "receipts",
            max_receipt_bytes=receipt_bytes,
            _id_source=lambda: bytes.fromhex("11" * 16),
        ),
        events,
    )
    service = ToolPackActivationService(
        permission_store=store,
        inventory=inventory,
        importer=_Importer(
            fresh_review or review,
            events,
            expected_request=review,
        ),
        reference_checker=references,
        receipt_store=receipts,
        lifecycle=_Lifecycle(events),
        now=now if callable(now) else lambda: now,
    )
    return service, store, references, receipts, events


def test_exact_match_activation_receipt_accepts_no_manual_mappings() -> None:
    """An exact-match import must not invent a manual server mapping."""
    receipt = ToolPackReceipt(
        schema=RECEIPT_SCHEMA,
        kind="import",
        profile_id="research",
        pack_digest="a" * 64,
        archive_digest="b" * 64,
        producer=("test-producer", "1"),
        imported_at="2026-09-01T12:00:00Z",
        reviewed_mappings=(),
        matched=(("mcp", "local:docs", "search"),),
    )

    assert receipt.reviewed_mappings == ()


def test_compile_imported_profile_uses_safe_defaults_and_current_runtime_hash() -> None:
    tool = _tool()
    portable_hash = portable_contract_sha256(tool)
    rule = PortableToolRule("mcp", "local:docs", "search", "allow", portable_hash)
    review = _review(
        matched=(
            MappedToolRule(
                rule,
                ("mcp", "local:docs", "search"),
                portable_hash,
                True,
            ),
        ),
        fallbacks=(
            PortableFallback("mcp", "*", "ask"),
            PortableFallback("builtin", "agent:builtin", "deny"),
            PortableFallback("mcp", "local:docs", "ask"),
        ),
    )

    compiled = compile_imported_profile(
        review,
        _inventory(tool),
        receipt=_receipt(),
        imported_at=_NOW,
    )

    assert compiled["global_default"] == "ask"
    assert compiled["servers"]["agent:builtin"] == {"default": "deny"}
    assert compiled["servers"]["local:docs"]["default"] == "ask"
    assert compiled["servers"]["local:docs"]["tools"]["search"] == {
        "state": "allow",
        "definition_hash": definition_hash(tool.description, tool.input_schema),
    }
    assert portable_hash not in json.dumps(compiled, sort_keys=True)
    lifecycle = compiled["tool_pack_lifecycle"]
    assert compiled["profile_kind"] == "tool_pack_imported"
    assert lifecycle == {
        "schema": "tldw.tool-pack-lifecycle/v1",
        "origin": "imported",
        "pack_digest": review.content_digest,
        "imported_at": "2026-09-01T12:00:00Z",
        "first_bind_confirmation_required": True,
        "receipt_id": _RECEIPT_ID,
        "receipt_digest": _RECEIPT_DIGEST,
        "counts": {"matched": 1, "omitted": 0, "pending_deny": 0},
        "policy_digest": profile_policy_digest(compiled),
        "revision": 1,
    }


def test_compile_keeps_only_reviewed_effective_exceptions() -> None:
    search = _tool(name="search")
    builtin = _tool(server_key="builtin:tldw_chatbook", name="notes_search")
    search_hash = portable_contract_sha256(search)
    builtin_hash = portable_contract_sha256(builtin)
    matched = (
        MappedToolRule(
            PortableToolRule("mcp", "local:docs", "search", "ask", search_hash),
            ("mcp", "local:docs", "search"),
            search_hash,
            True,
        ),
        MappedToolRule(
            PortableToolRule(
                "mcp",
                "builtin:tldw_chatbook",
                "notes_search",
                "allow",
                builtin_hash,
            ),
            ("mcp", "builtin:tldw_chatbook", "notes_search"),
            builtin_hash,
            True,
        ),
    )
    omitted = (
        PortableToolRule("mcp", "source:gone", "changed", "allow", "3" * 64),
        PortableToolRule("mcp", "source:gone", "missing", "ask", "4" * 64),
    )
    pending = (
        PortableToolRule("mcp", "source:block", "covered", "deny", None),
        PortableToolRule("mcp", "source:orphan", "blocked", "deny", None),
    )
    review = _review(
        matched=matched,
        fallbacks=(
            PortableFallback("mcp", "*", "ask"),
            PortableFallback("builtin", "agent:builtin", "ask"),
            PortableFallback("mcp", "builtin:tldw_chatbook", "ask"),
            PortableFallback("mcp", "local:docs", "ask"),
            PortableFallback("mcp", "source:block", "deny"),
        ),
        changed=(omitted[0],),
        missing=(omitted[1], pending[0], pending[1]),
        pending_denies=pending,
        omitted_allow_ask=omitted,
    )

    compiled = compile_imported_profile(
        review,
        _inventory(search, builtin),
        receipt=_receipt(),
        imported_at=_NOW,
    )

    assert compiled["servers"]["local:docs"] == {"default": "ask"}
    assert compiled["servers"]["source:block"] == {"default": "deny"}
    assert compiled["servers"]["source:orphan"] == {
        "tools": {"blocked": {"state": "deny"}}
    }
    assert compiled["servers"]["builtin:tldw_chatbook"] == {
        "default": "ask",
        "tools": {"notes_search": {"state": "allow"}},
    }
    serialized = json.dumps(compiled, sort_keys=True)
    assert "changed" not in serialized
    assert '"missing"' not in serialized
    assert compiled["tool_pack_lifecycle"]["counts"] == {
        "matched": 2,
        "omitted": 2,
        "pending_deny": 2,
    }


def test_compile_rejects_destination_contract_change() -> None:
    reviewed_tool = _tool(description="reviewed")
    changed_tool = _tool(description="changed")
    reviewed_hash = portable_contract_sha256(reviewed_tool)
    review = _review(
        matched=(
            MappedToolRule(
                PortableToolRule("mcp", "local:docs", "search", "allow", reviewed_hash),
                ("mcp", "local:docs", "search"),
                reviewed_hash,
                True,
            ),
        ),
    )

    with pytest.raises(ToolPackError, match=r"review_stale$"):
        compile_imported_profile(
            review,
            _inventory(changed_tool),
            receipt=_receipt(),
            imported_at=_NOW,
        )


def test_activation_public_models_and_receipt_first_unbound_install(
    tmp_path: Path, monkeypatch
) -> None:
    assert InstalledToolProfile
    assert ToolPackActivationResult
    events: list[str] = []
    review = _review()
    inventory = _inventory()

    def capture(_registry: object) -> PermissionInventorySnapshot:
        events.append("inventory")
        return inventory

    monkeypatch.setattr(activation_module, "capture_v1_inventory", capture)
    references = _References(events)
    store = _Store(events, review)
    receipts = _RecordingReceiptStore(
        ToolPackReceiptStore(
            tmp_path / "receipts",
            _id_source=lambda: bytes.fromhex("11" * 16),
        ),
        events,
    )
    service = ToolPackActivationService(
        permission_store=store,
        inventory=inventory,
        importer=_Importer(review, events),
        reference_checker=references,
        receipt_store=receipts,
        lifecycle=_Lifecycle(events),
        now=lambda: _NOW,
    )

    result = service.install(review)

    assert events == [
        "reinspect",
        "inventory",
        "receipt_durable",
        "coordinator",
        "store_fence",
        "reference_check",
        "install",
    ]
    assert result.installed == InstalledToolProfile(
        "research",
        store.profiles["research"]["tool_pack_lifecycle"]["policy_digest"],  # type: ignore[index]
        1,
        "tp-" + "11" * 16,
    )
    assert result.reconciled is False
    assert references.referenced is False
    assert (
        receipts.read(
            result.installed.receipt_id,
            expected_digest=store.profiles["research"]["tool_pack_lifecycle"][
                "receipt_digest"
            ],  # type: ignore[index]
        ).receipt.reviewed_mappings
        == ()
    )
    assert store.profiles["default"] == {
        "global_default": "allow",
        "servers": {},
    }


def test_activation_receipt_preserves_overlapping_review_categories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = PortableToolRule("mcp", "source:docs", "changed", "allow", "3" * 64)
    missing = PortableToolRule("mcp", "source:docs", "blocked", "deny", None)
    review = _review(
        changed=(changed,),
        missing=(missing,),
        pending_denies=(missing,),
        omitted_allow_ask=(changed,),
    )
    service, store, _references, receipts, _events = _service(
        tmp_path,
        monkeypatch,
        review=review,
    )

    result = service.install(review)

    lifecycle = store.profiles["research"]["tool_pack_lifecycle"]  # type: ignore[index]
    verified = receipts.read(
        result.installed.receipt_id,
        expected_digest=lifecycle["receipt_digest"],
    ).receipt
    changed_identity = ("mcp", "source:docs", "changed")
    missing_identity = ("mcp", "source:docs", "blocked")
    assert verified.changed == verified.omitted == (changed_identity,)
    assert verified.missing == verified.pending_deny == (missing_identity,)


def test_activation_receipt_is_live_owned_until_authority_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = _review()
    inventory = _inventory()
    monkeypatch.setattr(
        activation_module,
        "capture_v1_inventory",
        lambda _registry: inventory,
    )
    events: list[str] = []
    store = _Store(events, review)
    observed: list[frozenset[str]] = []
    service: ToolPackActivationService

    def references_profile(_profile_id: str) -> bool:
        observed.append(service.live_receipt_ids())
        return False

    service = ToolPackActivationService(
        permission_store=store,
        inventory=inventory,
        importer=_Importer(review, events),
        reference_checker=references_profile,
        receipt_store=ToolPackReceiptStore(
            tmp_path / "receipts",
            _id_source=lambda: bytes.fromhex("11" * 16),
        ),
        lifecycle=_Lifecycle(events),
        now=lambda: _NOW,
    )

    result = service.install(review)

    assert observed == [frozenset({result.installed.receipt_id})]
    assert service.live_receipt_ids() == frozenset()


@pytest.mark.parametrize(
    ("fresh_review", "category"),
    [
        (replace(_review(), archive_sha256="9" * 64), "review_stale"),
        (
            replace(
                _review(),
                mappings=(ServerMapping("source:docs", "local:docs"),),
            ),
            "review_stale",
        ),
        (replace(_review(), inventory_digest="changed"), "review_stale"),
        (
            replace(_review(), store_generation="sha256:" + "9" * 64),
            "store_changed",
        ),
        (replace(_review(), destination_id="other"), "review_stale"),
    ],
)
def test_activation_rejects_changed_review_evidence_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_review: ToolPackImportReview,
    category: str,
) -> None:
    review = _review()
    service, store, _references, _receipts, events = _service(
        tmp_path,
        monkeypatch,
        review=review,
        fresh_review=fresh_review,
    )

    with pytest.raises(ToolPackError) as error:
        service.install(review)

    assert error.value.category == category
    assert "receipt_durable" not in events
    assert store.install_calls == 0


def test_activation_rejects_expired_review_before_reinspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = _review()
    service, store, _references, _receipts, events = _service(
        tmp_path,
        monkeypatch,
        review=review,
        now=review.expires_at,
    )

    with pytest.raises(ToolPackError, match=r"review_stale$"):
        service.install(review)

    assert events == []
    assert store.install_calls == 0


def test_activation_rechecks_expiry_under_final_authority_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = _review()
    clock = iter((_NOW, review.expires_at))
    service, store, _references, _receipts, events = _service(
        tmp_path,
        monkeypatch,
        review=review,
        now=lambda: next(clock),
    )

    with pytest.raises(ToolPackError, match=r"review_stale$"):
        service.install(review)

    assert events[-3:] == ["receipt_durable", "coordinator", "store_fence"]
    assert "reference_check" not in events
    assert store.install_calls == 0


def test_activation_rejects_fresh_inventory_change_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = _review()
    service, store, _references, _receipts, events = _service(
        tmp_path,
        monkeypatch,
        review=review,
        inventory=_inventory(digest="changed"),
    )

    with pytest.raises(ToolPackError, match=r"review_stale$"):
        service.install(review)

    assert events == ["reinspect", "inventory"]
    assert store.install_calls == 0


def test_activation_rechecks_destination_references_after_durable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    review = _review()
    store = _Store(events, review)
    references = _References(events, referenced=True)
    service, store, _references, _receipts, events = _service(
        tmp_path,
        monkeypatch,
        review=review,
        store=store,
        references=references,
    )

    with pytest.raises(ToolPackError, match=r"destination_referenced$"):
        service.install(review)

    assert events[-4:] == [
        "receipt_durable",
        "coordinator",
        "store_fence",
        "reference_check",
    ]
    assert store.install_calls == 0
    assert "research" not in store.profiles


def test_activation_reports_final_strict_store_failure_as_store_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    review = _review()
    store = _Store(events, review, fail_strict_read_at=1)
    service, store, _references, _receipts, events = _service(
        tmp_path,
        monkeypatch,
        review=review,
        store=store,
    )

    with pytest.raises(ToolPackError, match=r"store_invalid$"):
        service.install(review)

    assert "receipt_durable" in events
    assert store.install_calls == 0


def test_activation_rejects_store_change_under_final_authority_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = _review()
    service, store, _references, _receipts, events = _service(
        tmp_path,
        monkeypatch,
        review=review,
    )
    store.generation = "sha256:" + "9" * 64

    with pytest.raises(ToolPackError, match=r"store_changed$"):
        service.install(review)

    assert events[-3:] == ["receipt_durable", "coordinator", "store_fence"]
    assert "reference_check" not in events
    assert store.install_calls == 0


def test_activation_rejects_occupied_destination_under_final_authority_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = _review()
    service, store, _references, _receipts, events = _service(
        tmp_path,
        monkeypatch,
        review=review,
    )
    store.profiles["Research"] = {"global_default": "deny", "servers": {}}

    with pytest.raises(ToolPackError, match=r"destination_referenced$"):
        service.install(review)

    assert events[-3:] == ["receipt_durable", "coordinator", "store_fence"]
    assert "reference_check" not in events
    assert store.install_calls == 0


def test_activation_rejects_receipt_capacity_before_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = _review()
    service, store, _references, _receipts, events = _service(
        tmp_path,
        monkeypatch,
        review=review,
        receipt_bytes=1,
    )

    with pytest.raises(ToolPackError, match=r"capacity_exceeded$"):
        service.install(review)

    assert events == ["reinspect", "inventory"]
    assert store.install_calls == 0


@pytest.mark.parametrize("outcome", ["profile_limit", "store_bytes_limit"])
def test_activation_maps_profile_and_store_caps_to_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    events: list[str] = []
    review = _review()
    store = _Store(events, review, install_outcome=outcome)
    service, store, _references, _receipts, _events = _service(
        tmp_path,
        monkeypatch,
        review=review,
        store=store,
    )

    with pytest.raises(ToolPackError, match=r"capacity_exceeded$"):
        service.install(review)

    assert store.install_calls == 1
    assert "research" not in store.profiles


@pytest.mark.parametrize(
    ("outcome", "fail_reconciliation", "category", "reconciled"),
    [
        ("before", False, "activation_failed", False),
        ("after", False, None, True),
        ("third", False, "activation_uncertain", False),
        ("before", True, "activation_uncertain", False),
    ],
)
def test_activation_strictly_reconciles_ambiguous_install_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    fail_reconciliation: bool,
    category: str | None,
    reconciled: bool,
) -> None:
    events: list[str] = []
    review = _review()
    store = _Store(
        events,
        review,
        install_outcome=outcome,
        fail_reconciliation_read=fail_reconciliation,
    )
    service, store, _references, _receipts, _events = _service(
        tmp_path,
        monkeypatch,
        review=review,
        store=store,
    )

    if category is None:
        result = service.install(review)
        assert result.reconciled is reconciled
        assert result.installed.profile_id == "research"
    else:
        with pytest.raises(ToolPackError) as error:
            service.install(review)
        assert error.value.category == category

    assert store.install_calls == 1


def test_import_install_is_unbound_and_preserves_existing_workspace_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    permission_store = MCPPermissionStore(tmp_path / "permissions.json")
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
    )
    registry.create_workspace(
        workspace_id="w-default",
        name="Default policy",
        assistant_defaults=WorkspaceAssistantDefaults(
            assistant_id="persona-default",
            tool_policy_profile_id="default",
        ),
    )
    registry.create_workspace(
        workspace_id="w-local",
        name="Local policy",
        assistant_defaults=WorkspaceAssistantDefaults(
            assistant_id="persona-local",
        ),
    )

    def references_profile(profile_id: str) -> bool:
        return any(
            record.assistant_defaults is not None
            and record.assistant_defaults.tool_policy_profile_id == profile_id
            for record in registry.list_workspaces(include_archived=True)
        )

    inventory = _inventory()
    monkeypatch.setattr(
        activation_module,
        "capture_v1_inventory",
        lambda _registry: inventory,
    )
    review = replace(
        _review(),
        store_generation=permission_store.read_snapshot_strict().generation,
    )
    service = ToolPackActivationService(
        permission_store=permission_store,
        inventory=inventory,
        importer=_Importer(review, []),
        reference_checker=references_profile,
        receipt_store=ToolPackReceiptStore(tmp_path / "receipts"),
        now=lambda: _NOW,
    )
    tool = _tool(server_key="future:server", name="future")
    before_defaults = tuple(
        (record.workspace_id, record.assistant_defaults)
        for record in registry.list_workspaces(include_archived=True)
    )
    before_payload = permission_store.read_snapshot_strict().payload
    before_effective = (
        resolve_effective_state(before_payload, tool, profile_id="default"),
        resolve_effective_state(before_payload, tool, profile_id="default"),
    )

    result = service.install(review)

    after_defaults = tuple(
        (record.workspace_id, record.assistant_defaults)
        for record in registry.list_workspaces(include_archived=True)
    )
    after_payload = permission_store.read_snapshot_strict().payload
    after_effective = (
        resolve_effective_state(after_payload, tool, profile_id="default"),
        resolve_effective_state(after_payload, tool, profile_id="default"),
    )
    lifecycle = after_payload["profiles"]["research"]["tool_pack_lifecycle"]
    assert after_defaults == before_defaults
    assert after_effective == before_effective
    assert references_profile("research") is False
    assert result.installed.profile_id == "research"
    assert lifecycle["first_bind_confirmation_required"] is True
