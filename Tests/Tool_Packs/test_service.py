"""Application facade tests for portable Tool Packs."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import MCPPermissionStore, definition_hash
from tldw_chatbook.Tool_Packs import service as service_module
from tldw_chatbook.Tool_Packs import export as export_module
from tldw_chatbook.Tool_Packs.binding import (
    ToolProfileLifecycleCoordinator,
    profile_policy_digest,
)
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventorySnapshot,
    PermissionInventoryTool,
)
from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.Tool_Packs.export import ToolPackExportReview, ToolPackExportService
from tldw_chatbook.Tool_Packs.publication import (
    CapturedToolPackDestination,
    publish_tool_pack,
)
from tldw_chatbook.Tool_Packs.receipt_store import (
    RECEIPT_SCHEMA,
    ToolPackReceipt,
    ToolPackReceiptStore,
)
from tldw_chatbook.Tool_Packs.service import ToolPackService, ToolProfileListing
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults, WorkspaceRecord


NOW = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)


class Delegate:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[str] = []

    def call(self, name: str) -> object:
        self.calls.append(name)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    def capture(self, **_kwargs):
        return self.call("capture")

    def inspect_archive(self, *_args, **_kwargs):
        return self.call("inspect")

    def install(self, *_args, **_kwargs):
        return self.call("install")

    def review(self, *_args, **_kwargs):
        return self.call("review")

    def confirm(self, *_args, **_kwargs):
        return self.call("confirm")

    def remove(self, *_args, **_kwargs):
        return self.call("remove")

    def live_receipt_ids(self):
        return self.call("live")


class Registry:
    def __init__(self, records: tuple[WorkspaceRecord, ...] = ()) -> None:
        self.records = records
        self.guard = None

    def list_workspaces(self, *, include_archived: bool = False):
        return (
            self.records
            if include_archived
            else tuple(row for row in self.records if not row.archived)
        )

    def get_workspace(self, workspace_id: str):
        return next(
            (row for row in self.records if row.workspace_id == workspace_id), None
        )

    def attach_tool_profile_guard(self, guard):
        self.guard = guard


def tool(server_key: str, name: str) -> HubTool:
    return HubTool(
        server_key,
        "safe",
        "local",
        name,
        f"{name} description",
        {"type": "object"},
        (),
        False,
        True,
    )


def inventory() -> PermissionInventorySnapshot:
    tools = (tool("agent:builtin", "calculate"), tool("local:docs", "search"))
    return PermissionInventorySnapshot(
        tuple(
            PermissionInventoryTool(
                "builtin" if item.server_key == "agent:builtin" else "mcp",
                item,
                "f" * 64,
            )
            for item in tools
        ),
        (("builtin", "agent:builtin"), ("mcp", "local:docs")),
        (),
        "i" * 64,
    )


def receipt(store: ToolPackReceiptStore, profile_id: str):
    data = ToolPackReceipt(
        schema=RECEIPT_SCHEMA,
        kind="import",
        profile_id=profile_id,
        pack_digest="a" * 64,
        archive_digest="b" * 64,
        producer=("tests", "1"),
        imported_at="2026-09-01T12:00:00Z",
        reviewed_mappings=(),
        matched=(),
    ).to_bytes()
    with store.reserve(len(data)) as reservation:
        return reservation.commit(data)


def imported(handle, *, first_bind: bool = True) -> dict[str, object]:
    search = tool("local:docs", "search")
    profile: dict[str, object] = {
        "global_default": "deny",
        "servers": {
            "agent:builtin": {"default": "ask"},
            "local:docs": {
                "default": "ask",
                "tools": {
                    "search": {
                        "state": "allow",
                        "definition_hash": definition_hash(
                            search.description, search.input_schema
                        ),
                    }
                },
            },
        },
        "profile_kind": "tool_pack_imported",
    }
    profile["tool_pack_lifecycle"] = {
        "schema": "tldw.tool-pack-lifecycle/v1",
        "origin": "imported",
        "pack_digest": "a" * 64,
        "imported_at": "2026-09-01T12:00:00Z",
        "first_bind_confirmation_required": first_bind,
        "receipt_id": handle.receipt_id,
        "receipt_digest": handle.digest,
        "counts": {"matched": 1, "omitted": 0, "pending_deny": 0},
        "policy_digest": "0" * 64,
        "revision": 1,
    }
    profile["tool_pack_lifecycle"]["policy_digest"] = profile_policy_digest(profile)
    return profile


def tombstone(handle) -> dict[str, object]:
    profile: dict[str, object] = {
        "global_default": "deny",
        "servers": {"agent:builtin": {"default": "deny"}},
        "profile_kind": "tool_pack_tombstone",
    }
    profile["tool_pack_lifecycle"] = {
        "schema": "tldw.tool-pack-lifecycle/v1",
        "origin": "tombstone",
        "pack_digest": "a" * 64,
        "imported_at": "2026-09-01T12:00:00Z",
        "removed_at": "2026-09-02T12:00:00Z",
        "first_bind_confirmation_required": False,
        "receipt_id": handle.receipt_id,
        "receipt_digest": handle.digest,
        "policy_digest": "0" * 64,
        "revision": 2,
    }
    profile["tool_pack_lifecycle"]["policy_digest"] = profile_policy_digest(profile)
    return profile


def facade(tmp_path: Path, *, store=None, registry=None, activation=None, **deps):
    receipts = deps.pop("receipts", ToolPackReceiptStore(tmp_path / "receipts"))
    return ToolPackService(
        permission_store=store or MCPPermissionStore(tmp_path / "permissions.json"),
        inventory=object(),
        workspace_registry=registry or Registry(),
        receipt_store=receipts,
        exporter=deps.pop("exporter", Delegate(object())),
        importer=deps.pop("importer", Delegate(object())),
        activation=activation or Delegate(frozenset()),
        binding_guard=deps.pop("binding", Delegate(object())),
        removal=deps.pop("removal", Delegate(object())),
        publisher=deps.pop("publisher", lambda *_args, **_kwargs: object()),
        lifecycle=ToolProfileLifecycleCoordinator(),
        now=lambda: NOW,
    )


def test_facade_exposes_separate_review_and_commit_operations(tmp_path: Path) -> None:
    exporter, importer, activation = (
        Delegate("capture"),
        Delegate("inspect"),
        Delegate("install"),
    )
    binding, removal = Delegate("bind"), Delegate("remove")
    publications = []
    service = facade(
        tmp_path,
        exporter=exporter,
        importer=importer,
        activation=activation,
        binding=binding,
        removal=removal,
        publisher=lambda snapshot, destination, **kwargs: (
            publications.append((snapshot, destination, kwargs)) or "publish"
        ),
    )

    assert isinstance(service.lifecycle, ToolProfileLifecycleCoordinator)
    defaults = WorkspaceAssistantDefaults(
        assistant_id="p", tool_policy_profile_id="research"
    )

    assert (
        service.capture_export("default", display_name="D", suggested_id="d")
        == "capture"
    )
    assert (
        service.inspect_import(Path("x.tldw-tool-pack"), destination_id="research")
        == "inspect"
    )
    assert activation.calls == []
    assert service.import_unbound("review") == "install"
    assert service.review_first_bind("w", defaults, action="set") == "bind"
    assert service.confirm_first_bind("review") == "bind"
    assert service.remove_profile("research", expected_revision=1) == "remove"
    assert publications == []
    assert not hasattr(service, "set_tool_state")
    assert not hasattr(service, "edit_profile")


def test_real_capture_result_can_be_published_by_the_facade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MCPPermissionStore(tmp_path / "permissions.json")
    monkeypatch.setattr(
        export_module, "capture_v1_inventory", lambda _registry: inventory()
    )
    service = facade(
        tmp_path,
        store=store,
        exporter=ToolPackExportService(store, object()),
        publisher=publish_tool_pack,
    )
    review = service.capture_export(
        "default", display_name="Default", suggested_id="default"
    )
    destination = CapturedToolPackDestination.capture(
        tmp_path / "default.tldw-tool-pack"
    )

    assert type(review) is ToolPackExportReview
    result = service.publish_export(review, destination)

    assert result.committed is True
    assert destination.path.is_file()


def test_capture_export_fences_an_imported_profile_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts = ToolPackReceiptStore(tmp_path / "receipts")
    handle = receipt(receipts, "research")
    store = MCPPermissionStore(tmp_path / "permissions.json")
    payload = store.load()
    payload["profiles"]["research"] = imported(handle)
    store.save(payload)
    lifecycle = payload["profiles"]["research"]["tool_pack_lifecycle"]
    monkeypatch.setattr(
        export_module, "capture_v1_inventory", lambda _registry: inventory()
    )
    service = facade(
        tmp_path,
        store=store,
        receipts=receipts,
        exporter=ToolPackExportService(store, object()),
    )

    review = service.capture_export(
        "research",
        display_name="Research",
        suggested_id="research",
        expected_revision=lifecycle["revision"],
        expected_policy_digest=lifecycle["policy_digest"],
    )
    assert type(review) is ToolPackExportReview

    with pytest.raises(ToolPackError, match=r"profile_invalid$"):
        service.capture_export(
            "research",
            display_name="Research",
            suggested_id="research",
            expected_revision=lifecycle["revision"] + 1,
            expected_policy_digest=lifecycle["policy_digest"],
        )


def test_capture_export_fences_a_local_profile_policy_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MCPPermissionStore(tmp_path / "permissions.json")
    payload = store.load()
    captured_digest = profile_policy_digest(payload["profiles"]["default"])
    monkeypatch.setattr(
        export_module, "capture_v1_inventory", lambda _registry: inventory()
    )
    service = facade(
        tmp_path,
        store=store,
        exporter=ToolPackExportService(store, object()),
    )

    review = service.capture_export(
        "default",
        display_name="Default",
        suggested_id="default",
        expected_policy_digest=captured_digest,
    )
    assert type(review) is ToolPackExportReview

    store.set_global_default("deny")
    with pytest.raises(ToolPackError, match=r"profile_invalid$"):
        service.capture_export(
            "default",
            display_name="Default",
            suggested_id="default",
            expected_policy_digest=captured_digest,
        )


def test_publish_export_forwards_the_worker_cancellation_probe(tmp_path: Path) -> None:
    published: list[object] = []

    def publisher(_snapshot, _destination, **kwargs):
        published.append(kwargs["cancelled"])
        return "published"

    service = facade(tmp_path, publisher=publisher)
    review = ToolPackExportReview(
        SimpleNamespace(), "i" * 64, (), (), ()  # type: ignore[arg-type]
    )

    def probe() -> bool:
        return False

    assert (
        service.publish_export(review, object(), cancelled=probe) == "published"
    )
    assert published == [probe]


def test_publish_rejects_values_other_than_the_captured_review(tmp_path: Path) -> None:
    with pytest.raises(ToolPackError, match=r"publication_failed$"):
        facade(tmp_path).publish_export(object(), object())


def test_listing_is_immutable_current_and_hides_tombstones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = iter((bytes.fromhex("11" * 16), bytes.fromhex("22" * 16)))
    receipts = ToolPackReceiptStore(tmp_path / "receipts", _id_source=lambda: next(ids))
    linked, retired = receipt(receipts, "research"), receipt(receipts, "retired")
    store = MCPPermissionStore(tmp_path / "permissions.json")
    payload = store.load()
    broken = imported(linked)
    broken["tool_pack_lifecycle"]["policy_digest"] = "0" * 64
    bad_tombstone = tombstone(retired)
    bad_tombstone["tool_pack_lifecycle"]["policy_digest"] = "0" * 64
    payload["profiles"].update(
        {
            "bad-tombstone": bad_tombstone,
            "research": imported(linked),
            "retired": tombstone(retired),
            "ws-managed": {"global_default": "deny", "servers": {}},
            "broken": broken,
        }
    )
    store.save(payload)
    records = (
        WorkspaceRecord(
            "active",
            "Active",
            assistant_defaults=WorkspaceAssistantDefaults(
                assistant_id="p1", tool_policy_profile_id="research"
            ),
        ),
        WorkspaceRecord(
            "archived",
            "Archived",
            archived=True,
            assistant_defaults=WorkspaceAssistantDefaults(
                assistant_id="p2", tool_policy_profile_id="research"
            ),
        ),
    )
    monkeypatch.setattr(
        service_module, "capture_v1_inventory", lambda _registry: inventory()
    )
    service = facade(
        tmp_path, store=store, registry=Registry(records), receipts=receipts
    )

    listing = service.list_profiles()
    row = listing.by_id("research")
    assert isinstance(listing, ToolProfileListing)
    assert tuple(item.profile_id for item in listing.profiles) == (
        "bad-tombstone",
        "broken",
        "default",
        "research",
        "ws-managed",
    )
    assert listing.by_id("retired") is None
    assert row.origin == "imported"
    assert row.lifecycle_valid is True
    assert row.binding_state == "bound"
    assert row.first_bind_confirmation_required is True
    assert row.reference_counts == (1, 1)
    assert row.posture_counts == (1, 1, 0)
    assert row.receipt_health == "available"
    assert row.removal_eligible is False and row.removal_blocker == "referenced"
    assert listing.by_id("default").origin == "local"
    assert listing.by_id("default").policy_digest == profile_policy_digest(
        payload["profiles"]["default"]
    )
    assert listing.by_id("ws-managed").origin == "workspace-managed"
    assert listing.by_id("ws-managed").policy_digest == profile_policy_digest(
        payload["profiles"]["ws-managed"]
    )
    assert listing.by_id("broken").lifecycle_valid is False
    assert listing.by_id("bad-tombstone").origin == "imported"
    assert listing.by_id("bad-tombstone").lifecycle_valid is False
    with pytest.raises(FrozenInstanceError):
        row.origin = "local"


@pytest.mark.parametrize(
    ("failure", "category"),
    (
        ("authority", "authority_unavailable"),
        ("references", "references_unavailable"),
        ("inventory", "inventory_unavailable"),
    ),
)
def test_listing_failures_are_stable_and_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, category: str
) -> None:
    store, registry = MCPPermissionStore(tmp_path / "permissions.json"), Registry()
    if failure == "authority":
        store = SimpleNamespace(
            read_snapshot_strict=lambda: (_ for _ in ()).throw(
                RuntimeError("/private/path API_KEY=secret")
            )
        )
    if failure == "references":
        registry.list_workspaces = lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("credential /private/db")
        )
    monkeypatch.setattr(
        service_module,
        "capture_v1_inventory",
        (lambda _registry: (_ for _ in ()).throw(RuntimeError("command secret")))
        if failure == "inventory"
        else (lambda _registry: inventory()),
    )
    result = facade(tmp_path, store=store, registry=registry).list_profiles()
    assert result.profiles == () and result.unavailable_category == category
    assert (
        "private" not in repr(result).casefold()
        and "secret" not in repr(result).casefold()
    )


def test_missing_receipt_degrades_provenance_but_not_first_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MCPPermissionStore(tmp_path / "permissions.json")
    payload = store.load()
    payload["profiles"]["research"] = imported(
        SimpleNamespace(receipt_id="tp-" + "1" * 32, digest="2" * 64)
    )
    store.save(payload)
    monkeypatch.setattr(
        service_module, "capture_v1_inventory", lambda _registry: inventory()
    )
    row = facade(tmp_path, store=store).list_profiles().by_id("research")
    assert row.first_bind_confirmation_required is True
    assert row.receipt_health == "unavailable"
    assert row.removal_blocker == "receipt_unavailable"


def test_listing_applies_the_raw_shell_ask_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = tool("local:__local__", "shell_exec")
    shell_inventory = PermissionInventorySnapshot(
        (PermissionInventoryTool("mcp", shell, "f" * 64),),
        (("mcp", "local:__local__"),),
        (),
        "i" * 64,
    )
    store = MCPPermissionStore(tmp_path / "permissions.json")
    payload = store.load()
    payload["profiles"]["raw-shell"] = {
        "global_default": "deny",
        "servers": {
            "local:__local__": {
                "default": "deny",
                "tools": {
                    "shell_exec": {
                        "state": "allow",
                        "definition_hash": definition_hash(
                            shell.description, shell.input_schema
                        ),
                    }
                },
            }
        },
    }
    store.save(payload)
    monkeypatch.setattr(
        service_module, "capture_v1_inventory", lambda _registry: shell_inventory
    )

    row = facade(tmp_path, store=store).list_profiles().by_id("raw-shell")

    assert row is not None
    assert row.posture_counts == (0, 1, 0)


def test_compose_shares_lifecycle_reference_authority_and_receipt_root(
    tmp_path: Path,
) -> None:
    service = ToolPackService.compose(
        permission_store=MCPPermissionStore(tmp_path / "permissions.json"),
        inventory=object(),
        workspace_registry=Registry(),
        receipt_root=tmp_path / "receipts",
    )
    assert service.receipt_root == (tmp_path / "receipts").resolve()
    assert service.binding_guard.lifecycle is service._lifecycle
    assert service._activation._lifecycle is service._lifecycle
    assert service._removal._lifecycle is service._lifecycle
    assert service._importer._reference_checker is service._references
    assert service._activation._reference_checker is service._references
    assert service._removal._reference_checker is service._references


def test_reconcile_removes_only_expired_unprotected_regular_orphan(
    tmp_path: Path,
) -> None:
    ids = iter(bytes.fromhex(f"{number:02x}" * 16) for number in range(1, 7))
    receipts = ToolPackReceiptStore(tmp_path / "receipts", _id_source=lambda: next(ids))
    linked, retired, live, fresh, expired = (
        receipt(receipts, name)
        for name in ("linked", "retired", "live", "fresh", "expired")
    )
    corrupt_id = "tp-" + "e" * 32
    corrupt_path = receipts.root / corrupt_id
    corrupt_path.write_bytes(b"corrupt but referenced")
    corrupt_path.chmod(0o600)
    old = (NOW - timedelta(days=2)).timestamp()
    for path in (linked.path, retired.path, live.path, expired.path, corrupt_path):
        os.utime(path, (old, old))
    symlink = receipts.root / ("tp-" + "f" * 32)
    symlink.symlink_to(expired.path)
    (receipts.root / "unknown").write_text("x")
    store = MCPPermissionStore(tmp_path / "permissions.json")
    payload = store.load()
    payload["profiles"]["linked"] = imported(linked)
    payload["profiles"]["retired"] = tombstone(retired)
    payload["profiles"]["corrupt"] = imported(
        SimpleNamespace(receipt_id=corrupt_id, digest="9" * 64)
    )
    store.save(payload)
    service = facade(
        tmp_path,
        store=store,
        receipts=receipts,
        activation=Delegate(frozenset({live.receipt_id})),
    )
    result = service.reconcile_receipts()
    assert result.removed_ids == (expired.receipt_id,)
    assert result.unavailable_category is None
    assert linked.path.exists() and retired.path.exists()
    assert live.path.exists() and fresh.path.exists() and corrupt_path.exists()
    assert symlink.is_symlink() and (receipts.root / "unknown").exists()


def test_reconcile_failure_is_fail_safe_and_private(tmp_path: Path) -> None:
    ids = iter((bytes.fromhex("11" * 16),))
    receipts = ToolPackReceiptStore(tmp_path / "receipts", _id_source=lambda: next(ids))
    orphan = receipt(receipts, "orphan")
    old = (NOW - timedelta(days=2)).timestamp()
    os.utime(orphan.path, (old, old))
    registry = Registry()
    registry.list_workspaces = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("/private/db credential")
    )
    result = facade(tmp_path, registry=registry, receipts=receipts).reconcile_receipts()
    assert (
        result.removed_ids == ()
        and result.unavailable_category == "references_unavailable"
    )
    assert orphan.path.exists()


def test_reconcile_reports_a_stable_incomplete_result_at_the_entry_budget(
    tmp_path: Path,
) -> None:
    receipts = ToolPackReceiptStore(tmp_path / "receipts", max_reconcile_entries=2)
    orphan = receipt(receipts, "orphan")
    old = (NOW - timedelta(days=2)).timestamp()
    os.utime(orphan.path, (old, old))
    for index in range(2):
        (receipts.root / f"unknown-{index}").write_text("x")

    result = facade(tmp_path, receipts=receipts).reconcile_receipts()

    assert result.removed_ids == ()
    assert result.unavailable_category == "receipt_store_incomplete"
    assert orphan.path.exists()


@pytest.mark.parametrize(
    ("failure", "category"),
    (
        ("authority", "authority_unavailable"),
        ("live", "live_owners_unavailable"),
    ),
)
def test_other_reconcile_owner_capture_failures_reclaim_nothing(
    tmp_path: Path, failure: str, category: str
) -> None:
    ids = iter((bytes.fromhex("11" * 16),))
    receipts = ToolPackReceiptStore(tmp_path / "receipts", _id_source=lambda: next(ids))
    orphan = receipt(receipts, "orphan")
    old = (NOW - timedelta(days=2)).timestamp()
    os.utime(orphan.path, (old, old))
    store: object = MCPPermissionStore(tmp_path / "permissions.json")
    activation: object = Delegate(frozenset())
    if failure == "authority":
        store = SimpleNamespace(
            read_snapshot_strict=lambda: (_ for _ in ()).throw(
                RuntimeError("secret authority")
            )
        )
    else:
        activation = Delegate(RuntimeError("secret live owner"))

    result = facade(
        tmp_path, store=store, activation=activation, receipts=receipts
    ).reconcile_receipts()

    assert result.removed_ids == () and result.unavailable_category == category
    assert orphan.path.exists()


def test_unexpected_delegate_error_is_stable_and_private(tmp_path: Path) -> None:
    service = facade(
        tmp_path,
        exporter=Delegate(RuntimeError("/private/path --token=credential payload")),
    )
    with pytest.raises(ToolPackError) as caught:
        service.capture_export("default", display_name="x", suggested_id="x")
    assert str(caught.value) == "tool_pack.export.profile_invalid"
