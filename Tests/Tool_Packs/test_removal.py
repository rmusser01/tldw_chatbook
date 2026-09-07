"""Fail-closed removal and runtime-lease tests for imported Tool profiles."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import threading

import pytest

from tldw_chatbook.MCP.permission_store import (
    MCPPermissionStore,
    resolve_builtin_state,
    resolve_effective_state_by_key,
)
from tldw_chatbook.Tool_Packs.binding import (
    ProfileMutationError,
    ToolProfileBindingGuard,
    ToolProfileLifecycleCoordinator,
    profile_policy_digest,
)
from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.Tool_Packs.receipt_store import (
    RECEIPT_SCHEMA,
    ToolPackReceipt,
    ToolPackReceiptStore,
)
from tldw_chatbook.Tool_Packs.removal import ToolProfileRemovalService
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults


_MAX_STORE_BYTES = 8 * 1024 * 1024


class _References:
    def __init__(
        self,
        referenced: bool = False,
        records: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.referenced = referenced
        self.records = list(records)
        self.calls: list[tuple[str, bool]] = []

    def references_profile(self, profile_id: str, *, include_archived: bool) -> bool:
        self.calls.append((profile_id, include_archived))
        return self.referenced or any(
            candidate == profile_id and (kind != "archived" or include_archived)
            for kind, candidate in self.records
        )

    def add(self, kind: str, profile_id: str) -> None:
        self.records.append((kind, profile_id))


class _BlockingReferences(_References):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def references_profile(self, profile_id: str, *, include_archived: bool) -> bool:
        self.calls.append((profile_id, include_archived))
        self.entered.set()
        assert self.release.wait(2)
        return False


def _binding_guard(
    store: MCPPermissionStore,
    lifecycle: ToolProfileLifecycleCoordinator,
) -> ToolProfileBindingGuard:
    return ToolProfileBindingGuard(
        permission_store=store,
        inventory=object(),
        workspace_defaults_reader=lambda _workspace_id: None,
        lifecycle=lifecycle,
    )


def _intended_defaults() -> WorkspaceAssistantDefaults:
    return WorkspaceAssistantDefaults(
        assistant_id="persona-1",
        tool_policy_profile_id="research",
    )


def _import_receipt(store: ToolPackReceiptStore, profile_id: str = "research"):
    receipt = ToolPackReceipt(
        schema=RECEIPT_SCHEMA,
        kind="import",
        profile_id=profile_id,
        pack_digest="a" * 64,
        archive_digest="b" * 64,
        producer=("test-producer", "1"),
        imported_at="2026-09-01T12:00:00Z",
        reviewed_mappings=(),
        matched=(("mcp", "local:docs", "search"),),
    )
    data = receipt.to_bytes()
    with store.reserve(len(data)) as reservation:
        return reservation.commit(data)


def _imported_profile(receipt) -> dict:
    profile = {
        "global_default": "ask",
        "servers": {
            "agent:builtin": {"default": "ask"},
            "local:docs": {
                "default": "deny",
                "tools": {
                    "search": {
                        "state": "allow",
                        "definition_hash": "c" * 64,
                    }
                },
            },
        },
        "profile_kind": "tool_pack_imported",
        "tool_pack_lifecycle": {
            "schema": "tldw.tool-pack-lifecycle/v1",
            "origin": "imported",
            "pack_digest": "a" * 64,
            "imported_at": "2026-09-01T12:00:00Z",
            "first_bind_confirmation_required": True,
            "receipt_id": receipt.receipt_id,
            "receipt_digest": receipt.digest,
            "counts": {"matched": 1, "omitted": 0, "pending_deny": 0},
            "policy_digest": "0" * 64,
            "revision": 1,
        },
    }
    profile["tool_pack_lifecycle"]["policy_digest"] = profile_policy_digest(profile)
    return profile


def _installed_service(
    tmp_path: Path,
    *,
    references: object | None = None,
    lifecycle: ToolProfileLifecycleCoordinator | None = None,
    profile_id: str = "research",
):
    ids = iter((bytes.fromhex("11" * 16), bytes.fromhex("22" * 16)))
    receipts = ToolPackReceiptStore(
        tmp_path / "receipts",
        _id_source=lambda: next(ids),
    )
    import_receipt = _import_receipt(receipts, profile_id)
    profile = _imported_profile(import_receipt)
    permission_store = MCPPermissionStore(tmp_path / "permissions.json")
    permission_store.install_profile_if_absent(
        profile_id,
        profile,
        expected_generation=permission_store.read_snapshot_strict().generation,
        max_profiles=128,
        max_store_bytes=_MAX_STORE_BYTES,
    )
    payload = permission_store.load()
    lifecycle_payload = payload["profiles"][profile_id]["tool_pack_lifecycle"]
    lifecycle_payload["first_bind_confirmation_required"] = False
    lifecycle_payload["revision"] = 3
    permission_store.save(payload)
    reference_checker = references or _References()
    lifecycle = lifecycle or ToolProfileLifecycleCoordinator()
    service = ToolProfileRemovalService(
        permission_store=permission_store,
        receipt_store=receipts,
        reference_checker=reference_checker,
        lifecycle=lifecycle,
    )
    return (
        service,
        permission_store,
        receipts,
        reference_checker,
        lifecycle,
        import_receipt,
    )


def test_exact_profile_lease_is_immutable_and_counted_until_exit() -> None:
    lifecycle = ToolProfileLifecycleCoordinator()

    with lifecycle.lease("research") as lease:
        assert lease.profile_id == "research"
        assert lease.lease_id
        assert lifecycle.active_lease_count("research") == 1
        assert lifecycle.active_lease_count("other") == 0
        with pytest.raises(FrozenInstanceError):
            lease.profile_id = "other"  # type: ignore[misc]

    assert lifecycle.active_lease_count("research") == 0


@pytest.mark.parametrize("profile_id", [None, "", " research ", 1, True])
def test_exact_profile_lease_rejects_invalid_profile_ids(profile_id: object) -> None:
    lifecycle = ToolProfileLifecycleCoordinator()

    with pytest.raises(ValueError, match="profile_id"):
        with lifecycle.lease(profile_id):  # type: ignore[arg-type]
            pass


def test_remove_replaces_imported_authority_with_exact_permanent_deny(
    tmp_path: Path,
) -> None:
    service, store, receipts, references, _lifecycle, old_receipt = _installed_service(
        tmp_path
    )
    before_count = len(store.read_snapshot_strict().payload["profiles"])

    result = service.remove("research", expected_revision=3)

    snapshot = store.read_snapshot_strict()
    profile = snapshot.payload["profiles"]["research"]
    lifecycle = profile["tool_pack_lifecycle"]
    assert result.tombstone.profile_id == "research"
    assert result.tombstone.profile_kind == "tool_pack_tombstone"
    assert result.tombstone.revision == 4
    assert result.tombstone.policy_digest == profile_policy_digest(profile)
    assert result.store_generation == snapshot.generation
    assert result.reconciled is False
    assert profile["global_default"] == "deny"
    assert profile["servers"] == {"agent:builtin": {"default": "deny"}}
    assert profile["profile_kind"] == "tool_pack_tombstone"
    assert lifecycle["origin"] == "tombstone"
    assert lifecycle["pack_digest"] == "a" * 64
    assert lifecycle["imported_at"] == "2026-09-01T12:00:00Z"
    assert lifecycle["first_bind_confirmation_required"] is False
    assert lifecycle["receipt_id"] == result.tombstone.receipt_id
    assert lifecycle["receipt_digest"] == result.tombstone.receipt_digest
    assert lifecycle["revision"] == 4
    assert set(lifecycle) == {
        "schema",
        "origin",
        "pack_digest",
        "imported_at",
        "removed_at",
        "first_bind_confirmation_required",
        "receipt_id",
        "receipt_digest",
        "policy_digest",
        "revision",
    }
    persisted_profile = store.load()["profiles"]["research"]
    assert not any(
        state in json.dumps(persisted_profile, sort_keys=True)
        for state in ('"allow"', '"ask"')
    )
    assert len(snapshot.payload["profiles"]) == before_count
    with pytest.raises(ProfileMutationError, match="profile_exists"):
        store.install_profile_if_absent(
            "research",
            _imported_profile(old_receipt),
            expected_generation=snapshot.generation,
            max_profiles=128,
            max_store_bytes=_MAX_STORE_BYTES,
        )
    assert receipts.exists(old_receipt.receipt_id)
    assert receipts.exists(result.tombstone.receipt_id)
    compact = receipts.read(
        result.tombstone.receipt_id,
        expected_digest=result.tombstone.receipt_digest,
    ).receipt
    assert compact.kind == "compact_tombstone"
    assert compact.pack_digest == "a" * 64
    assert compact.prior_receipt_digest == old_receipt.digest
    assert compact.removed_at == lifecycle["removed_at"]
    assert references.calls == [("research", True)]
    assert (
        resolve_effective_state_by_key(
            store.load(), "local:any", "future", profile_id="research"
        ).origin
        == "tombstone"
    )
    assert (
        resolve_builtin_state(
            store.load(),
            type(
                "Tool",
                (),
                {
                    "server_key": "agent:builtin",
                    "name": "future",
                    "description": "future",
                    "input_schema": {"type": "object"},
                    "tags": (),
                },
            )(),
            profile_id="research",
        ).state
        == "deny"
    )


@pytest.mark.parametrize("reference_kind", ["active", "archived", "dangling"])
def test_any_active_archived_or_dangling_reference_blocks_removal(
    tmp_path: Path, reference_kind: str
) -> None:
    references = _References(records=((reference_kind, "research"),))
    service, store, receipts, _references, _lifecycle, old_receipt = _installed_service(
        tmp_path, references=references
    )
    before = store.path.read_bytes()

    with pytest.raises(ToolPackError, match=r"referenced$"):
        service.remove("research", expected_revision=3)

    assert references.calls == [("research", True)]
    assert store.path.read_bytes() == before
    assert receipts.exists(old_receipt.receipt_id)
    assert len(tuple(receipts.root.iterdir())) == 1


@pytest.mark.parametrize("bad_result", [None, 1, "yes"])
def test_reference_authority_errors_and_non_bool_results_fail_closed(
    tmp_path: Path, bad_result: object
) -> None:
    def references(_profile_id: str):
        if bad_result is None:
            raise OSError("workspace registry unavailable")
        return bad_result

    service, *_ = _installed_service(tmp_path, references=references)

    with pytest.raises(ToolPackError, match=r"referenced$"):
        service.remove("research", expected_revision=3)


def test_exact_profile_lease_blocks_removal_but_other_profile_does_not(
    tmp_path: Path,
) -> None:
    lifecycle = ToolProfileLifecycleCoordinator()
    service, *_ = _installed_service(tmp_path, lifecycle=lifecycle)

    with lifecycle.lease("other"):
        assert service.remove("research", expected_revision=3).tombstone.revision == 4

    service, *_ = _installed_service(tmp_path / "second", lifecycle=lifecycle)
    with lifecycle.lease("research"):
        with pytest.raises(ToolPackError, match=r"in_use$"):
            service.remove("research", expected_revision=3)


def test_binding_wins_removal_race_and_removal_observes_current_reference(
    tmp_path: Path,
) -> None:
    lifecycle = ToolProfileLifecycleCoordinator()
    references = _References()
    service, store, *_ = _installed_service(
        tmp_path,
        references=references,
        lifecycle=lifecycle,
    )
    guard = _binding_guard(store, lifecycle)
    binding_entered = threading.Event()
    release_binding = threading.Event()
    removal_attempted = threading.Event()
    removal_finished = threading.Event()
    binding_errors: list[BaseException] = []
    removal_errors: list[BaseException] = []

    def bind() -> None:
        try:
            with guard.mutation_scope(
                action="set",
                workspace_id="w-1",
                current_defaults=None,
                intended_defaults=_intended_defaults(),
                confirmation_token=None,
            ):
                references.add("active", "research")
                binding_entered.set()
                assert release_binding.wait(2)
        except BaseException as error:  # pragma: no cover - asserted below
            binding_errors.append(error)

    def remove() -> None:
        removal_attempted.set()
        try:
            service.remove("research", expected_revision=3)
        except BaseException as error:  # pragma: no branch - asserted below
            removal_errors.append(error)
        finally:
            removal_finished.set()

    binding_thread = threading.Thread(target=bind)
    binding_thread.start()
    assert binding_entered.wait(2)
    removal_thread = threading.Thread(target=remove)
    removal_thread.start()
    assert removal_attempted.wait(2)
    assert not removal_finished.is_set()

    release_binding.set()
    binding_thread.join(2)
    removal_thread.join(2)

    assert not binding_thread.is_alive()
    assert not removal_thread.is_alive()
    assert binding_errors == []
    assert len(removal_errors) == 1
    assert isinstance(removal_errors[0], ToolPackError)
    assert removal_errors[0].category == "referenced"
    assert references.calls == [("research", True)]


def test_removal_wins_binding_race_and_binding_observes_tombstone(
    tmp_path: Path,
) -> None:
    lifecycle = ToolProfileLifecycleCoordinator()
    references = _BlockingReferences()
    service, store, *_ = _installed_service(
        tmp_path,
        references=references,
        lifecycle=lifecycle,
    )
    guard = _binding_guard(store, lifecycle)
    binding_attempted = threading.Event()
    binding_finished = threading.Event()
    removal_results = []
    removal_errors: list[BaseException] = []
    binding_errors: list[BaseException] = []

    def remove() -> None:
        try:
            removal_results.append(service.remove("research", expected_revision=3))
        except BaseException as error:  # pragma: no cover - asserted below
            removal_errors.append(error)

    def bind() -> None:
        binding_attempted.set()
        try:
            with guard.mutation_scope(
                action="set",
                workspace_id="w-1",
                current_defaults=None,
                intended_defaults=_intended_defaults(),
                confirmation_token=None,
            ):
                pass
        except BaseException as error:  # pragma: no branch - asserted below
            binding_errors.append(error)
        finally:
            binding_finished.set()

    removal_thread = threading.Thread(target=remove)
    removal_thread.start()
    assert references.entered.wait(2)
    binding_thread = threading.Thread(target=bind)
    binding_thread.start()
    assert binding_attempted.wait(2)
    assert not binding_finished.is_set()

    references.release.set()
    removal_thread.join(2)
    binding_thread.join(2)

    assert not removal_thread.is_alive()
    assert not binding_thread.is_alive()
    assert removal_errors == []
    assert len(removal_results) == 1
    assert len(binding_errors) == 1
    assert isinstance(binding_errors[0], ToolPackError)
    assert binding_errors[0].operation == "bind"
    assert binding_errors[0].category == "lifecycle_invalid"


def test_removal_wins_lease_race_and_late_runtime_gate_observes_tombstone(
    tmp_path: Path,
) -> None:
    lifecycle = ToolProfileLifecycleCoordinator()
    references = _BlockingReferences()
    service, store, *_ = _installed_service(
        tmp_path,
        references=references,
        lifecycle=lifecycle,
    )
    lease_attempted = threading.Event()
    lease_acquired = threading.Event()
    removal_results = []
    removal_errors: list[BaseException] = []
    lease_states = []

    def remove() -> None:
        try:
            removal_results.append(service.remove("research", expected_revision=3))
        except BaseException as error:  # pragma: no cover - asserted below
            removal_errors.append(error)

    def run_with_lease() -> None:
        lease_attempted.set()
        with lifecycle.lease("research"):
            lease_acquired.set()
            lease_states.append(
                resolve_effective_state_by_key(
                    store.load(),
                    "local:any",
                    "future",
                    profile_id="research",
                )
            )

    removal_thread = threading.Thread(target=remove)
    removal_thread.start()
    assert references.entered.wait(2)
    lease_thread = threading.Thread(target=run_with_lease)
    lease_thread.start()
    assert lease_attempted.wait(2)
    assert not lease_acquired.is_set()

    references.release.set()
    removal_thread.join(2)
    lease_thread.join(2)

    assert not removal_thread.is_alive()
    assert not lease_thread.is_alive()
    assert removal_errors == []
    assert len(removal_results) == 1
    assert [(state.state, state.origin) for state in lease_states] == [
        ("deny", "tombstone")
    ]
    assert lifecycle.active_lease_count("research") == 0


@pytest.mark.parametrize("profile_id", ["default", "ws-w-1", "legacy", "invalid"])
def test_non_imported_profiles_are_not_removable(
    tmp_path: Path, profile_id: str
) -> None:
    service, store, *_ = _installed_service(tmp_path)
    if profile_id != "default":
        payload = store.load()
        payload["profiles"][profile_id] = (
            {
                "profile_kind": "tool_pack_imported",
                "servers": {},
            }
            if profile_id == "invalid"
            else {"servers": {}}
        )
        store.save(payload)

    with pytest.raises(ToolPackError, match=r"non_removable$"):
        service.remove(profile_id, expected_revision=1)


def test_stale_revision_and_already_removed_profile_fail_closed(tmp_path: Path) -> None:
    service, *_ = _installed_service(tmp_path)

    with pytest.raises(ToolPackError, match=r"stale$"):
        service.remove("research", expected_revision=2)

    service.remove("research", expected_revision=3)
    with pytest.raises(ToolPackError, match=r"non_removable$"):
        service.remove("research", expected_revision=4)


class _AmbiguousStore:
    def __init__(self, real: MCPPermissionStore, outcome: str) -> None:
        self.real = real
        self.outcome = outcome

    @contextmanager
    def mutation_fence(self) -> Iterator[None]:
        with self.real.mutation_fence():
            yield

    def read_snapshot_strict(self):
        return self.real.read_snapshot_strict()

    def replace_profile_with_tombstone(self, profile_id: str, profile: dict, **kwargs):
        if self.outcome == "prior":
            raise OSError("replace failed before commit")
        self.real.replace_profile_with_tombstone(profile_id, profile, **kwargs)
        if self.outcome == "tombstone":
            raise OSError("replace committed before error")
        payload = self.real.load()
        payload["profiles"][profile_id] = {"global_default": "deny", "servers": {}}
        self.real.save(payload)
        raise OSError("replace left an unexpected state")


@pytest.mark.parametrize(
    ("outcome", "category", "reconciled"),
    [
        ("prior", "non_removable", False),
        ("tombstone", None, True),
        ("third", "outcome_uncertain", False),
    ],
)
def test_replacement_outcomes_are_strictly_reconciled_and_keep_receipts(
    tmp_path: Path,
    outcome: str,
    category: str | None,
    reconciled: bool,
) -> None:
    service, real_store, receipts, references, lifecycle, old_receipt = (
        _installed_service(tmp_path)
    )
    service = ToolProfileRemovalService(
        permission_store=_AmbiguousStore(real_store, outcome),
        receipt_store=receipts,
        reference_checker=references,
        lifecycle=lifecycle,
    )

    if category is None:
        result = service.remove("research", expected_revision=3)
        assert result.reconciled is reconciled
        compact_id = result.tombstone.receipt_id
    else:
        with pytest.raises(ToolPackError, match=rf"{category}$"):
            service.remove("research", expected_revision=3)
        compact_ids = [
            path.name
            for path in receipts.root.iterdir()
            if path.name != old_receipt.receipt_id
        ]
        assert len(compact_ids) == 1
        compact_id = compact_ids[0]

    assert receipts.exists(old_receipt.receipt_id)
    assert receipts.exists(compact_id)
