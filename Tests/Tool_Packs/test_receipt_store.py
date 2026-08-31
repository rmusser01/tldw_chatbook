from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import stat

import pytest

from tldw_chatbook.Tool_Packs.contracts import ToolPackError, canonical_json_bytes
from tldw_chatbook.Tool_Packs.receipt_store import (
    ToolPackReceipt,
    ToolPackReceiptStore,
)


_ZERO_HASH = "0" * 64
_ONE_HASH = "1" * 64
_TWO_HASH = "2" * 64
_NOW = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)


def _identity(
    tool_name: str = "search", *, server_key: str = "local:docs"
) -> dict[str, str]:
    return {
        "authority": "mcp",
        "server_key": server_key,
        "tool_name": tool_name,
    }


def _import_payload(*, identities: int = 1) -> dict[str, object]:
    matched = [
        _identity(f"search-{index:04d}") for index in range(identities)
    ]
    return {
        "schema": "tldw.tool-pack-receipt/v1",
        "kind": "import",
        "profile_id": "research",
        "pack_digest": _ZERO_HASH,
        "archive_digest": _ONE_HASH,
        "producer": {"name": "tldw_chatbook", "version": "1.0.0"},
        "imported_at": "2026-08-31T12:00:00Z",
        "reviewed_mappings": [
            {
                "source_server_key": "remote:docs",
                "destination_server_key": "local:docs",
            }
        ],
        "matched": matched,
        "changed": [],
        "missing": [],
        "pending_deny": [],
        "omitted": [],
    }


def _receipt_bytes(*, identities: int = 1) -> bytes:
    return canonical_json_bytes(_import_payload(identities=identities))


def _store(
    root: Path,
    *,
    max_receipt_bytes: int = 16_384,
    max_total_bytes: int = 32_768,
    fault=None,
    ids: list[bytes] | None = None,
) -> ToolPackReceiptStore:
    id_values = iter(ids or [bytes.fromhex("ab" * 16)])
    return ToolPackReceiptStore(
        root,
        max_receipt_bytes=max_receipt_bytes,
        max_total_bytes=max_total_bytes,
        _fault=fault,
        _id_source=lambda: next(id_values),
    )


def _commit(store: ToolPackReceiptStore, data: bytes | None = None):
    receipt_bytes = data or _receipt_bytes()
    with store.reserve(len(receipt_bytes)) as reservation:
        return reservation.commit(receipt_bytes)


def test_receipt_union_strictly_round_trips_immutable_import_and_tombstone() -> None:
    imported = ToolPackReceipt.from_dict(_import_payload())
    compact_raw = {
        "schema": "tldw.tool-pack-receipt/v1",
        "kind": "compact_tombstone",
        "profile_id": "research",
        "pack_digest": _ZERO_HASH,
        "removed_at": "2026-08-31T13:00:00Z",
        "prior_receipt_digest": _TWO_HASH,
    }
    compact = ToolPackReceipt.from_dict(compact_raw)

    assert imported.to_dict() == _import_payload()
    assert imported.to_bytes() == _receipt_bytes()
    assert compact.to_dict() == compact_raw
    with pytest.raises(FrozenInstanceError):
        imported.kind = "compact_tombstone"  # type: ignore[misc]


def test_direct_receipt_construction_rejects_unsorted_mapping_alias() -> None:
    with pytest.raises(ToolPackError, match=r"payload_invalid$"):
        ToolPackReceipt(
            schema="tldw.tool-pack-receipt/v1",
            kind="import",
            profile_id="research",
            pack_digest=_ZERO_HASH,
            archive_digest=_ONE_HASH,
            producer=("tldw_chatbook", "1.0.0"),
            imported_at="2026-08-31T12:00:00Z",
            reviewed_mappings=(("z", "local:z"), ("a", "local:a")),
            matched=(("mcp", "local:docs", "search"),),
        )


def test_receipt_rejects_casefolded_identity_collision_across_groups() -> None:
    raw = _import_payload()
    raw["changed"] = [_identity("SEARCH-0000")]

    with pytest.raises(ToolPackError, match=r"payload_invalid$"):
        ToolPackReceipt.from_dict(raw)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("description",), "secret description"),
        (("input_schema",), {"type": "object"}),
        (("configuration",), {"endpoint": "https://secret.invalid"}),
        (("secret",), "token"),
        (("workspace_id",), "ws-private"),
        (("persona",), "private persona"),
        (("authority_state",), "allow"),
        (("matched", 0, "state"), "allow"),
        (("producer", "endpoint"), "https://secret.invalid"),
        (("reviewed_mappings", 0, "command"), "run-me"),
    ],
)
def test_receipt_rejects_privacy_prohibited_or_unknown_fields(path, value) -> None:
    raw = _import_payload()
    owner: object = raw
    for key in path[:-1]:
        owner = owner[key]  # type: ignore[index]
    owner[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.payload_invalid$"):
        ToolPackReceipt.from_dict(raw)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.pop("profile_id"),
        lambda raw: raw.update(schema="tldw.tool-pack-receipt/v2"),
        lambda raw: raw.update(kind="other"),
        lambda raw: raw.update(pack_digest="A" * 64),
        lambda raw: raw.update(imported_at="2026-08-31 12:00:00"),
        lambda raw: raw.update(matched=[_identity("z"), _identity("a")]),
        lambda raw: raw.update(matched=[_identity(), _identity()]),
        lambda raw: raw.update(reviewed_mappings=[]),
    ],
)
def test_receipt_rejects_missing_malformed_unsorted_or_mismatched_fields(mutation) -> None:
    raw = _import_payload()
    mutation(raw)

    with pytest.raises(ToolPackError):
        ToolPackReceipt.from_dict(raw)


def test_receipt_is_private_reserved_and_digest_authenticated(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    store = _store(root)

    with store.reserve(len(_receipt_bytes())) as reservation:
        handle = reservation.commit(_receipt_bytes())

    verified = store.read(handle.receipt_id, expected_digest=handle.digest)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(handle.path.stat().st_mode) == 0o600
    assert handle.receipt_id == "tp-" + "ab" * 16
    assert verified.digest == handle.digest
    assert verified.receipt.profile_id == "research"
    assert verified.handle == handle


def test_store_rejects_a_symlinked_receipt_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "receipts"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ToolPackError, match=r"activation_failed$"):
        _store(linked)


def test_read_rejects_receipt_with_relaxed_file_mode(tmp_path: Path) -> None:
    store = _store(tmp_path / "receipts")
    handle = _commit(store)
    handle.path.chmod(0o644)

    with pytest.raises(ToolPackError, match=r"payload_invalid$"):
        store.read(handle.receipt_id, expected_digest=handle.digest)


def test_capacity_counts_live_reservations_across_store_instances(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    first = _store(root, max_receipt_bytes=500, max_total_bytes=700)
    second = _store(
        root,
        max_receipt_bytes=500,
        max_total_bytes=700,
        ids=[bytes.fromhex("cd" * 16)],
    )

    reservation = first.reserve(400)
    with pytest.raises(ToolPackError, match=r"capacity_exceeded$"):
        second.reserve(301)
    reservation.release()
    reservation.release()
    second.reserve(301).release()


def test_capacity_enforces_projection_actual_and_committed_files(tmp_path: Path) -> None:
    data = _receipt_bytes()
    store = _store(
        tmp_path / "receipts",
        max_receipt_bytes=len(data),
        max_total_bytes=len(data) * 2,
        ids=[bytes.fromhex("aa" * 16), bytes.fromhex("bb" * 16)],
    )

    with pytest.raises(ToolPackError, match=r"capacity_exceeded$"):
        store.reserve(len(data) + 1)
    with store.reserve(len(data) - 1) as too_small:
        with pytest.raises(ToolPackError, match=r"capacity_exceeded$"):
            too_small.commit(data)
    _commit(store, data)
    _commit(store, data)
    with pytest.raises(ToolPackError, match=r"capacity_exceeded$"):
        store.reserve(1)


def test_authenticated_name_collision_retries_without_truncating_existing_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "receipts"
    root.mkdir(mode=0o700)
    collision = root / ("tp-" + "aa" * 16)
    collision.write_bytes(b"existing receipt must survive")
    store = _store(
        root,
        ids=[bytes.fromhex("aa" * 16), bytes.fromhex("bb" * 16)],
    )

    handle = _commit(store)

    assert collision.read_bytes() == b"existing receipt must survive"
    assert handle.receipt_id == "tp-" + "bb" * 16


def test_context_exception_before_authority_link_releases_capacity_idempotently(
    tmp_path: Path,
) -> None:
    data = _receipt_bytes()
    store = _store(
        tmp_path / "receipts",
        max_receipt_bytes=len(data),
        max_total_bytes=len(data) * 2,
        ids=[bytes.fromhex("aa" * 16), bytes.fromhex("bb" * 16)],
    )

    with pytest.raises(RuntimeError, match="authority link failed"):
        with store.reserve(len(data)) as reservation:
            reservation.commit(data)
            raise RuntimeError("authority link failed")

    with store.reserve(len(data)) as follow_up:
        follow_up.release()
        follow_up.release()


def test_failure_before_replace_cleans_private_residue_and_releases_capacity(
    tmp_path: Path,
) -> None:
    def fail(stage: str) -> None:
        if stage == "before_replace":
            raise OSError("injected")

    root = tmp_path / "receipts"
    unrelated = root / "tp-near-miss"
    store = _store(root, fault=fail)
    unrelated.write_text("untouched")

    with store.reserve(len(_receipt_bytes())) as reservation:
        with pytest.raises(ToolPackError, match=r"activation_failed$"):
            reservation.commit(_receipt_bytes())
        reservation.release()
        reservation.release()

    assert sorted(path.name for path in root.iterdir()) == ["tp-near-miss"]
    assert unrelated.read_text() == "untouched"
    store.reserve(len(_receipt_bytes())).release()


def test_failure_after_replace_reports_uncertain_and_keeps_visible_receipt(
    tmp_path: Path,
) -> None:
    def fail(stage: str) -> None:
        if stage == "after_replace":
            raise OSError("injected")

    root = tmp_path / "receipts"
    store = _store(root, fault=fail)
    digest = hashlib.sha256(_receipt_bytes()).hexdigest()

    with store.reserve(len(_receipt_bytes())) as reservation:
        with pytest.raises(ToolPackError, match=r"activation_uncertain$"):
            reservation.commit(_receipt_bytes())

    entries = list(root.iterdir())
    assert len(entries) == 1
    receipt_id = entries[0].name
    assert store.read(receipt_id, expected_digest=digest).digest == digest
    store.reserve(len(_receipt_bytes())).release()


@pytest.mark.parametrize(
    "receipt_id",
    ["../tp-" + "a" * 32, "tp-" + "A" * 32, "tp-" + "a" * 31, "other"],
)
def test_read_and_exists_reject_unauthenticated_names(
    tmp_path: Path, receipt_id: str
) -> None:
    store = _store(tmp_path / "receipts")

    with pytest.raises(ToolPackError, match=r"payload_invalid$"):
        store.read(receipt_id, expected_digest=_ZERO_HASH)
    with pytest.raises(ToolPackError, match=r"payload_invalid$"):
        store.exists(receipt_id)


def test_read_rejects_digest_mismatch_noncanonical_bytes_and_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "receipts"
    store = _store(root)
    handle = _commit(store)

    with pytest.raises(ToolPackError, match=r"payload_invalid$"):
        store.read(handle.receipt_id, expected_digest=_ZERO_HASH)

    noncanonical = root / ("tp-" + "cd" * 16)
    noncanonical.write_bytes(_receipt_bytes().replace(b'"archive_digest"', b' "archive_digest"'))
    noncanonical.chmod(0o600)
    with pytest.raises(ToolPackError, match=r"payload_invalid$"):
        store.read(
            noncanonical.name,
            expected_digest=hashlib.sha256(noncanonical.read_bytes()).hexdigest(),
        )

    symlink = root / ("tp-" + "ef" * 16)
    symlink.symlink_to(handle.path)
    with pytest.raises(ToolPackError, match=r"payload_invalid$"):
        store.read(symlink.name, expected_digest=handle.digest)


def test_reconcile_removes_only_old_authenticated_unowned_regular_receipts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "receipts"
    store = _store(root)
    old = _NOW - timedelta(hours=24, seconds=1)
    recent = _NOW - timedelta(hours=23, minutes=59)
    names = {
        "orphan": "tp-" + "01" * 16,
        "linked": "tp-" + "02" * 16,
        "live": "tp-" + "03" * 16,
        "recent": "tp-" + "04" * 16,
        "corrupt_linked": "tp-" + "05" * 16,
    }
    for label, name in names.items():
        path = root / name
        path.write_bytes(_receipt_bytes() if label != "corrupt_linked" else b"corrupt")
        os.utime(path, (old.timestamp(), old.timestamp()))
    os.utime(root / names["recent"], (recent.timestamp(), recent.timestamp()))
    unknown = root / "tp-unknown"
    unknown.write_text("unknown")
    os.utime(unknown, (old.timestamp(), old.timestamp()))
    directory = root / ("tp-" + "06" * 16)
    directory.mkdir()
    symlink = root / ("tp-" + "07" * 16)
    symlink.symlink_to(root / names["orphan"])

    removed = store.reconcile_orphans(
        {names["linked"], names["corrupt_linked"]},
        {names["live"]},
        now=_NOW,
    )

    assert removed == (names["orphan"],)
    assert not (root / names["orphan"]).exists()
    for name in set(names.values()) - {names["orphan"]}:
        assert (root / name).exists()
    assert unknown.exists() and directory.is_dir() and symlink.is_symlink()


def test_reconcile_grace_boundary_is_exactly_twenty_four_hours(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    store = _store(root)
    before = root / ("tp-" + "11" * 16)
    exact = root / ("tp-" + "12" * 16)
    before.write_bytes(_receipt_bytes())
    exact.write_bytes(_receipt_bytes())
    os.utime(before, ((_NOW - timedelta(hours=24) + timedelta(microseconds=1)).timestamp(),) * 2)
    os.utime(exact, ((_NOW - timedelta(hours=24)).timestamp(),) * 2)

    assert store.reconcile_orphans(set(), set(), now=_NOW) == (exact.name,)
    assert before.exists() and not exact.exists()


def test_compaction_validates_source_lineage_and_is_smaller(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    store = _store(
        root,
        max_receipt_bytes=64_000,
        max_total_bytes=128_000,
        ids=[bytes.fromhex("aa" * 16), bytes.fromhex("bb" * 16)],
    )
    source = _commit(store, _receipt_bytes(identities=100))

    compact = store.write_compact_tombstone(source, profile_id="research")
    verified = store.read(compact.receipt_id, expected_digest=compact.digest)

    assert compact.size < source.size
    assert verified.receipt.to_dict() == {
        "schema": "tldw.tool-pack-receipt/v1",
        "kind": "compact_tombstone",
        "profile_id": "research",
        "pack_digest": _ZERO_HASH,
        "removed_at": verified.receipt.removed_at,
        "prior_receipt_digest": source.digest,
    }
    with pytest.raises(ToolPackError, match=r"non_removable$"):
        store.write_compact_tombstone(source, profile_id="different")


def test_compaction_rejects_tampered_or_already_compact_source(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    store = _store(
        root,
        max_receipt_bytes=64_000,
        max_total_bytes=128_000,
        ids=[
            bytes.fromhex("aa" * 16),
            bytes.fromhex("bb" * 16),
            bytes.fromhex("cc" * 16),
        ],
    )
    source = _commit(store, _receipt_bytes(identities=10))
    compact = store.write_compact_tombstone(source, profile_id="research")

    source.path.write_bytes(b"tampered")
    with pytest.raises(ToolPackError, match=r"payload_invalid$"):
        store.write_compact_tombstone(source, profile_id="research")
    with pytest.raises(ToolPackError, match=r"non_removable$"):
        store.write_compact_tombstone(compact, profile_id="research")
