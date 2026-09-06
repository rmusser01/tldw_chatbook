"""Behavioral tests for immutable Canvas runtime-profile admission."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from tldw_chatbook.Canvas.profiles import (
    ProfileRecord,
    load_profile_snapshot,
    resolve_profile,
    runtime_snapshot_id,
)

STATIC = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "Canvas" / "static"


def test_removing_diagrams_retains_parent_profile(profile_snapshot):
    result = resolve_profile(
        profile_snapshot,
        operation="update",
        parent_profile="canvas-v2-mermaid-1",
        has_diagrams=False,
    )

    assert result.profile_id == "canvas-v2-mermaid-1"
    assert result.executable


@pytest.mark.parametrize(
    ("operation", "parent", "has_diagrams", "expected"),
    [
        ("create", None, False, "canvas-v1"),
        ("create", None, True, "canvas-v2-mermaid-1"),
        ("update", "canvas-v1", False, "canvas-v1"),
        ("update", "canvas-v1", True, "canvas-v2-mermaid-1"),
        ("update", "canvas-v2-mermaid-1", False, "canvas-v2-mermaid-1"),
        ("update", "canvas-v2-mermaid-1", True, "canvas-v2-mermaid-1"),
        ("rename", "canvas-v1", True, "canvas-v1"),
        ("load", "canvas-v2-mermaid-1", False, "canvas-v2-mermaid-1"),
    ],
)
def test_profile_selection_uses_exact_parent_or_admitted_default(
    profile_snapshot, operation, parent, has_diagrams, expected
):
    result = resolve_profile(
        profile_snapshot,
        operation=operation,
        parent_profile=parent,
        has_diagrams=has_diagrams,
    )

    assert result.profile_id == expected
    assert result.executable is True
    assert result.reason is None


@pytest.mark.parametrize("operation", ["update", "rename", "load"])
@pytest.mark.parametrize(
    ("profile_id", "reason"),
    [
        ("canvas-retired", "profile-retired"),
        ("canvas-revoked", "profile-revoked"),
        ("canvas-unknown", "profile-unavailable"),
    ],
)
def test_identity_preserving_operations_keep_unavailable_profile_source_only(
    profile_snapshot, operation, profile_id, reason
):
    records = profile_snapshot.profiles
    if profile_id != "canvas-unknown":
        records += (ProfileRecord(profile_id, "e" * 64, False, reason, 0),)
    snapshot = replace(profile_snapshot, profiles=records)

    result = resolve_profile(
        snapshot,
        operation=operation,
        parent_profile=profile_id,
        has_diagrams=False,
    )

    assert result.profile_id == profile_id
    assert result.executable is False
    assert result.reason == reason


def test_missing_admitted_diagram_default_does_not_fall_back_to_v1(
    profile_snapshot,
):
    snapshot = replace(profile_snapshot, default_diagram_profile=None)

    result = resolve_profile(
        snapshot, operation="create", parent_profile=None, has_diagrams=True
    )

    assert result.profile_id == "profile-unavailable"
    assert result.executable is False
    assert result.reason == "profile-unavailable"


@pytest.mark.parametrize(
    ("operation", "parent"),
    [("update", None), ("rename", None), ("load", None), ("delete", "canvas-v1")],
)
def test_invalid_operation_or_missing_required_parent_is_rejected(
    profile_snapshot, operation, parent
):
    with pytest.raises(ValueError):
        resolve_profile(
            profile_snapshot,
            operation=operation,
            parent_profile=parent,
            has_diagrams=False,
        )


def test_candidate_profile_identity_is_validated_before_resolution(profile_snapshot):
    with pytest.raises(ValueError):
        resolve_profile(
            profile_snapshot,
            operation="load",
            parent_profile="../canvas-v1",
            has_diagrams=False,
        )


def test_snapshot_identity_is_canonical_and_covers_execution_policy(profile_snapshot):
    reordered = replace(
        profile_snapshot, profiles=tuple(reversed(profile_snapshot.profiles))
    )
    expected = hashlib.sha256(
        json.dumps(
            {
                "build_id": "a" * 64,
                "default_diagram_profile": "canvas-v2-mermaid-1",
                "policy_id": "b" * 64,
                "profiles": [
                    {
                        "executable": True,
                        "library_bytes": 0,
                        "manifest_sha256": "c" * 64,
                        "profile_id": "canvas-v1",
                        "reason": None,
                    },
                    {
                        "executable": True,
                        "library_bytes": 77817,
                        "manifest_sha256": "d" * 64,
                        "profile_id": "canvas-v2-mermaid-1",
                        "reason": None,
                    },
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert runtime_snapshot_id(profile_snapshot) == expected
    assert runtime_snapshot_id(reordered) == expected
    assert (
        runtime_snapshot_id(replace(profile_snapshot, policy_id="f" * 64)) != expected
    )


def test_profile_records_and_snapshots_are_immutable(profile_snapshot):
    with pytest.raises(FrozenInstanceError):
        profile_snapshot.default_diagram_profile = None
    with pytest.raises(FrozenInstanceError):
        profile_snapshot.profiles[0].executable = False


def _isolated_static(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from tldw_chatbook.Canvas import profiles, runtime_assets

    package_root = tmp_path / "Canvas"
    shutil.copytree(STATIC, package_root / "static")
    monkeypatch.setattr(profiles, "files", lambda _package: package_root)
    monkeypatch.setattr(runtime_assets, "files", lambda _package: package_root)
    return package_root / "static"


def test_packaged_snapshot_admits_only_v1_until_v2_is_qualified():
    snapshot = load_profile_snapshot()

    assert snapshot.default_diagram_profile is None
    assert [(record.profile_id, record.executable) for record in snapshot.profiles] == [
        ("canvas-v1", True)
    ]
    assert snapshot.profiles[0].library_bytes == 0


def test_owned_snapshot_does_not_reread_mutated_packaged_inputs(tmp_path, monkeypatch):
    static = _isolated_static(tmp_path, monkeypatch)
    snapshot = load_profile_snapshot()
    identity = runtime_snapshot_id(snapshot)

    static.joinpath("runtime-manifest.json").write_bytes(b"tampered")
    static.joinpath("profile-catalog.json").write_bytes(b"tampered")

    assert runtime_snapshot_id(snapshot) == identity
    assert resolve_profile(
        snapshot, operation="load", parent_profile="canvas-v1", has_diagrams=False
    ).executable


@pytest.mark.parametrize(
    "mutation",
    [
        "byte-tamper",
        "missing-manifest",
        "duplicate-key",
        "duplicate-id",
        "unknown-field",
        "id-reuse",
        "missing-contract",
        "bad-library-inventory",
    ],
)
def test_packaged_snapshot_rejects_untrusted_or_ambiguous_inputs(
    tmp_path, monkeypatch, mutation
):
    static = _isolated_static(tmp_path, monkeypatch)
    catalog_path = static / "profile-catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    manifest_path = static / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if mutation == "byte-tamper":
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    elif mutation == "missing-manifest":
        manifest_path.unlink()
    elif mutation == "duplicate-key":
        catalog_path.write_text(
            catalog_path.read_text(encoding="utf-8").replace(
                '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
            ),
            encoding="utf-8",
        )
    elif mutation == "duplicate-id":
        catalog["profiles"].append(catalog["profiles"][0])
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    elif mutation == "unknown-field":
        catalog["profiles"][0]["surprise"] = True
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    elif mutation == "id-reuse":
        catalog["profiles"][0]["profile_id"] = "canvas-v2"
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    elif mutation == "missing-contract":
        del manifest["profile_contract"]["unicode"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        catalog["profiles"][0]["manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    else:
        catalog["profiles"][0]["library"] = {
            "bytes": 1,
            "files": {"missing.js": {"bytes": 1, "sha256": "f" * 64}},
        }
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(
        ValueError, match="Canvas runtime profile catalog is unavailable"
    ):
        load_profile_snapshot()


def test_oversized_catalog_is_read_only_to_runtime_manifest_ceiling(
    tmp_path, monkeypatch
):
    static = _isolated_static(tmp_path, monkeypatch)
    from tldw_chatbook.Canvas import profiles

    static.joinpath("profile-catalog.json").write_bytes(
        b"{" + b" " * profiles.PROFILE_CATALOG_BYTES + b"}"
    )

    with pytest.raises(
        ValueError, match="Canvas runtime profile catalog is unavailable"
    ):
        load_profile_snapshot()
